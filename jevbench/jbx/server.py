"""POST /v1/systemone (TypeSafe shape) and GET /health around jbx.engine.JbxEngine. One request at a time.

    python -m jbx.server --port 8090 [--backend vllm|hf] [--model /models/gemma] [--option twice_max_tokens=800 ...]

Request:  {"state": ..., "model": ..., "questions": {"decision": {"type", "instructions", "criteria"}}}
Response: {"model", "answers": {"decision": {...}}, "usage": {"input_tokens", "output_tokens"}}
A question the engine cannot answer gets HTTP 422; other errors 500.

POST /v1/raw (our experiments, tools/run_local.py --server): {"state", "questions", "options": {read-policy options and
"lora": adapter directory}} -> {"response", "raw", "latency_s"}; the options last for that request only (JbxEngine.options).
Adapters must lie under ADAPTER_ROOTS (env JBX_ADAPTER_ROOTS). Start with --option enable_lora=true to switch adapters per request.
"""
import argparse
import json
import os
import threading
import time
from http.server import BaseHTTPRequestHandler, HTTPServer

from .engine import REPO, REVISION, JbxEngine
from .layout import QuestionError

# per-request LoRA adapters are accepted only under these directories (os.pathsep-separated; none by default)
ADAPTER_ROOTS = tuple(os.path.join(os.path.abspath(p), "") for p in os.environ.get("JBX_ADAPTER_ROOTS", "").split(os.pathsep) if p) or ("\0",)


def parse_options(pairs):
    out = {}
    for p in pairs or []:
        k, _, v = p.partition("=")
        try:
            out[k] = json.loads(v)
        except json.JSONDecodeError:
            out[k] = v
    return out


def reload_engine(engine, overrides=None):
    """Re-import jbx.layout / jbx.backends / jbx.engine and rebuild the engine around the loaded vLLM model, so code
    changes take effect without giving the GPU back; `overrides` replace engine options (e.g. the submitted setting,
    {"preset": "submission"}). -> the new engine."""
    import importlib
    import sys

    from . import backends as _b, engine as _e, layout as _l

    for mod in (_l, _b, _e):
        importlib.reload(mod)
    old = engine.backend
    new_backend = sys.modules["jbx.backends"].VllmBackend(llm=old.llm, sampling_params=old.SamplingParams, enable_lora=old.lora_enabled,
                                                         max_lora_rank=old.max_lora_rank) if hasattr(old, "llm") else old
    opts = dict(engine.init_options)
    overrides = dict(overrides or {})
    if overrides.pop("preset", None) == "submission":
        opts.update(sys.modules["jbx.engine"].SUBMISSION)
    opts.update(overrides)
    return sys.modules["jbx.engine"].JbxEngine(tokenizer=engine.tok, backend_obj=new_backend, **opts)


def make_handler(engine):
    lock = threading.Lock()
    holder = {"engine": engine}

    class Handler(BaseHTTPRequestHandler):
        def _send(self, status, body):
            data = json.dumps(body, allow_nan=False).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self):
            engine = holder["engine"]
            if self.path.rstrip("/") == "/health":
                self._send(200, {"ok": True, "model": engine.model_id, "engine": engine.provenance})
            else:
                self._send(404, {"error": "not found"})

        def do_POST(self):
            engine = holder["engine"]
            route = self.path.rstrip("/")
            if route not in ("/v1/systemone", "/v1/raw", "/v1/think", "/admin/reload"):
                return self._send(404, {"error": "not found"})
            try:
                body = json.loads(self.rfile.read(int(self.headers.get("Content-Length") or 0)) or b"{}")
                if route == "/admin/reload":
                    with lock:
                        holder["engine"] = reload_engine(engine, body.get("options"))
                    return self._send(200, {"ok": True, "engine": holder["engine"].provenance})
                if route == "/v1/think":
                    opts = dict(body.get("options") or {})
                    if opts.get("lora") and not os.path.abspath(str(opts["lora"])).startswith(ADAPTER_ROOTS):
                        return self._send(422, {"error": "adapter outside the allowed roots"})
                    with lock, engine.options(**opts):
                        t0 = time.perf_counter()
                        raw = engine.think_raw(body.get("state"), body.get("question"), int(body.get("max_think", 512)), bool(body.get("twice", False)))
                        out = {"raw": raw, "latency_s": time.perf_counter() - t0}
                elif route == "/v1/raw":
                    opts = dict(body.get("options") or {})
                    if opts.get("lora") and not os.path.abspath(str(opts["lora"])).startswith(ADAPTER_ROOTS):
                        return self._send(422, {"error": "adapter outside the allowed roots"})
                    with lock, engine.options(**opts):
                        t0 = time.perf_counter()
                        resp, raw = engine.predict_raw(body.get("state"), body.get("questions"))
                        out = {"response": resp, "raw": raw, "latency_s": time.perf_counter() - t0}
                else:
                    with lock:
                        out = engine.predict(body.get("state"), body.get("questions"))
            except QuestionError as exc:
                return self._send(422, {"error": str(exc)[:300]})
            except (json.JSONDecodeError, AttributeError, TypeError, ValueError) as exc:
                return self._send(422, {"error": f"bad request: {type(exc).__name__}: {str(exc)[:200]}"})
            except Exception as exc:  # noqa: BLE001
                return self._send(500, {"error": type(exc).__name__})
            self._send(200, out)

        def log_message(self, *args):
            pass

    return Handler


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8090)
    ap.add_argument("--model", default=REPO)
    ap.add_argument("--revision", default=REVISION)
    ap.add_argument("--backend", default="vllm", choices=("vllm", "hf"))
    ap.add_argument("--option", action="append", help="engine option key=value (JSON values)")
    ap.add_argument("--preset", default="submission", choices=("submission", "none"),
                    help="start from the submitted setting (jbx.engine.SUBMISSION, as load() does) or the plain engine")
    a = ap.parse_args()
    revision = None if a.revision in ("", "none", "None") or os.path.isdir(a.model) else a.revision
    from .engine import SUBMISSION, pod_environment
    pod_environment()
    opts = {**(SUBMISSION if a.preset == "submission" else {}), **parse_options(a.option)}
    engine = JbxEngine(model=a.model, revision=revision, backend=a.backend, **opts)
    print(json.dumps({"serving": f"http://{a.host}:{a.port}/v1/systemone", "engine": engine.provenance}), flush=True)
    HTTPServer((a.host, a.port), make_handler(engine)).serve_forever()


if __name__ == "__main__":
    main()

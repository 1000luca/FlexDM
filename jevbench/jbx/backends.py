"""Log-probabilities of given token ids at the position after each prompt, from one model step.

Both backends return the same quantity: log_softmax over the whole vocabulary of Gemma's final logits after its soft cap
(30 * tanh(x / 30)), at the last prompt position, for the requested ids. That is what vLLM returns with
logprobs_mode="raw_logprobs" (jevlab.vllm_reader reads the same), so temperatures fitted on one carry over to the other
up to numerics (FP8 vs bf16).

VllmBackend: vllm.LLM in-process, prompt token ids, max_tokens=1, logprob_token_ids (at most 128 per request, wider
questions are read in chunks with the same prompt). Prefix caching shares the chat head and state across rows.
HfBackend: transformers, for machines without vLLM (for example a Jetson Thor). No prefix cache; one forward per row.
"""
import math
import os

MAX_LABEL_IDS = 128            # vLLM's MAX_LOGPROB_TOKEN_IDS
LENGTH_MARKERS = ("maximum model length", "maximum context length", "longer than the maximum")


class TooLong(ValueError):
    pass


ALLOWED_LORA_RANKS = (1, 8, 16, 32, 64, 128, 256, 320, 512)


class VllmBackend:
    """With `lora` (a directory in train/average_adapters.py --vllm-out form: Gemma-4's full-attention k_proj LoRA copied
    to v_proj), vLLM starts with LoRA enabled and every generate call carries the adapter, as jevlab.vllm_lora_reader.
    With enable_lora and no adapter (a long-lived server), set_adapter() switches adapters per request."""
    name = "vllm"

    def __init__(self, llm=None, sampling_params=None, lora=None, max_lora_rank=0, enable_lora=False, max_loras=1, **llm_kwargs):
        self.lora_request, self.lora_info, self._requests = None, None, {}
        self.lora_enabled = bool(lora) or str(enable_lora).lower() in ("1", "true", "yes", "on")
        self.max_lora_rank = 0
        if self.lora_enabled:
            rank = _adapter_rank(lora)[0] if lora else 0
            self.max_lora_rank = int(max_lora_rank) or next(r for r in ALLOWED_LORA_RANKS if r >= max(rank, 256 if not lora else rank))
            if self.max_lora_rank not in ALLOWED_LORA_RANKS or self.max_lora_rank < rank:
                raise ValueError(f"max_lora_rank must be one of {ALLOWED_LORA_RANKS} and >= the adapter rank {rank}")
            llm_kwargs.update(enable_lora=True, max_lora_rank=self.max_lora_rank, max_loras=int(max_loras))
        if llm is None:
            if not llm_kwargs.pop("deep_gemm", False):
                # block-FP8 on sm_120 picks DeepGEMM, which fails there (vLLM #60261, #51884)
                os.environ.setdefault("VLLM_USE_DEEP_GEMM", "0")
            from vllm import LLM

            llm = LLM(**llm_kwargs)
        if sampling_params is None:
            from vllm import SamplingParams as sampling_params
        self.llm, self.SamplingParams = llm, sampling_params
        if lora:
            self.set_adapter(lora)

    def set_adapter(self, path):
        """Serve the base model (None) or one adapter directory on the following reads."""
        if not path:
            self.lora_request, self.lora_info = None, None
            return
        if not self.lora_enabled:
            raise ValueError("this backend was started without LoRA (enable_lora)")
        path = os.path.abspath(os.path.expanduser(str(path)))
        rank, cfg = _adapter_rank(path)
        if rank > self.max_lora_rank:
            raise ValueError(f"adapter rank {rank} is above max_lora_rank {self.max_lora_rank}")
        if path not in self._requests:
            from vllm.lora.request import LoRARequest

            self._requests[path] = LoRARequest(f"adapter{len(self._requests) + 1}", len(self._requests) + 1, path)
        self.lora_request = self._requests[path]
        self.lora_info = {"path": path, "rank": rank, "lora_alpha": cfg.get("lora_alpha"), "max_lora_rank": self.max_lora_rank}

    def read(self, prompts, label_ids):
        """-> (one list of log-probabilities per prompt in the order of its label_ids, cached prefix tokens per prompt)."""
        found = [{} for _ in prompts]
        cached = [None] * len(prompts)
        width = max(map(len, label_ids), default=0)
        for start in range(0, width, MAX_LABEL_IDS):
            rows = [k for k, ids in enumerate(label_ids) if len(ids) > start]
            params = [self.SamplingParams(max_tokens=1, temperature=0.0, logprob_token_ids=label_ids[k][start:start + MAX_LABEL_IDS], detokenize=False)
                      for k in rows]
            try:
                extra = {"lora_request": self.lora_request} if self.lora_request is not None else {}
                outs = self.llm.generate([{"prompt_token_ids": prompts[k]} for k in rows], params, use_tqdm=False, **extra)
            except ValueError as exc:
                if any(m in str(exc) for m in LENGTH_MARKERS):
                    raise TooLong(str(exc)) from exc
                raise
            if len(outs) != len(rows):
                raise RuntimeError(f"vLLM returned {len(outs)} outputs for {len(rows)} prompts")
            for k, o in zip(rows, outs):
                got = o.prompt_token_ids if isinstance(o.prompt_token_ids, list) else list(o.prompt_token_ids)
                if got != prompts[k]:
                    raise RuntimeError("vLLM read other token ids than the ones sent")
                step = o.outputs[0].logprobs[0]
                for t in label_ids[k][start:start + MAX_LABEL_IDS]:
                    found[k][t] = float(step[t].logprob)
                if cached[k] is None:
                    cached[k] = int(o.num_cached_tokens or 0)
        return [[found[k][t] for t in ids] for k, ids in enumerate(label_ids)], cached

    def generate(self, prompt, max_tokens, stop_token_ids=()):
        """Greedy continuation of one prompt -> (generated token ids, finished on a stop token)."""
        params = self.SamplingParams(max_tokens=int(max_tokens), temperature=0.0, stop_token_ids=list(stop_token_ids), detokenize=False)
        extra = {"lora_request": self.lora_request} if self.lora_request is not None else {}
        try:
            out = self.llm.generate([{"prompt_token_ids": prompt}], [params], use_tqdm=False, **extra)[0].outputs[0]
        except ValueError as exc:
            if any(m in str(exc) for m in LENGTH_MARKERS):
                raise TooLong(str(exc)) from exc
            raise
        ids = list(out.token_ids)
        return ids, out.finish_reason == "stop"

    def runtime(self):
        info = {"backend": self.name, "lora": self.lora_info}
        try:
            import torch
            import vllm

            info.update(vllm=vllm.__version__, torch=torch.__version__, cuda=torch.version.cuda)
            from vllm.platforms import current_platform

            info["gpu"] = current_platform.get_device_name(0)
        except Exception as exc:
            info["note"] = f"runtime details unavailable: {type(exc).__name__}"
        return info

    def close(self):
        core = getattr(getattr(self.llm, "llm_engine", None), "engine_core", None)
        if core is not None and hasattr(core, "shutdown"):
            core.shutdown()


class HfBackend:
    name = "hf"

    def __init__(self, model, revision=None, dtype="bfloat16", device=None, max_len=131072, chunk_tokens=8192, attn_implementation="sdpa"):
        import torch
        from transformers import AutoModelForCausalLM

        self.torch = torch
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = AutoModelForCausalLM.from_pretrained(model, revision=revision, dtype=getattr(torch, dtype),
                                                          attn_implementation=attn_implementation).to(self.device).eval()
        self.decoder = self.model.model
        self.head = self.model.get_output_embeddings()
        cfg = self.model.config.get_text_config()
        self.softcap = getattr(cfg, "final_logit_softcapping", None)
        self.max_len, self.chunk = int(max_len), int(chunk_tokens)

    def _last_logits(self, ids):
        torch = self.torch
        from transformers import DynamicCache

        cache, out = DynamicCache(), None
        with torch.inference_mode():
            for s in range(0, len(ids), self.chunk):
                piece = ids[s:s + self.chunk]
                out = self.decoder(input_ids=torch.tensor([piece], device=self.device),
                                   attention_mask=torch.ones(1, s + len(piece), dtype=torch.long, device=self.device),
                                   past_key_values=cache, use_cache=True)
                cache = out.past_key_values
            x = self.head(out.last_hidden_state[0, -1]).float()
        if self.softcap:
            x = self.softcap * torch.tanh(x / self.softcap)
        return torch.log_softmax(x, -1)

    def read(self, prompts, label_ids):
        res = []
        for ids, labels in zip(prompts, label_ids):
            if len(ids) >= self.max_len:
                raise TooLong(f"maximum model length {self.max_len}: prompt of {len(ids)} tokens")
            lp = self._last_logits(ids)
            res.append([float(lp[t]) for t in labels])
        return res, [0] * len(prompts)

    def runtime(self):
        import transformers

        info = {"backend": self.name, "torch": self.torch.__version__, "transformers": transformers.__version__, "device": self.device}
        if self.device == "cuda":
            info.update(cuda=self.torch.version.cuda, gpu=self.torch.cuda.get_device_name())
        return info

    def close(self):
        pass


def _adapter_rank(path):
    import json

    with open(os.path.join(os.path.abspath(os.path.expanduser(str(path))), "adapter_config.json")) as f:
        cfg = json.load(f)
    return int(cfg["r"]), cfg


def softmax(scores):
    top = max(scores)
    w = [math.exp(x - top) for x in scores]
    s = sum(w)
    return [v / s for v in w]

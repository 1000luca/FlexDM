"""Letter readout of google/gemma-4-31B-it served by vLLM in-process, with the same prompts and options as jevlab.reader.

Same token ids as Reader: the rows are built by jevlab.reader.Layout (decider or header layout, labels A..J, single-token
wide labels above 10 options), and the read policy (read_twice, twice_max_tokens, reread_below, reverse_below) follows
Reader.__call__ step for step. tools/vllm_check.py checks both on CPU against Reader with a mocked vLLM.

What differs is the readout. Each row is one vLLM request with max_tokens=1 and `logprob_token_ids` set to the labels of
its options (stock vLLM SamplingParams, at most 128 ids per request, so wider questions are read in chunks of 128 with
the same prompt). vLLM returns log-probabilities of the logits after Gemma's final soft cap (30 * tanh(x / 30)), not the
raw logits Reader reads, so Reader's T(n) = 10.124 - 1.633 ln(n) does not carry over. The probabilities are a softmax of
the label log-probabilities at a temperature per question type:

    choice: T = temperature_choice + temperature_choice_log_slope * ln(n options)
    noul:   T = temperature_noul, and noul_bias is added to the "yes" score after dividing by T

The defaults (choice 4.263, noul 7.609, no slope, no bias) are wcole3/jev-brain's published fit for this model in the
header layout on vLLM (MIT, data/gemma-4-31b/header/calibration.json, fitted outside the suite). They are a starting
point; refit them on our own off-suite data. temperature_scheme="options" applies Reader's T(n) instead (for checks).

Why in-process (vllm.LLM with prompt token ids) and not an OpenAI-compatible server: the kit runs one request at a time
in one process, as the board measures it; the engine loads like Reader (`--engine jevlab.vllm_reader:VllmReader`), with
no second process, port or HTTP client to manage on a shared machine, and token ids go to vLLM unchanged (checked on
every output). vLLM's own engine core still runs in a child process; latency is the wall time of the generate calls.

All rows of one read pass of a request go to vLLM in one generate call. With prefix caching, the rows share the KV blocks
of their common prefix (chat head, question list, state); the re-read and reversed passes hit the same cache. Rows above
the token limit (max_model_len - 1) are refused as Unsupported, never truncated. max_model_len defaults to 131,072 (the
board's maintainers lift context caps to that or more for their private runs; the model's window is 262,144): with FP8
weights a request of that length needs about 23 GiB of KV cache, which the default gpu_memory_utilization=0.70 holds.

    python -m decision_index run --engine jevlab.vllm_reader:VllmReader --option layout=header \
        --rows sample-2000.jsonl.gz --out runs/vllm-header-sample2000
"""
import math
import os

from decision_index.engines.base import Engine, Unsupported

from jevlab.reader import LAYOUTS, Layout, _flag, render_question

REPO, REVISION = "google/gemma-4-31B-it", "842da3794eaa0b77d5f08bae87a17459d91ff475"
MAX_LABEL_IDS = 128            # vLLM's MAX_LOGPROB_TOKEN_IDS (vllm/sampling_params.py)
TEMPERATURE_SCHEMES = ("type", "options")
# Reader options that only mean something for the transformers engine; accepted so the same command lines work.
HF_ONLY_OPTIONS = ("device", "chunk_tokens", "cache_prefix", "cudnn_sdp", "pack_questions", "pack_tokens", "attn_block")
LENGTH_MARKERS = ("maximum model length", "maximum context length", "longer than the maximum")


def softmax(scores):
    top = max(scores)
    weights = [math.exp(x - top) for x in scores]
    total = sum(weights)
    return [w / total for w in weights]


class VllmBackend:
    """Log-probabilities of given token ids at the position after each prompt, from one vLLM decoding step.
    `llm` and `sampling_params` can be stand-ins with the same interface (tools/vllm_check.py)."""

    def __init__(self, llm=None, sampling_params=None, **llm_kwargs):
        if llm is None:
            from vllm import LLM

            llm = LLM(**llm_kwargs)
        if sampling_params is None:
            from vllm import SamplingParams as sampling_params
        self.llm, self.SamplingParams = llm, sampling_params

    def read(self, prompts, label_ids):
        """-> (one list of log-probabilities per prompt, in the order of its label_ids; prompt tokens served from the
        prefix cache per prompt). Label ids go in chunks of 128; each chunk is sent with the prompts that read one."""
        found = [{} for _ in prompts]
        cached = [None] * len(prompts)
        width = max(map(len, label_ids), default=0)
        for start in range(0, width, MAX_LABEL_IDS):
            rows = [k for k, ids in enumerate(label_ids) if len(ids) > start]
            params = [self.SamplingParams(max_tokens=1, temperature=0.0, logprob_token_ids=label_ids[k][start:start + MAX_LABEL_IDS], detokenize=False)
                      for k in rows]
            outs = self.llm.generate([{"prompt_token_ids": prompts[k]} for k in rows], params, use_tqdm=False)
            if len(outs) != len(rows):
                raise RuntimeError(f"vLLM returned {len(outs)} outputs for {len(rows)} prompts")
            for k, o in zip(rows, outs):
                got = o.prompt_token_ids if isinstance(o.prompt_token_ids, list) else list(o.prompt_token_ids)
                if got != prompts[k]:
                    raise RuntimeError("vLLM read other token ids than the ones sent (order or added tokens)")
                step = o.outputs[0].logprobs[0]
                for t in label_ids[k][start:start + MAX_LABEL_IDS]:
                    found[k][t] = float(step[t].logprob)
                if cached[k] is None:
                    cached[k] = int(o.num_cached_tokens or 0)
        return [[found[k][t] for t in ids] for k, ids in enumerate(label_ids)], cached

    def close(self):
        engine = getattr(self.llm, "llm_engine", None)
        core = getattr(engine, "engine_core", None)
        if core is not None and hasattr(core, "shutdown"):
            core.shutdown()


class VllmReader(Engine):
    name = "jevlab-vllm-reader"
    latency = ("In-process request wall time including prompt construction: one vLLM generate call per read pass of a request "
               "(all its rows together, shared prefix from vLLM's prefix cache); excludes model loading.")

    def __init__(self, model=REPO, revision=REVISION, quantization="fp8_per_block", dtype="bfloat16", max_model_len=131072, max_tokens=None,
                 gpu_memory_utilization=0.70, kv_cache_dtype="auto", attention_backend="TRITON_ATTN", prefix_caching=True,
                 prefix_cache_retention_interval=32, max_num_batched_tokens=8192, max_num_seqs=256, enforce_eager=False, language_model_only=True,
                 deep_gemm=False, seed=0, llm_options=None, layout="decider", read_twice=False, twice_max_tokens=0, reread_below=0.0, reverse_below=0.0,
                 temperature_scheme="type", temperature_choice=4.263, temperature_choice_log_slope=0.0, temperature_noul=7.609, noul_bias=0.0,
                 temperature=10.124, temperature_log_slope=-1.633, backend=None, tokenizer=None, **options):
        ignored = {k: options.pop(k) for k in HF_ONLY_OPTIONS if k in options}
        if options:
            raise ValueError(f"unknown VllmReader options {sorted(options)}")
        super().__init__(**ignored)
        if layout not in LAYOUTS:
            raise ValueError(f"layout must be one of {LAYOUTS}, not {layout!r}")
        if temperature_scheme not in TEMPERATURE_SCHEMES:
            raise ValueError(f"temperature_scheme must be one of {TEMPERATURE_SCHEMES}, not {temperature_scheme!r}")
        if isinstance(quantization, str) and quantization.strip().lower() in ("", "none", "null", "bf16", "bfloat16"):
            quantization = None
        if isinstance(attention_backend, str) and attention_backend.strip().lower() in ("", "auto", "none", "null"):
            attention_backend = None
        self.max_model_len = int(max_model_len)
        self.limit = int(max_tokens) if max_tokens else self.max_model_len - 1
        if not 0 < self.limit < self.max_model_len:
            raise ValueError(f"max_tokens must be below max_model_len ({self.max_model_len}): one position is the answer slot")
        self.header = layout == "header"
        self.read_twice = _flag(read_twice)
        self.twice_max_tokens, self.reread_below = int(twice_max_tokens), float(reread_below)
        self.reverse_below = float(reverse_below)
        self.scheme = temperature_scheme
        self.t_choice, self.t_choice_slope = float(temperature_choice), float(temperature_choice_log_slope)
        self.t_noul, self.noul_bias = float(temperature_noul), float(noul_bias)
        self.temperature, self.temperature_log_slope = float(temperature), float(temperature_log_slope)
        self.model_id = model
        if tokenizer is None:
            from transformers import AutoTokenizer

            tokenizer = AutoTokenizer.from_pretrained(model, revision=revision)
        self.tok = tokenizer
        self.layout = Layout(tokenizer)
        self.llm_kwargs = dict(model=model, revision=revision, tokenizer_revision=revision, dtype=dtype, quantization=quantization,
                               max_model_len=self.max_model_len, gpu_memory_utilization=float(gpu_memory_utilization), kv_cache_dtype=kv_cache_dtype,
                               enable_prefix_caching=_flag(prefix_caching), prefix_cache_retention_interval=prefix_cache_retention_interval,
                               max_num_batched_tokens=int(max_num_batched_tokens), max_num_seqs=int(max_num_seqs), enforce_eager=_flag(enforce_eager),
                               language_model_only=_flag(language_model_only), attention_backend=attention_backend, seed=int(seed),
                               logprobs_mode="raw_logprobs", generation_config="vllm", **dict(llm_options or {}))
        if backend is None and not _flag(deep_gemm):
            # block-FP8 on sm_120 picks DeepGEMM, which fails there; off, vLLM uses its CUTLASS block-FP8 kernel (vLLM #60261, #51884)
            os.environ.setdefault("VLLM_USE_DEEP_GEMM", "0")
        self.backend = backend if backend is not None else VllmBackend(**self.llm_kwargs)
        temps = ({"choice": f"{self.t_choice} + {self.t_choice_slope} * ln(n options)", "noul": self.t_noul, "noul_bias": self.noul_bias}
                 if self.scheme == "type" else {"all": f"max({self.temperature} + {self.temperature_log_slope} * ln(n options), 0.05)"})
        self.provenance = {
            "kind": "inference technique",
            "repo": model,
            "revision": revision,
            "serving": "vLLM offline LLM, in-process, prompt token ids",
            "vllm_options": {k: v for k, v in self.llm_kwargs.items() if k not in ("model", "revision", "tokenizer_revision")},
            "VLLM_USE_DEEP_GEMM": os.environ.get("VLLM_USE_DEEP_GEMM"),
            "max_row_tokens": self.limit,
            "read_twice": self.read_twice,
            "twice_max_tokens": self.twice_max_tokens,
            "reread_below": self.reread_below,
            "reverse_below": self.reverse_below,
            "layout": layout,
            "temperature_scheme": self.scheme,
            "temperatures": temps,
            "ignored_options": sorted(ignored),
            "policy": ("The state is preceded by the list of every question of the request ('Questions that will be asked about the context:'), "
                       "the header layout of wcole3/jev-brain. " if self.header else "")
                      + "One row per question in the model's chat template with thinking off, ending at 'Answer: ('; vLLM reads one step and returns "
                      "the log-probabilities (after the final soft cap) of the option labels (logprob_token_ids), which go through a softmax at a "
                      + ("temperature per question type. " if self.scheme == "type" else "temperature set by the number of options. ")
                      + "Options keep the request's order. Rows above the token limit are refused, never cut."
                      + (" The user turn (state, question, options) is written twice, joined by 'Read the same input again before answering:'"
                         + (f", in requests of at most {self.twice_max_tokens} tokens read once" if self.twice_max_tokens > 0 else "") + "." if self.read_twice else "")
                      + (f" Questions of the other requests whose most likely option has probability below {self.reread_below} are read again, written twice "
                         "the same way, after the first read." if self.reread_below > 0 else "")
                      + (f" A question whose most likely option then has probability below {self.reverse_below} is read once more with its options in "
                         "reverse order, and the two distributions are averaged." if self.reverse_below > 0 else ""),
        }

    def runtime(self):
        info = {}
        try:
            import torch
            import vllm

            info.update(vllm=vllm.__version__, torch=torch.__version__, cuda=torch.version.cuda)
            from vllm.platforms import current_platform

            info["gpu"] = current_platform.get_device_name(0)       # NVML, so the parent process opens no CUDA context
        except Exception as exc:                                     # the mocked backend of tools/vllm_check.py has no vLLM
            info.setdefault("note", f"runtime details unavailable: {type(exc).__name__}")
        try:
            import transformers

            info["transformers"] = transformers.__version__
        except ImportError:
            pass
        return info

    def close(self):
        if hasattr(self.backend, "close"):
            self.backend.close()

    def _read(self, ctx, pieces, ns):
        """-> [(label log-probabilities, prompt tokens served from the prefix cache)] for the rows ctx + piece."""
        try:
            scores, cached = self.backend.read([ctx + piece for piece in pieces], [self.layout.label_ids[:n] for n in ns])
        except ValueError as exc:
            if any(m in str(exc) for m in LENGTH_MARKERS):
                raise Unsupported(f"maximum context length: {exc}") from exc
            raise
        return list(zip(scores, cached))

    def _probs(self, scores, kind, reverse=False):
        """Softmax over the shown options; with reverse the options are in reverse order (yes first for noul)."""
        n = len(scores)
        if self.scheme == "options":
            t, bias = max(self.temperature + self.temperature_log_slope * math.log(n), 0.05), 0.0
        elif kind == "noul":
            t, bias = self.t_noul, self.noul_bias
        else:
            t, bias = max(self.t_choice + self.t_choice_slope * math.log(n), 0.05), 0.0
        x = [s / t for s in scores]
        if bias:
            x[0 if reverse else 1] += bias
        return softmax(x)

    def _check(self, ctx, piece):
        if len(ctx) + len(piece) > self.limit:
            raise Unsupported(f"maximum context length: a row of {len(ctx) + len(piece)} tokens exceeds the declared limit of {self.limit}")

    def __call__(self, state, questions):
        # the same steps as jevlab.reader.Reader.__call__, with log-probabilities from vLLM in place of label logits
        rendered = [(key, q["type"], *render_question(q)) for key, q in questions.items()]
        texts = [text for _, _, text, _, _ in rendered] if self.header else None
        ctx = self.layout.context(state, texts)
        again = self.layout.again(state, texts) if self.read_twice or self.reread_below > 0 else None
        rows = [(key, kind, self.layout.question(text, options), self.layout.question(text, options, again) if again else None, keys)
                for key, kind, text, options, keys in rendered]
        # a request whose rows written twice would pass the token limit is read once
        fits = [again is not None and len(ctx) + len(twice) <= self.limit for _, _, _, twice, _ in rows]
        once_tokens = len(ctx) + sum(len(once) for _, _, once, _, _ in rows)
        whole = self.read_twice and all(fits) and (self.twice_max_tokens <= 0 or once_tokens <= self.twice_max_tokens)
        first = [twice if whole else once for _, _, once, twice, _ in rows]
        for piece in first:
            self._check(ctx, piece)
        ns = [len(keys) for *_, keys in rows]
        kinds = [kind for _, kind, *_ in rows]
        read = self._read(ctx, first, ns)
        pieces = list(first)
        reread = self.reread_below > 0 and not whole
        unsure = [i for i, (scores, _) in enumerate(read) if reread and fits[i] and max(self._probs(scores, kinds[i])) < self.reread_below]
        firsts = {}
        if unsure:
            for i, again_read in zip(unsure, self._read(ctx, [rows[i][3] for i in unsure], [ns[i] for i in unsure])):
                firsts[i] = read[i][0]
                read[i] = again_read
                pieces[i] = rows[i][3]
        probs = [self._probs(scores, kind) for (scores, _), kind in zip(read, kinds)]
        flipped = {}
        if self.reverse_below > 0:
            low = [i for i, pr in enumerate(probs) if max(pr) < self.reverse_below]
            if low:
                twice = [whole or i in firsts for i in low]
                rev = [self.layout.question(rendered[i][2], rendered[i][3][::-1], again if t else None) for i, t in zip(low, twice)]
                for piece in rev:
                    self._check(ctx, piece)
                for i, (scores, _) in zip(low, self._read(ctx, rev, [ns[i] for i in low])):
                    back = self._probs(scores, kinds[i], reverse=True)[::-1]    # reversed positions back to the caller's order
                    flipped[i] = scores
                    probs[i] = [(a + b) / 2 for a, b in zip(probs[i], back)]
        answers, raw = {}, {}
        for i, ((key, kind, _, _, keys), (scores, cached)) in enumerate(zip(rows, read)):
            raw[key] = {"label_logprobs": [round(x, 6) for x in scores], "prompt_tokens": len(ctx) + len(pieces[i]), "cached_prefix_tokens": cached,
                        "read_twice": whole or i in firsts}
            if i in firsts:
                raw[key]["first_label_logprobs"] = [round(x, 6) for x in firsts[i]]
            if i in flipped:
                raw[key]["reversed_label_logprobs"] = [round(x, 6) for x in flipped[i]]
            probs_i = probs[i]
            if kind == "noul":
                answers[key] = {"type": "noul", "noul": probs_i[1]}
            else:
                answers[key] = {"type": "choice", "choice": keys[max(range(len(keys)), key=probs_i.__getitem__)], "probabilities": dict(zip(keys, probs_i))}
        return {"model": self.model_id, "answers": answers, "usage": {"input_tokens": sum(v["prompt_tokens"] for v in raw.values())}}, raw

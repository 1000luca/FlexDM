"""FlexDM for JevBench: letter readout of google/gemma-4-31B-it, typed answers in the TypeSafe /v1/systemone shape.

    from jbx.engine import load
    eng = load("/models/gemma-4-31b-it")                  # the submitted setting (SUBMISSION); JbxEngine(...) for others
    eng.predict(state, {"decision": {"type": "score", "instructions": "...", "criteria": ["...", "..."]}})
    -> {"model": ..., "answers": {"decision": {"type": "score", "score": 1.2, "probabilities": {"0": .., "1": ..}}},
        "usage": {"input_tokens": 812, "output_tokens": 0}}

Per question one row (jbx.layout, header layout); the model's log-probabilities of the option labels at 'Answer: ('
go through a softmax at a temperature per question type:

    choice: T = temperature_choice + temperature_choice_log_slope * ln(n options)
    noul:   T = temperature_noul, noul_bias added to the "yes" score after dividing by T
    score:  T = temperature_score + temperature_score_log_slope * ln(K levels)

With score_median=1 a Score distribution is then exponentially tilted (p_i * e^(l i)) so that its expected level equals
its own median. JevBench's Intelligence scores the expected level by absolute error, which the median minimises; the
tilt moves only the centre and keeps the spread that calibration scores.

Read policy. Read twice (FlexDM) only when the request read once is at most twice_max_tokens: JevBench caps the cost at
about 718 input tokens per decision on average for this base model, and usage counts both copies. twice_max_by_type
("choice=600,noul=300,score=1000") sets the limit per question type (a request uses its strictest type's limit). A state above
max_state_tokens (0 = never) is cut to its head and tail with a marker; a row still above the model's length limit is
refused with QuestionError (HTTP 422 in the pod: wrong, no tokens).

Answer shaping. JevBench scores a Noul answer as an abstention (wrong) unless P(yes) >= 0.80 or <= 0.20. With
noul_commit=c (0 = off) a P(yes) between 1-c and c moves to the edge of the band on its own side; the top answer never
changes. Score answers carry the distribution over level keys "0".."K-1" and its expected level.

diag=True also reads the other policy (once vs twice) of every question whose twice row fits and returns both
log-probability vectors in the raw record, so read policies and temperatures can be compared offline from one run.
Usage still reports only the policy's rows.
"""
import contextlib
import math

from .backends import HfBackend, TooLong, VllmBackend, softmax
from .layout import Layout, QuestionError, render_question

REPO, REVISION = "google/gemma-4-31B-it", "842da3794eaa0b77d5f08bae87a17459d91ff475"
# Per-type temperatures fitted for JevBench's objective on Jev-style data outside the JevBench public items
# (tools/fit_temps.py on shadow-v2 rows read by FlexDM, header layout, vLLM FP8; 10/28): choice minimises top-label ECE,
# noul the ECE of P(yes) after the 0.8 commit, score maximises O1S CC + calibration. FlexDM's Decision Index fit was
# choice 10.9757 - 1.7016 ln n, noul 10.732.
DEFAULTS = dict(temperature_choice=6.75, temperature_choice_log_slope=-0.5, temperature_noul=13.0, noul_bias=0.0,
                temperature_score=7.25, temperature_score_log_slope=-0.75, score_median=0.0)


def median_level(p):
    """Interpolated median of a distribution over levels 0..K-1 (the point that minimises expected absolute error)."""
    c = 0.0
    for i, v in enumerate(p):
        if c + v >= 0.5:
            return i - 0.5 + (0.5 - c) / v if v > 0 else float(i)
        c += v
    return float(len(p) - 1)


def tilt_mean(p, target):
    """Exponential tilt p_i * exp(l i) with l found by bisection so that the mean level equals target."""
    k = len(p)
    target = min(max(target, 1e-6), k - 1 - 1e-6)
    logp = [math.log(max(v, 1e-300)) for v in p]

    def tilted(l):
        q = softmax([lp + l * i for i, lp in enumerate(logp)])
        return sum(i * v for i, v in enumerate(q)), q

    lo, hi = -60.0, 60.0
    for _ in range(80):
        mid = (lo + hi) / 2
        if tilted(mid)[0] < target:
            lo = mid
        else:
            hi = mid
    return tilted((lo + hi) / 2)[1]


# The submitted setting (2026-10-29 search on three local sets, see README): header layout, a system turn,
# read twice per type up to 500 (choice) / 300 (noul) / 1,000 (score) tokens, Score centred on its median. The class
# defaults stay the plain engine; load() (the pod loader) applies this.
SYSTEM_TURN = ("You are a decision engine. Each request gives a state (the context) and one typed question: choose one option, decide yes or no, or pick a level on a scale. Base the answer only on the state and on the question's own rules, and check names, numbers and dates exactly.")
SUBMISSION = dict(system=SYSTEM_TURN, twice_max_by_type="choice=500,noul=300,score=1000", temperature_score_log_slope=-0.5,
                  score_median=True, noul_commit=0.801, layout="header", score_labels="digits", max_state_tokens=0)
# noul_commit 0.801 rather than 0.8: a committed P(yes) lands just inside the decisive region (>= 0.80 / <= 0.20) even
# if a scorer compares strictly or after float rounding (1 - 0.8 = 0.19999999999999996).


def _flag(v):
    return v.strip().lower() in ("1", "true", "yes", "on") if isinstance(v, str) else bool(v)


def _type_limits(v):
    """Per-type read-twice limits from a dict or "choice=600,noul=300,score=1000" -> {type: tokens}; empty = none."""
    if not v:
        return {}
    if isinstance(v, str):
        v = dict(kv.split("=") for kv in v.replace(" ", "").split(",") if kv)
    out = {k: int(x) for k, x in dict(v).items()}
    if set(out) - {"choice", "noul", "score"}:
        raise ValueError("twice_max_by_type keys must be choice, noul, score")
    return out


def type_probs(scores, kind, temps):
    """Softmax of the label log-probabilities at the question type's temperature."""
    n, t = len(scores), temps
    if kind == "noul":
        x = [s / t["temperature_noul"] for s in scores]
        x[1] += t["noul_bias"]
        return softmax(x)
    if kind == "score":
        temp = max(t["temperature_score"] + t["temperature_score_log_slope"] * math.log(n), 0.05)
        p = softmax([s / temp for s in scores])
        return tilt_mean(p, median_level(p)) if t.get("score_median") else p
    temp = max(t["temperature_choice"] + t["temperature_choice_log_slope"] * math.log(n), 0.05)
    return softmax([s / temp for s in scores])


def shape_answer(kind, keys, probs, noul_commit=0.0):
    """TypeSafe answer of one question; a Noul P(yes) inside (1-c, c) moves to the edge on its own side."""
    if kind == "noul":
        p, c = probs[1], noul_commit
        if c and 1 - c < p < c:
            p = c if p >= 0.5 else 1 - c
        return {"type": "noul", "noul": p}
    if kind == "score":
        return {"type": "score", "score": sum(i * p for i, p in enumerate(probs)), "probabilities": dict(zip(keys, probs))}
    top = keys[max(range(len(keys)), key=probs.__getitem__)]
    return {"type": "choice", "choice": top, "probabilities": dict(zip(keys, probs))}


class JbxEngine:
    def __init__(self, model=REPO, revision=REVISION, backend="vllm", tokenizer=None, backend_obj=None, layout="header",
                 read_twice=True, twice_max_tokens=800, max_state_tokens=0, cut_head=0.5, noul_commit=0.8, score_labels="digits",
                 max_model_len=131072, diag=False, temperature_choice=None, temperature_choice_log_slope=None, temperature_noul=None,
                 noul_bias=None, temperature_score=None, temperature_score_log_slope=None, score_median=None, served_id="flexdm-jevbench",
                 quantization="fp8_per_block", dtype="bfloat16", gpu_memory_utilization=0.70, max_num_batched_tokens=8192,
                 attention_backend="TRITON_ATTN", enforce_eager=False, seed=0, reread_below=0.0, reread_max_tokens=3000, system=None,
                 twice_max_by_type=None, **backend_options):
        if layout not in ("header", "decider"):
            raise ValueError("layout must be 'header' or 'decider'")
        self.reread_below, self.reread_max_tokens = float(reread_below), int(reread_max_tokens)
        self.init_options = dict(layout=layout, read_twice=read_twice, twice_max_tokens=twice_max_tokens, max_state_tokens=max_state_tokens,
                                 reread_below=reread_below, reread_max_tokens=reread_max_tokens,
                                 cut_head=cut_head, noul_commit=noul_commit, score_labels=score_labels, max_model_len=max_model_len, diag=diag,
                                 served_id=served_id, system=system, twice_max_by_type=twice_max_by_type)
        self.init_options.update({k: v for k, v in dict(
            temperature_choice=temperature_choice, temperature_choice_log_slope=temperature_choice_log_slope, temperature_noul=temperature_noul,
            noul_bias=noul_bias, temperature_score=temperature_score, temperature_score_log_slope=temperature_score_log_slope,
            score_median=score_median).items() if v is not None})
        self.header = layout == "header"
        self.read_twice, self.twice_max_tokens = _flag(read_twice), int(twice_max_tokens)
        self.twice_max_by_type = _type_limits(twice_max_by_type)
        self.max_state_tokens = int(max_state_tokens)
        self.cut_head = int(cut_head * self.max_state_tokens) if isinstance(cut_head, float) and cut_head < 1 else int(cut_head)
        self.noul_commit = float(noul_commit)
        if self.noul_commit and not 0.5 < self.noul_commit < 1:
            raise ValueError("noul_commit must be 0 (off) or in (0.5, 1)")
        self.max_model_len = int(max_model_len)
        self.limit = self.max_model_len - 1
        self.diag = _flag(diag)
        t = dict(DEFAULTS)
        for k, v in dict(temperature_choice=temperature_choice, temperature_choice_log_slope=temperature_choice_log_slope,
                         temperature_noul=temperature_noul, noul_bias=noul_bias, temperature_score=temperature_score,
                         temperature_score_log_slope=temperature_score_log_slope, score_median=score_median).items():
            if v is not None:
                t[k] = float(_flag(v)) if k == "score_median" else float(v)
        self.temps = t
        self.model_id = served_id
        if tokenizer is None:
            from transformers import AutoTokenizer

            tokenizer = AutoTokenizer.from_pretrained(model, revision=revision)
        self.tok = tokenizer
        self.layouts = {}
        if score_labels not in ("digits", "letters"):
            raise ValueError("score_labels must be 'digits' or 'letters'")
        self.score_labels, self.system = score_labels, system or None
        self.layout = self.layout_for(score_labels, self.system)
        if backend_obj is not None:
            self.backend = backend_obj
        elif backend == "vllm":
            if isinstance(quantization, str) and quantization.strip().lower() in ("", "none", "bf16", "bfloat16"):
                quantization = None
            self.backend = VllmBackend(model=model, revision=revision, tokenizer_revision=revision, dtype=dtype, quantization=quantization,
                                       max_model_len=self.max_model_len, gpu_memory_utilization=float(gpu_memory_utilization),
                                       enable_prefix_caching=True, max_num_batched_tokens=int(max_num_batched_tokens),
                                       attention_backend=attention_backend or None, enforce_eager=_flag(enforce_eager), seed=int(seed),
                                       language_model_only=True, logprobs_mode="raw_logprobs", generation_config="vllm", **backend_options)
        elif backend == "hf":
            self.backend = HfBackend(model, revision=revision, dtype=dtype, max_len=self.max_model_len, **backend_options)
        else:
            raise ValueError("backend must be 'vllm' or 'hf'")
        self.provenance = {
            "base": REPO, "revision": revision, "backend": getattr(self.backend, "name", "custom"), "layout": layout,
            "read_twice": self.read_twice, "twice_max_tokens": self.twice_max_tokens, "max_state_tokens": self.max_state_tokens,
            "cut_head": self.cut_head, "noul_commit": self.noul_commit, "reread_below": self.reread_below, "reread_max_tokens": self.reread_max_tokens, "score_labels": score_labels, "max_model_len": self.max_model_len,
            "system": self.system, "twice_max_by_type": self.twice_max_by_type or None,
            "temperatures": self.temps, "lora": getattr(self.backend, "lora_info", None),
        }

    # -- per-request options (long-lived server) -----------------------------------------------------------------
    POLICY = ("layout", "read_twice", "twice_max_tokens", "max_state_tokens", "cut_head", "diag", "score_labels", "noul_commit", "lora",
              "reread_below", "reread_max_tokens", "system", "twice_max_by_type", "preset") + tuple(DEFAULTS)

    def layout_for(self, score_labels, system):
        """Prompt rows for a Score label style and an optional system turn (cached; needs only the tokenizer)."""
        key = (score_labels, system or None)
        if key not in self.layouts:
            self.layouts[key] = Layout(self.tok, score_labels=score_labels, system=system or None)
        return self.layouts[key]

    @contextlib.contextmanager
    def options(self, **opts):
        """Apply read-policy options (and an adapter, `lora`) for one request, then restore the engine's own."""
        if opts.get("preset") not in (None, "submission"):
            raise ValueError("preset must be 'submission'")
        if opts.pop("preset", None):
            opts = {**SUBMISSION, **opts}
        unknown = set(opts) - set(self.POLICY)
        if unknown:
            raise ValueError(f"unknown request options {sorted(unknown)}")
        temps_saved = self.temps
        temp_opts = {k: (float(_flag(v)) if k == "score_median" else float(v)) for k, v in opts.items() if k in DEFAULTS}
        if temp_opts:
            self.temps = {**self.temps, **temp_opts}
        saved = (self.header, self.read_twice, self.twice_max_tokens, self.max_state_tokens, self.cut_head, self.diag,
                 self.score_labels, self.layout, self.noul_commit, self.reread_below, self.reread_max_tokens, self.system,
                 self.twice_max_by_type)
        lora_saved = getattr(self.backend, "lora_info", None)
        try:
            if "layout" in opts:
                if opts["layout"] not in ("header", "decider"):
                    raise ValueError("layout must be 'header' or 'decider'")
                self.header = opts["layout"] == "header"
            if "read_twice" in opts:
                self.read_twice = _flag(opts["read_twice"])
            if "twice_max_tokens" in opts:
                self.twice_max_tokens = int(opts["twice_max_tokens"])
            if "twice_max_by_type" in opts:
                self.twice_max_by_type = _type_limits(opts["twice_max_by_type"])
            if "max_state_tokens" in opts:
                self.max_state_tokens = int(opts["max_state_tokens"])
                self.cut_head = self.max_state_tokens // 2
            if "cut_head" in opts:
                self.cut_head = int(opts["cut_head"])
            if "diag" in opts:
                self.diag = _flag(opts["diag"])
            if "score_labels" in opts or "system" in opts:
                labels, system = opts.get("score_labels", self.score_labels), opts.get("system", self.system)
                if labels not in ("digits", "letters"):
                    raise ValueError("score_labels must be 'digits' or 'letters'")
                self.score_labels, self.system, self.layout = labels, system or None, self.layout_for(labels, system)
            if "noul_commit" in opts:
                self.noul_commit = float(opts["noul_commit"])
            if "reread_below" in opts:
                self.reread_below = float(opts["reread_below"])
            if "reread_max_tokens" in opts:
                self.reread_max_tokens = int(opts["reread_max_tokens"])
            if "lora" in opts:
                self.backend.set_adapter(opts["lora"])
            yield self
        finally:
            (self.header, self.read_twice, self.twice_max_tokens, self.max_state_tokens, self.cut_head, self.diag,
             self.score_labels, self.layout, self.noul_commit, self.reread_below, self.reread_max_tokens, self.system,
             self.twice_max_by_type) = saved
            if "lora" in opts:
                self.backend.set_adapter(lora_saved["path"] if lora_saved else None)
            self.temps = temps_saved

    # -- probabilities ---------------------------------------------------------------------------------------------
    def probs(self, scores, kind):
        return type_probs(scores, kind, self.temps)

    def answer(self, kind, keys, probs):
        return shape_answer(kind, keys, probs, self.noul_commit)

    # -- one request -----------------------------------------------------------------------------------------------
    def predict_raw(self, state, questions):
        """-> (TypeSafe response, raw record per question key)."""
        if not isinstance(questions, dict) or not questions:
            raise QuestionError("questions must be a non-empty map")
        rendered = [(key, *render_question(q)) for key, q in questions.items()]
        texts = [text for _, _, text, _, _ in rendered] if self.header else None
        state_text, removed = self.layout.cut_state(state, self.max_state_tokens, self.cut_head)
        ctx = self.layout.context(state_text, texts)
        again = self.layout.again(state_text, texts) if (self.read_twice or self.diag or self.reread_below > 0) else None
        rows = []
        for key, kind, text, options, keys in rendered:
            once = self.layout.question(kind, text, options)
            if len(ctx) + len(once) > self.limit:
                raise QuestionError(f"a row of {len(ctx) + len(once)} tokens exceeds the limit of {self.limit}")
            twice = self.layout.question(kind, text, options, again) if again else None
            rows.append((key, kind, keys, once, twice, self.layout.option_ids(kind, len(keys))))
        once_tokens = len(ctx) + sum(len(r[3]) for r in rows)
        fits = all(r[4] is not None and len(ctx) + len(r[4]) <= self.limit for r in rows)
        limit = min(self.twice_max_by_type.get(r[1], self.twice_max_tokens) for r in rows)   # strictest type of the request
        whole = self.read_twice and fits and (limit <= 0 or once_tokens <= limit)
        pieces = [r[4] if whole else r[3] for r in rows]
        prompts = [ctx + p for p in pieces]
        label_ids = [r[5] for r in rows]
        if self.diag and fits:                     # the other policy in the same call, for offline comparison
            prompts += [ctx + (r[3] if whole else r[4]) for r in rows]
            label_ids += label_ids
        try:
            scores, cached = self.backend.read(prompts, label_ids)
        except TooLong as exc:
            raise QuestionError(str(exc)) from None
        n = len(rows)
        reread = {}
        if self.reread_below > 0 and not whole and fits and once_tokens <= self.reread_max_tokens:
            unsure = [i for i in range(n) if max(self.probs(scores[i], rows[i][1])) < self.reread_below]
            if unsure:
                try:
                    again_scores, _ = self.backend.read([ctx + rows[i][4] for i in unsure], [rows[i][5] for i in unsure])
                except TooLong as exc:
                    raise QuestionError(str(exc)) from None
                reread = dict(zip(unsure, again_scores))
        answers, raw = {}, {}
        for i, (key, kind, keys, once, twice, _) in enumerate(rows):
            sc = reread.get(i, scores[i])
            pr = self.probs(sc, kind)
            answers[key] = self.answer(kind, keys, pr)
            raw[key] = {"type": kind, "logprobs": [round(x, 6) for x in sc], "prompt_tokens": len(prompts[i]) + (len(ctx) + len(twice) if i in reread else 0),
                        "read_twice": whole or i in reread, "cached_prefix_tokens": cached[i], "once_tokens": once_tokens, "cut_tokens": removed}
            if i in reread:
                raw[key]["first_logprobs"] = [round(x, 6) for x in scores[i]]
            if self.diag and fits:
                j = i + len(rows)
                raw[key]["other_logprobs"] = [round(x, 6) for x in scores[j]]
                raw[key]["other_prompt_tokens"] = len(prompts[j])
        usage = {"input_tokens": sum(raw[r[0]]["prompt_tokens"] for r in rows), "output_tokens": 0}
        return {"model": self.model_id, "answers": answers, "usage": usage}, raw

    def predict(self, state, questions):
        return self.predict_raw(state, questions)[0]

    def think_raw(self, state, question, max_think=512, twice=False):
        """Experiment: let the model think (greedy, at most max_think tokens), close the thought and read the option labels.
        -> raw record with the label log-probabilities after thinking, thought length and token usage."""
        kind, text, options, keys = render_question(question)
        texts = [text] if self.header else None
        state_text, removed = self.layout.cut_state(state, self.max_state_tokens, self.cut_head)
        ctx = self.layout.context(state_text, texts)
        row = ctx + self.layout.question(kind, text, options, self.layout.again(state_text, texts) if twice else None)
        prompt = self.layout.think_prompt(row)
        if len(prompt) + max_think + 8 > self.limit:
            raise QuestionError("too long to think")
        try:
            gen, finished = self.backend.generate(prompt, max_think, stop_token_ids=self.layout.think_close)
            if not gen or gen[-1] != self.layout.think_close[0]:
                gen = gen + self.layout.think_close           # cut the thought where it stands
            read = prompt + gen + self.layout.answer
            scores, _ = self.backend.read([read], [self.layout.option_ids(kind, len(keys))])
        except TooLong as exc:
            raise QuestionError(str(exc)) from None
        return {"type": kind, "logprobs": [round(x, 6) for x in scores[0]], "thought_tokens": len(gen), "finished": finished,
                "input_tokens": len(prompt) + len(self.layout.answer), "output_tokens": len(gen), "cut_tokens": removed,
                "thought": self.tok.decode(gen[:-1])}

    def runtime(self):
        return self.backend.runtime() if hasattr(self.backend, "runtime") else {}

    def close(self):
        if hasattr(self.backend, "close"):
            self.backend.close()


def pod_environment():
    """Environment for an offline, read-only pod (set only where the operator has not): no Hub calls, caches under /tmp
    when HOME is not writable, and DeepGEMM off (the FP8 kernels the submitted numbers were measured with)."""
    import os
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("VLLM_USE_DEEP_GEMM", "0")
    home = os.path.expanduser("~")
    if not os.access(home, os.W_OK):
        for var, sub in (("XDG_CACHE_HOME", ""), ("VLLM_CACHE_ROOT", "vllm"), ("TRITON_CACHE_DIR", "triton"),
                         ("TORCHINDUCTOR_CACHE_DIR", "inductor"), ("HF_HOME", "hf")):
            os.environ.setdefault(var, os.path.join("/tmp/jbx-cache", sub))


def load(model_dir, **kwargs):
    """JevBench python_inprocess loader: loader('/models/<dir>', **load_kwargs).predict(state, questions). The submitted
    setting (SUBMISSION) unless load_kwargs override it."""
    pod_environment()
    kwargs = {**SUBMISSION, **kwargs}
    kwargs.setdefault("revision", None)
    return JbxEngine(model=model_dir, **kwargs)

"""Letter readout of a stock chat model for the Decision Index. Nothing is generated.

The baseline prompt is the "chat layout" of decider-ai 1.6.0 (Apache-2.0, github.com/Mapika/decider), which is how the
board entry "Decider chat · Gemma-4-31B" reads the stock model. One row per question:

    <chat template head>Context:\n<state>\n\nQuestion: <instructions>\nOptions:\n(A) ...\n(B) ...<template tail, thinking off>Answer: (

The answer is read at the final " (": the last hidden state there is multiplied by the output-embedding rows of the
labels of the question's options, in float32, and the result goes through a softmax at a temperature set by the number
of options. These are the logits before Gemma's final soft cap (30 * tanh(x / 30)); the capped logits the model returns
squeeze every confident answer into the same narrow band below 30. Up to 10 options are labelled A..J; above that every
option gets its own single-token label (A..Z, then two-letter strings), up to 255.

By default the state is read once per request and the request's questions are read side by side after it in one pass
(pack_questions); each question sees only the state and itself, so this equals one full row per question up to bf16
rounding. With read_twice the user turn is written twice, as shisa-ai/shisa-de-2 does:

    Context:\n<state>\n\nQuestion: ...\nOptions: ...\n\nRead the same input again before answering:\nContext:\n<state>\n\nQuestion: ...\nOptions: ...

so the second copy of the state is read with the question and its options already in view. Reading twice costs about
twice the tokens, so it can be limited: twice_max_tokens reads twice only requests of at most that many tokens read once
(state plus all its questions), and reread_below re-reads, written twice, the questions of the other requests whose most
likely option has a probability below that value after the first read; the state stays cached between the two reads. A
row that written twice would pass the token limit is read once.

    python -m decision_index run --engine jevlab.reader:Reader --rows sample.jsonl.gz --out runs/reader-sample
"""
import json
import math
import string

from decision_index.engines.base import Engine, Unsupported

LETTERS = "ABCDEFGHIJ"
MAX_OPTIONS = 255
ANNOTATE_MIN = 8
NOUL_WITHOUT_INSTRUCTIONS = "Which answer fits the context?"
READ_AGAIN = "\n\nRead the same input again before answering:\n"
QUESTIONS_HEAD = "Questions that will be asked about the context:\n"
QUESTIONS_ITEM = "- "
LAYOUTS = ("decider", "header")
SENTINEL = "@@JEVLAB_USER_CONTENT@@"
THINKING_BLOCKS = (("<think>\n", "</think>\n\n"), ("<think>", "</think>\n\n"), ("<|channel>thought\n", "<channel|>"))


def _txt(v):
    return v if isinstance(v, str) else json.dumps(v, ensure_ascii=False)


def _flag(v):
    return v.strip().lower() in ("1", "true", "yes", "on") if isinstance(v, str) else bool(v)


def annotate_indices(x, min_len=ANNOTATE_MIN):
    """Write each element's position into long arrays, so a question naming `records[47]` is a lookup, not a count."""
    if isinstance(x, list):
        if len(x) >= min_len:
            return [{"_index": i, **annotate_indices(v, min_len)} if isinstance(v, dict) else {"_index": i, "value": annotate_indices(v, min_len)} for i, v in enumerate(x)]
        return [annotate_indices(v, min_len) for v in x]
    if isinstance(x, dict):
        return {k: annotate_indices(v, min_len) for k, v in x.items()}
    return x


def render_state(state):
    return state if isinstance(state, str) else json.dumps(annotate_indices(state), ensure_ascii=False)


def render_question(q):
    """-> (question text, option texts, option keys). A noul question is the two options no/yes, keyed False/True."""
    kind, crit, raw = q["type"], q.get("criteria"), q.get("instructions", "")
    if kind == "choice":
        if isinstance(crit, (list, tuple)):
            crit = {str(c): None for c in crit}
        if not isinstance(crit, dict) or not 2 <= len(crit) <= MAX_OPTIONS:
            raise Unsupported(f"options per choice: a map of 2..{MAX_OPTIONS} options is required")
        keys = list(crit)
        return _txt(raw), [k if crit[k] in (None, "") else f"{k}: {_txt(crit[k])}" for k in keys], keys
    if kind == "noul":
        crit = crit if isinstance(crit, dict) else {}
        no, yes = crit.get("false", crit.get(False)), crit.get("true", crit.get(True))
        text = NOUL_WITHOUT_INSTRUCTIONS if raw in (None, "") else _txt(raw)
        return text, ["no" if no in (None, "") else f"no: {_txt(no)}", "yes" if yes in (None, "") else f"yes: {_txt(yes)}"], [False, True]
    raise Unsupported("Unsupported question type " + str(kind))


class Layout:
    """Token ids of the prompt rows. Needs only the tokenizer."""

    def __init__(self, tok):
        if not getattr(tok, "chat_template", None):
            raise ValueError("the tokenizer has no chat template")
        self.tok = tok
        kw = {"enable_thinking": False} if "enable_thinking" in tok.chat_template else {}
        head, tail = tok.apply_chat_template([{"role": "user", "content": SENTINEL}], tokenize=False, add_generation_prompt=True, **kw).split(SENTINEL)
        for opened, closed in THINKING_BLOCKS:
            if tail.endswith(opened):
                tail += closed
        self.head_text, self.tail_text = head, tail
        self.head, self.tail = self._enc(head), self._enc(tail)
        self.answer = self._enc("Answer: (")
        self.open = self._enc("\n(")
        names = list(string.ascii_uppercase) + [a + b for a in string.ascii_uppercase for b in string.ascii_uppercase]
        single = [(n, t[0]) for n, t in ((n, self._enc(n)) for n in names) if len(t) == 1][:MAX_OPTIONS]
        self.labels, self.label_ids = [n for n, _ in single], [i for _, i in single]
        if len(self.label_ids) != MAX_OPTIONS or len(set(self.label_ids)) != MAX_OPTIONS:
            raise ValueError("the tokenizer does not have 255 distinct single-token labels")
        # the labels must be what the model would write next at the answer slot, and the slot must not merge with the template
        for n, i in single[: len(LETTERS)]:
            if self._enc("Answer: (" + n) != self.answer + [i]:
                raise ValueError(f"the label {n!r} is not one token after 'Answer: ('")
        if self._enc(tail + "Answer: (") != self.tail + self.answer:
            raise ValueError("the chat template tail merges with the answer slot")

    def _enc(self, text):
        return self.tok.encode(text, add_special_tokens=False)

    @staticmethod
    def listing(texts):
        """The header layout's list of every question of the request, placed before the state."""
        return QUESTIONS_HEAD + "\n".join(QUESTIONS_ITEM + t for t in texts) + "\n\n"

    def context(self, state, texts=None):
        """Ids up to the end of the state. With `texts` (the request's question texts) the header layout: the list of
        questions, then the state, each part encoded on its own as wcole3/jev-brain builds it."""
        ids = self.head + (self._enc(self.listing(texts)) if texts is not None else [])
        return ids + self._enc("Context:\n" + render_state(state))

    def again(self, state, texts=None):
        """Ids that start the second copy of the user turn, up to the end of its state."""
        return self._enc(READ_AGAIN + (self.listing(texts) if texts is not None else "") + "Context:\n" + render_state(state))

    def question(self, text, options, again=None):
        """Ids after the context: the question, its labelled options, the template tail and the answer slot. With `again`
        (from again()), the user turn is written a second time before the tail."""
        ids = self._enc(f"\n\nQuestion: {text}\nOptions:")
        if len(options) <= len(LETTERS):
            ids += self._enc("".join(f"\n({LETTERS[j]}) {o}" for j, o in enumerate(options)))
        else:
            for j, o in enumerate(options):
                ids += self.open + [self.label_ids[j]] + self._enc(f") {o}")
        if again:
            ids = ids + again + ids
        return ids + self.tail + self.answer


class Reader(Engine):
    name = "jevlab-reader"
    latency = "In-process request wall time including prompt construction and reuse of the state's KV cache across the questions of one request; excludes model loading."

    def __init__(self, model="google/gemma-4-31B-it", revision="842da3794eaa0b77d5f08bae87a17459d91ff475", device=None, dtype="bfloat16", temperature=10.124, temperature_log_slope=-1.633,
                 max_tokens=32768, chunk_tokens=8192, cache_prefix=False, cudnn_sdp=False, pack_questions=True, pack_tokens=4096, read_twice=False, twice_max_tokens=0, reread_below=0.0, reverse_below=0.0, layout="decider", attn_block=512, **options):
        super().__init__(**options)
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.torch = torch
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        # decider-ai reports that on Blackwell with torch 2.14 the cuDNN attention backend returns wrong output when a suffix
        # attends to a cached prefix (engine.set_attention_backend_policy). Our own runs came out identical with it on and
        # off, so this is a precaution, not a measured fix.
        if not cudnn_sdp and hasattr(torch.backends.cuda, "enable_cudnn_sdp"):
            torch.backends.cuda.enable_cudnn_sdp(False)
        self.tok = AutoTokenizer.from_pretrained(model, revision=revision)
        self.layout = Layout(self.tok)
        self.model = AutoModelForCausalLM.from_pretrained(model, revision=revision, dtype=getattr(torch, dtype)).to(self.device).eval()
        self.decoder = self.model.model                                       # everything below the output layer
        head = self.model.get_output_embeddings()
        if head.bias is not None:
            raise ValueError("the output layer has a bias; the label readout assumes it has none")
        self.label_rows = head.weight.detach()[torch.tensor(self.layout.label_ids, device=self.device)].float()
        self.attn_block = int(attn_block)
        if self.attn_block > 0:
            from jevlab.attention import use_blocked_attention
            use_blocked_attention(self.model, self.attn_block)
        self.model_id = model
        # T(n) = temperature + temperature_log_slope * ln(n options). The defaults are the fit published for the board entry
        # "Decider chat · Gemma-4-31B" (Mapika/decision-index-results README); they are here to reproduce that entry.
        self.temperature, self.temperature_log_slope = float(temperature), float(temperature_log_slope)
        self.limit = int(max_tokens)
        self.chunk = int(chunk_tokens)
        self.cache_prefix = _flag(cache_prefix)
        self.pack, self.pack_tokens = _flag(pack_questions), int(pack_tokens)
        self.read_twice = _flag(read_twice)
        self.twice_max_tokens, self.reread_below = int(twice_max_tokens), float(reread_below)
        self.reverse_below = float(reverse_below)
        if layout not in LAYOUTS:
            raise ValueError(f"layout must be one of {LAYOUTS}, not {layout!r}")
        self.header = layout == "header"
        cfg = self.model.config.get_text_config()
        self.layer_types, self.window = set(cfg.layer_types), getattr(cfg, "sliding_window", None)
        self.cache, self.cached_ids = None, []
        self.provenance = {
            "kind": "inference technique",
            "repo": model,
            "revision": revision or getattr(self.model.config, "_commit_hash", None),
            "device": self.device,
            "dtype": dtype,
            "temperature": self.temperature,
            "temperature_log_slope": self.temperature_log_slope,
            "max_row_tokens": self.limit,
            "cudnn_sdp": bool(cudnn_sdp),
            "pack_questions": self.pack,
            "read_twice": self.read_twice,
            "twice_max_tokens": self.twice_max_tokens,
            "reread_below": self.reread_below,
            "reverse_below": self.reverse_below,
            "layout": "header" if self.header else "decider",
            "attention": f"sdpa in query blocks of {self.attn_block}, each over the span of keys its mask allows" if self.attn_block > 0 else "sdpa",
            "policy": ("The state is preceded by the list of every question of the request ('Questions that will be asked about the context:'), "
                       "the header layout of wcole3/jev-brain. " if self.header else "")
                      + "One row per question in the model's chat template with thinking off, ending at 'Answer: ('; the last hidden state there times the "
                      "output-embedding rows of the option labels (float32, before the final soft cap) goes through a softmax. Options keep the request's "
                      "order. Rows above the token limit are refused, never cut."
                      + (" The user turn (state, question, options) is written twice, joined by 'Read the same input again before answering:'"
                         + (f", in requests of at most {self.twice_max_tokens} tokens read once" if self.twice_max_tokens > 0 else "") + "." if self.read_twice else "")
                      + (f" Questions of the other requests whose most likely option has probability below {self.reread_below} are read again, written twice "
                         "the same way, after the first read." if self.reread_below > 0 else "")
                      + (f" A question whose most likely option then has probability below {self.reverse_below} is read once more with its options in "
                         "reverse order, and the two distributions are averaged." if self.reverse_below > 0 else "")
                      + (" The state is read once per request; the questions of the request are read side by side after it in one pass, each seeing only "
                         "the state and itself." if self.pack else ""),
        }
        if self.pack:
            self.latency = "In-process request wall time including prompt construction; the state is read once and the request's questions are read together after it; excludes model loading."
        elif not self.cache_prefix:
            self.latency = "In-process request wall time including prompt construction; every question row is read in full; excludes model loading."

    def runtime(self):
        torch = self.torch
        import transformers

        info = {"torch": torch.__version__, "transformers": transformers.__version__, "device": self.device}
        if self.device == "cuda":
            info.update(cuda=torch.version.cuda, gpu=torch.cuda.get_device_name())
        return info

    def synchronize(self):
        if self.device == "cuda":
            self.torch.cuda.synchronize()

    def reset_cache(self):
        self.cache, self.cached_ids = None, []

    def _forward(self, ids, chunk):
        """Run `ids`, reusing the cached prefix it shares with the previous row. -> (hidden state at the last position, cached prefix length)."""
        torch = self.torch
        from transformers import DynamicCache

        p = 0
        if self.cache_prefix and self.cache is not None:
            n = min(len(ids) - 1, len(self.cached_ids))
            while p < n and ids[p] == self.cached_ids[p]:
                p += 1
        if p == 0:
            self.cache = DynamicCache()
        elif len(self.cached_ids) > p:
            self.cache.crop(p - len(self.cached_ids))
        self.cached_ids = ids[:p]
        with torch.inference_mode():
            for s in range(p, len(ids), chunk):
                piece = ids[s:s + chunk]
                out = self.decoder(input_ids=torch.tensor([piece], device=self.device), attention_mask=torch.ones(1, s + len(piece), dtype=torch.long, device=self.device),
                                   past_key_values=self.cache, use_cache=True)
                self.cache = out.past_key_values
                self.cached_ids = ids[:s + len(piece)]
        return out.last_hidden_state[0, -1], p

    def _pack_masks(self, p, lengths):
        """Masks for questions laid side by side after a cached prefix of p tokens. Each question token sees the prefix and
        the earlier tokens of its own question, at the positions it would have in its own row; on sliding-window layers only
        keys less than `window` positions back, the rule transformers applies to a full row (masking_utils.sliding_window_overlay)."""
        torch, dev = self.torch, self.device
        own = torch.repeat_interleave(torch.arange(len(lengths), device=dev), torch.tensor(lengths, device=dev))
        qpos = p + torch.cat([torch.arange(m, device=dev) for m in lengths])
        kpos = torch.cat([torch.arange(p, device=dev), qpos])
        kown = torch.cat([torch.full((p,), -1, device=dev), own])
        allowed = (kown[None, :] == -1) | ((kown[None, :] == own[:, None]) & (kpos[None, :] <= qpos[:, None]))
        masks = {}
        for t in self.layer_types:
            if t == "full_attention":
                masks[t] = allowed[None, None]
            elif t == "sliding_attention":
                masks[t] = (allowed & (qpos[:, None] - kpos[None, :] < self.window))[None, None]
            else:
                raise ValueError(f"no packed mask for layer type {t!r}")
        return masks, qpos

    def _packed(self, p, pieces):
        """Hidden states at the last token of each piece, the pieces read side by side after the cached prefix of p tokens."""
        torch = self.torch
        lengths = [len(x) for x in pieces]
        masks, qpos = self._pack_masks(p, lengths)
        ids = torch.tensor([[t for x in pieces for t in x]], device=self.device)
        ends = torch.tensor(lengths, device=self.device).cumsum(0) - 1
        try:
            with torch.inference_mode():
                out = self.decoder(input_ids=ids, attention_mask=masks, position_ids=qpos[None], past_key_values=self.cache, use_cache=True)
            return out.last_hidden_state[0, ends]
        finally:
            extra = self.cache.get_seq_length() - p
            if extra > 0:
                self.cache.crop(-extra)

    def _joint(self, ctx, pieces):
        """Hidden states at the last token of each piece, the state and all pieces read in one pass: the state causally,
        each piece after it seeing the state and itself. The state's keys and values stay in the cache afterwards."""
        torch, dev = self.torch, self.device
        from transformers import DynamicCache

        p, lengths = len(ctx), [len(x) for x in pieces]
        own = torch.cat([torch.full((p,), -1, device=dev), torch.repeat_interleave(torch.arange(len(lengths), device=dev), torch.tensor(lengths, device=dev))])
        pos = torch.cat([torch.arange(p, device=dev)] + [p + torch.arange(m, device=dev) for m in lengths])
        allowed = (pos[None, :] <= pos[:, None]) & ((own[None, :] == -1) | (own[None, :] == own[:, None]))
        masks = {}
        for t in self.layer_types:
            if t == "full_attention":
                masks[t] = allowed[None, None]
            elif t == "sliding_attention":
                masks[t] = (allowed & (pos[:, None] - pos[None, :] < self.window))[None, None]
            else:
                raise ValueError(f"no packed mask for layer type {t!r}")
        ids = torch.tensor([ctx + [t for x in pieces for t in x]], device=dev)
        ends = p + torch.tensor(lengths, device=dev).cumsum(0) - 1
        self.cache, self.cached_ids = DynamicCache(), []
        with torch.inference_mode():
            out = self.decoder(input_ids=ids, attention_mask=masks, position_ids=pos[None], past_key_values=self.cache, use_cache=True)
        extra = self.cache.get_seq_length() - p
        if extra > 0:
            self.cache.crop(-extra)
        self.cached_ids = list(ctx)
        return out.last_hidden_state[0, ends]

    def _extend(self, p, piece, chunk):
        """Hidden state at the last token of one long piece, read in chunks after the cached prefix of p tokens."""
        torch = self.torch
        try:
            with torch.inference_mode():
                for s in range(0, len(piece), chunk):
                    part = piece[s:s + chunk]
                    out = self.decoder(input_ids=torch.tensor([part], device=self.device), attention_mask=torch.ones(1, p + s + len(part), dtype=torch.long, device=self.device),
                                       past_key_values=self.cache, use_cache=True)
            return out.last_hidden_state[0, -1]
        finally:
            extra = self.cache.get_seq_length() - p
            if extra > 0:
                self.cache.crop(-extra)

    def packed_label_logits(self, ctx, pieces, ns, reuse=False):
        """float32 label logits of every question of a request: the state once, then the questions in packs of about
        pack_tokens tokens. A question longer than that is read on its own, in chunks. A request of at most pack_tokens
        tokens in all is read in one pass. With reuse, a state still cached from the first read of the same request is not
        read again."""
        budget = self.pack_tokens
        while True:
            try:
                if not reuse and not self.cache_prefix and len(ctx) + sum(map(len, pieces)) <= budget:
                    hidden = self._joint(ctx, pieces)
                    return [(self.label_rows[:n] @ h.float()).cpu() for h, n in zip(hidden, ns)], 0
                if reuse and self.cache is not None and self.cached_ids == ctx and self.cache.get_seq_length() == len(ctx):
                    p0 = len(ctx)
                else:
                    _, p0 = self._forward(ctx, self.chunk)
                p = len(ctx)
                hidden, group, size = {}, [], 0
                for i, piece in enumerate(pieces):
                    if len(piece) > budget:
                        hidden[i] = self._extend(p, piece, budget)
                        continue
                    if group and size + len(piece) > budget:
                        hidden.update(zip(group, self._packed(p, [pieces[j] for j in group])))
                        group, size = [], 0
                    group.append(i)
                    size += len(piece)
                if group:
                    hidden.update(zip(group, self._packed(p, [pieces[j] for j in group])))
                return [(self.label_rows[:n] @ hidden[i].float()).cpu() for i, n in enumerate(ns)], p0
            except self.torch.OutOfMemoryError:
                self.reset_cache()
                if self.device == "cuda":
                    self.torch.cuda.empty_cache()
                if budget <= 256:
                    raise
                budget //= 2
            except Exception:
                self.reset_cache()
                raise

    def label_logits(self, ids, n):
        """float32 logits (before the soft cap) of the first n labels at the answer slot of one row."""
        torch = self.torch
        chunk = self.chunk
        while True:
            try:
                hidden, p = self._forward(ids, chunk)
                return (self.label_rows[:n] @ hidden.float()).cpu(), p
            except torch.OutOfMemoryError:
                self.reset_cache()
                if self.device == "cuda":
                    torch.cuda.empty_cache()
                if chunk <= 512:
                    raise
                chunk //= 2
            except Exception:
                self.reset_cache()
                raise

    def _read(self, ctx, pieces, ns, reuse=False):
        """-> [(float32 label logits, cached prefix tokens)] for the pieces after ctx."""
        if self.pack:
            logits, p0 = self.packed_label_logits(ctx, pieces, ns, reuse)
            return [(x, p0 if i == 0 else len(ctx)) for i, x in enumerate(logits)]
        return [self.label_logits(ctx + piece, n) for piece, n in zip(pieces, ns)]

    def _probs(self, logits, n):
        t = max(self.temperature + self.temperature_log_slope * math.log(n), 0.05)
        return self.torch.softmax(logits.double() / t, -1).tolist()

    def __call__(self, state, questions):
        rendered = [(key, q["type"], *render_question(q)) for key, q in questions.items()]
        texts = [text for _, _, text, _, _ in rendered] if self.header else None
        ctx = self.layout.context(state, texts)
        again = self.layout.again(state, texts) if self.read_twice or self.reread_below > 0 else None
        rows = []
        for key, kind, text, options, keys in rendered:
            rows.append((key, kind, self.layout.question(text, options), self.layout.question(text, options, again) if again else None, keys))
        # a request whose rows written twice would pass the token limit is read once
        fits = [again is not None and len(ctx) + len(twice) <= self.limit for _, _, _, twice, _ in rows]
        once_tokens = len(ctx) + sum(len(once) for _, _, once, _, _ in rows)
        whole = self.read_twice and all(fits) and (self.twice_max_tokens <= 0 or once_tokens <= self.twice_max_tokens)
        first = [twice if whole else once for _, _, once, twice, _ in rows]
        for piece in first:
            if len(ctx) + len(piece) > self.limit:
                raise Unsupported(f"maximum context length: a row of {len(ctx) + len(piece)} tokens exceeds the declared limit of {self.limit}")
        ns = [len(keys) for *_, keys in rows]
        read = self._read(ctx, first, ns)
        pieces = list(first)
        reread = self.reread_below > 0 and not whole
        unsure = [i for i, ((logits, _), n) in enumerate(zip(read, ns)) if reread and fits[i] and max(self._probs(logits, n)) < self.reread_below]
        firsts = {}
        if unsure:
            again_read = self._read(ctx, [rows[i][3] for i in unsure], [ns[i] for i in unsure], reuse=True)
            for i, (logits, _) in zip(unsure, again_read):
                firsts[i] = read[i][0]
                read[i] = (logits, len(ctx))
                pieces[i] = rows[i][3]
        probs = [self._probs(logits, n) for (logits, _), n in zip(read, ns)]
        flipped = {}
        if self.reverse_below > 0:
            low = [i for i, pr in enumerate(probs) if max(pr) < self.reverse_below]
            if low:
                twice = [whole or i in firsts for i in low]
                rev = [self.layout.question(rendered[i][2], rendered[i][3][::-1], again if t else None) for i, t in zip(low, twice)]
                for i, piece in zip(low, rev):
                    if len(ctx) + len(piece) > self.limit:
                        raise Unsupported(f"maximum context length: a row of {len(ctx) + len(piece)} tokens exceeds the declared limit of {self.limit}")
                for i, (logits, _) in zip(low, self._read(ctx, rev, [ns[i] for i in low], reuse=True)):
                    back = self._probs(logits, ns[i])[::-1]                   # reversed positions back to the caller's order
                    flipped[i] = logits
                    probs[i] = [(a + b) / 2 for a, b in zip(probs[i], back)]
        answers, raw = {}, {}
        for i, ((key, kind, _, _, keys), (logits, p)) in enumerate(zip(rows, read)):
            raw[key] = {"label_logits": [round(x, 4) for x in logits.tolist()], "prompt_tokens": len(ctx) + len(pieces[i]), "cached_prefix_tokens": p,
                        "read_twice": whole or i in firsts}
            if i in firsts:
                raw[key]["first_label_logits"] = [round(x, 4) for x in firsts[i].tolist()]
            if i in flipped:
                raw[key]["reversed_label_logits"] = [round(x, 4) for x in flipped[i].tolist()]
            probs_i = probs[i]
            if kind == "noul":
                answers[key] = {"type": "noul", "noul": probs_i[1]}
            else:
                answers[key] = {"type": "choice", "choice": keys[max(range(len(keys)), key=probs_i.__getitem__)], "probabilities": dict(zip(keys, probs_i))}
        return {"model": self.model_id, "answers": answers, "usage": {"input_tokens": sum(v["prompt_tokens"] for v in raw.values())}}, raw

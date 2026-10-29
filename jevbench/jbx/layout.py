"""Prompt rows for JevBench decisions. Nothing is generated; the answer is read at 'Answer: ('.

Same rows as FlexDM (jevlab.reader.Layout, header layout) for Choice and Noul, plus Score questions, without the
Decision Index kit, so the package runs alone in a JevBench pod. tools/check_jbx.py checks that Choice and Noul rows
are token for token the rows jevlab.reader builds.

    <chat head>Questions that will be asked about the context:\n- <question>\n\nContext:\n<state>
    \n\nQuestion: <question>\nOptions:\n(A) <option>\n(B) ...<chat tail, thinking off>Answer: (

Choice options are "key: description" (or the bare key); Noul options are "no: <false>" and "yes: <true>". Score levels
are listed in order and labelled with their level digit, "(0) <level>", "(1) ...", because Score instructions number
the levels the same way ("0 = none, 1 = one ..."); above 10 levels, or with score_labels="letters", they are lettered
like Choice options. Read twice writes the user turn a second time, joined by 'Read the same input again before
answering:', so the second copy of the state is read with the question and its options already in view.

An optional system turn (`system`, off by default) goes before the user turn; the user turn and the tail are unchanged.

A state above max_state_tokens is cut to its first `head` and last tokens with a marker in between (opt-in; JevBench
allows disclosed truncation). Index annotation of long JSON arrays follows jevlab.reader.
"""
import json
import string

LETTERS = "ABCDEFGHIJ"
DIGITS = "0123456789"
MAX_OPTIONS = 255
ANNOTATE_MIN = 8
NOUL_WITHOUT_INSTRUCTIONS = "Which answer fits the context?"
READ_AGAIN = "\n\nRead the same input again before answering:\n"
QUESTIONS_HEAD = "Questions that will be asked about the context:\n"
QUESTIONS_ITEM = "- "
CUT_MARK = "\n[... {n} tokens of the middle of this context omitted ...]\n"
SENTINEL = "@@JBX_USER_CONTENT@@"
THINKING_BLOCKS = (("<think>\n", "</think>\n\n"), ("<think>", "</think>\n\n"), ("<|channel>thought\n", "<channel|>"))
TYPES = ("choice", "noul", "score")


class QuestionError(ValueError):
    """A question the engine cannot answer. The JevBench pod driver counts it as HTTP 422 (wrong, no tokens)."""


def _txt(v):
    return v if isinstance(v, str) else json.dumps(v, ensure_ascii=False)


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
    """-> (type, question text, option texts, answer keys). Noul keys are False/True; Score keys are "0".."K-1"."""
    kind, crit, raw = q.get("type"), q.get("criteria"), q.get("instructions", "")
    if kind == "choice":
        if isinstance(crit, (list, tuple)):
            crit = {str(c): None for c in crit}
        if not isinstance(crit, dict) or not 2 <= len(crit) <= MAX_OPTIONS:
            raise QuestionError(f"a choice needs a map of 2..{MAX_OPTIONS} options")
        keys = list(crit)
        return kind, _txt(raw), [k if crit[k] in (None, "") else f"{k}: {_txt(crit[k])}" for k in keys], keys
    if kind == "noul":
        crit = crit if isinstance(crit, dict) else {}
        no, yes = crit.get("false", crit.get(False)), crit.get("true", crit.get(True))
        text = NOUL_WITHOUT_INSTRUCTIONS if raw in (None, "") else _txt(raw)
        return kind, text, ["no" if no in (None, "") else f"no: {_txt(no)}", "yes" if yes in (None, "") else f"yes: {_txt(yes)}"], [False, True]
    if kind == "score":
        if isinstance(crit, dict):          # {"0": "...", "1": "..."} in level order
            try:
                crit = [crit[k] for k in sorted(crit, key=int)]
            except (TypeError, ValueError):
                raise QuestionError("score criteria must be a list of levels or a map keyed 0..K-1") from None
        if not isinstance(crit, (list, tuple)) or not 2 <= len(crit) <= MAX_OPTIONS:
            raise QuestionError(f"a score needs a list of 2..{MAX_OPTIONS} levels")
        return kind, _txt(raw), [_txt(c) for c in crit], [str(i) for i in range(len(crit))]
    raise QuestionError(f"unsupported question type {kind!r}")


class Layout:
    """Token ids of the prompt rows. Needs only the tokenizer."""

    def __init__(self, tok, score_labels="digits", system=None):
        if not getattr(tok, "chat_template", None):
            raise ValueError("the tokenizer has no chat template")
        if score_labels not in ("digits", "letters"):
            raise ValueError("score_labels must be 'digits' or 'letters'")
        self.tok, self.score_digits, self.system = tok, score_labels == "digits", system or None
        kw = {"enable_thinking": False} if "enable_thinking" in tok.chat_template else {}
        turns = ([{"role": "system", "content": self.system}] if self.system else []) + [{"role": "user", "content": SENTINEL}]
        head, tail = tok.apply_chat_template(turns, tokenize=False, add_generation_prompt=True, **kw).split(SENTINEL)
        if self.system and self.system not in head:
            raise ValueError("the chat template dropped the system turn")
        for opened, closed in THINKING_BLOCKS:
            if tail.endswith(opened):
                tail += closed
        self.head, self.tail = self._enc(head), self._enc(tail)
        self.answer = self._enc("Answer: (")
        self.open = self._enc("\n(")
        names = list(string.ascii_uppercase) + [a + b for a in string.ascii_uppercase for b in string.ascii_uppercase]
        single = [(n, t[0]) for n, t in ((n, self._enc(n)) for n in names) if len(t) == 1][:MAX_OPTIONS]
        self.labels, self.label_ids = [n for n, _ in single], [i for _, i in single]
        if len(self.label_ids) != MAX_OPTIONS or len(set(self.label_ids)) != MAX_OPTIONS:
            raise ValueError("the tokenizer does not have 255 distinct single-token labels")
        self.digit_ids = []
        for d in DIGITS:
            t = self._enc(d)
            if len(t) != 1 or self._enc("Answer: (" + d) != self.answer + t:
                raise ValueError(f"the digit {d!r} is not one token after 'Answer: ('")
            self.digit_ids.append(t[0])
        for n, i in single[: len(LETTERS)]:
            if self._enc("Answer: (" + n) != self.answer + [i]:
                raise ValueError(f"the label {n!r} is not one token after 'Answer: ('")
        if self._enc(tail + "Answer: (") != self.tail + self.answer:
            raise ValueError("the chat template tail merges with the answer slot")
        # thinking on: the template adds a system turn with the think flag and leaves the model turn open; the model
        # writes "<|channel>thought\n...<channel|>" and then its answer
        self.think_head = self.think_tail = self.think_close = None
        if "enable_thinking" in tok.chat_template:
            th_head, th_tail = tok.apply_chat_template([{"role": "user", "content": SENTINEL}], tokenize=False, add_generation_prompt=True,
                                                       enable_thinking=True).split(SENTINEL)
            close = self._enc("<channel|>")
            if th_head != head and len(close) == 1:
                self.think_head, self.think_tail, self.think_close = self._enc(th_head), self._enc(th_tail), close

    def _enc(self, text):
        return self.tok.encode(text, add_special_tokens=False)

    def cut_state(self, state, max_state_tokens, head):
        """-> (state text, tokens removed). Keeps the first `head` and the last (max_state_tokens - head) tokens."""
        text = render_state(state)
        if max_state_tokens <= 0:
            return text, 0
        ids = self._enc(text)
        if len(ids) <= max_state_tokens:
            return text, 0
        h = min(int(head), max_state_tokens)
        t = max_state_tokens - h
        removed = len(ids) - h - t
        return self.tok.decode(ids[:h]) + CUT_MARK.format(n=removed) + (self.tok.decode(ids[-t:]) if t else ""), removed

    @staticmethod
    def listing(texts):
        return QUESTIONS_HEAD + "\n".join(QUESTIONS_ITEM + t for t in texts) + "\n\n"

    def context(self, state_text, texts=None):
        """Ids up to the end of the state; with `texts` (the request's question texts) the header layout."""
        ids = self.head + (self._enc(self.listing(texts)) if texts is not None else [])
        return ids + self._enc("Context:\n" + state_text)

    def again(self, state_text, texts=None):
        return self._enc(READ_AGAIN + (self.listing(texts) if texts is not None else "") + "Context:\n" + state_text)

    def option_ids(self, kind, n):
        """Label token ids read at the answer slot, in option order."""
        if kind == "score" and self.score_digits and n <= len(DIGITS):
            return self.digit_ids[:n]
        return self.label_ids[:n]

    def think_prompt(self, row_ids):
        """A thinking-mode prompt with the same user turn as a read row (head + user turn + tail + answer slot)."""
        if self.think_head is None:
            raise ValueError("this chat template has no thinking mode")
        user = row_ids[len(self.head): len(row_ids) - len(self.tail) - len(self.answer)]
        return self.think_head + user + self.think_tail

    def question(self, kind, text, options, again=None):
        """Ids after the context: the question, its labelled options, the chat tail and the answer slot."""
        ids = self._enc(f"\n\nQuestion: {text}\nOptions:")
        if kind == "score" and self.score_digits and len(options) <= len(DIGITS):
            ids += self._enc("".join(f"\n({DIGITS[j]}) {o}" for j, o in enumerate(options)))
        elif len(options) <= len(LETTERS):
            ids += self._enc("".join(f"\n({LETTERS[j]}) {o}" for j, o in enumerate(options)))
        else:
            for j, o in enumerate(options):
                ids += self.open + [self.label_ids[j]] + self._enc(f") {o}")
        if again:
            ids = ids + again + ids
        return ids + self.tail + self.answer

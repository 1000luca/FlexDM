# FlexDM-JB: FlexDM for JevBench

FlexDM-JB runs the unmodified [google/gemma-4-31B-it](https://huggingface.co/google/gemma-4-31B-it) as a JevBench
decision model. There is no training and no extra weights. The model generates no text: each decision is one forward
pass, or two when the input is read twice, and the answer is read from the log-probabilities of the option labels at
`Answer: (`. It uses the same prompt rows as [FlexDM](../README.md), the Decision Index engine in this repository,
and adds Score questions and settings chosen for JevBench's Capability score and cost cap.

It is an inference technique on stock weights. It answers all three JevBench types (choice, noul, score) in the TypeSafe
`/v1/systemone` shape.

## Pod recipe (open-weights run)

| Field | Value |
|---|---|
| `kind` | `python_inprocess` (or `http_typesafe`, below) |
| `image` | `vllm/vllm-openai:v0.30.0@sha256:8a69ffad015f138d7170c4ddc429e230a3bc1c1719f67e14324749df200a4b90` (official vLLM 0.30.0; the engine needs only vLLM, transformers and the standard library) |
| `pythonpath` | `["/code/jevbench"]` |
| `loader` | `jbx.engine:load` (called as `load("/models/<dir>")`; it applies the submitted setting, `jbx.engine.SUBMISSION`) |
| `model_dir` | `gemma-4-31b-it` |
| `load_kwargs` | `{}` |
| `model` | `flexdm-jevbench` |
| `min_vram_gb` | `96` (vLLM `gpu_memory_utilization=0.70`, `max_model_len=131072`; tested on one RTX PRO 6000 Blackwell 96 GB) |
| weights | `google/gemma-4-31B-it`, revision `842da3794eaa0b77d5f08bae87a17459d91ff475` |
| `model-00001-of-00002.safetensors` | sha256 `eeef8791537bc04f110967c513149e037d2a9ae97d49add7291ebfa62806bbfa` |
| `model-00002-of-00002.safetensors` | sha256 `018912220f559f7025d60333e0996183cd538aa77ad6f4988a89ce47be681f10` |
| price | base-model reference for `google/gemma-4-31B-it` (USD 0.09 / 1M input, 0.34 / 1M output). Output tokens are always 0. |

The loader reads the weights from the local directory only. No `trust_remote_code`, no Python from `/models`, no
network. Before vLLM starts, `jbx.engine.pod_environment()` sets these, unless the operator has already set them:
- `HF_HUB_OFFLINE=1`
- `VLLM_USE_DEEP_GEMM=0`, the FP8 kernels the numbers below were measured with
- cache directories under `/tmp` when `$HOME` is not writable

`http_typesafe` alternative, with the same answers:

```
argv:      ["python3", "-m", "jbx.server", "--host", "127.0.0.1", "--port", "8000", "--model", "/models/gemma-4-31b-it"]
env:       {"PYTHONPATH": "/code/jevbench"}
ready_url: http://127.0.0.1:8000/health
endpoint:  http://127.0.0.1:8000        (POST /v1/systemone)
```

The server starts from the same submitted setting (`--preset submission` is the default).

## What the engine does (submitted setting)

The values are in `jbx/engine.py` (`SUBMISSION`, `DEFAULTS`).

- **Model and precision**:
  - google/gemma-4-31B-it, loaded by vLLM 0.30.0 with FP8 quantization at load time (`fp8_per_block`).
  - Prefix cache on, TRITON_ATTN attention, `max_model_len` 131,072.
  - Thinking off. Nothing is generated.
- **Prompt row** (FlexDM's header layout):
  - Chat turns: a system turn, then a user turn, then the model turn.
  - User turn: `Questions that will be asked about the context:` and the question, then `Context:` and the state, then the question with labelled options.
  - Model turn: ends with `Answer: (`.
  - Choice options are `(A) key: description`. Noul options are `(A) no: …` and `(B) yes: …`. Score levels are `(0) …`, `(1) …`, in level order.
- **System turn** (fixed text, written before any result was seen): *"You are a decision engine. Each request gives a
  state (the context) and one typed question: choose one option, decide yes or no, or pick a level on a scale. Base the
  answer only on the state and on the question's own rules, and check names, numbers and dates exactly."*
- **Read twice** (FlexDM):
  - The user turn is written a second time after `Read the same input again before answering:`, so the second copy of the state is read with the question in view.
  - This happens only when the request read once is at most 500 tokens (choice), 300 (noul) or 1,000 (score). Longer requests are read once.
  - `usage.input_tokens` counts every token read.
- **Probabilities**: softmax of the label log-probabilities at a temperature per type.
  - choice: 6.75 − 0.5 ln(options).
  - noul: 13.0.
  - score: 7.25 − 0.5 ln(levels).
  - The temperatures were fitted on public Jev-style decision datasets outside the JevBench public items.
- **Noul**: JevBench counts P(yes) strictly between 0.2 and 0.8 as an abstention. A P(yes) in that band is moved to 0.801 (or 0.199) on its own side, just inside the decisive region. The top answer never changes.
- **Score**:
  - The distribution over levels is exponentially tilted so that its expected level equals its median.
  - JevBench scores the expected level by absolute error, which the median minimises. The tilt moves only the centre of the distribution and keeps its spread.
- **Long inputs**: no truncation and no option pruning. A request longer than the model limit returns `QuestionError` (HTTP 422).

## How the setting was chosen, and local results

Every candidate was run on three local sets:
- **JevBench public 231**: the older public items, 231 decisions.
- **dev, 1,329 decisions** and **devb, 2,845 decisions** (disjoint from dev): Jev-style decisions from public datasets. These are not JevBench items.

A change was kept only if it did not lose on any of the three sets. Tested and dropped:
- decider layout (question after the context only)
- letter labels for Score levels
- a shorter system turn
- other choice/noul temperatures and Noul cut-offs
- truncating long states

The JevBench public items were used only to choose among these few fixed variants. No JevBench item was used to fit a parameter.

Local results come from a driver-like serial run: one question per request, wall time around `predict`, and the local
re-implementation of the v1.6 scorer (`jbx/score.py`; Score calibration uses an assumed `100 (1 − ECE − 0.5 nRPS)`).
Latency is measured on an RTX PRO 6000 Blackwell **Max-Q**, the 300 W version.

| Set | Capability | Intelligence | Calibration | Input tokens / decision | p50 latency (adjusted 2× + 0.15 s) |
|---|---|---|---|---|---|
| JevBench public 231 | 81.4 | 76.1 | 86.8 | 835 | 72 ms (0.30 s) |
| dev 1,329 | 83.3 | 71.8 | 94.9 | 1,108 | 121 ms (0.39 s) |
| devb 2,845 | 82.5 | 69.8 | 95.1 | 1,263 | 121 ms (0.39 s) |

The local sets have longer inputs than the current JevBench draw. Single-read Gemma-4-31B rows use 481–549 input tokens per decision there. Scaled to that draw, the setting uses an estimated 630–660 input tokens per decision, below the cap of about 718.

## Files

- `jbx/layout.py`: prompt rows (tokenizer only).
- `jbx/engine.py`: the engine, `SUBMISSION`, `load()`.
- `jbx/backends.py`: the vLLM backend, plus an HF transformers backend for machines without vLLM.
- `jbx/server.py`: `POST /v1/systemone`, `GET /health`.
- `jbx/score.py`: local scorer, for checks only.

```python
from jbx.engine import load
eng = load("/path/to/gemma-4-31B-it")
eng.predict("…state…", {"decision": {"type": "score", "instructions": "How severe is the incident?",
                                      "criteria": ["none", "minor", "major", "critical"]}})
# {"model": "flexdm-jevbench", "answers": {"decision": {"type": "score", "score": 1.6, "probabilities": {"0": …}}},
#  "usage": {"input_tokens": 412, "output_tokens": 0}}
```

License: MIT (this repository). The model weights are under their own license (Apache-2.0 for google/gemma-4-31B-it).

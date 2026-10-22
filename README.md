# FlexDM

**Stock Gemma-4-31B-it, no training, read better.**

FlexDM is an inference-technique entry for the [Jev Decision Index](https://huggingface.co/spaces/multimodalart/jev-decision-index).
It reads the unmodified `google/gemma-4-31B-it` checkpoint with two prompt changes, questions first and a second read of
short inputs, and returns a probability for every option in a single prefill. It generates no text, and it has no
trained weights, adapters or heads.

## Results

Decision Index 0.4.1, board of 2026-10-21 ([PR #246](https://github.com/apolinario/decision-index/pull/246)):

| | FlexDM |
|---|---|
| **Full score** | **67.56**, 8th of 223 models |
| Among entries that are not full fine-tunes (inference techniques, LoRA, LoRA + head) | **tied 1st** with AJev lm6q (LoRA, 67.57) |
| Among the 26 inference techniques | **1st** (next: Aleria Judge 1.1.1, 66.67) |
| Private tests, same skills (60% of Full) | 62.98 |
| Private tests, new domains (40% of Full) | **58.52**: highest of any entry that is not a full fine-tune, 4th overall |
| Public 0.3 index (shown on the board, not counted since 0.4) | 63.63 |
| Maintainers' rerun with our code | 100.00% decision agreement on 831 rows (FP8 bar: 96%) |
| Latency, board measurement (one RTX PRO 6000, one request at a time) | median 109 ms, mean 301 ms, p95 1,106 ms |

FlexDM is the highest-ranked entry on the board that uses no trained weights. The board counts Full scores less than 0.9
apart as a tie. Full results of the public run are in
[1000luca/FlexDM-decision-index](https://huggingface.co/datasets/1000luca/FlexDM-decision-index).

### Where the gain comes from

Three board entries run the same stock Gemma-4-31B-it weights with different reading methods:

| Entry | How it reads | Same skills | New domains | Full |
|---|---|---:|---:|---:|
| Decider chat | state first, then the question | 56.88 | 54.01 | 61.63 |
| jev-brain | the request's questions listed before the state | 61.56 | 54.29 | 64.61 |
| **FlexDM** | questions first, and inputs up to 3,000 tokens read twice | **62.98** | **58.52** | **67.56** |

The three engines also differ in serving details, so the steps are indicative, not a controlled ablation. In our own
paired test on a 2,000-row public sample, reading twice added +1.79 to the sample index (90% interval +0.17 to +3.40).

## How it works

Gemma is a causal model: a token can only attend to the tokens before it. If the state comes first, the model reads it
without knowing what will be asked. FlexDM changes the order of the input and lets the model read short inputs a
second time.

```
<chat head>Questions that will be asked about the context:      1. questions first
- <question 1>
- <question 2>

Context:
<state>

Question: <question i>
Options:
(A) <option>
(B) <option>

Read the same input again before answering:                      2. second copy, for inputs of up to 3,000 tokens
<question list, state, question and options again>
<chat tail, thinking off>Answer: (                               3. read the option labels here
```

1. **Questions first** (the header layout of [jev-brain](https://github.com/wcole3/jev-brain)). Every row starts with
   the list of all the request's questions, so the state is read with the questions already in view.
2. **Read twice, for short inputs.** If a request read once has at most 3,000 tokens, the whole user turn is written
   a second time (as in shisa-ai/shisa-de-2). Every token of the second copy can see the complete first copy, questions
   and options included. On our public sample this covers 15.7% of questions. Longer requests are read once, because
   the second copy cannot be shared between questions and would cost far more time there.
3. **Label readout, no generation.** The row ends at `Answer: (`. vLLM runs one step and returns the log-probabilities
   of the option labels (A, B, C, ...; single-token labels up to 255 options). A yes/no (`noul`) question has the
   options `no` and `yes`, and its answer is P(yes).
4. **Calibrated probabilities.** A softmax at a temperature per question type turns label scores into probabilities:
   choice `T = 10.9757 - 1.7016 * ln(number of options)`, noul `T = 10.732`. The temperatures were fitted on our own
   data from outside the suite. They change confidence, never the top answer (board ECE 0.044).
5. **Fast serving.** vLLM 0.30.0, FP8 weights quantized at load, prefix caching. All rows of a request share one
   cached prefix (chat head, question list, state), so a request with many questions pays for its state once.

Every benchmark uses this same rendering. There are no per-benchmark prompts, rules or post-processing.

## Quick start

Requirements: Linux x86_64, Python 3.12, one GPU with about 70 GB free (we used an RTX PRO 6000 Blackwell 96 GB), an
NVIDIA driver for CUDA 13.0 or newer, and [uv](https://docs.astral.sh/uv/).

```sh
git clone https://github.com/1000luca/FlexDM && cd FlexDM
uv venv .venv --python 3.12 && source .venv/bin/activate
uv pip install -r requirements.txt --torch-backend=cu130
hf download google/gemma-4-31B-it --revision 842da3794eaa0b77d5f08bae87a17459d91ff475   # gated: accept the Gemma terms first
```

Build the 0.3 suite with the [kit](https://github.com/apolinario/decision-index) (`suite rebuild`, then `suite import`;
the suite is not redistributed here). Then run it with the submitted settings:

```sh
./run_flexdm.sh runs/flexdm                                  # the public 0.3 suite in ./suite-0.3
./run_flexdm.sh runs/flexdm-other --rows other-rows.jsonl.gz  # any other rows file
```

`run_flexdm.sh` picks the GPU with the most free memory (it refuses to start below 72,000 MiB), sets the environment
and calls the kit's runner. Score with
`CUDA_VISIBLE_DEVICES="" python -m decision_index score --edition 0.3 --suite-dir suite-0.3 --engine jevlab.vllm_reader:VllmReader --results runs/flexdm/results.jsonl`.

## Reproducing the submitted run

Tag [`v1.0`](https://github.com/1000luca/FlexDM/tree/v1.0) holds the code of the submitted run. Later commits on `main`
change documentation only. The engine files are byte-identical to the ones the run loaded (sha256 recorded in the
run's `driver.json`):

    e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855  jevlab/__init__.py
    9ee4dbf437531d6093fa52bbcab3e9143f62e451c1450714fe1cb8411351e03c  jevlab/reader.py
    d69770c5e1e947b3b7aaedc1ed0d435fa441dc2ff08863d4bc3cf88ff22c71b8  jevlab/vllm_reader.py

| | |
|---|---|
| Engine | `jevlab.vllm_reader:VllmReader`, loaded in-process by the kit's unmodified runner |
| Options | `layout=header read_twice=true twice_max_tokens=3000 temperature_choice=10.9757 temperature_choice_log_slope=-1.7016 temperature_noul=10.732 noul_bias=0` (everything else at engine defaults) |
| Model | `google/gemma-4-31B-it` @ `842da3794eaa0b77d5f08bae87a17459d91ff475`, tokenizer at the same revision |
| Kit | `apolinario/decision-index` @ `f8552a13b643d39f61185f5951bb5fed16bedf51`, frozen 0.3 suite (`suite verify` matches) |
| Key packages | vllm 0.30.0, torch 2.13.0+cu130, transformers 5.16.1, tokenizers 0.23.1, flashinfer-python 0.6.18.post1, triton 3.7.1, ninja 1.13.2, huggingface-hub 1.33.0 (full freeze in `requirements.txt`) |
| Run | one process, one request at a time, `--compact`; 140,178 of 140,178 scored requests answered, none unsupported, no errors; the 442 rows the scorer excludes were run too |

Run it by hand instead of `run_flexdm.sh`:

```sh
source .venv/bin/activate            # .venv/bin must come first on PATH: FlashInfer's first-run JIT calls its ninja
export PYTHONPATH=$PWD HF_HUB_OFFLINE=1 HF_HUB_DISABLE_XET=1 CUDA_VISIBLE_DEVICES=0
python -m decision_index run --engine jevlab.vllm_reader:VllmReader --edition 0.3 --suite-dir suite-0.3 \
    --option layout=header --option read_twice=true --option twice_max_tokens=3000 \
    --option temperature_choice=10.9757 --option temperature_choice_log_slope=-1.7016 \
    --option temperature_noul=10.732 --option noul_bias=0 \
    --compact --out runs/flexdm
```

The engine stops on an unknown option rather than ignoring it.

<details>
<summary>vLLM settings (as recorded in <code>environment.json</code>)</summary>

| setting | value |
|---|---|
| quantization / dtype | `fp8_per_block` (online at load: weights in 128x128 blocks, activations 1x128) / `bfloat16` |
| max_model_len / largest accepted row | 131,072 / 131,071 tokens |
| gpu_memory_utilization | 0.70 |
| kv_cache_dtype | `auto` (bf16) |
| attention_backend | `TRITON_ATTN` |
| enable_prefix_caching / prefix_cache_retention_interval | true / 32 |
| max_num_batched_tokens / max_num_seqs | 8192 / 256 |
| enforce_eager / language_model_only / seed | false / true / 0 |
| logprobs_mode / generation_config | `raw_logprobs` (after Gemma's final soft cap) / `vllm` |
| environment set by the engine | `VLLM_USE_DEEP_GEMM=0` (block FP8 with DeepGEMM fails on sm_120; vLLM uses its CUTLASS kernel) |

Questions with more than 128 options are read in chunks of 128 labels with the same prompt. A state that is not a
string is written as JSON (`ensure_ascii=False`), and arrays of 8 or more elements get `_index` positions, as in decider.

</details>

<details>
<summary>Limits, memory and latency on our hardware</summary>

- **No truncation.** Rows of up to 131,071 tokens and 2-255 options per question are accepted. Anything larger is
  recorded as unsupported, never cut. No public 0.3 request reaches either limit (the longest is about 54,000 tokens).
  The 3,000-token cut only decides whether a request is written once or twice. Every request is read in full.
- **Memory.** vLLM reserves 70% of the card at start (about 68,500 MiB on a 96 GB card) and stops if that much is not
  free: about 32 GiB of FP8 weights plus about 32 GB of KV cache. The first start reads the 62 GB bf16 checkpoint,
  quantizes it and compiles kernels (5-15 minutes); later starts take about a minute.
- **Latency on our card**, an RTX PRO 6000 Blackwell **Max-Q** (300 W), over the 140,178 scored rows of the full public
  run: median 146 ms, mean 403 ms, p80 534 ms. Engine time was 15.7 hours for the scored rows plus 6.9 hours for the
  442 excluded rows. Requests of 33,000-54,000 tokens (some
  ToolRet and BRIGHT rows, mostly excluded from scoring) take tens of seconds each.

</details>

<details>
<summary>Files</summary>

| file | purpose |
|---|---|
| `jevlab/vllm_reader.py` | the submitted engine, `VllmReader` |
| `jevlab/reader.py` | the prompt layout (`Layout`, `render_question`), shared with our transformers engine |
| `jevlab/__init__.py` | empty package marker |
| `requirements.txt` | the exact environment of the run (frozen), kit included |
| `run_flexdm.sh` | picks a GPU, sets the environment, runs the kit with the submitted options |

`jevlab/reader.py` also defines `Reader`, a transformers engine we used for experiments. It is not part of this entry,
and with its default `attn_block=512` it imports `jevlab/attention.py`, which this repository leaves out. The vLLM path
never imports it.

</details>

## Disclosures

- **No training and no Jev outputs.** These are the stock weights, quantized to FP8 at load, so the quantized
  reproduction bar applies. No Jev API answers were used anywhere.
- **Choices made on public data.** We adopted the header layout after jev-brain's public result. We then chose it, and
  the 3,000-token read-twice cut, with runs on a 2,000-row stratified sample of the public 0.3 suite:
  - sample index 55.39 with the decider chat layout and 60.98 with the header layout;
  - 62.82 reading twice at 3,000 tokens and 62.92 at 6,000 tokens (slower, so we kept 3,000).

  We also checked the choices on our own shadow index, built from public datasets outside the suite (66.25 header once,
  67.25 with read twice, on our own scale).
- **Tried and dropped:** re-reading low-confidence questions with reversed options, averaging two option orders,
  an option-position bias correction, and mixing layouts. None of them helped.
- **Temperatures** were fitted on 2,871 off-suite requests from 16 public datasets, disjoint from the shadow evaluation
  rows, with no suite rows. Checked on a held-out half, ECE went from 0.123 to 0.030. The fit's noul bias (-1.03) was
  set to 0, because all of its noul questions came from a single dataset. The engine's built-in defaults are
  jev-brain's published fit, and this entry overrides them.

## Credits

- Header layout and the per-type temperature scheme: [wcole3/jev-brain](https://github.com/wcole3/jev-brain) (MIT,
  Copyright 2026 Will Cole). Our token ids match its layout exactly.
- Chat layout (`Context:` / `Question:` / `Options:` / `Answer: (`) and `_index` for long arrays: decider-ai 1.6.0,
  [Mapika/decider](https://github.com/Mapika/decider) (Apache-2.0).
- Reading the user turn twice: shisa-ai/shisa-de-2.
- Evaluation kit: [apolinario/decision-index](https://github.com/apolinario/decision-index). Model: Google's Gemma 4,
  under the Gemma terms of use, not redistributed here.

## License

MIT for the code in this repository (see `LICENSE`).

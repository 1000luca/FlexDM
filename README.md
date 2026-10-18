# FlexDM: letter readout of stock Gemma-4-31B-it on vLLM

FlexDM is an **inference technique** entry for the [Decision Index](https://github.com/apolinario/decision-index)
(edition 0.3, Layer 1). It serves the stock checkpoint `google/gemma-4-31B-it` @
`842da3794eaa0b77d5f08bae87a17459d91ff475` through our own engine. Nothing is trained and no weights of our own are
published. The engine quantizes the bf16 checkpoint to FP8 when it loads, and it generates no text: each question is a
single prefill that ends at `Answer: (`, and the answer is a softmax over the option labels' log-probabilities.

- Engine: `jevlab.vllm_reader:VllmReader`, a `decision_index.engines.Engine` subclass that the kit's own runner loads
  in-process (vLLM 0.30.0 offline `LLM`, prompt token ids, one request at a time).
- Options of the submitted run: `layout=header read_twice=true twice_max_tokens=3000 temperature_choice=10.9757
  temperature_choice_log_slope=-1.7016 temperature_noul=10.732 noul_bias=0`. All other settings are engine defaults.

## How it answers

1. **Header layout.** Each request becomes one row per question, in Gemma's chat template with thinking off. The
   row starts with the list of all the request's questions, then the state, then the question and its options:

       <chat head>Questions that will be asked about the context:
       - <question 1>
       - <question 2>

       Context:
       <state>

       Question: <question i>
       Options:
       (A) <option>
       (B) <option><chat tail, thinking off>Answer: (

   Each part is tokenized on its own, as wcole3/jev-brain builds it. A state that is not a string is written as JSON
   (`json.dumps`, `ensure_ascii=False`), and arrays of 8 or more elements get their positions written in (`_index`),
   following decider. Options keep the
   request's order. Questions with up to 10 options use labels A-J. Wider questions give each option its own
   single-token label (A-Z, then two-letter labels), up to 255. A yes/no (`noul`) question has the two options `no`
   and `yes`, and its answer is P(yes). Every benchmark uses this same rendering: there are no per-benchmark prompts,
   rules or post-processing.
2. **Label readout at `Answer: (`.** vLLM runs one step (`max_tokens=1`, `logprob_token_ids` = the labels of the
   shown options, `logprobs_mode=raw_logprobs`) and returns the log-probabilities of the labels after Gemma's final
   soft cap. Questions with more than 128 options are read in chunks of 128 labels, with the same prompt. All rows of
   one request go to vLLM in a single `generate` call, and prefix caching shares their common prefix (chat head,
   question list, state).
3. **Read twice, for requests of up to 3,000 tokens.** If a request read once has at most 3,000 tokens (the shared
   prefix plus every question row), the user turn is written twice in each of its rows, following
   shisa-ai/shisa-de-2. The second copy (question list, state, question and options) comes after
   `\n\nRead the same input again before answering:\n`. Larger requests are read once. On our 2,000-row public
   sample this applied to 15.7% of questions.
4. **Temperatures per question type, fitted on off-suite calibration data.** choice:
   `T = 10.9757 - 1.7016 * ln(number of options)` (floored at 0.05). noul: `T = 10.732`, with no bias. A temperature
   only rescales the probabilities: it never changes the top answer.
5. **FP8 vLLM serving.** vLLM 0.30.0 with online `fp8_per_block` quantization at load (weights in 128x128 blocks,
   activations in 1x128), a bf16 KV cache, the `TRITON_ATTN` attention backend, prefix caching and
   `max_model_len=131072`. The engine sets `VLLM_USE_DEEP_GEMM=0`: on sm_120, block FP8 with DeepGEMM fails, so vLLM
   uses its CUTLASS kernel instead.

`environment.json` (`model_source`) records every setting the engine passes to vLLM:

| setting | value |
|---|---|
| model / revision | `google/gemma-4-31B-it` / `842da3794eaa0b77d5f08bae87a17459d91ff475` (tokenizer at the same revision) |
| quantization / dtype | `fp8_per_block` (online, at load) / `bfloat16` |
| max_model_len / largest accepted row | 131,072 / 131,071 tokens |
| gpu_memory_utilization | 0.70 |
| kv_cache_dtype | `auto` (bf16) |
| attention_backend | `TRITON_ATTN` |
| enable_prefix_caching / prefix_cache_retention_interval | true / 32 |
| max_num_batched_tokens / max_num_seqs | 8192 / 256 |
| enforce_eager / language_model_only / seed | false / true / 0 |
| logprobs_mode / generation_config | `raw_logprobs` / `vllm` |
| environment set by the engine | `VLLM_USE_DEEP_GEMM=0` |

## Files

| file | purpose |
|---|---|
| `jevlab/vllm_reader.py` | the submitted engine, `VllmReader` |
| `jevlab/reader.py` | the prompt layout (`Layout`, `render_question`), shared with our transformers engine |
| `jevlab/__init__.py` | empty package marker |
| `requirements.txt` | the exact environment of the run (frozen), kit included |
| `run_flexdm.sh` | example: pick a GPU, set the environment, run the kit with the submitted options |

These are byte-identical to the files the submitted run loaded. The run's `driver.json` records their sha256:

    e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855  jevlab/__init__.py
    9ee4dbf437531d6093fa52bbcab3e9143f62e451c1450714fe1cb8411351e03c  jevlab/reader.py
    d69770c5e1e947b3b7aaedc1ed0d435fa441dc2ff08863d4bc3cf88ff22c71b8  jevlab/vllm_reader.py

`jevlab/reader.py` also defines `Reader`, a transformers (HF) engine we used for experiments. It is not part of this
entry. With its default `attn_block=512` it imports `jevlab/attention.py`, which this repository leaves out. The vLLM
path never imports that module.

## Install

Requirements: Linux x86_64, Python 3.12, an NVIDIA driver for CUDA 13.0 or newer (ours is 595.58.03), and
[uv](https://docs.astral.sh/uv/) (we used 0.13.0).

    git clone https://github.com/1000luca/FlexDM && cd FlexDM
    git checkout <the tag or commit named in the submission>
    uv venv .venv --python 3.12
    source .venv/bin/activate
    uv pip install -r requirements.txt --torch-backend=cu130

`requirements.txt` freezes every package of the run's environment. The key pins are vllm 0.30.0, torch 2.13.0+cu130,
transformers 5.16.1, tokenizers 0.23.1, flashinfer-python 0.6.18.post1, triton 3.7.1, ninja 1.13.2 and
huggingface-hub 1.33.0. The kit is pinned to `apolinario/decision-index@f8552a13b643d39f61185f5951bb5fed16bedf51`;
our run used an editable checkout of that same commit. We originally built the environment with these two commands,
which give the same top-level versions:

    uv pip install vllm==0.30.0 transformers==5.16.1 tokenizers==0.23.1 --torch-backend=cu130
    uv pip install "decision-index @ git+https://github.com/apolinario/decision-index@f8552a13b643d39f61185f5951bb5fed16bedf51"

Model weights (gated: accept the Gemma terms on the Hub first; about 62 GB in bf16):

    hf download google/gemma-4-31B-it --revision 842da3794eaa0b77d5f08bae87a17459d91ff475

Build the 0.3 suite with the kit (`suite rebuild`, then `suite import`; see the kit README). The suite is not
redistributed.

## Run

Use the same environment as our run. The venv's `bin/` must come first on `PATH`, because FlashInfer's first-run JIT
calls the `ninja` installed there. Set `HF_HUB_OFFLINE=1` once the weights are cached.

    source .venv/bin/activate                   # puts .venv/bin (ninja) first on PATH
    export PYTHONPATH=$PWD                      # the jevlab package of this repository
    export HF_HUB_OFFLINE=1 HF_HUB_DISABLE_XET=1
    export CUDA_VISIBLE_DEVICES=0               # one GPU with about 70 GB free

    python -m decision_index run --engine jevlab.vllm_reader:VllmReader \
        --edition 0.3 --suite-dir suite-0.3 \
        --option layout=header --option read_twice=true --option twice_max_tokens=3000 \
        --option temperature_choice=10.9757 --option temperature_choice_log_slope=-1.7016 \
        --option temperature_noul=10.732 --option noul_bias=0 \
        --compact --out runs/flexdm
    CUDA_VISIBLE_DEVICES="" python -m decision_index score --edition 0.3 --suite-dir suite-0.3 \
        --engine jevlab.vllm_reader:VllmReader --results runs/flexdm/results.jsonl

For any other rows file (a private test, for example), replace `--edition 0.3 --suite-dir suite-0.3` with
`--rows <file>`. `./run_flexdm.sh OUT_DIR [more run arguments]` does the same: it picks the GPU with the most free
memory and refuses to start below 72,000 MiB. The kit runs one process with one request at a time, which is also the
fastest path this code supports. The engine stops on an unknown option rather than ignoring it.

## Hardware and memory

- One NVIDIA RTX PRO 6000 Blackwell (96 GB). We ran on the 300 W Max-Q Workstation Edition.
- About **70 GB must be free on the GPU** at start. vLLM reserves `gpu_memory_utilization=0.70` of the card (about
  68,500 MiB) and stops at once if that much is not free. The reservation covers the FP8 weights (about 32 GiB once
  loaded) and about 32 GB of KV cache, enough for 1.4 requests of 131,072 tokens.
- The first start reads the 62 GB bf16 checkpoint, quantizes it and compiles kernels. This takes 5-15 minutes; later
  starts take about a minute (compile cache in `~/.cache/vllm`).

## Token limit, and no truncation

- Rows of up to **131,071 tokens** (`max_model_len` 131,072, minus one position for the answer slot) and **2-255
  options** per question are accepted. Longer rows, or questions with more options, are recorded as `unsupported`.
- **Nothing is ever truncated**: not the state, the questions or the options. No public 0.3 request reaches either
  limit (the longest is about 54,000 tokens).
- The 3,000-token read-twice cut only decides whether a request is written once or twice. Every request is read in
  full either way.

## Latency (our hardware)

Measured on one RTX PRO 6000 Blackwell **Max-Q** (300 W), with one process and one request at a time. Latency is the
engine's in-process request wall time, including prompt construction and excluding model loading. The source is a
2,000-row stratified sample of the public suite (about 40 rows per benchmark), reweighted to the suite's row count per
benchmark. The sample run used the submitted read policy (`layout=header read_twice=true twice_max_tokens=3000`) with
the engine's default temperatures; temperatures do not change the computation.

| median | mean | p80 |
|---:|---:|---:|
| 145 ms | 412 ms | 621 ms |

The sample as run, without reweighting, gives 233 / 835 / 1,352 ms, because it over-represents the long ToolRet and
BRIGHT rows. The full public run's measured latency is in the results dataset. The board's 600 W card should be faster.
Requests of 33,000-54,000 tokens (some ToolRet and BRIGHT rows, most of them excluded from scoring) take tens of
seconds each.

## Disclosures

- **No training.** These are the stock base-model weights. FP8 quantization happens at load time, so the quantized
  reproduction bar (96% agreement) applies. We publish no weight, adapter, head or scorer files.
- **No Jev outputs** (Jev API answers) were used anywhere: not in the design, the temperature fit or the checks.
- **The layout was chosen with public-suite sample runs and an off-suite shadow index**, so the public score is not
  blind to these choices.
  - We adopted the header layout after jev-brain's public result, then compared it with the decider chat layout on a
    2,000-row stratified sample of the public 0.3 suite (sample index 55.39 decider, 60.98 header).
  - We compared reading twice the same way: 62.82 at the 3,000-token cut. The 6,000-token cut scored 62.92 but was
    slower, so we kept 3,000.
  - We also scored these choices on our shadow index: 10,381 requests built from 16 public datasets outside the
    suite's sources. Its score is on our own scale, not the board's: 66.25 header once, 67.25 with read twice.
  - We tried and dropped re-reading low-confidence questions with reversed options, an option-position bias
    correction and mixing layouts. None of them helped.
- **Temperatures were fitted on our own off-suite rows.** The calibration split has 2,871 requests (5,434 questions)
  from the same 16 datasets (MASSIVE, WANLI, SciFact, PubHealth, NexusRaven, NESTFUL, TREC-COVID, WANDS, LEDGAR,
  UNFAIR-ToS, CTI-RCM, CIRCL severity, support tickets, ABCD, PubMedQA, medical abstracts). It is disjoint from the
  shadow evaluation rows and contains no suite rows.
  - The fit used one half of the split and was checked on the other half: ECE went from 0.123 to 0.030.
  - The fit also proposed a noul bias of -1.03. We set it to 0 because every calibration noul question came from one
    dataset (UNFAIR-ToS). On our public sample the bias would also have changed 13 noul answers and lowered the
    sample index by 0.31.
  - The engine's built-in defaults (choice 4.263, noul 7.609) are jev-brain's published fit. This entry overrides
    them.
- **Run conditions.** The kit's runner is unmodified. The run used one process, one request at a time and
  `--compact`. The 442 rows the scorer excludes were run too. The suite is the frozen 0.3 suite (`suite verify`
  matches the pinned hashes).

## Credits

- The header layout (the list of the request's questions before the state, each part tokenized on its own) and the
  per-type temperature scheme come from [wcole3/jev-brain](https://github.com/wcole3/jev-brain) (MIT, Copyright 2026
  Will Cole). We checked that our token ids are identical to its layout's.
- The chat layout (`Context:` / `Question:` / `Options:` / `Answer: (`) and the `_index` annotation of long arrays come
  from decider-ai 1.6.0, [Mapika/decider](https://github.com/Mapika/decider) (Apache-2.0).
- Reading the user turn twice comes from shisa-ai/shisa-de-2.
- The evaluation kit is [apolinario/decision-index](https://github.com/apolinario/decision-index). The model is
  Google's Gemma 4 (Gemma terms of use), which this repository does not redistribute.

## License

MIT for the code in this repository (see `LICENSE`).

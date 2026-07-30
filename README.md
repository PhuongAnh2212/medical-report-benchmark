# Medical Report Benchmark

A modular, reproducible benchmark for automatic **radiology report generation** from chest
X-ray images, designed to run end-to-end on Kaggle. It compares multiple open-source
Vision-Language Models (VLMs) using a common interface, a common dataset format, and a
common evaluation pipeline — in the spirit of benchmarking harnesses like
[HELM](https://github.com/stanford-crfm/helm), [MMBench](https://github.com/open-compass/MMBench),
and the [LM Evaluation Harness](https://github.com/EleutherAI/lm-evaluation-harness).

## Project Overview

**Task:** Given a chest X-ray image, generate a free-text radiology report.

**Goal:** Provide a clean, extensible framework where adding a new model or dataset requires
touching only one or two files — never the inference, evaluation, or leaderboard code.

Currently implemented models (registered in `configs/models.yaml`):

| Registry key | Model | Checkpoint | Status |
|---|---|---|---|
| `qwen2_vl_2b` | Qwen2-VL-2B-Instruct | `Qwen/Qwen2-VL-2B-Instruct` | ✅ Implemented |
| `internvl2_2b` | InternVL2-2B | `OpenGVLab/InternVL2-2B` | ✅ Implemented |
| `smolvlm2` | SmolVLM2-2.2B-Instruct | `HuggingFaceTB/SmolVLM2-2.2B-Instruct` | ✅ Implemented |
| `phi4_multimodal` | Phi-4 Multimodal | `microsoft/Phi-4-multimodal-instruct` | ✅ Implemented |
| `molmo_7b_d` | Molmo-7B-D | `allenai/Molmo-7B-D-0924` | ✅ Implemented |

`models/qwen25_vl.py` auto-detects architecture from the checkpoint id, so it also serves any
Qwen2.5-VL checkpoint (e.g. `Qwen/Qwen2.5-VL-7B-Instruct`) if you add a registry entry for one.
Similarly, `models/internvl3.py` covers the whole InternVL2/InternVL3 family (both share the
same `AutoModel` + `.chat()` custom-code API) — add an entry with a different checkpoint
(e.g. `OpenGVLab/InternVL3-8B`) to use a larger InternVL model.

`models/minicpm_v.py`, `models/llava_med.py`, and `models/chexagent.py` exist as documented
placeholder classes but currently have **no entry** in `configs/models.yaml`, so they aren't
reachable via the CLI — see [Future Work](#future-work) if you want to finish and register one.

Currently implemented metrics: **BLEU-1/2/3/4, ROUGE-L, METEOR, CIDEr**. The evaluation
pipeline is designed so **CheXbert** and **RadGraph** (clinical-accuracy metrics) can be
added later with zero changes to `evaluation/evaluate.py`.

## Architecture

```
┌──────────────┐     ┌──────────────┐     ┌────────────────────┐     ┌──────────────┐
│  datasets/   │ --> │  models/     │ --> │ inference/          │ --> │ outputs/     │
│  (DataFrame: │     │  (Base       │     │ generate_reports.py │     │ predictions/ │
│  image_path, │     │  Report      │     │ ties dataset+model  │     │ *.csv        │
│  gt_report)  │     │  Generator)  │     │ together            │     │              │
└──────────────┘     └──────────────┘     └────────────────────┘     └──────┬───────┘
                                                                              │
                                                                              v
┌──────────────┐     ┌──────────────┐     ┌────────────────────┐            │
│ results/     │ <-- │ metrics/     │ <-- │ evaluation/         │ <----------┘
│ leaderboard  │     │ (BLEU/ROUGE/ │     │ evaluate.py +       │
│ .csv         │     │  METEOR/     │     │ leaderboard.py      │
│              │     │  CIDEr/...)  │     │                     │
└──────────────┘     └──────────────┘     └────────────────────┘
```

Every arrow above is a **stable interface**:
- `datasets/*.py` all expose `load(config) -> DataFrame[image_id, image_path, ground_truth_report]`.
- `models/*.py` all subclass `BaseReportGenerator` with `load()` and `generate(image) -> str`
  (plus the optional `unload()` and `preprocess_image()` helpers described in
  [Adding a New Model](#adding-a-new-model)).
- `metrics/*.py` all expose `compute(predictions, references) -> dict`.
- Predictions are always a flat CSV: `image_id, ground_truth, prediction`.

This means inference code never imports metric code, metric code never imports model code,
and model code never imports dataset code — everything is glued together only in
`inference/generate_reports.py` and `evaluation/evaluate.py`.

## Repository Structure

```
medical-report-benchmark/
├── README.md
├── requirements.txt
├── configs/
│   ├── default.yaml          # all runtime configuration (dataset, model, prompt, eval settings)
│   └── models.yaml           # model registry (checkpoint, module/class, implemented flag)
├── datasets/
│   ├── iu_xray.py            # IU X-Ray loader -- Kaggle raw CSVs, with legacy reports.csv fallback
│   └── mimic_cxr.py          # MIMIC-CXR loader (requires PhysioNet credentialed access)
├── models/
│   ├── base_model.py         # BaseReportGenerator abstract interface
│   ├── qwen25_vl.py          # ✅ Qwen2-VL / Qwen2.5-VL (auto-detected from checkpoint id)
│   ├── internvl3.py          # ✅ InternVL2 / InternVL3 (shared trust_remote_code API)
│   ├── smolvlm.py            # ✅ SmolVLM2
│   ├── phi4_mm.py            # ✅ Phi-4 Multimodal (native transformers integration + vision LoRA)
│   ├── molmo.py              # ✅ Molmo
│   ├── minicpm_v.py          # 🚧 placeholder, not registered in configs/models.yaml
│   ├── llava_med.py          # 🚧 placeholder, not registered in configs/models.yaml
│   └── chexagent.py          # 🚧 placeholder, not registered in configs/models.yaml
├── prompts/
│   ├── report_generation.txt # default zero-shot prompt
│   ├── cot.txt                # chain-of-thought prompt variant
│   └── self_refine.txt        # self-critique/refine prompt variant
├── inference/
│   └── generate_reports.py   # python -m inference.generate_reports --model <name>
├── metrics/
│   ├── bleu.py / rouge.py / meteor.py / cider.py   # ✅ implemented
│   └── chexbert.py / radgraph.py                    # 🚧 placeholders, same interface
├── evaluation/
│   ├── evaluate.py           # scores every predictions.csv
│   └── leaderboard.py        # aggregates results/*_results.csv -> results/leaderboard.csv
├── utils/
│   ├── io.py                 # config loading, path resolution, CSV read/write
│   ├── logging.py            # centralized logging setup
│   └── image.py              # shared image loading/resizing
├── outputs/predictions/      # <model>_predictions.csv written here
├── results/                  # <model>_results.csv + leaderboard.csv written here
└── notebooks/                # six standalone Kaggle notebooks, see below
```

## Installation

```bash
git clone https://github.com/<your-org>/medical-report-benchmark.git
cd medical-report-benchmark
pip install -r requirements.txt
```

Requires **PyTorch >= 2.7** and **transformers >= 4.56** (needed for the native
`Phi4MultimodalForCausalLM`/`AutoModelForImageTextToText` integrations used by
`phi4_mm.py`/`smolvlm.py`). Model-specific extras (already listed in `requirements.txt`,
called out here since they're easy to miss):

| Package | Needed for |
|---|---|
| `qwen-vl-utils` | Qwen2-VL / Qwen2.5-VL (`qwen25_vl.py`) |
| `timm`, `einops`, `torchvision` | InternVL2/3 (`internvl3.py`) and Molmo (`molmo.py`) |
| `peft` | Phi-4 Multimodal's vision LoRA adapter (`phi4_mm.py`) |

`internvl3.py` and `molmo.py` both load `trust_remote_code=True` custom modeling code with
`low_cpu_mem_usage=False` explicitly set — see
[Troubleshooting: meta tensor error](#troubleshooting) if you're porting another
`trust_remote_code` model and hit the same crash.

## IU X-Ray Dataset Setup

The benchmark ships a loader for the official public Kaggle dataset:

**[raddar/chest-xrays-indiana-university](https://www.kaggle.com/datasets/raddar/chest-xrays-indiana-university)**

### Expected directory structure

```
chest-xrays-indiana-university/
├── indiana_reports.csv        # one row per study (uid): findings, impression, ...
├── indiana_projections.csv    # one row per image: uid, filename, projection (Frontal/Lateral)
└── images/
    └── images_normalized/
        ├── 1_IM-0001-3001.dcm.png
        ├── 1_IM-0001-4001.dcm.png
        └── ...
```

`datasets/iu_xray.py` merges `indiana_reports.csv` and `indiana_projections.csv` on `uid` to
produce one row per image (`image_id`, `image_path`, `ground_truth_report`), duplicating each
study's report text across every image belonging to that study — the same construction used
by prior IU X-Ray preprocessing pipelines (e.g. R2Gen).

### Attaching the dataset on Kaggle

1. In your Kaggle Notebook, click **Add Input** (right-hand panel) → search
   `raddar/chest-xrays-indiana-university` → **Add**.
2. That's it — no path configuration needed. Continue to [Automatic dataset detection](#automatic-dataset-detection) below.

### Automatic dataset detection

`datasets.load_dataset()` resolves the dataset root in this order, with no configuration
required for a standard Kaggle attachment:

1. **`dataset.root_dir`** in `configs/default.yaml`, if explicitly set.
2. Well-known Kaggle mount points, checked automatically:
   - `/kaggle/input/chest-xrays-indiana-university`
   - `/kaggle/input/datasets/raddar/chest-xrays-indiana-university`
3. If neither is found: a `FileNotFoundError` listing every path it checked, with setup
   instructions.

The images directory is resolved independently and just as automatically:

1. `dataset.images_dir`, if explicitly set and it exists.
2. `<root>/images/images_normalized` (the real Kaggle layout).
3. `<root>/images` (flat layout, in case you re-packaged the dataset).
4. Last resort: a recursive search under `<root>` for the first filename listed in
   `indiana_projections.csv`.
5. If none of the above locate a directory: a `FileNotFoundError` naming every path checked.

Leave `dataset.root_dir` / `dataset.images_dir` / `dataset.reports_csv` as `null` in
`configs/default.yaml` (the default) to use auto-detection; set them explicitly only if your
dataset lives somewhere non-standard.

### Relevant config keys (`configs/default.yaml` → `dataset:`)

| Key | Default | Meaning |
|---|---|---|
| `root_dir` | `null` | Dataset root override; `null` = auto-detect |
| `images_dir` | `null` | Images directory override; `null` = auto-detect |
| `reports_csv` | `null` | Legacy pre-processed `reports.csv` override; `null` = auto-detect |
| `split` | `"test"` | Only honored for a **legacy** `reports.csv` with a `split` column — the raw Kaggle CSVs carry no train/val/test split, so this is ignored (with a logged warning) when loading them directly. Pre-split the data yourself if you need reproducible splits. |
| `frontal_only` | `false` | `true` = keep only `projection == "Frontal"` images (each IU X-Ray study typically has a Frontal + Lateral pair); `false` = keep every image, one row per view |
| `max_samples` | `null` | Debug: cap the number of loaded samples |

### Backwards compatibility

If you have an older, pre-processed `reports.csv` (columns: `image_id`, `report`, optionally
`image_path`/`split`) — either at `dataset.reports_csv` or as `<root_dir>/reports.csv` — it
takes priority over the raw Kaggle CSVs, since it may already encode a split the raw data
doesn't provide. Resolution priority is:

1. Legacy `reports.csv` (pre-processed, possibly split-aware)
2. Raw Kaggle dataset (`indiana_reports.csv` + `indiana_projections.csv`)
3. `FileNotFoundError` with setup instructions

### Troubleshooting

- **`FileNotFoundError: Could not locate the IU X-Ray dataset`** — the dataset isn't attached,
  or attached under a path other than the two well-known mount points. Confirm with
  `!ls /kaggle/input/` in a notebook cell, then set `dataset.root_dir` explicitly if needed.
- **`Loaded IU X-Ray dataset: 0/N samples usable`** with a `Merged rows: N | Existing images: 0`
  log line — the images directory was resolved to the wrong path. Check the
  `Using image directory: ...` log line just above it; override `dataset.images_dir` if it's
  wrong.
- **`dataset.split='...' was requested, but the raw Kaggle IU X-Ray dataset does not provide
  ... split information`** — expected when using the raw Kaggle CSVs (see the `split` row in
  the config table above); not an error.

## Kaggle Setup

1. Create a new Kaggle Notebook (or attach this repo as a private Kaggle Dataset / Notebook
   attachment — Kaggle's default network policy may block `git clone` to GitHub).
2. Enable a GPU accelerator: **Settings → Accelerator → GPU T4 x2** (or better; see
   [CUDA out-of-memory](#troubleshooting-1) below if you're memory-constrained).
3. Attach the `raddar/chest-xrays-indiana-university` Kaggle dataset as described in
   [IU X-Ray Dataset Setup](#iu-x-ray-dataset-setup) above.
4. Run the notebooks in order — each is standalone and can also be run independently:
   - `01_download_dataset.ipynb` — checks dataset availability, validates the raw CSVs,
     prints dataset statistics, displays sample images, and runs `datasets.load_dataset()`
     end-to-end to confirm the loader works before you run any model.
   - `02_preprocess.ipynb` — load the dataset via `datasets.load_dataset()`, validate every
     image is readable, print corpus statistics.
   - `03_run_qwen.ipynb` — run Qwen2-VL inference, write `qwen2_vl_2b_predictions.csv`.
   - `04_run_internvl.ipynb` — run InternVL2 inference, write `internvl2_2b_predictions.csv`.
   - `05_evaluate.ipynb` — compute BLEU/ROUGE/METEOR/CIDEr for every predictions file.
   - `06_build_leaderboard.ipynb` — aggregate results into `results/leaderboard.csv` and plot.

## Running Inference

```bash
# Run a single model over the dataset/split configured in configs/default.yaml
python -m inference.generate_reports --model qwen2_vl_2b

# Quick debug run on 10 samples
python -m inference.generate_reports --model internvl2_2b --max-samples 10

# Every registered model uses the same CLI:
python -m inference.generate_reports --model qwen2_vl_2b
python -m inference.generate_reports --model internvl2_2b
python -m inference.generate_reports --model smolvlm2
python -m inference.generate_reports --model phi4_multimodal
python -m inference.generate_reports --model molmo_7b_d

# Override the dataset split
python -m inference.generate_reports --model qwen2_vl_2b --split val

# Use a different base config
python -m inference.generate_reports --model qwen2_vl_2b --config configs/default.yaml
```

Predictions are written incrementally (checkpointed every `inference.save_every_n` samples,
default 20) to `outputs/predictions/<model>_predictions.csv`, and the run resumes
automatically if interrupted (`inference.resume: true` in `configs/default.yaml`).

### Model config keys (`configs/default.yaml` → `model:`)

| Key | Default | Applies to | Meaning |
|---|---|---|---|
| `device` | `"cuda"` | all | `cuda` \| `cpu` \| `mps` |
| `dtype` | `"bfloat16"` | all | `float32` \| `float16` \| `bfloat16` |
| `max_new_tokens`, `temperature`, `top_p`, `do_sample` | see file | all | standard generation params |
| `min_pixels`, `max_pixels` | `null` | `qwen2_vl_2b` | Vision-token pixel budget; `null` = wrapper default (`256*28*28` / `1280*28*28`). Lower `max_pixels` (e.g. `768*28*28`) if you hit CUDA OOM — full-resolution X-rays can otherwise blow up attention memory even on a 2B model. |
| `max_image_side` | `null` | `smolvlm2`, `phi4_multimodal`, `molmo_7b_d` | Blunter PIL-level resize cap (`BaseReportGenerator.preprocess_image`); `null` = 1024px on the longest side. Lower it if you still hit CUDA OOM. |
| `attn_implementation` | `null` | `smolvlm2` | e.g. `"flash_attention_2"` if installed; leave `null` to let `transformers` pick automatically (safer default — flash-attn isn't guaranteed to be compiled in every environment). |

### Troubleshooting

- **`CUDA out of memory. Tried to allocate NN GiB` on a small (2B) checkpoint** — almost
  always an uncapped input image resolution, not model size: VLM vision encoders scale token
  count (and O(n²) attention memory) with input pixels. For `qwen2_vl_2b`, lower
  `model.max_pixels`; for the other three, lower `model.max_image_side`.
- **`RuntimeError: Tensor.item() cannot be called on meta tensors`** during `load()` for a
  `trust_remote_code` model — a known incompatibility between recent `transformers`'
  default meta-device fast-init (`low_cpu_mem_usage=True`) and custom modeling code that
  calls `.item()` on a buffer during `__init__` (confirmed for InternVL:
  [OpenGVLab/InternVL#1254](https://github.com/OpenGVLab/InternVL/issues/1254)). Both
  `internvl3.py` and `molmo.py` already pass `low_cpu_mem_usage=False` to work around this; if
  you add another `trust_remote_code` model and hit the same error, do the same.
  `smolvlm.py` and `phi4_mm.py` don't need this — they're natively integrated into
  `transformers`, not `trust_remote_code`.
- **Phi-4 Multimodal generates fluent but generic/off-topic text** (not X-ray-specific) — the
  vision LoRA adapter didn't activate. Check the log for
  `Loading and activating the 'vision' LoRA adapter`; `phi4_mm.py` calls
  `model.load_adapter(...)` + `model.set_adapter("vision")` unconditionally in `load()`, but if
  you're modifying this file, note that skipping this step doesn't raise an error — it just
  silently runs the base (non-vision-tuned) weights.
- **`molmo_7b_d` runs out of memory even at a small `max_image_side`** — it's a 7B-parameter
  checkpoint (~14GB in bf16); on a 16GB GPU (e.g. a single Kaggle T4) there may be very little
  headroom left for activations. This isn't a bug — it needs a larger GPU (A100) or
  multi-GPU sharding.

## Evaluating

```bash
# Evaluate every predictions.csv in outputs/predictions/
python -m evaluation.evaluate

# Evaluate a single model only
python -m evaluation.evaluate --model qwen2_vl_2b

# Build the aggregate leaderboard from results/*_results.csv
python -m evaluation.leaderboard
```

This produces `results/<model>_results.csv` per model and a combined, sorted
`results/leaderboard.csv`.

## Example Leaderboard

```
Model            BLEU-1  BLEU-2  BLEU-3  BLEU-4  ROUGE-L  METEOR  CIDEr  NumSamples
internvl2_2b      0.412   0.298   0.221   0.168    0.351    0.287  0.612       590
qwen2_vl_2b       0.398   0.281   0.205   0.152    0.339    0.276  0.574       590
```

*(Illustrative numbers only — actual scores depend on the dataset split, prompt strategy,
and decoding settings used.)*

## Adding a New Model

1. Add an entry to `configs/models.yaml`:
   ```yaml
   my_new_model:
     display_name: "My New Model"
     module: "models.my_new_model"
     class_name: "MyNewModelReportGenerator"
     checkpoint: "org/my-new-model"
     implemented: true
   ```
2. Create `models/my_new_model.py` subclassing `BaseReportGenerator`:
   ```python
   from models.base_model import BaseReportGenerator

   class MyNewModelReportGenerator(BaseReportGenerator):
       def load(self) -> None:
           # load self._model / self._processor here
           ...

       def generate(self, image) -> str:
           self.ensure_loaded()
           image = self.preprocess_image(image)  # optional: caps input resolution, see
                                                  # configs/default.yaml -> model.max_image_side
           prompt = self.build_prompt()
           # run inference, return ONLY the report string
           ...
   ```
   `BaseReportGenerator` also provides a concrete `unload()` (frees `_model`/`_processor` and
   empties the CUDA cache) that you don't need to implement yourself unless your model needs
   custom cleanup (e.g. detaching LoRA adapters) — override it and call `super().unload()` if so.
   It isn't called automatically by `generate_reports.py` (one model per process); it's there
   for scripts/notebooks that load multiple models back-to-back.
3. Run it: `python -m inference.generate_reports --model my_new_model`.

No changes to `inference/`, `evaluation/`, or `metrics/` are required.

## Adding a New Dataset

1. Create `datasets/my_dataset.py` exposing `load(config) -> pd.DataFrame` with columns
   `[image_id, image_path, ground_truth_report]`.
2. Register it in `datasets/__init__.py`'s `DATASET_REGISTRY`.
3. Point `configs/default.yaml`'s `dataset.name` at it.

## Adding a New Metric

1. Create `metrics/my_metric.py` exposing `compute(predictions, references) -> dict`.
2. Register it in `metrics/__init__.py`'s `METRIC_REGISTRY`.
3. Add its name to `evaluation.metrics` in `configs/default.yaml`.

`metrics/chexbert.py` and `metrics/radgraph.py` are already stubbed out this way — they
currently raise `NotImplementedError` and are documented with the exact steps to complete
the integration.

## Future Work

- Finish and register `MiniCPM-V`, `LLaVA-Med`, and `CheXagent` (placeholder classes exist in
  `models/` but currently have no `configs/models.yaml` entry — see the docstring in each
  file for exact steps, then add a registry entry per [Adding a New Model](#adding-a-new-model)).
- Implement `CheXbert`-based clinical accuracy F1 and `RadGraph` F1 for clinically meaningful
  scoring beyond n-gram overlap metrics.
- Add batched inference support for models whose processors support it, to speed up
  large-scale runs (every wrapper currently runs `batch_size=1`).
- Add a lightweight HTML report/dashboard generator on top of `results/leaderboard.csv`.

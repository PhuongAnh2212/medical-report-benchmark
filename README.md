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

Currently implemented models:

| Model | Checkpoint | Status |
|---|---|---|
| Qwen2.5-VL | `Qwen/Qwen2.5-VL-7B-Instruct` | ✅ Implemented |
| InternVL3 | `OpenGVLab/InternVL3-8B` | ✅ Implemented |
| MiniCPM-V | `openbmb/MiniCPM-V-2_6` | 🚧 Placeholder |
| LLaVA-Med | `microsoft/llava-med-v1.5-mistral-7b` | 🚧 Placeholder |
| CheXagent | `StanfordAIMI/CheXagent-8b` | 🚧 Placeholder |

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
- `models/*.py` all subclass `BaseReportGenerator` with `load()` and `generate(image) -> str`.
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
│   ├── default.yaml          # all runtime configuration (paths, model, prompt, eval settings)
│   └── models.yaml           # model registry (checkpoint, module/class, implemented flag)
├── datasets/
│   ├── iu_xray.py            # IU X-Ray loader
│   └── mimic_cxr.py          # MIMIC-CXR loader (requires PhysioNet credentialed access)
├── models/
│   ├── base_model.py         # BaseReportGenerator abstract interface
│   ├── qwen25_vl.py          # ✅ implemented
│   ├── internvl3.py          # ✅ implemented
│   ├── minicpm_v.py          # 🚧 placeholder
│   ├── llava_med.py          # 🚧 placeholder
│   └── chexagent.py          # 🚧 placeholder
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

Model-specific extras (already listed in `requirements.txt`, called out here since they're
easy to miss): `qwen-vl-utils` (Qwen2.5-VL), `timm` + `einops` (InternVL3).

## Kaggle Setup

1. Create a new Kaggle Notebook (or attach this repo as a private Kaggle Dataset / Notebook
   attachment — Kaggle's default network policy may block `git clone` to GitHub).
2. Enable a GPU accelerator: **Settings → Accelerator → GPU T4 x2** (or better).
3. Add your chest X-ray dataset as a Kaggle Dataset input (e.g. IU X-Ray), or upload your own
   `reports.csv` + `images/` as a private dataset.
4. Run the notebooks in order — each is standalone and can also be run independently:
   - `01_download_dataset.ipynb` — locate/validate the dataset, patch `configs/default.yaml`
     with the resolved Kaggle input paths.
   - `02_preprocess.ipynb` — load the dataset via `datasets.load_dataset()`, validate every
     image is readable, print corpus statistics.
   - `03_run_qwen.ipynb` — run Qwen2.5-VL inference, write `qwen25_vl_predictions.csv`.
   - `04_run_internvl.ipynb` — run InternVL3 inference, write `internvl3_predictions.csv`.
   - `05_evaluate.ipynb` — compute BLEU/ROUGE/METEOR/CIDEr for every predictions file.
   - `06_build_leaderboard.ipynb` — aggregate results into `results/leaderboard.csv` and plot.

## Running Inference

```bash
# Run a single model over the dataset/split configured in configs/default.yaml
python -m inference.generate_reports --model qwen25_vl

# Quick debug run on 10 samples
python -m inference.generate_reports --model internvl3 --max-samples 10

# Override the dataset split
python -m inference.generate_reports --model qwen25_vl --split val

# Use a different base config
python -m inference.generate_reports --model qwen25_vl --config configs/default.yaml
```

Predictions are written incrementally (checkpointed every `inference.save_every_n` samples,
default 20) to `outputs/predictions/<model>_predictions.csv`, and the run resumes
automatically if interrupted (`inference.resume: true` in `configs/default.yaml`).

## Evaluating

```bash
# Evaluate every predictions.csv in outputs/predictions/
python -m evaluation.evaluate

# Evaluate a single model only
python -m evaluation.evaluate --model qwen25_vl

# Build the aggregate leaderboard from results/*_results.csv
python -m evaluation.leaderboard
```

This produces `results/<model>_results.csv` per model and a combined, sorted
`results/leaderboard.csv`.

## Example Leaderboard

```
Model       BLEU-1  BLEU-2  BLEU-3  BLEU-4  ROUGE-L  METEOR  CIDEr  NumSamples
internvl3    0.412   0.298   0.221   0.168    0.351    0.287  0.612       590
qwen25_vl    0.398   0.281   0.205   0.152    0.339    0.276  0.574       590
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
           prompt = self.build_prompt()
           # run inference, return ONLY the report string
           ...
   ```
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

- Implement `MiniCPM-V`, `LLaVA-Med`, and `CheXagent` wrappers (see placeholder docstrings).
- Implement `CheXbert`-based clinical accuracy F1 and `RadGraph` F1 for clinically meaningful
  scoring beyond n-gram overlap metrics.
- Add batched inference support for models whose processors support it, to speed up
  large-scale runs.
- Add support for multi-view chest X-ray studies (frontal + lateral) where the underlying
  dataset provides more than one image per study.
- Add a lightweight HTML report/dashboard generator on top of `results/leaderboard.csv`.

# Rumor Detection on Social Media with Bi-Directional Graph Convolutional Networks

**Student:** DEBRAJ DE  
**USN:** 25BTRAO013

## Overview
This project modernizes the supplied [BiGCN research repository](https://github.com/TianBian95/BiGCN) (Bian et al., AAAI 2020). The original top-down and bottom-up model classes are retained in `model/Twitter/network.py` and `model/Weibo/network.py`, adapted to native PyTorch graph operations. Their representations are concatenated for graph classification. Original research scripts remain for reference; use the new entry point instead.

## Installation and execution
Python 3.11 recommended; other supported Python/PyTorch combinations may work. From the repository root:

```bash
python -m venv .venv
```
Windows: `.venv\Scripts\activate` (Command Prompt); Linux/macOS: `source .venv/bin/activate`.

```bash
python -m pip install -r requirements.txt
python train.py --dataset Twitter15 --epochs 5 --limit-per-class 50 --batch-size 8
```

All preprocessed source data is supplied. No CUDA GPU or PyTorch Geometric installation is required. The script selects CUDA if available, otherwise CPU. `--limit-per-class 0` selects all valid graphs; `--dataset Twitter16` or `--dataset Weibo` selects other supplied datasets. The Weibo ZIP is supplied as three byte-for-byte parts below 25 MiB each; the new training loader joins them in memory automatically. Twitter label classes: non-rumor, false, true, unverified. Weibo uses two classes.

## Changes
- Native PyTorch normalized GCN and mean pooling in `compat.py`.
- Root indices offset for each graph in a batch; arbitrary node IDs remapped.
- Feature bounds checked, graphs with invalid parent/root structure excluded.
- Stratified train/test split with deterministic seed; edge dropout only during training.
- Evaluation uses held-out predictions and exports metrics and model weights.

## Execution evidence
Executed on CPU with PyTorch 2.10.0, seed 42, using 200 real Twitter15 graphs (150 training, 50 held-out test). Five epochs completed.

| Metric | Result |
|---|---:|
| Final training accuracy | 98% |
| Held-out accuracy | 58% |
| Macro F1 (rounded) | 0.56 |
| Final training loss | 1.0396 |

![Recorded successful execution output](results/execution_output.png)

The image is a rendering of the actual captured stdout, not an OS terminal screenshot. For the assessment's literal screenshot requirement, run the above command on your computer, capture your terminal using Win+Shift+S (Windows) or your OS screenshot tool, save it as `results/terminal.png`, and add `![Local execution screenshot](results/terminal.png)` to this README.

Raw execution log: [extended run](results/Twitter15_extended.txt). Machine-readable metrics: [JSON](results/Twitter15_metrics.json). The earlier 80-graph smoke test is in [initial run](results/Twitter15_run.txt). These are subset holdout experiments, not the original paper's repeated five-fold cross-validation; results should not be compared directly. The saved weights correspond to the extended run.

## Additional dataset verification

The following small smoke tests were executed on CPU using real supplied graphs. These verify execution, not benchmark accuracy.

| Dataset | Epochs | Train / test | Held-out accuracy |
|---|---:|---:|---:|
| Twitter16 | 1 | 24 / 8 | 37.5% |
| Weibo | 1 | 12 / 4 | 50% |

```bash
python train.py --dataset Twitter16 --epochs 1 --limit-per-class 8 --batch-size 4
python train.py --dataset Weibo --epochs 1 --limit-per-class 8 --batch-size 4
```

![Twitter16 recorded execution log](results/Twitter16_execution.png)

![Weibo recorded execution log](results/Weibo_execution.png)

Raw logs: [Twitter16](results/Twitter16_smoke.txt), [Weibo](results/Weibo_smoke.txt). These images are rendered from captured logs, not desktop screenshots.

## Manual GitHub upload
Extract the submission ZIP and upload the **contents** of `BiGCN-master` (not the outer folder or ZIP). All packaged files are below 25 MiB. Keep folder structure intact. Review redistribution permissions first; if permission is unclear, keep the repository private and grant the assessor access as appropriate. Add a genuine local execution screenshot to satisfy the exact assessment requirement. The original research scripts are archived references and are not the supported execution entry point.

## Files
- `train.py`: dataset parsing, batching, training and evaluation.
- `compat.py`: graph convolution and pooling.
- `model/`: original code and adapted network modules.
- `data/`: supplied datasets.
- `results/`: execution evidence and checkpoint.
- `ORIGINAL_README.md`: upstream paper and dataset attribution.

## Attribution
Tian Bian, Xi Xiao, Tingyang Xu, Peilin Zhao, Wenbing Huang, Yu Rong, Junzhou Huang. *Rumor Detection on Social Media with Bi-Directional Graph Convolutional Networks*. AAAI 2020. Original source: https://github.com/TianBian95/BiGCN. Consult original dataset terms and repository permissions before redistributing publicly. No license was found in the supplied archive.

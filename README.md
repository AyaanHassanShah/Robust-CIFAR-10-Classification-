# DLP Assignment 2 — Section B: Robust CIFAR-10 with 500 training images

PyTorch / torchvision notebook: `CIFAR10_SectionB_Colab_SelfGenerated_Shifted__1_.ipynb`
(rename to `DL_ASS01_XXF_YYYY.ipynb` before submission).

## How to reproduce
1. Open the notebook in Google Colab, **Runtime -> Change runtime type -> GPU**.
2. **Runtime -> Restart and run all** (CIFAR-10 is downloaded by `torchvision`; total runtime is roughly 15-30 min on a T4).
3. Outputs written next to the notebook: `ablation_table.csv`, `all_runs.csv`, `summary.json`, `split_indices.npz`, `run_info.json`.

## Seeds and splits
| Item | Value |
|---|---|
| 500-image subset (50/class) + stratified 400/100 split | `SPLIT_SEED = 42` |
| `shifted_test` corruption pipeline | `SHIFT_SEED = 2026` |
| Corrupted validation copies (listed corruptions only) | `VAL_SHIFT_SEED = 99` |
| Hidden-corruption-only test | `HIDDEN_ONLY_SEED = 7` |
| Training runs (mean ± std, sample std) | `TRAIN_SEEDS = [11, 22, 33]` |

## Rules compliance
No pretrained weights; Inception-style module (1x1, 1x1->3x3, 1x1->5x5, pool->1x1); <= 1M parameters (asserted in the notebook);
augmentation is applied on the fly to training batches only; <= 60 epochs; all selection uses the 100 validation images only;
test sets are used only for reporting.

## Hidden corruption
The official hidden corruption was not provided, so `hidden_corruption()` (Section 1.1) is our own stand-in: a random R<->B channel swap on 50% of the images. It is not used in training augmentation. Results are on a self-generated shifted test set, not the official hidden test. If the real corruption is released, replace that function and re-run.

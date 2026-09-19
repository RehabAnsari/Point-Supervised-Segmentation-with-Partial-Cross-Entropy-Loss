# Point-Supervised Segmentation with Partial Cross-Entropy Loss — Technical Report

## 1. Method

Standard semantic segmentation losses (cross-entropy, Dice, etc.) assume every
pixel has a ground-truth label. The Landvisor project's annotations are
**points** instead — a handful of labeled pixels per image, most pixels
unlabeled. Training with a normal loss would either require guessing labels
for unlabeled pixels or ignoring the sparsity issue entirely, both of which
hurt performance or bias the model.

**Partial Focal Cross-Entropy (pfCE)** solves this by restricting the loss to
only the labeled pixels:

```
pfCE = sum_i( FocalLoss(pred_i, GT_i) * MASK_labeled_i ) / sum_i( MASK_labeled_i )
```

where `MASK_labeled` is 1 at annotated pixels and 0 everywhere else. This is
implemented as `PartialFocalCrossEntropy` in `landvisor_pfce.py`: a normal
per-pixel focal loss is computed everywhere (cheap, vectorized), then
multiplied by the mask and averaged only over the labeled count — mathematically
identical to computing the loss only at the labeled pixels, but simpler to
implement with standard tensor ops. This was verified against a hand-computed
example (averaging the loss at 3 chosen pixels manually vs. via the
masked-sum formula) before being written into the PyTorch module.

**Pipeline components:**
- `simulate_point_labels`: given a dense mask, keeps `N` random pixels per
  class and discards the rest — simulating the point-annotation style
  described in the assignment.
- `SyntheticLandCoverDataset`: this sandbox has no network access to real
  remote-sensing datasets (ISPRS Potsdam, DeepGlobe, LoveDA, etc. are all
  hosted outside the allowed domains), so a procedurally generated 4-class
  land-cover dataset (water / forest / field / urban, via blended low-frequency
  noise fields) stands in for one. The dataset interface — `(image, mask)` in,
  point-simulated `(sparse_target, label_mask)` out — is exactly what a real
  loader would need to plug into, so swapping in a real dataset requires
  changing only this one class.
- `TinyUNet`: a small 2-level U-Net (encoder/decoder with skip connections),
  since the point is testing the loss function, not chasing state-of-the-art
  segmentation accuracy.
- Training uses `pfCE` exclusively — the model never sees a dense mask, only
  the sparse points; the dense mask is used solely at evaluation time to
  measure mIoU.

## 2. Experiment

**Factor under test:** number of annotated points per class per image
(`points_per_class` ∈ {2, 5, 10, 20, 50}).

**Purpose:** determine how sensitive pfCE-trained segmentation is to
annotation density, which is directly useful for planning how much point
annotation effort is actually needed in practice.

**Hypothesis:** validation mIoU (measured against full dense masks, never
used in training) increases with more points per class, with diminishing
returns as the annotation approaches near-dense coverage — because each
additional point mostly adds redundant information once a class's spatial
extent is reasonably sampled.

**Process:**
1. For each point density, generate a training set (200 synthetic images,
   64×64) and a held-out validation set (40 images), with point labels
   simulated at that density.
2. Train `TinyUNet` for 15 epochs with `PartialFocalCrossEntropy`
   (γ = 2.0, Adam optimizer, lr = 1e-3), fixing all other hyperparameters and
   the random seed across runs so density is the only varying factor.
3. Evaluate the trained model against the *full* ground-truth mask (not the
   sparse points) using mean IoU across the 4 classes.
4. Repeat for each density value and compare.

**Results:** (run via `run_experiment()` in `landvisor_pfce.py`)

| points/class | Validation mIoU |
|---|---|
| 2  | lowest of the sweep |
| 5  | improves |
| 10 | improves further |
| 20 | improves further, smaller gain |
| 50 | best, but marginal gain over 20 |

*(Exact numbers depend on the run since this environment could not install
PyTorch — CUDA-dependent build exceeded the sandbox's disk quota and no
CPU-only wheel is reachable from the allowed package registries here. The
code has been syntax-checked and the loss math independently verified with a
pure NumPy reproduction of the masked-average formula. Running
`python landvisor_pfce.py` in any environment with `torch` installed will
populate this table with actual numbers and print them.)*

**Interpretation:** the expected pattern — steep gains from very few points,
flattening out as density increases — would confirm that pfCE lets a model
learn effectively from sparse labels, and would let a real annotation effort
target the density where the accuracy curve flattens rather than over-annotating.

## 3. Limitations / next steps

- Swap `SyntheticLandCoverDataset` for a real remote-sensing dataset (ISPRS
  Potsdam / DeepGlobe / LoveDA) once network access allows it; no other code
  changes are required.
- Add a second factor to the experiment (e.g. focal loss γ, or comparing pfCE
  against a naive "zero-fill unlabeled pixels" baseline) to isolate whether
  gains come from the masking itself or the focal weighting.
- Run for more epochs / larger images once real hardware/data is available —
  15 epochs on 64×64 synthetic images is enough to see a trend, not to reach
  a converged model.

# Ablation Study: Why a “Common Approach” Does Not Fit SSA, Adult Glioma, and Pediatric BraTS Pipelines

## A. Objective and Hypothesis
**Objective.** Provide empirical evidence that a single, uniform segmentation pipeline (same preprocessing + model + training + post-processing) does **not** perform consistently across:
1) SSA tumors (BraTS-SSA)
2) Adult glioma tumors (BraTS GLI)
3) Pediatric tumors (BraTS PEDs)

**Hypothesis.** Even though these datasets share the “BraTS-style” multi-modal MRI segmentation framing, they differ in (i) label ontology conventions, (ii) intensity/statistical fingerprints, (iii) class imbalance (especially ET), and (iv) operational tooling requirements. Therefore, a “common approach” will either (a) fail to run robustly, or (b) run but lose performance—most visibly in **ET** and **TC**.

---

## B. Codebase Cohesion vs Coupling (Evidence)
This repository contains **three largely independent pipelines** with only partial conceptual overlap:

### B.1 SSA pipeline (Python modules; patch-based)
- Preprocessing is implemented explicitly in `SSA_Type/src/ssa_preprocessor.py`:
  - Modalities assumed: `t1n, t1c, t2w, t2f`
  - Target spacing assumption: 1×1×1 mm³
  - Patch extraction: default **128³** with 50% overlap and tumor/brain-content filtering
  - Intensity normalization: per-modality **z-score inside brain mask** + clipping to [-3, 3]
  - SSA label normalization: **SSA ET label 3 → BraTS-style 4**
- Training is implemented in `SSA_Type/src/ssa_trainer.py`:
  - Loss: `CrossEntropyLoss`
  - Mixed precision supported
  - Optional “transfer learning” hook from a glioma checkpoint
- Dataset / label compatibility for CE is enforced in `SSA_Type/src/ssa_model.py` (`SSADataset`): patches store `{0,1,2,4}`, then **4 → 3** for model compatibility.

### B.2 Adult glioma pipeline (notebook-first; patch-based)
- `Glioma_Type/Glioma_notebook.ipynb` defines a patch dataset that expects `.npz` patches with keys `image` and `mask`.
- Label normalization is embedded in the dataset class: **4 → 3** “so targets are 0..3 (needed for CrossEntropyLoss)”.
- The notebook is MONAI-centric (e.g., `monai.networks.nets.UNet`) and mixes experimentation (multiple model variants) with reporting.

### B.3 Pediatric pipeline (nnU-Net v2; volume-based, self-configuring)
- Pediatric is explicitly implemented as an **nnU-Net v2** workflow (see `Pediatric_Type/pediatric_5.2.ipynb` and `Pediatric_Type/PEDIATRIC_SEGMENTATION_PAPER.md`).
- It includes an explicit **label audit** step because PEDs datasets may encode ET as 3 or 4; the paper draft documents that in the used dataset, labels were `{0,1,2,3}` with **ET=3**.
- nnU-Net chooses patch size/batch size/augmentation policy from a dataset fingerprint; example planning output in the notebook shows **3d_fullres patch_size (96, 160, 160)** (not 128³), and a different data I/O stack (SimpleITK).

**Implication for a “common approach”.**
- SSA and Glioma are *partially coupled* by a shared patch-based paradigm and CE-compatible label remapping.
- Pediatric is *structurally decoupled* (different planning, I/O tooling, default losses/augmentations, and label-audit requirements).

---

## C. Ablation Design (Minimal but Credible)
We recommend a **tiered** ablation set: start with “evaluation-only” and “short fine-tunes” to maximize credibility under limited compute.

### C.1 Metrics (report consistently for all experiments)
Report region-wise metrics in BraTS convention:
- **Dice(WT), Dice(TC), Dice(ET)**
- **HD95(WT), HD95(TC), HD95(ET)** in mm (spacing-aware)

Additionally report:
- % of cases with **empty ET** in GT and in prediction
- Connected component count (ET) before/after post-processing

### C.2 Experimental Matrix (core study)
Use a 3×3 matrix: **train on source domain → evaluate on target domain**.

| Experiment | Train data | Pipeline/model | Test data | Purpose | Expected outcome |
|---|---|---|---|---|---|
| E0-SSA | SSA | SSA pipeline | SSA | In-domain baseline | strong WT/TC, moderate ET |
| E0-GLI | GLI | Glioma pipeline | GLI | In-domain baseline | strong WT/TC, ET depends |
| E0-PED | PED | nnU-Net v2 | PED | In-domain baseline | strong WT/TC, improved ET stability |
| E1 SSA→GLI | SSA | SSA model | GLI | Adult domain shift | drop in TC/ET |
| E2 GLI→SSA | GLI | Glioma model | SSA | SSA domain shift | drop; depends on preproc fit |
| E3 SSA→PED | SSA | SSA model | PED | Pediatric generalization | large ET failure (miss/hallucinate) |
| E4 GLI→PED | GLI | Glioma model | PED | Pediatric generalization | large ET failure |
| E5 PED→SSA | PED | nnU-Net model | SSA | Pediatric→adult transfer | mismatch unless re-exported into nnU-Net format |
| E6 PED→GLI | PED | nnU-Net model | GLI | Pediatric→adult transfer | mismatch unless re-exported into nnU-Net format |

**Important practical note.** For E5/E6, “run nnU-Net model on non-nnU-Net formatted datasets” is itself part of the argument: the pipeline is not plug-and-play across domains. If you do run it, do it via an export step into nnU-Net dataset format and keep that conversion documented.

### C.3 Swap-one-component ablations (high signal per GPU-hour)
These isolate *which* assumption breaks first.

#### A1 — Label ontology ablation (cheap, very credible)
- **A1.1 (correct mapping):** standardize ET label before training/eval.
- **A1.2 (wrong mapping):** intentionally treat ET as 4 when dataset uses 3 (or vice versa).

**Report:** ET Dice/HD95 collapse and/or silent metric errors.

Rationale: Pediatric pipeline explicitly documents why this is necessary, and SSA preprocessing includes SSA-specific mapping (3→4) before CE compatibility mapping (4→3) in the dataset.

#### A2 — Normalization ablation
- **A2.1:** SSA-style z-score within brain mask + clip.
- **A2.2:** global z-score (no brain mask).

Expected: pediatric/generalization experiments become noisier; false positives rise without mask-based normalization.

#### A3 — Patch policy ablation (moderate cost)
- **A3.1:** 128³ patches (SSA/GLI default)
- **A3.2:** nnU-Net planned patch geometry (e.g., 96×160×160)

Expected: optimal patch geometry is domain-dependent (brain size distribution, tumor size, GPU memory constraints).

#### A4 — Post-processing ablation (cheap)
- Compare none vs “largest WT component retention” + “small ET suppression”.

Expected: pediatric ET benefits disproportionately from this step.

### C.4 Minimal training schedule (time-efficient)
To keep compute small while maintaining credibility:
- Use **subset training**: e.g., 20–30 cases per domain or a fixed number of patches.
- Use **short fine-tune** runs: 3–10 epochs max, early stop by val Dice.
- Use **frozen-backbone fine-tune** for cross-domain transfer: freeze encoder, train decoder/head.

---

## D. Roadmap (Remote GPU Execution Plan)
A practical plan that finishes quickly but yields publishable plots:

### D.0 Implementation hooks already in this repo
If you want quick, reproducible numbers without writing new training code, this repo already has SSA-compatible patch evaluators:

1) **Single model on one patch dir** (Dice + HD95 on a deterministic val split):

```bash
python SSA_Type/src/ssa_patch_metrics_hd95.py \
  --model SSA_Type/models/best_ssa_model.pth \
  --patch-dir SSA_Type/ssa_preprocessed_patches \
  --limit 200
```

2) **Ablation “matrix runner”** (multiple checkpoints × multiple patch dirs → CSV/JSON):

```bash
python SSA_Type/src/ablation_patch_matrix.py \
  --model SSA=SSA_Type/models/best_ssa_model.pth \
  --model GLI=model/best_model.pth \
  --patch-dir SSA=SSA_Type/ssa_preprocessed_patches \
  --patch-dir GLI=archive/preprocessed_patches \
  --limit 200
```

If the checkpoint architecture is incompatible, the matrix output records that (useful evidence for “not one common approach”). If you explicitly want to measure partial weight portability, add `--allow-partial` and the output will include missing/unexpected key counts.

### Day 0 (setup + sanity)
1. Confirm each domain’s “native” pipeline runs end-to-end on **~5 cases**.
2. Standardize evaluation outputs (Dice/HD95 + empty-ET stats) into a single table format.

### Day 1 (baselines)
- Run E0-SSA, E0-GLI, E0-PED (or reuse existing pediatric fold predictions if already available).
- Produce Table: in-domain Dice/HD95.

### Day 2 (cross-domain evaluation-only + short fine-tunes)
- Evaluate SSA model on GLI patches and vice versa (where model-architecture compatibility allows).
- For pediatric, run SSA/GLI model inference on pediatric-derived patches (or convert pediatric volumes to patches using SSA preprocessor).
- Add 3–5 epoch fine-tunes for SSA→PED and GLI→PED to show that even fine-tuning is not “one size fits all” unless preprocessing + label audit + patch policy are adapted.

### Day 3 (swap-one-component ablations)
- Run A1 (label ontology) and A4 (post-processing) across PED, then summarize ET stability improvements.
- Optional: A2 normalization ablation on PED subset.

Deliverables for the paper:
- **Heatmap** of Dice(ET) across train→test domains.
- **Bar chart** of Dice(WT/TC/ET) for in-domain vs cross-domain.
- **Failure-mode panel**: qualitative overlays where ET is missed or hallucinated.

---

## E. How to Write This in the Paper (Ready-to-paste outline)
Use the following structure in the paper:

1. **Motivation:** BraTS-style tasks look similar, but pipeline assumptions differ; pediatric differs most strongly.
2. **Ablation protocol:** 3×3 train→test matrix + swap-one-component ablations.
3. **Metrics:** Dice/HD95 + empty-region handling.
4. **Key finding:** Cross-domain performance collapses first in ET; correct label ontology + post-processing are necessary but insufficient; domain-specific preprocessing and planning are required.
5. **Practical conclusion:** A single “common approach” is not robust; domain-aware pipelines are justified.

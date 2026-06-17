# Arousal Recognition from EDA Alone — K-EmoCon

Binary **arousal** recognition from a single wearable signal — the Empatica E4
**electrodermal activity (EDA)** — on the KAIST **K-EmoCon** dataset.
CS.50605 / DS.50202 · IoT Data Science.

**Author:** Gyeongmin Na

---

## Task
- Predict **binary arousal** (Low vs High, from `external.arousal`) using **EDA only**.
- 5-second segments at 4 Hz (20 samples each); 1,919 segments from 16 participants.
- Evaluation: predefined **participant-independent Group 4-Fold CV**, seed 42
  (no participant appears in both train and test).
- Baselines: course **DNN — 65.45%**, and a prior winning **EDA ConvLSTM — 74.78%**
  (used as an additional baseline).

## Key insight
EDA's *mean* predicts arousal at AUROC **0.80** when pooled across people, but only
**0.49 (chance)** after per-participant normalization. So the absolute EDA *level*
mostly encodes **which participant** it is (and each person's labels are heavily
skewed), while the genuine *within-person* arousal signal lives in EDA **variability**.
The features are therefore designed to capture **both** the between-person and
within-person views.

## Approach
- **58 hand-crafted EDA features:** 11 absolute (level + variability), 3 baseline-resting
  (elevation over each participant's pre-debate rest), three within-person views —
  *session-relative*, *session z-score*, *percentile-rank* (11 each) — and 11
  *neighbor-context* (mean of adjacent segments).
- **Model:** an L2-regularized **Logistic Regression**. With only 16 participants it
  generalizes across people, where deep models (ConvLSTM, BiLSTM) overfit.
- **Temporal smoothing:** a centered moving average of the predicted probabilities
  within each participant (window ≈ 9 segments / 45 s), exploiting the 84.9% lag-1
  agreement of consecutive arousal labels. It uses predictions only — never labels —
  so it is leakage-free.

## Result (Group 4-Fold CV, seed 42, EDA only)
| Model | Mean accuracy |
|---|---|
| Baseline DNN | 65.45% |
| Prior ConvLSTM (EDA) | 74.78% |
| Ours: Logistic Regression on EDA features | 76.19% |
| **Ours: + temporal smoothing (W = 9)** | **77.75%** |

macro-F1 ≈ 0.77, macro-AUROC ≈ 0.83; robust at 76.6–78.1% across the regularization
strength and the smoothing window.

## Notebook structure
`K_EmoCon_Gyeongmin Na.ipynb`
1. Preparation
2. Preprocessing
3. DNN model
4. Training & Evaluation
5. Bonus — the EDA-feature + Logistic Regression + temporal-smoothing pipeline (the contribution)

## Reproducibility
- Seed 42; fixed folds; the scaler and model are fit on the **training fold only** (no leakage).
- Deterministic; trains in under a second per fold, no GPU required.
- The **dataset is not included** (`K-EmoCon.CS592.pkl` is large and access-controlled).
  Set the dataset path at the top of the notebook, then **Run All** in Google Colab or Jupyter.

## References
- Park, C. Y., et al. (2020). *K-EmoCon, a multimodal sensor dataset for continuous emotion recognition in naturalistic conversations.* Scientific Data, 7, 293.
- Pedregosa, F., et al. (2011). *Scikit-learn: Machine Learning in Python.* JMLR, 12, 2825–2830.
- Virtanen, P., et al. (2020). *SciPy 1.0.* Nature Methods, 17, 261–272.
- Boucsein, W. (2012). *Electrodermal Activity* (2nd ed.). Springer.

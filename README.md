# Self-calibrated Probabilistic Variation Membership Inference Attack (SPV-MIA)

**SPV-MIA** is a neighborhood-based membership inference attack (MIA) that **self-calibrates** its decision statistic against a reference model to suppress false positives on globally “easy” text while preserving power on truly memorized samples.

> **TL;DR**: We compare how much *better* the target model fits a sample than expected from a typical model, **and** relative to its perturbed neighbors. This calibrated gap yields strong precision at very low FPR.

---

## Motivation

Classic score-based MIAs (loss/confidence) often flag **non-members** that happen to be easy or generic (“Thanks for your email.”). Neighborhood MIAs help by contrasting the sample with perturbed variants, but they can still trigger on globally easy text.

**SPV-MIA** corrects for this by subtracting what a *reference* model would achieve on the **same sample and the same neighborhood**—turning the statistic into a difficulty-aware signal. If a sentence is easy for everyone, the calibrated score shrinks; if only the target is unusually confident (e.g., due to memorization), the score pops.

---
## Method Overview

Let $LL_T(x)$ be the token-level log-likelihood (negative loss) of text $x$ under the **target** model $T$.  
Let $N(x)$ be a set of small, semantics-preserving perturbations (mask-fill edits).

**Neighborhood gap (target)**

$$
g_T(x) = LL_T(x) - \frac{1}{|N(x)|}\sum_{z \in N(x)} LL_T(z)
$$

**Neighborhood gap (reference model $R$)**

$$
g_R(x) = LL_R(x) - \frac{1}{|N(x)|}\sum_{z \in N(x)} LL_R(z)
$$

**Self-calibrated statistic**

$$
\Phi_{\mathrm{cal}}(x) = g_T(x) - g_R(x)
$$

**Z-normalized variant (better at low FPR)**

$$
\Phi_{\mathrm{zcal}}(x) = \frac{g_T(x)}{\sigma_T(x)} - \frac{g_R(x)}{\sigma_R(x)}
$$

> **Plain-text fallback:**  
> `g_T(x) = LL_T(x) - mean_{z in N(x)} LL_T(z)`  
> `g_R(x) = LL_R(x) - mean_{z in N(x)} LL_R(z)`  
> `Phi_cal(x) = g_T(x) - g_R(x)`  
> `Phi_zcal(x) = g_T(x)/sigma_T(x) - g_R(x)/sigma_R(x)`


## What’s in this repo

- `calibrated_neighborhood_attack.py` — **New orchestration script** for SPV-MIA that reuses your project utilities.  
  It wires up datasets, neighborhood generation, target/reference models, scoring, and metrics.
- `run_mia_unified.py` — Existing project utilities (model loading, neighbor generation, scoring, ROC/PR metrics).
- `custom_datasets.py` — Dataset loaders and preprocessing helpers.

> **Note:** If `--ref_model` is provided, the existing scoring in `run_mia_unified.py` can compute ΔLL internally (target − reference), so the standard neighborhood gap becomes **self-calibrated** automatically.

---

## Installation

```bash
# Python 3.10+ recommended
pip install -U torch transformers datasets numpy scikit-learn

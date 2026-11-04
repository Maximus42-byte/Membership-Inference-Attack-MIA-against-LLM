# RRN-MIA: Residual Rank Neighbourhood Membership Inference Attack

Calibrated neighbourhood-based membership inference attacks against fine-tuned
language models.

This repository accompanies the M.Sc. thesis *A Unified Calibration Framework for
Neighbourhood-Based Membership Inference Attacks against Large Language Models*
(Mahdi Saiedi, Sharif University of Technology, 2026). It extends the
neighbourhood comparison attack of Mattern et al. (ACL 2023) with reference-model
residual calibration and non-parametric rank-based testing, and benchmarks the
resulting family of attacks under a single evaluation setup.

## The problem

The original neighbourhood attack scores a target text `x` by comparing the target
model's log-likelihood of `x` against the mean log-likelihood of `k` perturbed
neighbours:

```
G_T(x) = LL_T(x) − (1/k) · Σ LL_T(x'_i)
```

A large gap suggests the model has seen `x` during training. The weakness is that
some texts are simply more fluent or more conventional than their perturbations, so
*any* language model scores them above their neighbours. The signal is then a
property of the text, not evidence of memorisation.

## The attack family

This repository implements five attacks under one interface, each adding one
correction to the one before it:

| Attack | Idea |
| --- | --- |
| `N-MIA` | Does the target model prefer the original text over its neighbours? |
| `ZN-MIA` | Normalise that preference by the neighbourhood's own variance. |
| `RN-MIA` | Is the preference specific to the target model, after subtracting a reference model? |
| `QN-MIA` | Is the preference extreme within the local neighbourhood (rank-based)? |
| `RRN-MIA` | Is the *target-specific* preference extreme within the local neighbourhood? |

RRN-MIA combines the residual calibration of RN-MIA with the rank-based test of
QN-MIA, and is the strongest of the five in the experiments below.

## How RRN-MIA works

For the target text and every neighbour, compute the residual log-likelihood
between the target model `T` and a small reference model `R`:

```
ΔLL(y) = LL_T(y) − LL_R(y)
```

Then measure how extreme the original text's residual is within its own
neighbourhood, as an empirical p-value:

```
p_RRN(x) = (1 + #{ i : ΔLL(x'_i) ≥ ΔLL(x) }) / (k + 1)

score(x) = 1 − p_RRN(x)
```

A high score means almost no neighbour shows a larger target-specific advantage
than the original text, which is evidence of membership.

### Full algorithm

```
Input: target text x, target model T, reference model R,
       mask-filling model M, neighbour count k

1. Generate neighbours N(x) = {x'_1, ..., x'_k} by masked span infilling with M
2. Compute LL_T over x and all neighbours
3. Compute LL_R over x and all neighbours
4. Form residuals ΔLL(y) = LL_T(y) − LL_R(y) for x and each neighbour
5. p_RRN(x) = (1 + #{ΔLL(x'_i) ≥ ΔLL(x)}) / (k + 1)
6. score(x) = 1 − p_RRN(x);  higher means more member-like
```

### Why `k` matters

RRN-MIA is rank-based, so the neighbour count sets the resolution of the score.
The smallest attainable p-value is `1 / (k + 1)`:

| k | Minimum p-value | Maximum score |
| ---: | ---: | ---: |
| 10 | 0.0909 | 0.9091 |
| 50 | 0.0196 | 0.9804 |
| 100 | 0.0099 | 0.9901 |
| 1000 | 0.0010 | 0.9990 |

With `k = 10` the score cannot resolve the low-FPR region at all, which is exactly
the region that matters for privacy. This is visible in the results: moving from
10 to 50 neighbours is what makes low-FPR performance non-zero.

## Results

Target model: DistilGPT-2 fine-tuned on CNN/DailyMail highlights. Reference model:
pre-trained `distilgpt2`. 1000 members (train split) and 1000 non-members
(validation split).

| Method | k | ROC-AUC | PR-AUC | TPR@1%FPR | TPR@0.1%FPR |
| --- | ---: | ---: | ---: | ---: | ---: |
| Original neighbourhood | 10 | 0.5918 | 0.5698 | 1.10% | 0.00% |
| ZN-MIA | 10 | 0.5876 | 0.5680 | 1.70% | 0.10% |
| RN-MIA | 10 | 0.6547 | 0.6238 | 1.70% | 0.00% |
| QN-MIA | 10 | 0.5098 | 0.5435 | 0.00% | 0.00% |
| QN-MIA | 50 | 0.5519 | 0.5393 | 1.00% | 0.00% |
| RRN-MIA | 10 | 0.6570 | 0.6773 | 0.00% | 0.00% |
| **RRN-MIA** | **50** | **0.6924** | **0.7119** | **8.20%** | **1.80%** |

Against the original neighbourhood attack, RRN-MIA with 50 neighbours improves
ROC-AUC by 0.101 and PR-AUC by 0.142, and raises TPR@1%FPR from 1.10% to 8.20%.

Two findings are worth separating. Residual calibration alone (RN-MIA) improves
ranking but not the low-FPR region. Rank-based testing alone (QN-MIA) is weak,
performing near chance at `k = 10`. Only the combination improves both, which
suggests the generic-text effect has to be removed *before* the local rank test is
meaningful.

### A note on TPR@0.01%FPR

This column is zero for every method, and that is an artefact of the evaluation
rather than a property of the attacks. With 1000 non-members the smallest
resolvable FPR step is 1/1000 = 0.1%, so a 0.01% threshold only admits the
`FPR = 0` operating point. Measuring that region properly requires substantially
more non-member samples.

## Repository layout

| File | Purpose |
| --- | --- |
| `RRN_neighborhood_attack.py` | Main entry point for running the attacks |
| `run_mia_unified.py` | Model loading, neighbour generation, likelihoods, metrics |
| `custom_datasets.py` | Dataset loading and preparation |
| `plot_curves.py` | ROC and PR curves from a result JSON |
| `results/` | Output JSON and plots |

## Setup

Requires Python 3.10 and a CUDA-capable GPU for reasonable runtimes.

```bash
conda create -n mia python=3.10 -y
conda activate mia
pip install -r requirements.txt
```

If the requirements file is incomplete, the core dependencies are:

```bash
pip install -U torch transformers datasets numpy scikit-learn matplotlib tqdm \
    accelerate sentencepiece protobuf
```

The experiments assume a DistilGPT-2 model already fine-tuned on CNN/DailyMail
highlights, at `./ft_distilgpt2_highlights`.

## Running the attack

### Experimental setup

| Component | Value |
| --- | --- |
| Target model | `./ft_distilgpt2_highlights` |
| Reference model | `distilgpt2` |
| Dataset | CNN/DailyMail v3.0.0, `highlights` field |
| Members | `train[:50000]`, 1000 sampled |
| Non-members | `validation`, 1000 sampled |
| Mask-filling model | `t5-small` |
| Mask percentage | 0.20 |
| Span length | 1 |
| Neighbours | 10, 50 |

### RRN-MIA with 50 neighbours

```bash
HF_DATASETS_OFFLINE=1 HF_HUB_OFFLINE=1 python RRN_neighborhood_attack.py \
  --cache_dir ./.hf_cache \
  --dataset_member cnn_dailymail_highlights --dataset_member_key highlights \
  --dataset_nonmember cnn_dailymail_highlights --dataset_nonmember_key highlights \
  --mask_filling_model_name t5-small \
  --pct_words_masked 0.20 \
  --span_length 1 \
  --n_perturbations 50 \
  --n_samples 1000 \
  --batch_size 50 \
  --chunk_size 20 \
  --base_model_name "$(realpath ./ft_distilgpt2_highlights)" \
  --ref_model distilgpt2 \
  --criterion rrn \
  --save_path results/rrn_mia_n50_1000.json
```

Set `--n_perturbations 10` for the 10-neighbour run. Other attacks in the family
are selected with `--criterion`, which accepts `n`, `zn`, `rn`, `qn` and `rrn`.

If you run out of GPU memory, reduce `--batch_size` to 25 and `--chunk_size` to 10.

### Plots

```bash
python plot_curves.py \
  --result_json results/rrn_mia_n50_1000.json \
  --out_dir results/rrn_mia_n50_plots
```

This writes `roc_curve.png` and `pr_curve.png` into the output directory.

## Limitations

1. RRN-MIA requires a reference model, which the original attack does not.
2. It is roughly twice as expensive per sample, since both the target and the
   reference model must score the text and all of its neighbours.
3. Rank resolution is bounded by `1 / (k + 1)`, so small neighbour counts cannot
   reach the low-FPR region at all.
4. Results depend directly on the quality of the generated neighbours.
5. Evaluating TPR at 0.01% FPR needs far more than 1000 non-members.
6. All experiments use a single target model and dataset; the findings have not
   been tested across architectures or domains.

   
## Credits

The neighbourhood comparison attack and parts of the evaluation code come from:

> Mattern, Mireshghallah, Jin, Schölkopf, Sachan and Berg-Kirkpatrick.
> *Membership Inference Attacks against Language Models via Neighbourhood
> Comparison.* ACL 2023. https://github.com/justusmattern27/neighbour-mia

`run_mia_unified.py` and `custom_datasets.py` originate from that repository and
are not covered by this repository's license.

The calibration framework, the ZN/RN/QN/RRN variants and the benchmark in this
repository are the contribution of this work. All other files are original.

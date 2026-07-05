<div dir="rtl" align="right">

# ZN-MIA: Z-score Neighbourhood Membership Inference Attack

## خلاصه کوتاه

**ZN-MIA** یا **Z-score Neighbourhood Membership Inference Attack** یک extension مستقیم از حمله‌ی اصلی **Neighbourhood Attack** است.

حمله‌ی اصلی فقط اختلاف log-likelihood متن اصلی با میانگین neighbourهایش را اندازه می‌گیرد. اما ZN-MIA علاوه بر میانگین، پراکندگی neighbourها را هم در نظر می‌گیرد.

به زبان ساده:

</div>

<div dir="ltr" align="left">

```text
N-MIA  : Is the original text better than its neighbours?
ZN-MIA : Is the original text unusually better than its neighbours,
         relative to local neighbour variance?
```

</div>

<div dir="rtl" align="right">

ایده‌ی اصلی این است که اگر متن اصلی فقط کمی بهتر از neighbourها باشد، ولی neighbourها خودشان خیلی پراکنده باشند، این سیگنال چندان قابل اعتماد نیست. اما اگر همان اختلاف در neighbourhood بسیار پایدار رخ دهد، احتمال membership قوی‌تر می‌شود.

---

## جایگاه ZN-MIA در خانواده حملات

</div>

<div dir="ltr" align="left">

```text
N-MIA   = Original Neighbourhood Attack
ZN-MIA  = Z-score Neighbourhood Attack
RN-MIA  = Residual Neighbourhood Attack
QN-MIA  = Quantile / Rank Neighbourhood Attack
RRN-MIA = Residual Rank Neighbourhood Attack
```

</div>

<div dir="rtl" align="right">

در این خانواده، ZN-MIA اولین upgrade مستقیم روی N-MIA است. این روش هنوز به reference model نیاز ندارد و فقط از همان neighbourهایی استفاده می‌کند که N-MIA هم تولید می‌کند.

---

## حمله اولیه Neighbourhood Attack

در حمله‌ی اصلی، برای هر متن هدف \(x\)، مجموعه‌ای از متن‌های مشابه یا neighbour ساخته می‌شود:

</div>

<div dir="ltr" align="left">

$$
N(x)=\{x'_1,x'_2,\dots,x'_k\}
$$

</div>

<div dir="rtl" align="right">

در پیاده‌سازی ما، این neighbourها با روش زیر ساخته می‌شوند:

1. بخشی از متن با span masking انتخاب می‌شود.
2. بخش‌های ماسک‌شده با مدل T5 پر می‌شوند.
3. چند نسخه‌ی perturb شده از متن اصلی ساخته می‌شود.

سپس target model روی متن اصلی و neighbourها ارزیابی می‌شود:

</div>

<div dir="ltr" align="left">

$$
LL_T(x), \quad LL_T(x'_1), \dots, LL_T(x'_k)
$$

</div>

<div dir="rtl" align="right">

در حمله‌ی اصلی، score به شکل زیر است:

</div>

<div dir="ltr" align="left">

$$
d(x)=LL_T(x)-\frac{1}{k}\sum_{i=1}^{k}LL_T(x'_i)
$$

</div>

<div dir="rtl" align="right">

اگر \(d(x)\) بزرگ باشد، یعنی target model متن اصلی را نسبت به neighbourهایش بهتر می‌شناسد. این می‌تواند نشانه‌ی membership باشد.

---

## مشکل d-score در N-MIA

score اصلی فقط فاصله از میانگین neighbourها را نگاه می‌کند، اما پراکندگی neighbourها را در نظر نمی‌گیرد.

### حالت اول: neighbourhood پایدار

</div>

<div dir="ltr" align="left">

```text
LL_T(x)      = -20
neighbour LL = [-25.1, -25.0, -24.9, -25.2]
mean         = -25.05
std          ≈ 0.11
d            = 5.05
```

</div>

<div dir="rtl" align="right">

اینجا neighbourها بسیار نزدیک به هم هستند. بنابراین اختلاف `5.05` بسیار معنادار است.

### حالت دوم: neighbourhood ناپایدار

</div>

<div dir="ltr" align="left">

```text
LL_T(x)      = -20
neighbour LL = [-30, -22, -27, -18]
mean         = -24.25
std          ≈ 4.5
d            = 4.25
```

</div>

<div dir="rtl" align="right">

اینجا `d` هنوز بزرگ است، اما neighbourها خودشان بسیار پراکنده‌اند. بنابراین اختلاف متن اصلی با میانگین neighbourها به اندازه‌ی حالت اول قابل اعتماد نیست.

مشکل اصلی N-MIA این است که فقط **فاصله از میانگین** را نگاه می‌کند، نه **معناداری فاصله نسبت به پراکندگی محلی**.

---

## ایده‌ی اصلی ZN-MIA

ZN-MIA همان neighbourhood gap را نگه می‌دارد، اما آن را با standard deviation neighbourها normalize می‌کند.

</div>

<div dir="ltr" align="left">

$$
z(x)=
\frac{LL_T(x)-\mu_N(x)}
{\sigma_N(x)+\epsilon}
$$

</div>

<div dir="rtl" align="right">

که در آن:

</div>

<div dir="ltr" align="left">

$$
\mu_N(x)=\frac{1}{k}\sum_{i=1}^{k}LL_T(x'_i)
$$

$$
\sigma_N(x)=std(LL_T(x'_1),\dots,LL_T(x'_k))
$$

</div>

<div dir="rtl" align="right">

و \(\epsilon\) یک مقدار کوچک مثل `1e-8` است تا از تقسیم بر صفر جلوگیری شود.

پس ZN-MIA یک حمله‌ی **sample-specific normalized neighbourhood attack** است.

---

## تفسیر Z-score در ZN-MIA

ZN-MIA می‌پرسد:

> متن اصلی چند standard deviation بالاتر از neighbourهای خودش قرار دارد؟

اگر مقدار \(z(x)\) بزرگ باشد، یعنی متن اصلی نه‌تنها از میانگین neighbourها بهتر است، بلکه این بهتر بودن نسبت به پراکندگی neighbourها هم معنادار است.

بنابراین:

- \(z(x)\) بزرگ‌تر → sample member-likeتر
- \(z(x)\) کوچک‌تر → sample non-member-likeتر

---

## تفاوت N-MIA و ZN-MIA

</div>

<div dir="ltr" align="left">

| Component | N-MIA | ZN-MIA |
|---|---|---|
| Score | Mean gap | Standardized mean gap |
| Formula | \(LL_T(x)-\mu_N(x)\) | \((LL_T(x)-\mu_N(x))/\sigma_N(x)\) |
| Calibration | Mean-based | Mean + variance-based |
| Reference model | No | No |
| Sensitive to noisy neighbourhoods | More | Less |
| Low-FPR behaviour | Moderate | Often better |

</div>

<div dir="rtl" align="right">

---

## تصمیم عضویت

بعد از محاسبه‌ی \(z(x)\)، یک threshold انتخاب می‌شود:

</div>

<div dir="ltr" align="left">

```text
if z(x) > gamma:
    predict member
else:
    predict non-member
```

</div>

<div dir="rtl" align="right">

در ارزیابی ما به جای انتخاب یک threshold ثابت، کل ROC curve و PR curve محاسبه می‌شود.

---

# نصب و آماده‌سازی محیط

## 1. ساخت محیط مجازی

اگر از `venv` استفاده می‌کنید:

</div>

<div dir="ltr" align="left">

```bash
python3 -m venv mia
source mia/bin/activate
```

</div>

<div dir="rtl" align="right">

اگر از `conda` استفاده می‌کنید:

</div>

<div dir="ltr" align="left">

```bash
conda create -n mia python=3.10 -y
conda activate mia
```

</div>

<div dir="rtl" align="right">

## 2. نصب requirements

اگر فایل `requirements.txt` در repo وجود دارد:

</div>

<div dir="ltr" align="left">

```bash
python -m pip install --upgrade pip setuptools wheel
pip install -r requirements.txt
```

</div>

<div dir="rtl" align="right">

اگر بعضی packageها missing بودند، حداقل dependencyهای زیر را نصب کنید:

</div>

<div dir="ltr" align="left">

```bash
pip install -U torch transformers datasets numpy scikit-learn matplotlib tqdm accelerate sentencepiece protobuf
```

</div>

<div dir="rtl" align="right">

برای اطمینان از نصب packageهای اصلی:

</div>

<div dir="ltr" align="left">

```bash
python - <<'PY'
import torch
import transformers
import datasets
import sklearn
import numpy
import matplotlib

print("torch:", torch.__version__)
print("transformers:", transformers.__version__)
print("datasets:", datasets.__version__)
print("sklearn:", sklearn.__version__)
print("numpy:", numpy.__version__)
print("matplotlib:", matplotlib.__version__)
print("CUDA available:", torch.cuda.is_available())
PY
```

</div>

<div dir="rtl" align="right">

---

# فایل‌های مهم شاخه

</div>

<div dir="ltr" align="left">

| File | Purpose |
|---|---|
| `calibrated_neighborhood_attack.py` | اسکریپت اصلی اجرای ZN-MIA |
| `run_mia_unified.py` | توابع مشترک برای load model، تولید neighbour، likelihood و metricها |
| `custom_datasets.py` | dataset utilities |
| `plot_curves.py` | استخراج ROC و PR curve از JSON خروجی |
| `results/` | محل ذخیره نتایج |

</div>

<div dir="rtl" align="right">

---

# Experimental Setup

در آزمایش نهایی ZN-MIA، setup زیر استفاده شده است:

</div>

<div dir="ltr" align="left">

| Component | Value |
|---|---|
| Target model | `./ft_distilgpt2_highlights` |
| Reference model | Not used |
| Dataset | CNN/DailyMail v3.0.0 |
| Member split | `train[:50000]`, field `highlights` |
| Non-member split | `validation`, field `highlights` |
| Number of member samples | 1000 |
| Number of non-member samples | 1000 |
| Mask-filling model | `t5-small` |
| Mask percentage | `0.20` |
| Span length | `1` |
| Number of neighbours | `10` |
| Criterion | `z` |

</div>

<div dir="rtl" align="right">

در این setup، target model قبلاً روی CNN/DailyMail highlights fine-tune شده است. بنابراین memberها از train split و non-memberها از validation split انتخاب می‌شوند.

---

# اجرای ZN-MIA

## اجرای ZN-MIA با 10 همسایه و 1000 نمونه

</div>

<div dir="ltr" align="left">

```bash
rm -f results/zn_mia_n10_1000.json

HF_DATASETS_OFFLINE=1 HF_HUB_OFFLINE=1 python calibrated_neighborhood_attack.py \
  --cache_dir ./.hf_cache \
  --dataset_member cnn_dailymail_highlights --dataset_member_key highlights \
  --dataset_nonmember cnn_dailymail_highlights --dataset_nonmember_key highlights \
  --mask_filling_model_name t5-small \
  --pct_words_masked 0.20 \
  --span_length 1 \
  --n_perturbations 10 \
  --n_samples 1000 \
  --batch_size 50 \
  --chunk_size 20 \
  --base_model_name "$(realpath ./ft_distilgpt2_highlights)" \
  --criterion z \
  --save_path results/zn_mia_n10_1000.json
```

</div>

<div dir="rtl" align="right">

در اجرای صحیح باید مواردی مشابه زیر دیده شود:

</div>

<div dir="ltr" align="left">

```text
Computing log likelihoods: 1000/1000
perturbation_10_z ROC AUC: ...
Saved results to results/zn_mia_n10_1000.json
```

</div>

<div dir="rtl" align="right">

---

# استخراج metricها از فایل خروجی

برای استخراج ROC-AUC، PR-AUC و TPR در FPRهای پایین:

</div>

<div dir="ltr" align="left">

```bash
python - <<'PY'
import json
import numpy as np

p = "results/zn_mia_n10_1000.json"
d = json.load(open(p))

if isinstance(d, list):
    d = d[0]

fpr = np.array(d["metrics"]["fpr"])
tpr = np.array(d["metrics"]["tpr"])

def tpr_at(alpha):
    mask = fpr <= alpha
    return float(tpr[mask].max()) if mask.any() else 0.0

print("file:", p)
print("name:", d.get("name"))
print("criterion:", d.get("criterion"))
print("num_real:", len(d["predictions"]["real"]))
print("num_samples:", len(d["predictions"]["samples"]))
print("raw_results:", len(d.get("raw_results", [])))
print("roc_auc:", d["metrics"]["roc_auc"])
print("pr_auc:", d["pr_metrics"]["pr_auc"])
print("TPR@1%FPR:", tpr_at(0.01))
print("TPR@0.1%FPR:", tpr_at(0.001))
print("TPR@0.01%FPR:", tpr_at(0.0001))
PY
```

</div>

<div dir="rtl" align="right">

خروجی معتبر آزمایش ما:

</div>

<div dir="ltr" align="left">

```text
name: neigh_z_n10
num_real: 1000
num_samples: 1000
raw_results: 1000
roc_auc: 0.587588
pr_auc: 0.5679556388163753
TPR@1%FPR: 0.017
TPR@0.1%FPR: 0.001
TPR@0.01%FPR: 0.001
```

</div>

<div dir="rtl" align="right">

---

# استخراج نمودارهای ROC و PR

برای تولید دو نمودار زیر:

</div>

<div dir="ltr" align="left">

```text
roc_curve.png
pr_curve.png
```

</div>

<div dir="rtl" align="right">

ابتدا فایل `plot_curves.py` را در ریشه پروژه بسازید:

</div>

<div dir="ltr" align="left">

```python
import os
import json
import argparse
import numpy as np
import matplotlib.pyplot as plt


def load_result(path):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if isinstance(data, list):
        data = data[0]

    return data


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--result_json",
        type=str,
        required=True,
        help="Path to result JSON file"
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        default=None,
        help="Directory to save roc_curve.png and pr_curve.png"
    )

    args = parser.parse_args()

    d = load_result(args.result_json)

    out_dir = args.out_dir or os.path.dirname(args.result_json)
    os.makedirs(out_dir, exist_ok=True)

    name = d.get("name", "MIA_Result")

    # ROC curve
    fpr = np.array(d["metrics"]["fpr"], dtype=float)
    tpr = np.array(d["metrics"]["tpr"], dtype=float)
    roc_auc = float(d["metrics"]["roc_auc"])

    plt.figure(figsize=(7, 5))
    plt.plot(fpr, tpr, label=f"{name} (ROC-AUC = {roc_auc:.4f})")
    plt.plot([0, 1], [0, 1], linestyle="--", label="Random baseline")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"ROC Curve - {name}")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "roc_curve.png"), dpi=200)
    plt.close()

    # Precision-Recall curve
    recall = np.array(d["pr_metrics"]["recall"], dtype=float)
    precision = np.array(d["pr_metrics"]["precision"], dtype=float)
    pr_auc = float(d["pr_metrics"]["pr_auc"])

    plt.figure(figsize=(7, 5))
    plt.plot(recall, precision, label=f"{name} (PR-AUC = {pr_auc:.4f})")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title(f"Precision-Recall Curve - {name}")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "pr_curve.png"), dpi=200)
    plt.close()

    print("Saved:", os.path.join(out_dir, "roc_curve.png"))
    print("Saved:", os.path.join(out_dir, "pr_curve.png"))


if __name__ == "__main__":
    main()
```

</div>

<div dir="rtl" align="right">

سپس برای ZN-MIA اجرا کنید:

</div>

<div dir="ltr" align="left">

```bash
python plot_curves.py \
  --result_json results/zn_mia_n10_1000.json \
  --out_dir results/zn_mia_n10_plots
```

</div>

<div dir="rtl" align="right">

خروجی:

</div>

<div dir="ltr" align="left">

```text
results/zn_mia_n10_plots/roc_curve.png
results/zn_mia_n10_plots/pr_curve.png
```

</div>

<div dir="rtl" align="right">

---

# نتایج ZN-MIA

نتیجه معتبر ZN-MIA با 1000 member و 1000 non-member:

</div>

<div dir="ltr" align="left">

| Method | k | Samples | ROC-AUC | PR-AUC | TPR@1%FPR | TPR@0.1%FPR | TPR@0.01%FPR |
|---|---:|---:|---:|---:|---:|---:|---:|
| ZN-MIA | 10 | 1000 | 0.587588 | 0.567956 | 1.70% | 0.10% | 0.10% |

</div>

<div dir="rtl" align="right">

---

# مقایسه با Original، RN، QN و RRN

جدول زیر نتایج فعلی روش‌های مختلف را نشان می‌دهد:

</div>

<div dir="ltr" align="left">

| Method | k | Samples | ROC-AUC | PR-AUC | TPR@1%FPR | TPR@0.1%FPR | TPR@0.01%FPR |
|---|---:|---:|---:|---:|---:|---:|---:|
| Original Neighbourhood | 10 | 1000 | 0.591787 | 0.569768 | 1.10% | 0.00% | 0.00% |
| ZN-MIA | 10 | 1000 | 0.587588 | 0.567956 | 1.70% | 0.10% | 0.10% |
| RN-MIA | 10 | 1000 | 0.654651 | 0.623765 | 1.70% | 0.00% | 0.00% |
| QN-MIA | 10 | 1000 | 0.509758 | 0.543458 | 0.00% | 0.00% | 0.00% |
| QN-MIA | 50 | 1000 | 0.551854 | 0.539276 | 1.00% | 0.00% | 0.00% |
| RRN-MIA | 10 | 1000 | 0.657015 | 0.677290 | 0.00% | 0.00% | 0.00% |
| RRN-MIA | 50 | 1000 | **0.692407** | **0.711859** | **8.20%** | **1.80%** | **0.00%** |

</div>

<div dir="rtl" align="right">

---

## مقایسه ZN-MIA با Original Neighbourhood Attack

</div>

<div dir="ltr" align="left">

| Method | k | ROC-AUC | PR-AUC | TPR@1%FPR | TPR@0.1%FPR | TPR@0.01%FPR |
|---|---:|---:|---:|---:|---:|---:|
| Original Neighbourhood | 10 | **0.591787** | **0.569768** | 1.10% | 0.00% | 0.00% |
| ZN-MIA | 10 | 0.587588 | 0.567956 | **1.70%** | **0.10%** | **0.10%** |

</div>

<div dir="rtl" align="right">

در این setup، Original از نظر ROC-AUC و PR-AUC کمی بهتر است:

</div>

<div dir="ltr" align="left">

```text
ROC-AUC difference = 0.591787 - 0.587588 = 0.004199
PR-AUC difference  = 0.569768 - 0.567956 = 0.001812
```

</div>

<div dir="rtl" align="right">

اما ZN-MIA در ناحیه low-FPR بهتر عمل کرده است:

</div>

<div dir="ltr" align="left">

```text
TPR@1%FPR:
Original = 1.10%
ZN-MIA   = 1.70%

TPR@0.1%FPR:
Original = 0.00%
ZN-MIA   = 0.10%

TPR@0.01%FPR:
Original = 0.00%
ZN-MIA   = 0.10%
```

</div>

<div dir="rtl" align="right">

بنابراین نتیجه‌ی علمی دقیق این است:

> ZN-MIA در این setup باعث افزایش global ROC-AUC نسبت به baseline اصلی نمی‌شود، اما در ناحیه‌ی low-FPR عملکرد بهتری دارد.

---

## مقایسه ZN-MIA با RN-MIA

RN-MIA برخلاف ZN-MIA از یک reference model استفاده می‌کند. بنابراین به جای اینکه فقط local variance را در target model در نظر بگیرد، تلاش می‌کند اثرهای عمومی متن را با reference model حذف کند.

</div>

<div dir="ltr" align="left">

| Method | k | Reference model | ROC-AUC | PR-AUC | TPR@1%FPR |
|---|---:|---|---:|---:|---:|
| ZN-MIA | 10 | No | 0.587588 | 0.567956 | 1.70% |
| RN-MIA | 10 | Yes, `distilgpt2` | **0.654651** | **0.623765** | 1.70% |

</div>

<div dir="rtl" align="right">

RN-MIA از نظر ROC-AUC و PR-AUC بهتر از ZN-MIA است:

</div>

<div dir="ltr" align="left">

```text
ROC-AUC gain of RN over ZN = 0.654651 - 0.587588 = +0.067063
PR-AUC gain of RN over ZN  = 0.623765 - 0.567956 = +0.055809
```

</div>

<div dir="rtl" align="right">

این نشان می‌دهد که reference-based residual calibration در این setup سیگنال membership قوی‌تری نسبت به variance normalization ایجاد کرده است.

---

## مقایسه ZN-MIA با QN-MIA

QN-MIA به جای z-score، از rank / quantile استفاده می‌کند. بنابراین non-parametric است، اما magnitude اختلاف را تا حدی از دست می‌دهد.

</div>

<div dir="ltr" align="left">

| Method | k | ROC-AUC | PR-AUC | TPR@1%FPR |
|---|---:|---:|---:|---:|
| ZN-MIA | 10 | **0.587588** | **0.567956** | **1.70%** |
| QN-MIA | 10 | 0.509758 | 0.543458 | 0.00% |
| QN-MIA | 50 | 0.551854 | 0.539276 | 1.00% |

</div>

<div dir="rtl" align="right">

در نتایج فعلی، ZN-MIA از QN-MIA بهتر عمل کرده است. دلیل احتمالی این است که QN-MIA فقط rank را نگه می‌دارد و اطلاعات magnitude اختلاف likelihood را از دست می‌دهد، درحالی‌که ZN-MIA هنوز magnitude را حفظ می‌کند ولی آن را با variance محلی normalize می‌کند.

---

## مقایسه ZN-MIA با RRN-MIA

RRN-MIA ترکیب residual calibration و rank-based testing است. بنابراین نسبت به ZN-MIA هم reference model دارد و هم rank-based local decision.

</div>

<div dir="ltr" align="left">

| Method | k | Reference model | ROC-AUC | PR-AUC | TPR@1%FPR | TPR@0.1%FPR |
|---|---:|---|---:|---:|---:|---:|
| ZN-MIA | 10 | No | 0.587588 | 0.567956 | 1.70% | 0.10% |
| RRN-MIA | 50 | Yes, `distilgpt2` | **0.692407** | **0.711859** | **8.20%** | **1.80%** |

</div>

<div dir="rtl" align="right">

RRN-MIA بهترین روش فعلی است، اما هزینه‌ی محاسباتی بیشتری دارد و به reference model نیاز دارد. ZN-MIA سبک‌تر است، چون reference model نمی‌خواهد و فقط از target model و neighbourهای همان نمونه استفاده می‌کند.

---

# تحلیل نتایج

در این setup، ZN-MIA از نظر ROC-AUC و PR-AUC تقریباً نزدیک به Original Neighbourhood Attack است، اما کمی پایین‌تر قرار می‌گیرد.

با این حال، ZN-MIA در low-FPR بهتر از Original عمل می‌کند:

- `TPR@1%FPR` از `1.10%` به `1.70%` افزایش یافته است.
- `TPR@0.1%FPR` از `0.00%` به `0.10%` افزایش یافته است.
- `TPR@0.01%FPR` از `0.00%` به `0.10%` افزایش یافته است.

این نشان می‌دهد که variance-aware normalization می‌تواند برای نقاط عملیاتی سخت‌تر مفید باشد، حتی اگر global AUC را افزایش ندهد.

---

# چرا ZN-MIA همیشه ROC-AUC را بهتر نمی‌کند؟

چند دلیل ممکن وجود دارد:

1. با `k=10` تخمین standard deviation neighbourhood ناپایدار است.
2. اگر `std` خیلی کوچک باشد، z-score ممکن است noisy شود.
3. اگر perturbationها کیفیت یکسانی نداشته باشند، variance normalization همیشه کمک نمی‌کند.
4. ZN-MIA reference model ندارد، پس generic difficulty متن را مثل RN-MIA حذف نمی‌کند.
5. z-score هنوز یک فرض scale-based دارد، در حالی که توزیع neighbourها ممکن است skewed یا heavy-tailed باشد.

---

# محدودیت‌ها

ZN-MIA چند محدودیت مهم دارد:

1. اگر تعداد neighbourها کم باشد، تخمین `std` قابل اعتماد نیست.
2. اگر `std` خیلی کوچک باشد، score ممکن است بیش‌ازحد بزرگ شود.
3. کیفیت perturbationها روی نتیجه اثر مستقیم دارد.
4. این روش generic difficulty را مثل RN-MIA با reference model حذف نمی‌کند.
5. نسبت به RRN-MIA، در این setup global و low-FPR performance ضعیف‌تری دارد.
6. برای بررسی بهتر، اجرای `k=50` یا `k=100` برای ZN-MIA هم می‌تواند مفید باشد.

---

# Expected Outputs

بعد از اجرای موفق ZN-MIA، فایل زیر ساخته می‌شود:

</div>

<div dir="ltr" align="left">

```text
results/zn_mia_n10_1000.json
```

</div>

<div dir="rtl" align="right">

بعد از اجرای `plot_curves.py`:

</div>

<div dir="ltr" align="left">

```text
results/zn_mia_n10_plots/roc_curve.png
results/zn_mia_n10_plots/pr_curve.png
```

</div>

<div dir="rtl" align="right">

---

# Summary

</div>

<div dir="ltr" align="left">

```text
ZN-MIA standardizes the original neighbourhood gap by the local standard deviation of the neighbour log-likelihoods. In the 1000-sample CNN/DailyMail highlights evaluation, ZN-MIA achieved ROC-AUC 0.5876 and PR-AUC 0.5680, which is slightly below the original neighbourhood baseline. However, ZN-MIA improved low-FPR detection, increasing TPR@1%FPR from 1.10% to 1.70% and achieving non-zero TPR at 0.1% and 0.01% FPR. This suggests that variance-aware neighbourhood normalization can improve privacy-relevant operating points even when it does not improve global ranking metrics.
```

</div>

<div dir="rtl" align="right">

---

# خلاصه نهایی

ZN-MIA یک extension سبک و مستقیم از حمله‌ی اصلی Neighbourhood Attack است.

حمله‌ی اصلی می‌پرسد:

</div>

<div dir="ltr" align="left">

```text
Is the original text better than its neighbours?
```

</div>

<div dir="rtl" align="right">

اما ZN-MIA می‌پرسد:

</div>

<div dir="ltr" align="left">

```text
Is the original text unusually better than its neighbours,
relative to local neighbour variance?
```

</div>

<div dir="rtl" align="right">

در آزمایش‌های فعلی، ZN-MIA نسبت به Original Neighbourhood Attack در ROC-AUC کمی ضعیف‌تر بود، اما در low-FPR بهتر عمل کرد. این روش نسبت به RN-MIA و RRN-MIA ساده‌تر و ارزان‌تر است، چون به reference model نیاز ندارد، اما همین موضوع باعث می‌شود نتواند generic text difficulty را به اندازه‌ی روش‌های residualized حذف کند.

</div>
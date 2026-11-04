<div dir="rtl" align="right">

# RRN-MIA: Residual Rank Neighbourhood Membership Inference Attack

## خلاصه کوتاه

**RRN-MIA** یا **Residual Rank Neighbourhood Membership Inference Attack** آخرین و قوی‌ترین variant پیشنهادی ما در خانواده‌ی حملات Neighbourhood-based Membership Inference است.

این روش دو ایده‌ی قبلی را ترکیب می‌کند:

1. **RN-MIA**: استفاده از یک مدل مرجع کوچک برای حذف سختی عمومی متن.
2. **QN-MIA**: استفاده از rank / quantile / empirical p-value برای تصمیم‌گیری محلی.

ایده‌ی اصلی RRN-MIA این است:

> اگر متن اصلی نسبت به neighbourهای خودش، بعد از حذف رفتار عمومی reference model، همچنان در tail توزیع residualها قرار بگیرد، احتمال membership آن بیشتر است.

به زبان ساده:

</div>

<div dir="ltr" align="left">

```text
N-MIA   : Does the target model prefer the original text over its neighbours?
RN-MIA  : Is this preference specific to the target model?
QN-MIA  : Is this preference extreme within the local neighbourhood?
RRN-MIA : Is the target-specific preference extreme within the local neighbourhood?
```

</div>

<div dir="rtl" align="right">

---

## جایگاه RRN-MIA در خانواده حملات

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

RRN-MIA از نظر مفهومی قوی‌ترین extension این خانواده است، چون همزمان سه ویژگی مهم دارد:

* **Local comparison**: هر متن با neighbourهای خودش مقایسه می‌شود.
* **Residual calibration**: اثرهای عمومی زبان با reference model کم می‌شود.
* **Rank-based testing**: تصمیم براساس جایگاه متن در توزیع residual-neighbourها گرفته می‌شود.

---

## حمله اولیه چه بود؟

در حمله‌ی اصلی Neighbourhood Attack، برای یک متن هدف (x)، مجموعه‌ای از neighbourها ساخته می‌شود:

</div>

<div dir="ltr" align="left">

$$
N(x)={x'_1,x'_2,\dots,x'_k}
$$

</div>

<div dir="rtl" align="right">

سپس target model روی متن اصلی و neighbourها ارزیابی می‌شود:

</div>

<div dir="ltr" align="left">

$$
LL_T(x), \quad LL_T(x'_1), \dots, LL_T(x'_k)
$$

</div>

<div dir="rtl" align="right">

و score اصلی به شکل زیر تعریف می‌شود:

</div>

<div dir="ltr" align="left">

$$
G_T(x)=LL_T(x)-\frac{1}{k}\sum_{i=1}^{k}LL_T(x'_i)
$$

</div>

<div dir="rtl" align="right">

اگر (G_T(x)) بزرگ باشد، یعنی target model متن اصلی را نسبت به neighbourهایش بهتر می‌شناسد. این می‌تواند نشانه‌ی membership باشد.

اما مشکل این است که بعضی متن‌ها ذاتاً طبیعی‌تر، رایج‌تر یا ساده‌تر از perturbationهایشان هستند. در این حالت، حتی یک مدل عمومی هم ممکن است متن اصلی را بهتر از neighbourها score بدهد. بنابراین سیگنال اصلی همیشه ناشی از memorization نیست.

---

## RN-MIA چه چیزی اضافه می‌کند؟

RN-MIA یک reference model کوچک اضافه می‌کند.

برای هر متن (y)، residual log-likelihood به شکل زیر تعریف می‌شود:

</div>

<div dir="ltr" align="left">

$$
\Delta LL(y)=LL_T(y)-LL_R(y)
$$

</div>

<div dir="rtl" align="right">

که در آن:

* (LL_T(y)): log-likelihood متن (y) تحت target model
* (LL_R(y)): log-likelihood متن (y) تحت reference model

اگر مقدار (\Delta LL(y)) بزرگ باشد، یعنی target model متن را بهتر از reference model می‌شناسد. این سیگنال نسبت به likelihood خام، target-specificتر است.

در RN-MIA، این residual score مستقیماً وارد neighbourhood gap می‌شود و سپس روی آن threshold زده می‌شود.

---

## QN-MIA چه چیزی اضافه می‌کند؟

QN-MIA به جای استفاده از mean/std یا threshold مستقیم، جایگاه متن اصلی را در توزیع neighbourهای خودش بررسی می‌کند.

یعنی می‌پرسد:

> آیا score متن اصلی از score تقریباً همه neighbourها extremeتر است؟

مزیت QN-MIA این است که non-parametric است و فرض Gaussian بودن توزیع neighbourها را لازم ندارد.

---

## RRN-MIA دقیقاً چه کاری انجام می‌دهد؟

در نسخه‌ی پیاده‌سازی‌شده‌ی RRN-MIA، ابتدا برای متن اصلی و همه‌ی neighbourها residual log-likelihood محاسبه می‌شود:

</div>

<div dir="ltr" align="left">

$$
\Delta LL(x)=LL_T(x)-LL_R(x)
$$

$$
\Delta LL(x'_i)=LL_T(x'_i)-LL_R(x'_i)
$$

</div>

<div dir="rtl" align="right">

سپس RRN-MIA بررسی می‌کند که residual متن اصلی نسبت به residual neighbourها چقدر extreme است.

empirical p-value به شکل زیر تعریف می‌شود:

</div>

<div dir="ltr" align="left">

$$
p_{RRN}(x)=
\frac{1+\sum_{i=1}^{k}\mathbf{1}[\Delta LL(x'_i)\geq \Delta LL(x)]}{k+1}
$$

</div>

<div dir="rtl" align="right">

چون در پیاده‌سازی ما score بزرگ‌تر یعنی member-likeتر، score نهایی به شکل زیر تعریف می‌شود:

</div>

<div dir="ltr" align="left">

$$
RRN_score(x)=1-p_{RRN}(x)
$$

</div>

<div dir="rtl" align="right">

پس اگر تقریباً هیچ neighbourی residual بالاتری از متن اصلی نداشته باشد، (p_{RRN}(x)) کوچک و (RRN_score(x)) بزرگ می‌شود. این یعنی متن اصلی از نظر target-specific advantage در tail محلی neighbourhood قرار دارد.

---

## تفاوت RRN-MIA با RN-MIA

RN-MIA روی residual neighbourhood gap یک threshold می‌زند.

اما RRN-MIA residual متن اصلی را نسبت به residualهای neighbourهای خودش rank می‌کند.

بنابراین:

</div>

<div dir="ltr" align="left">

```text
RN-MIA  = residual score-based attack
RRN-MIA = residual rank-based attack
```

</div>

<div dir="rtl" align="right">

RN-MIA می‌پرسد:

> آیا target-specific advantage متن بزرگ است؟

اما RRN-MIA می‌پرسد:

> آیا target-specific advantage متن نسبت به neighbourهای خودش extreme است؟

---

## تفاوت RRN-MIA با QN-MIA

QN-MIA rank را روی likelihoodهای target model می‌زند.

RRN-MIA rank را روی residual likelihoodهای target-reference می‌زند.

بنابراین QN-MIA می‌پرسد:

> آیا target model متن اصلی را نسبت به neighbourها extreme می‌بیند؟

اما RRN-MIA می‌پرسد:

> آیا target-specific advantage متن اصلی نسبت به neighbourها extreme است؟

این تفاوت مهم است، چون RRN-MIA اثرهای عمومی زبان را قبل از rank کردن کم می‌کند.

---

## نکته مهم درباره تعداد neighbourها

RRN-MIA یک روش rank-based است. بنابراین تعداد neighbourها مستقیماً روی resolution score اثر دارد.

کوچک‌ترین p-value ممکن برابر است با:

</div>

<div dir="ltr" align="left">

$$
p_{min}=\frac{1}{k+1}
$$

</div>

<div dir="rtl" align="right">

و بیشترین score ممکن برابر است با:

</div>

<div dir="ltr" align="left">

$$
RRN_score_{max}=1-\frac{1}{k+1}
$$

|    k | Minimum p-value | Maximum RRN score |
| ---: | --------------: | ----------------: |
|   10 |          0.0909 |            0.9091 |
|   50 |          0.0196 |            0.9804 |
|  100 |          0.0099 |            0.9901 |
| 1000 |          0.0010 |            0.9990 |

</div>

<div dir="rtl" align="right">

به همین دلیل، نسخه‌ی `k=10` برای low-FPR resolution کافی ندارد. در آزمایش‌های ما، وقتی تعداد neighbourها از 10 به 50 افزایش یافت، عملکرد low-FPR به شکل چشمگیری بهتر شد.

---

## الگوریتم RRN-MIA

</div>

<div dir="ltr" align="left">

```text
Input:
    x                  target text
    T                  target language model
    R                  small reference model
    M                  mask-filling model, e.g. T5
    k                  number of neighbours

Step 1:
    Generate neighbours:
        N(x) = {x'_1, ..., x'_k}

Step 2:
    Compute target log-likelihoods:
        LL_T(x), LL_T(x'_1), ..., LL_T(x'_k)

Step 3:
    Compute reference log-likelihoods:
        LL_R(x), LL_R(x'_1), ..., LL_R(x'_k)

Step 4:
    Compute residual log-likelihoods:
        DeltaLL(x)    = LL_T(x)    - LL_R(x)
        DeltaLL(x'_i) = LL_T(x'_i) - LL_R(x'_i)

Step 5:
    Compute empirical p-value:
        p_RRN(x) = (1 + count_i[DeltaLL(x'_i) >= DeltaLL(x)]) / (k + 1)

Step 6:
    Convert p-value to membership score:
        RRN_score(x) = 1 - p_RRN(x)

Step 7:
    Larger RRN_score means more member-like.
```

</div>

<div dir="rtl" align="right">

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

اگر نصب کامل نبود یا بعضی packageها missing بودند، حداقل dependencyهای زیر را نصب کنید:

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

| File                         | Purpose                                                               |
| ---------------------------- | --------------------------------------------------------------------- |
| `RRN_neighborhood_attack.py` | اسکریپت اصلی اجرای RRN-MIA                                            |
| `run_mia_unified.py`         | توابع مشترک برای load model، تولید neighbourها، likelihood و metricها |
| `custom_datasets.py`         | dataset utilities                                                     |
| `plot_curves.py`             | استخراج نمودارهای ROC و PR از JSON خروجی                              |
| `results/`                   | محل ذخیره نتایج                                                       |

</div>

<div dir="rtl" align="right">

---

# Experimental Setup

در آزمایش‌های نهایی این شاخه، setup زیر استفاده شده است:

</div>

<div dir="ltr" align="left">

| Component                    | Value                               |
| ---------------------------- | ----------------------------------- |
| Target model                 | `./ft_distilgpt2_highlights`        |
| Reference model              | `distilgpt2`                        |
| Dataset                      | CNN/DailyMail v3.0.0                |
| Member split                 | `train[:50000]`, field `highlights` |
| Non-member split             | `validation`, field `highlights`    |
| Number of member samples     | 1000                                |
| Number of non-member samples | 1000                                |
| Mask-filling model           | `t5-small`                          |
| Mask percentage              | `0.20`                              |
| Span length                  | `1`                                 |
| Neighbours tested            | `10`, `50`                          |
| Criterion                    | `rrn`                               |

</div>

<div dir="rtl" align="right">

در این setup، target model قبلاً روی CNN/DailyMail highlights fine-tune شده است. بنابراین memberها از train split و non-memberها از validation split انتخاب می‌شوند.

---

# اجرای آزمایش‌ها

## اجرای RRN-MIA با 10 همسایه

</div>

<div dir="ltr" align="left">

```bash
rm -f results/rrn_mia_n10_1000.json

HF_DATASETS_OFFLINE=1 HF_HUB_OFFLINE=1 python RRN_neighborhood_attack.py \
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
  --ref_model distilgpt2 \
  --criterion rrn \
  --save_path results/rrn_mia_n10_1000.json
```

</div>

<div dir="rtl" align="right">

## اجرای RRN-MIA با 50 همسایه

</div>

<div dir="ltr" align="left">

```bash
rm -f results/rrn_mia_n50_1000.json

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

</div>

<div dir="rtl" align="right">

اگر CUDA memory کم بود، مقدارهای زیر را کوچک‌تر کنید:

</div>

<div dir="ltr" align="left">

```bash
--batch_size 25 \
--chunk_size 10
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

p = "results/rrn_mia_n50_1000.json"
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
print("internal_score:", d.get("internal_score"))
print("num_real:", len(d["predictions"]["real"]))
print("num_samples:", len(d["predictions"]["samples"]))
print("raw_results:", len(d.get("raw_results", [])))
print("roc_auc:", d["metrics"]["roc_auc"])
print("pr_auc:", d["pr_metrics"]["pr_auc"])
print("TPR@1%FPR:", tpr_at(0.01))
print("TPR@0.1%FPR:", tpr_at(0.001))
print("TPR@0.01%FPR:", tpr_at(0.0001))

scores_real = np.array(d["predictions"]["real"])
scores_mem = np.array(d["predictions"]["samples"])

print("real score min/max:", float(scores_real.min()), float(scores_real.max()))
print("member score min/max:", float(scores_mem.min()), float(scores_mem.max()))
print("unique real scores:", len(np.unique(scores_real)))
print("unique member scores:", len(np.unique(scores_mem)))
PY
```

</div>

<div dir="rtl" align="right">

---

# استخراج نمودارهای ROC و PR

برای تولید نمودارهای زیر:

</div>

<div dir="ltr" align="left">

```text
roc_curve.png
pr_curve.png
```

</div>

<div dir="rtl" align="right">

ابتدا فایل زیر را با نام `plot_curves.py` در ریشه پروژه بسازید:

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

برای نتیجه‌ی `RRN_MIA_n50` اجرا کنید:

</div>

<div dir="ltr" align="left">

```bash
python plot_curves.py \
  --result_json results/rrn_mia_n50_1000.json \
  --out_dir results/rrn_mia_n50_plots
```

</div>

<div dir="rtl" align="right">

خروجی:

</div>

<div dir="ltr" align="left">

```text
results/rrn_mia_n50_plots/roc_curve.png
results/rrn_mia_n50_plots/pr_curve.png
```

</div>

<div dir="rtl" align="right">

برای نتیجه‌ی `RRN_MIA_n10`:

</div>

<div dir="ltr" align="left">

```bash
python plot_curves.py \
  --result_json results/rrn_mia_n10_1000.json \
  --out_dir results/rrn_mia_n10_plots
```

</div>

<div dir="rtl" align="right">

---

# نتایج RRN-MIA

نتایج معتبر با 1000 member و 1000 non-member:

</div>

<div dir="ltr" align="left">

| Method  |  k | Samples |      ROC-AUC |       PR-AUC | TPR@1%FPR | TPR@0.1%FPR | TPR@0.01%FPR |
| ------- | -: | ------: | -----------: | -----------: | --------: | ----------: | -----------: |
| RRN-MIA | 10 |    1000 |     0.657015 |     0.677290 |     0.00% |       0.00% |        0.00% |
| RRN-MIA | 50 |    1000 | **0.692407** | **0.711859** | **8.20%** |   **1.80%** |    **0.00%** |

</div>

<div dir="rtl" align="right">

افزایش تعداد neighbourها از 10 به 50 باعث شد:

</div>

<div dir="ltr" align="left">

```text
ROC-AUC gain = 0.692407 - 0.657015 = +0.035392
PR-AUC gain  = 0.711859 - 0.677290 = +0.034569

TPR@1%FPR increased from 0.00% to 8.20%
TPR@0.1%FPR increased from 0.00% to 1.80%
```

</div>

<div dir="rtl" align="right">

این نتیجه با ماهیت rank-based روش سازگار است، چون هرچه تعداد neighbourها بیشتر باشد، rank score resolution بهتری دارد.

---

# مقایسه با Original، ZN، RN و QN

جدول زیر نتایج فعلی روش‌های مختلف را نشان می‌دهد:

</div>

<div dir="ltr" align="left">

| Method                 |  k | Samples |      ROC-AUC |       PR-AUC | TPR@1%FPR | TPR@0.1%FPR | TPR@0.01%FPR |
| ---------------------- | -: | ------: | -----------: | -----------: | --------: | ----------: | -----------: |
| Original Neighbourhood | 10 |    1000 |     0.591787 |     0.569768 |     1.10% |       0.00% |        0.00% |
| ZN-MIA                 | 10 |    1000 |     0.587588 |     0.567956 |     1.70% |       0.10% |        0.10% |
| RN-MIA                 | 10 |    1000 |     0.654651 |     0.623765 |     1.70% |       0.00% |        0.00% |
| QN-MIA                 | 10 |    1000 |     0.509758 |     0.543458 |     0.00% |       0.00% |        0.00% |
| QN-MIA                 | 50 |    1000 |     0.551854 |     0.539276 |     1.00% |       0.00% |        0.00% |
| RRN-MIA                | 10 |    1000 |     0.657015 |     0.677290 |     0.00% |       0.00% |        0.00% |
| RRN-MIA                | 50 |    1000 | **0.692407** | **0.711859** | **8.20%** |   **1.80%** |    **0.00%** |

</div>

<div dir="rtl" align="right">

---

## مقایسه RRN-MIA با RN-MIA

RN-MIA فقط residual score را می‌سازد و سپس روی آن threshold می‌زند.

RRN-MIA یک قدم جلوتر می‌رود و residual متن اصلی را نسبت به residual neighbourهای همان متن rank می‌کند.

در نتایج فعلی:

</div>

<div dir="ltr" align="left">

| Method  |  k |      ROC-AUC |       PR-AUC | TPR@1%FPR | TPR@0.1%FPR |
| ------- | -: | -----------: | -----------: | --------: | ----------: |
| RN-MIA  | 10 |     0.654651 |     0.623765 |     1.70% |       0.00% |
| RRN-MIA | 50 | **0.692407** | **0.711859** | **8.20%** |   **1.80%** |

</div>

<div dir="rtl" align="right">

بهبود RRN-MIA نسبت به RN-MIA:

</div>

<div dir="ltr" align="left">

```text
ROC-AUC gain = 0.692407 - 0.654651 = +0.037756
PR-AUC gain  = 0.711859 - 0.623765 = +0.088094
TPR@1%FPR gain = 8.20% - 1.70% = +6.50 percentage points
TPR@0.1%FPR gain = 1.80% - 0.00% = +1.80 percentage points
```

</div>

<div dir="rtl" align="right">

این نشان می‌دهد که فقط residualization کافی نیست؛ rank-based local testing روی residualها باعث بهتر شدن سیگنال membership شده است.

---

## مقایسه RRN-MIA با QN-MIA

QN-MIA rank را روی scoreهای target model اعمال می‌کند.

RRN-MIA rank را روی residual score اعمال می‌کند:

</div>

<div dir="ltr" align="left">

```text
QN-MIA  : rank(LL_T)
RRN-MIA : rank(LL_T - LL_R)
```

</div>

<div dir="rtl" align="right">

یعنی QN-MIA فقط tail بودن در neighbourhood را بررسی می‌کند، اما RRN-MIA قبل از rank گرفتن، اثر generic بودن متن را هم کم می‌کند.

در نتایج فعلی:

</div>

<div dir="ltr" align="left">

| Method  |  k |      ROC-AUC |       PR-AUC | TPR@1%FPR | TPR@0.1%FPR |
| ------- | -: | -----------: | -----------: | --------: | ----------: |
| QN-MIA  | 10 |     0.509758 |     0.543458 |     0.00% |       0.00% |
| QN-MIA  | 50 |     0.551854 |     0.539276 |     1.00% |       0.00% |
| RRN-MIA | 50 | **0.692407** | **0.711859** | **8.20%** |   **1.80%** |

</div>

<div dir="rtl" align="right">

بهبود RRN-MIA نسبت به QN-MIA با 50 neighbour:

</div>

<div dir="ltr" align="left">

```text
ROC-AUC gain = 0.692407 - 0.551854 = +0.140553
PR-AUC gain  = 0.711859 - 0.539276 = +0.172583
TPR@1%FPR gain = 8.20% - 1.00% = +7.20 percentage points
TPR@0.1%FPR gain = 1.80% - 0.00% = +1.80 percentage points
```

</div>

<div dir="rtl" align="right">

بنابراین RRN-MIA به‌وضوح از QN-MIA بهتر عمل کرده است، چون rank-based testing را با reference-based residual calibration ترکیب می‌کند.

---

## مقایسه RRN-MIA با Original Neighbourhood Attack

در مقایسه با baseline اصلی:

</div>

<div dir="ltr" align="left">

| Method                 |  k |      ROC-AUC |       PR-AUC | TPR@1%FPR | TPR@0.1%FPR |
| ---------------------- | -: | -----------: | -----------: | --------: | ----------: |
| Original Neighbourhood | 10 |     0.591787 |     0.569768 |     1.10% |       0.00% |
| RRN-MIA                | 50 | **0.692407** | **0.711859** | **8.20%** |   **1.80%** |

</div>

<div dir="rtl" align="right">

بهبود:

</div>

<div dir="ltr" align="left">

```text
ROC-AUC gain = 0.692407 - 0.591787 = +0.100620
PR-AUC gain  = 0.711859 - 0.569768 = +0.142091
TPR@1%FPR gain = 8.20% - 1.10% = +7.10 percentage points
TPR@0.1%FPR gain = 1.80% - 0.00% = +1.80 percentage points
```

</div>

<div dir="rtl" align="right">

این نشان می‌دهد که RRN-MIA هم global ranking را بهتر می‌کند و هم در ناحیه‌ی privacy-sensitive یعنی low-FPR عملکرد قوی‌تری دارد.

---

# چرا TPR@0.01%FPR هنوز صفر است؟

در این آزمایش فقط 1000 نمونه non-member داریم. بنابراین کوچک‌ترین step برای FPR تقریباً برابر است با:

</div>

<div dir="ltr" align="left">

$$
\frac{1}{1000}=0.001=0.1%
$$

</div>

<div dir="rtl" align="right">

اما (0.01%) برابر است با:

</div>

<div dir="ltr" align="left">

$$
0.0001
$$

</div>

<div dir="rtl" align="right">

بنابراین با 1000 non-member، معیار `TPR@0.01%FPR` عملاً فقط حالت `FPR = 0` را می‌پذیرد. اگر در این operating point هیچ memberی انتخاب نشود، مقدار آن صفر می‌شود.

پس صفر بودن `TPR@0.01%FPR` الزاماً به معنی ضعف روش نیست؛ بلکه محدودیت resolution ارزیابی با 1000 non-member است.

---

# Expected Outputs

بعد از اجرای موفق RRN-MIA، فایل‌های زیر ساخته می‌شوند:

</div>

<div dir="ltr" align="left">

```text
results/rrn_mia_n10_1000.json
results/rrn_mia_n50_1000.json
```

</div>

<div dir="rtl" align="right">

بعد از اجرای `plot_curves.py`:

</div>

<div dir="ltr" align="left">

```text
results/rrn_mia_n50_plots/roc_curve.png
results/rrn_mia_n50_plots/pr_curve.png
```

</div>

<div dir="rtl" align="right">

---

# Limitations

RRN-MIA چند محدودیت مهم دارد:

1. به reference model نیاز دارد.
2. هزینه‌ی محاسباتی آن بیشتر از حمله‌ی اصلی است، چون target و reference model هر دو باید روی متن اصلی و neighbourها evaluate شوند.
3. با تعداد neighbour کم، rank score resolution محدود است.
4. کیفیت perturbationها روی نتیجه اثر مستقیم دارد.
5. برای ارزیابی دقیق‌تر `TPR@0.01%FPR` باید تعداد non-memberها بیشتر از 1000 باشد.
6. اجرای `k=50` زمان‌برتر از `k=10` است، چون تعداد perturbationها تقریباً پنج برابر می‌شود.

---

# Thesis-ready Summary

</div>

<div dir="ltr" align="left">

```text
RRN-MIA with 50 neighbours achieved the strongest overall performance among the evaluated attacks, reaching ROC-AUC 0.6924 and PR-AUC 0.7119 on 1000 CNN/DailyMail highlight samples. Compared with the original neighbourhood attack, RRN-MIA improved ROC-AUC by 0.1006 and PR-AUC by 0.1421. More importantly, it substantially improved low-FPR performance, increasing TPR@1%FPR from 1.10% to 8.20% and TPR@0.1%FPR from 0.00% to 1.80%. These results suggest that combining residual calibration with rank-based neighbourhood testing provides a stronger and more privacy-relevant membership signal than the original neighbourhood score.
```

</div>

<div dir="rtl" align="right">

---

# خلاصه نهایی

RRN-MIA ترکیب دو ایده‌ی مهم است:

* **Residual calibration** از RN-MIA
* **Rank-based local testing** از QN-MIA

این روش به جای اینکه فقط likelihood متن اصلی را بررسی کند، می‌پرسد:

> آیا target-specific local advantage متن اصلی در neighbourhood خودش extreme است؟

در آزمایش‌های فعلی، RRN-MIA با 50 همسایه بهترین نتیجه را به دست آورد و نسبت به Original، RN-MIA و QN-MIA هم در ROC-AUC، هم در PR-AUC و هم در low-FPR بهبود قابل توجهی ایجاد کرد.

</div>

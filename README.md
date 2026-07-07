<div dir="rtl" align="right">

# RN-MIA: Residual Neighbourhood Membership Inference Attack

## خلاصه کوتاه

**RN-MIA** یا **Residual Neighbourhood Membership Inference Attack** یک extension مستقیم از حمله‌ی اصلی **Neighbourhood Attack** است.

حمله‌ی اصلی برای هر متن \(x\)، متن اصلی را با neighbourهای خودش مقایسه می‌کند و بررسی می‌کند که آیا target model متن اصلی را بهتر از perturbationهای مشابهش می‌شناسد یا نه. اما این سیگنال همیشه membership نیست؛ گاهی متن اصلی ذاتاً طبیعی‌تر، روان‌تر یا ساده‌تر از neighbourهای تولیدشده است و حتی یک مدل عمومی نیز آن را بهتر score می‌دهد.

RN-MIA برای حل این مشکل، یک **reference model** کوچک و عمومی اضافه می‌کند. سپس advantage متن اصلی در target model را با advantage همان متن در reference model مقایسه می‌کند.

به زبان ساده:

</div>

<div dir="ltr" align="left">

```text
N-MIA  : Does the target model prefer the original text over its neighbours?
RN-MIA : Is this preference specific to the target model?
```

</div>

<div dir="rtl" align="right">

اگر target model متن اصلی را نسبت به neighbourها بهتر بداند، اما reference model چنین advantageای نداشته باشد، این سیگنال target-specificتر و membership-likeتر است.

---

## جایگاه RN-MIA در خانواده حملات

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

RN-MIA یک گام مهم از **local difficulty calibration** به سمت **target-specific local membership evidence** است. این روش از rank استفاده نمی‌کند، اما با reference model تلاش می‌کند اثرهای عمومی زبان و artifactهای مشترک بین مدل‌ها را حذف کند.

---

## حمله اولیه Neighbourhood Attack

در حمله‌ی اصلی، برای هر متن هدف \(x\)، یک مجموعه از neighbourها ساخته می‌شود:

</div>

<div dir="ltr" align="left">

$$
N(x)=\{x'_1,x'_2,\dots,x'_k\}
$$

</div>

<div dir="rtl" align="right">

در پیاده‌سازی ما، neighbourها با **span masking + T5 mask filling** ساخته می‌شوند:

1. چند span از متن اصلی mask می‌شود.
2. مدل mask-filling مثل T5 قسمت‌های mask شده را پر می‌کند.
3. \(k\) متن perturb شده ساخته می‌شود.

سپس target model روی متن اصلی و neighbourها ارزیابی می‌شود:

</div>

<div dir="ltr" align="left">

$$
LL_T(x), \quad LL_T(x'_1), \dots, LL_T(x'_k)
$$

</div>

<div dir="rtl" align="right">

در حمله‌ی اصلی، neighbourhood gap برای target model به شکل زیر است:

</div>

<div dir="ltr" align="left">

$$
G_T(x)=LL_T(x)-\frac{1}{k}\sum_{i=1}^{k}LL_T(x'_i)
$$

</div>

<div dir="rtl" align="right">

اگر \(G_T(x)\) بزرگ باشد، یعنی target model متن اصلی را نسبت به neighbourهایش بهتر می‌شناسد. این می‌تواند نشانه‌ی membership باشد.

اما مشکل این است که گاهی این advantage به خاطر memorization نیست، بلکه به خاطر طبیعی‌تر بودن متن اصلی نسبت به perturbationهاست.

---

## ایده اصلی RN-MIA

RN-MIA یک reference model کوچک و عمومی اضافه می‌کند. این reference model نباید روی training data خصوصی target fine-tune شده باشد. نقش آن این است که یک **generic language difficulty meter** باشد.

برای reference model نیز همان neighbourhood gap را محاسبه می‌کنیم:

</div>

<div dir="ltr" align="left">

$$
G_R(x)=LL_R(x)-\frac{1}{k}\sum_{i=1}^{k}LL_R(x'_i)
$$

</div>

<div dir="rtl" align="right">

سپس RN-MIA score به شکل residual تعریف می‌شود:

</div>

<div dir="ltr" align="left">

$$
S_{RN}(x)=G_T(x)-G_R(x)
$$

</div>

<div dir="rtl" align="right">

یعنی:

</div>

<div dir="ltr" align="left">

$$
S_{RN}(x)=\left(LL_T(x)-\frac{1}{k}\sum_{i=1}^{k}LL_T(x'_i)\right)-\left(LL_R(x)-\frac{1}{k}\sum_{i=1}^{k}LL_R(x'_i)\right)
$$

</div>

<div dir="rtl" align="right">

اگر \(S_{RN}(x)\) بزرگ باشد، یعنی local advantage متن اصلی بیشتر مخصوص target model است و احتمال membership بیشتر می‌شود.

---

## فرم معادل پیاده‌سازی‌شده

در پیاده‌سازی branch RN-MIA، تابع `get_ll` به صورت monkey-patch تغییر می‌کند تا به جای log-likelihood خام، residual log-likelihood را برگرداند:

</div>

<div dir="ltr" align="left">

$$
\Delta LL(y)=LL_T(y)-LL_R(y)
$$

</div>

<div dir="rtl" align="right">

سپس همان criterion داخلی `d` از کد اصلی استفاده می‌شود:

</div>

<div dir="ltr" align="left">

$$
d_{\Delta}(x)=
\Delta LL(x)-\frac{1}{k}\sum_{i=1}^{k}\Delta LL(x'_i)
$$

</div>

<div dir="rtl" align="right">

که دقیقاً معادل RN-MIA است:

</div>

<div dir="ltr" align="left">

$$
d_{\Delta}(x)=\left(LL_T(x)-\frac{1}{k}\sum_{i=1}^{k}LL_T(x'_i)\right)-\left(LL_R(x)-\frac{1}{k}\sum_{i=1}^{k}LL_R(x'_i)\right)
$$

</div>

<div dir="rtl" align="right">

بنابراین اگر در logها یا خروجی داخلی اسم criterion برابر `d` دیده شود، به این معنی نیست که حمله original اجرا شده است. در RN-MIA، `d` روی residual likelihood اجرا می‌شود.

---

## تفسیر score

با log-likelihood کار می‌کنیم، بنابراین مقدار بزرگ‌تر بهتر است:

- \(G_T(x)\) بزرگ: target model متن اصلی را نسبت به neighbourها بهتر می‌شناسد.
- \(G_R(x)\) بزرگ: reference model هم همین advantage را می‌بیند.
- \(S_{RN}(x)\) بزرگ: advantage بیشتر مخصوص target model است.

پس:

</div>

<div dir="ltr" align="left">

```text
large S_RN(x)  => stronger membership evidence
small S_RN(x)  => weaker membership evidence
```

</div>

<div dir="rtl" align="right">

اگر با loss کار شود، جهت علامت برعکس می‌شود. در این documentation فرض می‌کنیم scoreها بر اساس log-likelihood هستند.

---

## چرا reference model لازم است؟

در Neighbourhood Attack اولیه، متن اصلی ممکن است از neighbourها بهتر score بگیرد چون:

- متن اصلی طبیعی‌تر از perturbationهای T5 است.
- neighbourها کمی semantic shift دارند.
- متن اصلی ساده‌تر یا رایج‌تر است.
- perturbationها artifact زبانی دارند.
- local neighbourhood noisy است.

اگر این advantage برای reference model نیز وجود داشته باشد، احتمالاً advantage مربوط به خود متن یا perturbation artifact است، نه membership. RN-MIA این بخش مشترک را حذف می‌کند.

---

## چرا reference model باید کوچک و عمومی باشد؟

reference model در RN-MIA قرار نیست یک shadow model دقیق مثل LiRA باشد. هدف آن بازسازی training distribution مدل هدف نیست. هدف آن فقط تخمین difficulty عمومی متن است.

ویژگی‌های مطلوب reference model:

- کوچک باشد.
- عمومی باشد.
- public باشد.
- روی داده‌ی private target fine-tune نشده باشد.
- هزینه inference پایینی داشته باشد.
- fluency و difficulty عمومی متن را تقریب بزند.

نمونه‌های مناسب:

</div>

<div dir="ltr" align="left">

```text
distilgpt2
gpt2
EleutherAI/gpt-neo-125M
facebook/opt-125m
```

</div>

<div dir="rtl" align="right">

در آزمایش‌های فعلی ما از `distilgpt2` به عنوان reference model استفاده شده است.

---

## تفاوت RN-MIA با ZN-MIA

ZN-MIA gap مدل هدف را با standard deviation neighbourهای همان مدل normalize می‌کند:

</div>

<div dir="ltr" align="left">

$$
Z_T(x)=\frac{G_T(x)}{\sigma_T(N(x))}
$$

</div>

<div dir="rtl" align="right">

اما RN-MIA از reference model استفاده می‌کند:

</div>

<div dir="ltr" align="left">

$$
S_{RN}(x)=G_T(x)-G_R(x)
$$

</div>

<div dir="rtl" align="right">

پس تفاوت اصلی این است:

</div>

<div dir="ltr" align="left">

| Method | What it controls |
|---|---|
| ZN-MIA | Local variance of neighbour scores |
| RN-MIA | Generic text difficulty and shared perturbation artifacts |

</div>

<div dir="rtl" align="right">

ZN-MIA سبک‌تر است چون reference model ندارد. اما RN-MIA target-specificتر است، چون effectهای عمومی زبان را با reference model حذف می‌کند.

---

## تفاوت RN-MIA با QN-MIA

QN-MIA rank-based است و جایگاه متن اصلی را در توزیع neighbourهای target model می‌سنجد. اما RN-MIA score-based است و از reference model استفاده می‌کند.

</div>

<div dir="ltr" align="left">

| Method | Main idea | Reference model | Rank-based |
|---|---|---|---|
| QN-MIA | Rank of original among neighbours | No | Yes |
| RN-MIA | Target gap minus reference gap | Yes | No |

</div>

<div dir="rtl" align="right">

QN-MIA non-parametric است، اما magnitude اختلاف را از دست می‌دهد. RN-MIA magnitude را حفظ می‌کند و generic difficulty را حذف می‌کند.

---

## تفاوت RN-MIA با RRN-MIA

RRN-MIA ترکیب RN-MIA و QN-MIA است.

RN-MIA residual score را می‌سازد:

</div>

<div dir="ltr" align="left">

$$
S_{RN}(x)=G_T(x)-G_R(x)
$$

</div>

<div dir="rtl" align="right">

اما RRN-MIA residual likelihood متن اصلی را نسبت به residual likelihood neighbourها rank می‌کند:

</div>

<div dir="ltr" align="left">

$$
\Delta LL(y)=LL_T(y)-LL_R(y)
$$

$$
p_{RRN}(x)=
\frac{1+\sum_i\mathbf{1}[\Delta LL(x'_i)\geq\Delta LL(x)]}{k+1}
$$

$$
RRN\_score(x)=1-p_{RRN}(x)
$$

</div>

<div dir="rtl" align="right">

بنابراین:

</div>

<div dir="ltr" align="left">

```text
RN-MIA  = residual score-based attack
RRN-MIA = residual rank-based attack
```

</div>

<div dir="rtl" align="right">

---

## الگوریتم RN-MIA

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
    Compute target neighbourhood gap:
        G_T = LL_T(x) - mean_i LL_T(x'_i)

Step 5:
    Compute reference neighbourhood gap:
        G_R = LL_R(x) - mean_i LL_R(x'_i)

Step 6:
    Compute residual neighbourhood score:
        S_RN = G_T - G_R

Step 7:
    Larger S_RN means more member-like.
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
| `Residual_neighborhood_attack.py` | اسکریپت اصلی اجرای RN-MIA |
| `run_mia_unified.py` | توابع مشترک برای load model، تولید neighbour، likelihood و metricها |
| `custom_datasets.py` | dataset utilities |
| `plot_curves.py` | استخراج ROC و PR curve از JSON خروجی |
| `results/` | محل ذخیره نتایج |

</div>

<div dir="rtl" align="right">

---

# Experimental Setup

در آزمایش فعلی RN-MIA، setup زیر استفاده شده است:

</div>

<div dir="ltr" align="left">

| Component | Value |
|---|---|
| Target model | `./ft_distilgpt2_highlights` |
| Reference model | `distilgpt2` |
| Dataset | CNN/DailyMail v3.0.0 |
| Member split | `train[:50000]`, field `highlights` |
| Non-member split | `validation`, field `highlights` |
| Number of member samples | 1000 |
| Number of non-member samples | 1000 |
| Mask-filling model | `t5-small` |
| Mask percentage | `0.20` |
| Span length | `1` |
| Number of neighbours | `10` |
| Criterion | `rn` |
| Internal criterion | `d` on residual likelihoods |

</div>

<div dir="rtl" align="right">

در این setup، target model قبلاً روی CNN/DailyMail highlights fine-tune شده است. بنابراین memberها از train split و non-memberها از validation split انتخاب می‌شوند.

---

# اجرای RN-MIA

## اجرای RN-MIA با 10 همسایه و 1000 نمونه

</div>

<div dir="ltr" align="left">

```bash
rm -f results/rn_mia_n10_1000.json

HF_DATASETS_OFFLINE=1 HF_HUB_OFFLINE=1 python Residual_neighborhood_attack.py \
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
  --criterion rn \
  --save_path results/rn_mia_n10_1000.json
```

</div>

<div dir="rtl" align="right">

اگر branch قدیمی‌تر فقط مسیر `results/rn_mia_n10.json` را استفاده می‌کند، می‌توانید همان نام را نگه دارید:

</div>

<div dir="ltr" align="left">

```bash
--save_path results/rn_mia_n10.json
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
from pathlib import Path

p = Path("results/rn_mia_n10_1000.json")
if not p.exists():
    p = Path("results/rn_mia_n10.json")

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
print("internal_criterion:", d.get("internal_criterion"))
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

خروجی معتبر آزمایش فعلی:

</div>

<div dir="ltr" align="left">

```text
name: RN_MIA_n10
criterion: rn
internal_criterion: d
roc_auc: 0.6546505
pr_auc: 0.6237654590415775
TPR@1%FPR: 0.017
TPR@0.1%FPR: 0.0
TPR@0.01%FPR: 0.0
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

برای RN-MIA اجرا کنید:

</div>

<div dir="ltr" align="left">

```bash
python plot_curves.py \
  --result_json results/rn_mia_n10_1000.json \
  --out_dir results/rn_mia_n10_plots
```

</div>

<div dir="rtl" align="right">

اگر فایل شما با نام قدیمی ذخیره شده است:

</div>

<div dir="ltr" align="left">

```bash
python plot_curves.py \
  --result_json results/rn_mia_n10.json \
  --out_dir results/rn_mia_n10_plots
```

</div>

<div dir="rtl" align="right">

خروجی:

</div>

<div dir="ltr" align="left">

```text
results/rn_mia_n10_plots/roc_curve.png
results/rn_mia_n10_plots/pr_curve.png
```

</div>

<div dir="rtl" align="right">

---

# نتایج RN-MIA

نتیجه معتبر RN-MIA با 1000 member و 1000 non-member:

</div>

<div dir="ltr" align="left">

| Method | k | Samples | ROC-AUC | PR-AUC | TPR@1%FPR | TPR@0.1%FPR | TPR@0.01%FPR |
|---|---:|---:|---:|---:|---:|---:|---:|
| RN-MIA | 10 | 1000 | **0.654651** | **0.623765** | **1.70%** | 0.00% | 0.00% |

</div>

<div dir="rtl" align="right">

RN-MIA نسبت به Original و ZN-MIA بهبود واضحی در ROC-AUC و PR-AUC ایجاد کرده است. این نشان می‌دهد که reference-based residual calibration در این setup سیگنال membership قوی‌تری نسبت به neighbourhood gap خام یا z-normalized gap فراهم کرده است.

---

# مقایسه با Original، ZN، QN و RRN

جدول زیر نتایج فعلی روش‌های مختلف را نشان می‌دهد:

</div>

<div dir="ltr" align="left">

| Method | k | Samples | ROC-AUC | PR-AUC | TPR@1%FPR | TPR@0.1%FPR | TPR@0.01%FPR |
|---|---:|---:|---:|---:|---:|---:|---:|
| Original Neighbourhood | 10 | 1000 | 0.591787 | 0.569768 | 1.10% | 0.00% | 0.00% |
| ZN-MIA | 10 | 1000 | 0.587588 | 0.567956 | 1.70% | 0.10% | 0.10% |
| RN-MIA | 10 | 1000 | **0.654651** | **0.623765** | **1.70%** | 0.00% | 0.00% |
| QN-MIA | 10 | 1000 | 0.509758 | 0.543458 | 0.00% | 0.00% | 0.00% |
| QN-MIA | 50 | 1000 | 0.551854 | 0.539276 | 1.00% | 0.00% | 0.00% |
| RRN-MIA | 10 | 1000 | 0.657015 | 0.677290 | 0.00% | 0.00% | 0.00% |
| RRN-MIA | 50 | 1000 | **0.692407** | **0.711859** | **8.20%** | **1.80%** | 0.00% |

</div>

<div dir="rtl" align="right">

---

## مقایسه RN-MIA با Original Neighbourhood Attack

</div>

<div dir="ltr" align="left">

| Method | k | ROC-AUC | PR-AUC | TPR@1%FPR |
|---|---:|---:|---:|---:|
| Original Neighbourhood | 10 | 0.591787 | 0.569768 | 1.10% |
| RN-MIA | 10 | **0.654651** | **0.623765** | **1.70%** |

</div>

<div dir="rtl" align="right">

بهبود RN-MIA نسبت به Original:

</div>

<div dir="ltr" align="left">

```text
ROC-AUC gain = 0.654651 - 0.591787 = +0.062864
PR-AUC gain  = 0.623765 - 0.569768 = +0.053997
TPR@1%FPR gain = 1.70% - 1.10% = +0.60 percentage points
```

</div>

<div dir="rtl" align="right">

این نشان می‌دهد که حذف رفتار مشترک reference model باعث تقویت global membership signal شده است.

---

## مقایسه RN-MIA با ZN-MIA

</div>

<div dir="ltr" align="left">

| Method | k | Reference model | ROC-AUC | PR-AUC | TPR@1%FPR | TPR@0.1%FPR |
|---|---:|---|---:|---:|---:|---:|
| ZN-MIA | 10 | No | 0.587588 | 0.567956 | 1.70% | **0.10%** |
| RN-MIA | 10 | Yes, `distilgpt2` | **0.654651** | **0.623765** | 1.70% | 0.00% |

</div>

<div dir="rtl" align="right">

RN-MIA از نظر ROC-AUC و PR-AUC بهتر از ZN-MIA است:

</div>

<div dir="ltr" align="left">

```text
ROC-AUC gain = 0.654651 - 0.587588 = +0.067063
PR-AUC gain  = 0.623765 - 0.567956 = +0.055809
```

</div>

<div dir="rtl" align="right">

اما ZN-MIA در این run خاص در `TPR@0.1%FPR` و `TPR@0.01%FPR` مقدار غیرصفر دارد، درحالی‌که RN-MIA صفر است. بنابراین RN-MIA global ranking را بهتر کرده، اما low-FPR خیلی شدید هنوز محدود باقی مانده است.

---

## مقایسه RN-MIA با QN-MIA

</div>

<div dir="ltr" align="left">

| Method | k | Reference model | ROC-AUC | PR-AUC | TPR@1%FPR |
|---|---:|---|---:|---:|---:|
| QN-MIA | 10 | No | 0.509758 | 0.543458 | 0.00% |
| QN-MIA | 50 | No | 0.551854 | 0.539276 | 1.00% |
| RN-MIA | 10 | Yes, `distilgpt2` | **0.654651** | **0.623765** | **1.70%** |

</div>

<div dir="rtl" align="right">

RN-MIA به‌وضوح از QN-MIA بهتر است. این نشان می‌دهد که در این setup، حذف generic text difficulty با reference model مهم‌تر از rank-only scoring بوده است.

---

## مقایسه RN-MIA با RRN-MIA

RRN-MIA نسخه‌ی rank-based از ایده‌ی residual است. یعنی به جای threshold زدن روی residual neighbourhood gap، residual likelihood متن اصلی را نسبت به residual likelihood همسایه‌ها rank می‌کند.

</div>

<div dir="ltr" align="left">

| Method | k | Reference model | ROC-AUC | PR-AUC | TPR@1%FPR | TPR@0.1%FPR |
|---|---:|---|---:|---:|---:|---:|
| RN-MIA | 10 | Yes, `distilgpt2` | 0.654651 | 0.623765 | 1.70% | 0.00% |
| RRN-MIA | 10 | Yes, `distilgpt2` | 0.657015 | **0.677290** | 0.00% | 0.00% |
| RRN-MIA | 50 | Yes, `distilgpt2` | **0.692407** | **0.711859** | **8.20%** | **1.80%** |

</div>

<div dir="rtl" align="right">

RRN-MIA با 50 neighbour بهترین نتیجه را دارد. این نشان می‌دهد که residualization به تنهایی مفید است، اما ترکیب آن با rank-based local testing و افزایش تعداد neighbourها عملکرد را به‌طور قابل توجهی بهتر می‌کند.

---

# تحلیل نتایج

در این setup، RN-MIA نسبت به Original و ZN-MIA از نظر ROC-AUC و PR-AUC بهتر عمل کرده است:

- ROC-AUC از `0.591787` در Original به `0.654651` در RN-MIA افزایش یافته است.
- PR-AUC از `0.569768` در Original به `0.623765` در RN-MIA افزایش یافته است.
- TPR@1%FPR از `1.10%` به `1.70%` افزایش یافته است.

این نتیجه نشان می‌دهد که reference-based residual calibration یک membership signal قوی‌تر ایجاد کرده است.

با این حال، در FPRهای خیلی پایین‌تر، RN-MIA هنوز TPR صفر دارد. این یعنی RN-MIA global separability را بهتر کرده، اما برای low-FPR شدید، rank-based residual testing با تعداد neighbour بیشتر، یعنی RRN-MIA، بهتر عمل کرده است.

---

# چرا RN-MIA در low-FPR خیلی شدید هنوز محدود است؟

چند دلیل محتمل:

1. RN-MIA هنوز score-based است و threshold روی residual gap می‌زند.
2. با \(k=10\)، تخمین neighbourhood gap noisy است.
3. reference model ممکن است بخشی از signal واقعی را هم حذف کند.
4. low-FPR شدید با 1000 non-member resolution محدودی دارد.
5. RRN-MIA با rank-based testing و \(k=50\) در tail بهتر عمل می‌کند.

---

# Expected Outputs

بعد از اجرای موفق RN-MIA، فایل زیر ساخته می‌شود:

</div>

<div dir="ltr" align="left">

```text
results/rn_mia_n10_1000.json
```

</div>

<div dir="rtl" align="right">

یا در branchهای قدیمی‌تر:

</div>

<div dir="ltr" align="left">

```text
results/rn_mia_n10.json
```

</div>

<div dir="rtl" align="right">

بعد از اجرای `plot_curves.py`:

</div>

<div dir="ltr" align="left">

```text
results/rn_mia_n10_plots/roc_curve.png
results/rn_mia_n10_plots/pr_curve.png
```

</div>

<div dir="rtl" align="right">

---

# Limitations

RN-MIA چند محدودیت مهم دارد:

1. به reference model نیاز دارد.
2. اگر reference model خودش متن را memorized کرده باشد، ممکن است signal واقعی کم شود.
3. اگر reference model خیلی ضعیف باشد، residual noisy می‌شود.
4. اگر domain mismatch شدید باشد، reference model ممکن است difficulty عمومی متن را درست تخمین نزند.
5. کیفیت perturbationها روی نتیجه اثر مستقیم دارد.
6. هزینه محاسباتی آن از Original، ZN و QN بیشتر است، چون target و reference model هر دو باید evaluate شوند.
7. با \(k=10\)، neighbourhood gap ممکن است هنوز noisy باشد.
8. برای low-FPR شدید، RN-MIA در این setup از RRN-MIA ضعیف‌تر است.

---

# Summary

</div>

<div dir="ltr" align="left">

```text
RN-MIA extends the original neighbourhood attack by residualizing the target model's local neighbourhood advantage against the corresponding advantage of a small public reference language model. In the 1000-sample CNN/DailyMail highlights evaluation, RN-MIA achieved ROC-AUC 0.6547 and PR-AUC 0.6238 with 10 neighbours, improving substantially over the original neighbourhood baseline. This indicates that subtracting reference-model behaviour removes generic linguistic difficulty and perturbation artifacts that are visible to both models, leaving a more target-specific membership signal. However, RN-MIA did not improve the most stringent low-FPR operating points in this setup, motivating the residual rank-based extension RRN-MIA.
```

</div>

<div dir="rtl" align="right">

---

# خلاصه نهایی

RN-MIA یک variant مهم از خانواده‌ی حملات neighbourhood است.

تفاوت اصلی آن با حمله‌ی اولیه این است که فقط نمی‌پرسد:

> آیا target model متن اصلی را بهتر از neighbourها می‌شناسد؟

بلکه می‌پرسد:

> آیا target model متن اصلی را بیشتر از یک reference model عمومی بهتر از neighbourها می‌شناسد؟

در آزمایش‌های فعلی، RN-MIA نسبت به Original و ZN-MIA از نظر ROC-AUC و PR-AUC بهتر عمل کرد. این نشان می‌دهد که residual calibration با reference model سیگنال عضویت target-specificتری ایجاد می‌کند.

با این حال، برای low-FPR شدید، بهترین نتیجه مربوط به RRN-MIA با 50 neighbour بود. بنابراین RN-MIA یک مرحله‌ی مهم و موفق در مسیر رسیدن به RRN-MIA است.

</div>

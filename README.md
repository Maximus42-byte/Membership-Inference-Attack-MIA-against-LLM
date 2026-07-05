<div dir="rtl" align="right">

# QN-MIA: Quantile / Rank Neighbourhood Membership Inference Attack

## خلاصه کوتاه

**QN-MIA** یا **Quantile Neighbourhood Membership Inference Attack** یک variant غیرپارامتری از حمله‌ی اصلی **Neighbourhood Attack** است.

در حمله‌ی اصلی، متن اصلی با میانگین log-likelihood همسایه‌هایش مقایسه می‌شود. در ZN-MIA، این اختلاف با standard deviation همسایه‌ها نرمال می‌شود. اما در QN-MIA به جای تکیه بر mean و std، جایگاه یا **rank** متن اصلی در توزیع همسایه‌های خودش اندازه‌گیری می‌شود.

به زبان ساده:

</div>

<div dir="ltr" align="left">

```text
N-MIA  : Is the original text better than the mean of its neighbours?
ZN-MIA : Is this advantage large relative to local variance?
QN-MIA : Is the original text in the extreme tail of its neighbourhood?
```

</div>

<div dir="rtl" align="right">

هدف QN-MIA این است که حمله را از یک score میانگین‌محور به یک آزمون **rank-based** و **non-parametric** تبدیل کند. این ایده مخصوصاً برای privacy auditing و ناحیه‌ی **low-FPR** جذاب است، چون فقط وقتی یک نمونه member-like محسوب می‌شود که نسبت به تقریباً همه‌ی همسایه‌های خودش extreme باشد.

---

## جایگاه QN-MIA در خانواده حملات

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

QN-MIA از نظر مفهومی یک گام مهم است، چون حمله را از score عددی ساده به سمت **local hypothesis testing** می‌برد. با این حال، چون فقط rank را نگه می‌دارد و magnitude اختلاف likelihood را کنار می‌گذارد، ممکن است همیشه از روش‌های magnitude-based بهتر نباشد.

---

## حمله اولیه Neighbourhood Attack

در حمله‌ی اصلی، برای هر متن هدف \(x\)، مجموعه‌ای از neighbourها ساخته می‌شود:

</div>

<div dir="ltr" align="left">

$$
N(x)=\{x'_1,x'_2,\dots,x'_k\}
$$

</div>

<div dir="rtl" align="right">

در پیاده‌سازی ما، neighbourها با **span masking + T5 mask filling** ساخته می‌شوند:

1. چند span از متن اصلی ماسک می‌شود.
2. مدل mask-filling مثل T5 ماسک‌ها را پر می‌کند.
3. \(k\) متن perturb شده ساخته می‌شود.

سپس target model روی متن اصلی و neighbourها ارزیابی می‌شود:

</div>

<div dir="ltr" align="left">

$$
LL_T(x), \quad LL_T(x'_1), \dots, LL_T(x'_k)
$$

</div>

<div dir="rtl" align="right">

در N-MIA، score اصلی به شکل زیر است:

</div>

<div dir="ltr" align="left">

$$
d(x)=LL_T(x)-\frac{1}{k}\sum_{i=1}^{k}LL_T(x'_i)
$$

</div>

<div dir="rtl" align="right">

اگر \(d(x)\) بزرگ باشد، یعنی target model متن اصلی را نسبت به neighbourهای بسیار مشابهش بهتر می‌شناسد. این می‌تواند نشانه‌ی membership باشد.

---

## مشکل d-score و z-score

### مشکل d-score

در d-score فقط فاصله‌ی متن اصلی از میانگین neighbourها مهم است. اما اگر چند neighbour خراب یا outlier وجود داشته باشد، mean می‌تواند گمراه‌کننده شود.

</div>

<div dir="ltr" align="left">

```text
LL_T(x)       = -20.0
neighbour LL  = [-21.0, -21.5, -22.0, -40.0]
mean          = -26.125
d             = 6.125
```

</div>

<div dir="rtl" align="right">

اینجا یک neighbour خیلی بد داریم: `-40.0`. همین مقدار mean را پایین می‌کشد و باعث می‌شود \(d(x)\) بیش از حد بزرگ شود.

### مشکل z-score

در ZN-MIA، score به شکل زیر است:

</div>

<div dir="ltr" align="left">

$$
z(x)=\frac{LL_T(x)-\mu_N(x)}{\sigma_N(x)+\epsilon}
$$

</div>

<div dir="rtl" align="right">

این روش بهتر از d-score است، چون پراکندگی neighbourها را در نظر می‌گیرد. اما هنوز به mean و std وابسته است. اگر توزیع neighbourها skewed، heavy-tailed یا شامل outlier باشد، mean و std ممکن است نماینده‌ی خوبی برای neighbourhood نباشند.

پس QN-MIA می‌گوید:

> به جای اینکه عددهای دقیق mean و std را جدی بگیریم، فقط rank متن اصلی را در میان neighbourهای خودش نگاه می‌کنیم.

---

## ایده اصلی QN-MIA

QN-MIA یک روش **non-parametric** است. یعنی فرض نمی‌کند log-likelihood neighbourها Gaussian یا نرمال باشند.

برای log-likelihood، مقدار بزرگ‌تر بهتر است. بنابراین اگر \(LL_T(x)\) از همه یا تقریباً همه‌ی neighbourها بزرگ‌تر باشد، متن اصلی در extreme upper tail قرار دارد و membership signal قوی‌تر است.

---

## تعریف Quantile Score

برای هر متن \(x\)، مجموعه‌ی زیر را داریم:

</div>

<div dir="ltr" align="left">

$$
S(x)=\{LL_T(x),LL_T(x'_1),\dots,LL_T(x'_k)\}
$$

</div>

<div dir="rtl" align="right">

اگر log-likelihood بالاتر نشانه‌ی membership باشد، quantile score به شکل زیر تعریف می‌شود:

</div>

<div dir="ltr" align="left">

$$
q(x)=\frac{1+\sum_{i=1}^{k}\mathbf{1}[LL_T(x'_i)\leq LL_T(x)]}{k+1}
$$

</div>

<div dir="rtl" align="right">

تفسیر:

- اگر \(q(x)\) نزدیک 1 باشد، متن اصلی از اکثر neighbourها log-likelihood بالاتری دارد.
- اگر \(q(x)\) نزدیک 0.5 باشد، متن اصلی رفتار معمولی دارد.
- اگر \(q(x)\) پایین باشد، متن اصلی حتی از neighbourهای خودش هم بهتر نیست.

در پیاده‌سازی این شاخه، score اصلی QN به صورت quantile/rank تعریف شده است:

</div>

<div dir="ltr" align="left">

```text
QN(x) = (1 + count_z[LL(z) <= LL(x)]) / (|N(x)| + 1)
```

</div>

<div dir="rtl" align="right">

هرچه این score بزرگ‌تر باشد، نمونه member-likeتر است.

---

## تعریف معادل Empirical P-Value

همین ایده را می‌توان به صورت empirical p-value نیز نوشت:

</div>

<div dir="ltr" align="left">

$$
p(x)=\frac{1+\sum_{i=1}^{k}\mathbf{1}[LL_T(x'_i)\geq LL_T(x)]}{k+1}
$$

</div>

<div dir="rtl" align="right">

در این حالت:

- \(p(x)\) کوچک یعنی متن اصلی نسبت به neighbourهای خودش extreme است.
- \(q(x)\) بزرگ تقریباً معادل \(p(x)\) کوچک است.
- برای ROC/AUC بهتر است score طوری ذخیره شود که مقدار بزرگ‌تر یعنی member-likeتر باشد. بنابراین یا از \(q(x)\) استفاده می‌کنیم یا از \(-p(x)\).

---

## نکته مهم: تفاوت Loss و Log-Likelihood

در کد فعلی با log-likelihood کار می‌کنیم. در این حالت مقدار بزرگ‌تر بهتر است:

</div>

<div dir="ltr" align="left">

```text
LL_T(x) > LL_T(x'_i)  => more member-like
```

</div>

<div dir="rtl" align="right">

اما اگر در پیاده‌سازی دیگری با loss کار شود، جهت inequality عوض می‌شود:

</div>

<div dir="ltr" align="left">

```text
Loss_T(x) < Loss_T(x'_i)  => more member-like
```

</div>

<div dir="rtl" align="right">

بنابراین هنگام پیاده‌سازی QN-MIA باید دقیقاً مشخص باشد که score بر پایه‌ی `LL` است یا `loss`.

---

## محدودیت resolution وابسته به تعداد neighbourها

اگر \(k\) neighbour داشته باشیم، کوچک‌ترین empirical p-value ممکن برابر است با:

</div>

<div dir="ltr" align="left">

$$
p_{min}=\frac{1}{k+1}
$$

</div>

<div dir="rtl" align="right">

و تعداد سطح‌های ممکن برای rank score برابر با \(k+1\) است.

</div>

<div dir="ltr" align="left">

| k | Minimum p-value | Number of rank levels |
|---:|---:|---:|
| 10 | 0.0909 | 11 |
| 25 | 0.0385 | 26 |
| 50 | 0.0196 | 51 |
| 100 | 0.0099 | 101 |
| 1000 | 0.0010 | 1001 |

</div>

<div dir="rtl" align="right">

پس برای low-FPRهای بسیار شدید، مثل `0.1%` یا `0.01%`، تعداد neighbourها باید زیاد باشد. با این حال، حتی با \(k\) کمتر هم می‌توان quantile score را به عنوان ranking score استفاده کرد و ROC-AUC / PR-AUC را محاسبه کرد.

---

## الگوریتم QN-MIA

</div>

<div dir="ltr" align="left">

```text
Input:
    x                  target text
    T                  target language model
    M                  mask-filling perturbation model
    k                  number of neighbours

Step 1:
    Generate neighbours:
        N(x) = {x'_1, ..., x'_k}

Step 2:
    Compute log-likelihoods:
        ll_x = LL_T(x)
        ll_i = LL_T(x'_i), for i = 1..k

Step 3:
    Compute quantile score:
        q = (1 + count_i[ll_i <= ll_x]) / (k + 1)

Step 4:
    Use q as membership score:
        larger q => more member-like

Step 5:
    Evaluate the attack using ROC-AUC, PR-AUC, and low-FPR TPR.
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
| `Quantile_neighborhood_attack.py` | اسکریپت اصلی اجرای QN-MIA |
| `run_mia_unified.py` | توابع مشترک برای load model، تولید neighbour، likelihood و metricها |
| `custom_datasets.py` | dataset utilities |
| `plot_curves.py` | استخراج ROC و PR curve از JSON خروجی |
| `results/` | محل ذخیره نتایج |

</div>

<div dir="rtl" align="right">

---

# Experimental Setup

در آزمایش‌های فعلی QN-MIA، setup زیر استفاده شده است:

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
| Neighbours tested | `10`, `50` |
| Criterion | `qn` |

</div>

<div dir="rtl" align="right">

در این setup، target model قبلاً روی CNN/DailyMail highlights fine-tune شده است. بنابراین memberها از train split و non-memberها از validation split انتخاب می‌شوند.

---

# اجرای QN-MIA

## اجرای QN-MIA با 10 همسایه و 1000 نمونه

</div>

<div dir="ltr" align="left">

```bash
rm -f results/qn_mia_n10.json

HF_DATASETS_OFFLINE=1 HF_HUB_OFFLINE=1 python Quantile_neighborhood_attack.py \
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
  --criterion qn \
  --save_path results/qn_mia_n10.json
```

</div>

<div dir="rtl" align="right">

## اجرای QN-MIA با 50 همسایه و 1000 نمونه

</div>

<div dir="ltr" align="left">

```bash
rm -f results/qn_mia_n50.json

HF_DATASETS_OFFLINE=1 HF_HUB_OFFLINE=1 python Quantile_neighborhood_attack.py \
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
  --criterion qn \
  --save_path results/qn_mia_n50.json
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

p = "results/qn_mia_n50.json"
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

برای QN-MIA با 50 همسایه:

</div>

<div dir="ltr" align="left">

```bash
python plot_curves.py \
  --result_json results/qn_mia_n50.json \
  --out_dir results/qn_mia_n50_plots
```

</div>

<div dir="rtl" align="right">

خروجی:

</div>

<div dir="ltr" align="left">

```text
results/qn_mia_n50_plots/roc_curve.png
results/qn_mia_n50_plots/pr_curve.png
```

</div>

<div dir="rtl" align="right">

برای QN-MIA با 10 همسایه:

</div>

<div dir="ltr" align="left">

```bash
python plot_curves.py \
  --result_json results/qn_mia_n10.json \
  --out_dir results/qn_mia_n10_plots
```

</div>

<div dir="rtl" align="right">

---

# نتایج QN-MIA

نتایج معتبر با 1000 member و 1000 non-member:

</div>

<div dir="ltr" align="left">

| Method | k | Samples | ROC-AUC | PR-AUC | TPR@1%FPR | TPR@0.1%FPR | TPR@0.01%FPR |
|---|---:|---:|---:|---:|---:|---:|---:|
| QN-MIA | 10 | 1000 | 0.509758 | 0.543458 | 0.00% | 0.00% | 0.00% |
| QN-MIA | 50 | 1000 | 0.551854 | 0.539276 | 1.00% | 0.00% | 0.00% |

</div>

<div dir="rtl" align="right">

افزایش تعداد neighbourها از 10 به 50 باعث شد ROC-AUC بهتر شود و TPR@1%FPR از صفر به 1.00% برسد، اما PR-AUC کاهش کمی داشت.

</div>

<div dir="ltr" align="left">

```text
ROC-AUC gain = 0.551854 - 0.509758 = +0.042096
PR-AUC change = 0.539276 - 0.543458 = -0.004182
TPR@1%FPR increased from 0.00% to 1.00%
```

</div>

<div dir="rtl" align="right">

---

# مقایسه با Original، ZN، RN و RRN

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

## مقایسه QN-MIA با Original Neighbourhood Attack

</div>

<div dir="ltr" align="left">

| Method | k | ROC-AUC | PR-AUC | TPR@1%FPR |
|---|---:|---:|---:|---:|
| Original Neighbourhood | 10 | **0.591787** | **0.569768** | **1.10%** |
| QN-MIA | 10 | 0.509758 | 0.543458 | 0.00% |
| QN-MIA | 50 | 0.551854 | 0.539276 | 1.00% |

</div>

<div dir="rtl" align="right">

در این setup، QN-MIA از Original ضعیف‌تر است. دلیل اصلی این است که QN-MIA فقط rank را نگه می‌دارد و magnitude اختلاف likelihood را از دست می‌دهد. در حالی که Original از مقدار واقعی اختلاف با میانگین neighbourها استفاده می‌کند.

---

## مقایسه QN-MIA با ZN-MIA

</div>

<div dir="ltr" align="left">

| Method | k | ROC-AUC | PR-AUC | TPR@1%FPR | TPR@0.1%FPR |
|---|---:|---:|---:|---:|---:|
| ZN-MIA | 10 | **0.587588** | **0.567956** | **1.70%** | **0.10%** |
| QN-MIA | 10 | 0.509758 | 0.543458 | 0.00% | 0.00% |
| QN-MIA | 50 | 0.551854 | 0.539276 | 1.00% | 0.00% |

</div>

<div dir="rtl" align="right">

ZN-MIA در نتایج فعلی بهتر از QN-MIA است. این نشان می‌دهد که در این setup، حفظ magnitude اختلاف likelihood و normalize کردن آن با variance محلی مفیدتر از rank-only scoring بوده است.

---

## مقایسه QN-MIA با RN-MIA

RN-MIA از reference model استفاده می‌کند و اثرهای عمومی متن را حذف می‌کند. QN-MIA reference model ندارد و فقط rank متن را در neighbourhood target model می‌سنجد.

</div>

<div dir="ltr" align="left">

| Method | k | Reference model | ROC-AUC | PR-AUC | TPR@1%FPR |
|---|---:|---|---:|---:|---:|
| QN-MIA | 50 | No | 0.551854 | 0.539276 | 1.00% |
| RN-MIA | 10 | Yes, `distilgpt2` | **0.654651** | **0.623765** | **1.70%** |

</div>

<div dir="rtl" align="right">

RN-MIA به‌وضوح از QN-MIA بهتر عمل کرده است. این نشان می‌دهد که در این dataset، حذف generic text difficulty با reference model مهم‌تر از rank-only normalization بوده است.

---

## مقایسه QN-MIA با RRN-MIA

RRN-MIA ترکیب دو ایده است:

- residual calibration از RN-MIA
- rank-based local testing از QN-MIA

</div>

<div dir="ltr" align="left">

| Method | k | Reference model | ROC-AUC | PR-AUC | TPR@1%FPR | TPR@0.1%FPR |
|---|---:|---|---:|---:|---:|---:|
| QN-MIA | 50 | No | 0.551854 | 0.539276 | 1.00% | 0.00% |
| RRN-MIA | 50 | Yes, `distilgpt2` | **0.692407** | **0.711859** | **8.20%** | **1.80%** |

</div>

<div dir="rtl" align="right">

این مقایسه نشان می‌دهد که rank-based testing به تنهایی کافی نبود، اما وقتی همین ایده با residual calibration ترکیب شد، نتیجه‌ی بسیار قوی‌تری به دست آمد.

---

# تحلیل نتایج

در این setup، QN-MIA به عنوان یک روش rank-based ساده عملکرد محدودی داشت:

- با \(k=10\)، ROC-AUC تقریباً نزدیک random بود.
- با \(k=50\)، ROC-AUC بهتر شد، اما هنوز از Original و ZN-MIA پایین‌تر ماند.
- TPR@1%FPR با \(k=50\) به 1.00% رسید.
- در FPRهای پایین‌تر، نتیجه صفر باقی ماند.

این نشان می‌دهد که QN-MIA از نظر مفهومی مهم است، اما rank-only scoring ممکن است برای این setup کافی نباشد. ارزش اصلی QN-MIA در این پروژه این است که پایه‌ی روش قوی‌تر **RRN-MIA** را فراهم کرد.

---

# چرا QN-MIA در این setup ضعیف‌تر بود؟

چند دلیل محتمل:

1. با rank-only scoring، magnitude اختلاف likelihood از بین می‌رود.
2. با \(k=10\)، فقط 11 سطح score ممکن است.
3. حتی با \(k=50\)، score هنوز گسسته است.
4. perturbationهای T5 ممکن است کیفیت متفاوتی داشته باشند و rank را noisy کنند.
5. QN-MIA generic difficulty متن را مثل RN-MIA حذف نمی‌کند.
6. CNN/DailyMail highlights متن‌های کوتاه و خلاصه هستند؛ rank-only signal ممکن است در این نوع متن‌ها ضعیف‌تر باشد.

---

# چرا QN-MIA همچنان مهم است؟

اگرچه QN-MIA در نتایج نهایی بهترین روش نبود، از نظر علمی مهم است، چون نشان می‌دهد:

- فقط rank-based کردن Neighbourhood Attack کافی نیست.
- تعداد neighbourها برای روش‌های rank-based بسیار مهم است.
- ترکیب rank با residual calibration می‌تواند بسیار قوی‌تر باشد.
- QN-MIA به عنوان ablation نشان می‌دهد که بهبود RRN-MIA فقط به خاطر rank نیست، بلکه به خاطر ترکیب rank و reference-based residualization است.

---

# Expected Outputs

بعد از اجرای موفق QN-MIA، فایل‌های زیر ساخته می‌شوند:

</div>

<div dir="ltr" align="left">

```text
results/qn_mia_n10.json
results/qn_mia_n50.json
```

</div>

<div dir="rtl" align="right">

بعد از اجرای `plot_curves.py`:

</div>

<div dir="ltr" align="left">

```text
results/qn_mia_n10_plots/roc_curve.png
results/qn_mia_n10_plots/pr_curve.png

results/qn_mia_n50_plots/roc_curve.png
results/qn_mia_n50_plots/pr_curve.png
```

</div>

<div dir="rtl" align="right">

---

# Limitations

QN-MIA چند محدودیت مهم دارد:

1. برای p-valueهای خیلی کوچک به neighbour زیاد نیاز دارد.
2. با تعداد neighbour کم، score resolution محدود است.
3. rank-only scoring مقدار واقعی اختلاف likelihood را حذف می‌کند.
4. اگر neighbourها کیفیت بدی داشته باشند، rank نیز گمراه‌کننده می‌شود.
5. این روش generic difficulty را مثل RN-MIA با reference model حذف نمی‌کند.
6. برای این setup، QN-MIA نسبت به Original، ZN، RN و RRN ضعیف‌تر بود.
7. empirical p-value از نظر آماری فقط وقتی کاملاً معتبر است که neighbourها exchangeable و representative باشند؛ در عمل این فرض تقریباً برقرار است، نه دقیقاً.

---

#  Summary

</div>

<div dir="ltr" align="left">

```text
QN-MIA replaces mean- or variance-based neighbourhood scores with a non-parametric rank-based local test. In the 1000-sample CNN/DailyMail highlights evaluation, QN-MIA achieved ROC-AUC 0.5098 with 10 neighbours and 0.5519 with 50 neighbours. Increasing the number of neighbours improved ROC-AUC and enabled non-zero TPR@1%FPR, but QN-MIA remained weaker than the original neighbourhood baseline and the residualized variants. These results suggest that rank-based neighbourhood evidence alone is insufficient in this setup, but it provides an important ablation and motivates the stronger RRN-MIA method, where rank-based testing is applied to target-reference residual likelihoods.
```

</div>

<div dir="rtl" align="right">

---

# خلاصه نهایی

QN-MIA یک extension غیرپارامتری از Neighbourhood Attack است که به جای مقایسه‌ی متن اصلی با mean یا std neighbourها، جایگاه نسبی متن اصلی را در توزیع neighbourهای خودش بررسی می‌کند.

ایده اصلی:

> اگر target model متن اصلی را بهتر از تقریباً همه‌ی neighbourهایش بشناسد، این یک شواهد قوی برای membership است.

در آزمایش‌های فعلی، QN-MIA بهترین روش نبود، اما نقش مهمی در تحلیل دارد:

- نشان داد rank-only scoring به تنهایی کافی نیست.
- نشان داد تعداد neighbourها برای روش‌های rank-based مهم است.
- پایه‌ی مفهومی RRN-MIA را فراهم کرد.
- به عنوان ablation نشان داد که قدرت RRN-MIA از ترکیب rank و residual calibration می‌آید، نه فقط از rank.

</div>
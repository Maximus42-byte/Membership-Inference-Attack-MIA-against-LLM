<div dir="rtl" align="right">

# ZN-MIA: حمله استنتاج عضویت همسایگی با Z-Score

## 1. هدف این شاخه

این شاخه مربوط به نسخه‌ی **ZN-MIA** است؛ یعنی **Z-Score Neighborhood Membership Inference Attack**.

هدف ZN-MIA این است که حمله‌ی اولیه‌ی **Neighbourhood Attack / N-MIA** را از یک مقایسه‌ی ساده‌ی میانگین‌محور به یک مقایسه‌ی **variance-aware** تبدیل کند. در حمله‌ی اولیه، متن اصلی فقط با میانگین log-likelihood همسایه‌هایش مقایسه می‌شود. در ZN-MIA علاوه بر میانگین، پراکندگی log-likelihood همسایه‌ها نیز در نظر گرفته می‌شود.

به زبان ساده:

> **N-MIA** می‌پرسد: آیا متن اصلی از میانگین neighbourها بهتر score می‌گیرد؟
>
> **ZN-MIA** می‌پرسد: آیا این بهتر بودن، نسبت به پراکندگی neighbourها واقعاً معنادار است؟

---

## 2. پیش‌زمینه: حمله‌ی اولیه Neighbourhood Attack

در حمله‌ی اولیه، برای هر متن هدف `x`، ابتدا یک مجموعه از متن‌های مشابه ساخته می‌شود:

</div>

<div dir="ltr" align="left">

```text
N(x) = {x'_1, x'_2, ..., x'_k}
```

</div>

<div dir="rtl" align="right">

این متن‌ها **neighbour** یا **perturbation** هستند. در پیاده‌سازی فعلی، این neighbourها معمولاً با روش **span masking + T5 mask filling** ساخته می‌شوند:

1. بخشی از متن با توکن‌های `<extra_id_*>` ماسک می‌شود.
2. مدل mask-filling مثل T5 آن بخش‌ها را پر می‌کند.
3. چند نسخه‌ی perturb شده از متن اصلی ساخته می‌شود.

سپس مدل هدف روی متن اصلی و neighbourها ارزیابی می‌شود:

</div>

<div dir="ltr" align="left">

```text
LL_T(x)
LL_T(x'_1), LL_T(x'_2), ..., LL_T(x'_k)
```

</div>

<div dir="rtl" align="right">

در اینجا `LL_T` یعنی log-likelihood متن زیر **target model**.

در نسخه‌ی اصلی، score به شکل زیر است:

</div>

<div dir="ltr" align="left">

```text
d(x) = LL_T(x) - mean(LL_T(N(x)))
```

```text
d(x) = LL_T(x) - (1/k) * Σ_i LL_T(x'_i)
```

</div>

<div dir="rtl" align="right">

اگر `d(x)` بزرگ باشد، یعنی مدل هدف متن اصلی را نسبت به neighbourهای بسیار مشابهش خیلی بهتر می‌شناسد. این می‌تواند نشانه‌ی membership باشد.

---

## 3. مشکل d-score در N-MIA

score اولیه یعنی `d(x)` فقط اختلاف متن اصلی با میانگین neighbourها را می‌سنجد. اما این کافی نیست، چون همه‌ی neighbourhoodها به یک اندازه پایدار نیستند.

دو حالت را در نظر بگیرید.

### 3.1. حالت اول: neighbourhood پایدار

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

اینجا neighbourها خیلی نزدیک به هم هستند. پس اختلاف `5.05` بسیار معنادار است.

### 3.2. حالت دوم: neighbourhood ناپایدار

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

اینجا `d` هنوز بزرگ است، اما neighbourها خودشان بسیار پراکنده‌اند. پس اختلاف متن اصلی با میانگین neighbourها به اندازه‌ی حالت اول قابل اعتماد نیست.

مشکل اصلی N-MIA این است که فقط **فاصله از میانگین** را نگاه می‌کند، نه **معناداری فاصله نسبت به پراکندگی محلی**.

---

## 4. ایده‌ی اصلی ZN-MIA

ZN-MIA همان neighbourhood gap را نگه می‌دارد، اما آن را با standard deviation neighbourها normalize می‌کند.

فرمول اصلی:

</div>

<div dir="ltr" align="left">

```text
z(x) = (LL_T(x) - mean(LL_T(N(x)))) / std(LL_T(N(x)))
```

</div>

<div dir="rtl" align="right">

یا به صورت خلاصه:

</div>

<div dir="ltr" align="left">

```text
z(x) = d(x) / σ_N(x)
```

</div>

<div dir="rtl" align="right">

که در آن:

- `d(x)` همان neighbourhood gap است.
- `σ_N(x)` انحراف معیار log-likelihood همسایه‌های همان نمونه است.
- هر sample، normalization مخصوص خودش را دارد.

پس ZN-MIA یک **sample-specific normalized neighbourhood attack** است.

---

## 5. تفاوت دقیق ZN-MIA با N-MIA

| بخش | N-MIA | ZN-MIA |
|---|---|---|
| نوع score | اختلاف با میانگین neighbourها | اختلاف نرمال‌شده با std neighbourها |
| calibration | فقط mean-based | mean + variance-based |
| حساسیت به neighbourhood noisy | بیشتر | کمتر |
| مناسب برای low-FPR | متوسط | بهتر |
| فرض توزیعی | implicit و ساده | هنوز ساده، اما scale-aware |

تفاوت اصلی این است که N-MIA می‌گوید:

</div>

<div dir="ltr" align="left">

```text
How far is x from the neighbour mean?
```

</div>

<div dir="rtl" align="right">

اما ZN-MIA می‌گوید:

</div>

<div dir="ltr" align="left">

```text
How many neighbourhood standard deviations is x away from the neighbour mean?
```

</div>

<div dir="rtl" align="right">

---

## 6. تفسیر علمی ZN-MIA

ZN-MIA را می‌توان به عنوان یک تخمین ساده از **local standardized optimality** یا **normalized curvature** در اطراف متن `x` در نظر گرفت.

اگر متن اصلی در training set بوده باشد، انتظار داریم مدل هدف روی خود متن اصلی log-likelihood بالاتری بدهد، اما روی neighbourهای مصنوعی که احتمالاً در training نبودند، چنین مزیتی نداشته باشد.

در نتیجه:

- برای memberها، `LL_T(x)` معمولاً نسبت به neighbourها بالاتر است.
- اگر این اختلاف نسبت به پراکندگی neighbourها هم بزرگ باشد، `z(x)` بزرگ می‌شود.
- پس مقدار بزرگ‌تر `z(x)` نشانه‌ی قوی‌تری برای membership است.

---

## 7. تصمیم عضویت

در ZN-MIA، بعد از محاسبه‌ی `z(x)`، یک threshold انتخاب می‌شود:

</div>

<div dir="ltr" align="left">

```text
if z(x) > γ:
    predict member
else:
    predict non-member
```

</div>

<div dir="rtl" align="right">

در اینجا `γ` روی validation set یا با استفاده از ROC curve انتخاب می‌شود.

معیارهای اصلی گزارش:

- AUC
- TPR@FPR=5%
- TPR@FPR=1%
- TPR@FPR=0.1%
- ROC curve
- PR curve

---

## 8. جایگاه ZN-MIA در خانواده حملات پروژه

</div>

<div dir="ltr" align="left">

```text
N-MIA  →  baseline neighbourhood gap
ZN-MIA →  z-score normalized neighbourhood gap
RN-MIA →  residual neighbourhood gap with small reference model
QN-MIA →  rank / quantile neighbourhood test
RRN-MIA → residual rank-based neighbourhood test
```

</div>

<div dir="rtl" align="right">

در این مسیر، ZN-MIA اولین upgrade مستقیم روی N-MIA است. این روش هنوز به reference model نیاز ندارد و فقط از همان neighbourهایی استفاده می‌کند که N-MIA هم تولید می‌کند.

---

## 9. فایل‌های مهم در این شاخه

ساختار پیشنهادی شاخه:

</div>

<div dir="ltr" align="left">

```text
ZN-MIA/
├── README.md or ZN-MIA.md
├── run_mia_unified.py
├── custom_datasets.py
├── results/
├── cache/
└── figures/
```

</div>

<div dir="rtl" align="right">

فایل‌های اصلی:

| فایل | نقش |
|---|---|
| `run_mia_unified.py` | اجرای حملات، تولید perturbation، محاسبه likelihood، محاسبه d و z |
| `custom_datasets.py` | بارگذاری datasetهای سفارشی |
| `README.md` یا `ZN-MIA.md` | توضیح روش و دستور اجرای آزمایش‌ها |
| `results/` | خروجی‌ها، scoreها، ROC/PR، metadata |

---

## 10. پیش‌نیازها

### 10.1. ساخت محیط Python

</div>

<div dir="ltr" align="left">

```bash
conda create -n mia python=3.10 -y
conda activate mia
```

</div>

<div dir="rtl" align="right">

### 10.2. نصب کتابخانه‌ها

</div>

<div dir="ltr" align="left">

```bash
pip install torch torchvision torchaudio
pip install transformers datasets accelerate sentencepiece
pip install scikit-learn matplotlib tqdm numpy pandas
```

</div>

<div dir="rtl" align="right">

اگر GPU داری، نسخه‌ی مناسب PyTorch با CUDA را از سایت رسمی PyTorch نصب کن.

---

## 11. آماده‌سازی cache و مسیرها

پیشنهاد می‌شود مسیرهای حجیم در git ذخیره نشوند. این موارد باید در `.gitignore` باشند:

</div>

<div dir="ltr" align="left">

```gitignore
cache/
.hf_cache/
results/
outputs/
ft_distilgpt2_highlights/
*.pt
*.bin
*.safetensors
```

</div>

<div dir="rtl" align="right">

برای cache مدل‌های HuggingFace می‌توانی این مسیر را تنظیم کنی:

</div>

<div dir="ltr" align="left">

```bash
export HF_HOME=./.hf_cache
export TRANSFORMERS_CACHE=./.hf_cache
```

</div>

<div dir="rtl" align="right">

---

## 12. اجرای آزمایش ZN-MIA

نمونه دستور اجرا:

</div>

<div dir="ltr" align="left">

```bash
python run_mia_unified.py \
  --output_name zn_mia_experiment \
  --base_model_name EleutherAI/gpt-neo-2.7B \
  --mask_filling_model_name t5-3b \
  --n_perturbation_list 25 \
  --n_samples 2000 \
  --pct_words_masked 0.3 \
  --span_length 2 \
  --cache_dir cache \
  --dataset_member the_pile \
  --dataset_member_key text \
  --dataset_nonmember xsum \
  --ref_model gpt2-xl \
  --max_length 2000
```

</div>

<div dir="rtl" align="right">

نکته: در این branch باید مطمئن شوی criterion مربوط به `z` فعال است یا خروجی مربوط به z-score ذخیره می‌شود.

اگر در کد گزینه‌ای مثل `--criterion z` یا `--scoring z` وجود دارد، دستور را این‌گونه اجرا کن:

</div>

<div dir="ltr" align="left">

```bash
python run_mia_unified.py \
  --output_name zn_mia_experiment \
  --criterion z \
  --base_model_name EleutherAI/gpt-neo-2.7B \
  --mask_filling_model_name t5-3b \
  --n_perturbation_list 25 \
  --n_samples 2000 \
  --pct_words_masked 0.3 \
  --span_length 2 \
  --cache_dir cache \
  --dataset_member the_pile \
  --dataset_member_key text \
  --dataset_nonmember xsum \
  --max_length 2000
```

</div>

<div dir="rtl" align="right">

اگر چنین flagای در کد نیست، باید در بخش `run_perturbation_experiment` مطمئن شوی خروجی `z` محاسبه و ذخیره می‌شود.

---

## 13. محل محاسبه z-score در کد

منطق ZN-MIA باید در بخشی باشد که likelihood متن اصلی و perturbationها محاسبه شده‌اند.

شکل کلی محاسبه:

</div>

<div dir="ltr" align="left">

```python
original_ll = res["original_ll"]
perturbed_mean = res["perturbed_original_ll"]
perturbed_std = res["perturbed_original_ll_std"]

z_score = (original_ll - perturbed_mean) / perturbed_std
```

</div>

<div dir="rtl" align="right">

برای نمونه‌های non-member یا generated/sample نیز همین منطق باید اعمال شود:

</div>

<div dir="ltr" align="left">

```python
sampled_ll = res["sampled_ll"]
perturbed_sampled_mean = res["perturbed_sampled_ll"]
perturbed_sampled_std = res["perturbed_sampled_ll_std"]

z_sample = (sampled_ll - perturbed_sampled_mean) / perturbed_sampled_std
```

</div>

<div dir="rtl" align="right">

برای جلوگیری از تقسیم بر صفر:

</div>

<div dir="ltr" align="left">

```python
eps = 1e-8
z_score = (original_ll - perturbed_mean) / (perturbed_std + eps)
```

</div>

<div dir="rtl" align="right">

---

## 14. خروجی‌های مورد انتظار

بعد از اجرا، باید خروجی‌هایی شبیه موارد زیر داشته باشی:

</div>

<div dir="ltr" align="left">

```text
results/
└── zn_mia_experiment/
    ├── args.json
    ├── raw_results.json
    ├── predictions.json
    ├── roc_curve.png
    ├── pr_curve.png
    └── metrics.json
```

</div>

<div dir="rtl" align="right">

در `metrics.json` بهتر است حداقل این موارد ذخیره شوند:

</div>

<div dir="ltr" align="left">

```json
{
  "attack": "ZN-MIA",
  "score": "z",
  "auc": 0.0,
  "tpr_at_fpr_5": 0.0,
  "tpr_at_fpr_1": 0.0,
  "tpr_at_fpr_0_1": 0.0,
  "n_samples": 2000,
  "n_perturbations": 25,
  "base_model": "EleutherAI/gpt-neo-2.7B",
  "mask_model": "t5-3b"
}
```

</div>

<div dir="rtl" align="right">

---

## 15. پروتکل تکرار آزمایش‌ها

برای اینکه آزمایش‌ها reproducible باشند:

1. seedها ثابت باشند.
2. تعداد نمونه‌ها گزارش شود.
3. تعداد perturbationها گزارش شود.
4. مدل هدف و mask-filling model دقیقاً مشخص شوند.
5. dataset member و non-member مشخص شوند.
6. مقدار `pct_words_masked` و `span_length` گزارش شود.
7. metricهای low-FPR جداگانه گزارش شوند.

seedهای پیشنهادی:

</div>

<div dir="ltr" align="left">

```python
torch.manual_seed(0)
np.random.seed(0)
random.seed(0)
```

</div>

<div dir="rtl" align="right">

---

## 16. مقایسه با baseline

برای گزارش پایان‌نامه، ZN-MIA باید حداقل با موارد زیر مقایسه شود:

| Attack | توضیح |
|---|---|
| LOSS | threshold روی loss خام |
| N-MIA / d-score | neighbourhood gap بدون std |
| ZN-MIA / z-score | neighbourhood gap نرمال‌شده |

جدول پیشنهادی برای نتایج:

| Method | AUC | TPR@5% FPR | TPR@1% FPR | TPR@0.1% FPR |
|---|---:|---:|---:|---:|
| LOSS | - | - | - | - |
| N-MIA | - | - | - | - |
| ZN-MIA | - | - | - | - |

---

## 17. انتظار تجربی

انتظار اصلی از ZN-MIA این نیست که همیشه AUC را خیلی زیاد کند. انتظار مهم‌تر این است که در نقاط low-FPR بهتر عمل کند.

به طور خاص:

- اگر neighbourhoodها پایدار باشند، ZN-MIA باید سیگنال قوی‌تری بدهد.
- اگر neighbourhoodها noisy باشند، ZN-MIA باید false positiveهای ناشی از gapهای غیرقابل اعتماد را کاهش دهد.
- بیشترین اثر احتمالی در `TPR@1%FPR` و `TPR@0.1%FPR` دیده می‌شود.

---

## 18. محدودیت‌ها

ZN-MIA چند محدودیت دارد:

1. اگر `std` خیلی کوچک باشد، score ممکن است بسیار بزرگ شود.
2. اگر تعداد neighbourها کم باشد، تخمین `std` قابل اعتماد نیست.
3. اگر T5 perturbationهای بد بسازد، z-score هنوز ممکن است گمراه شود.
4. این روش generic difficulty را مثل RN-MIA با reference model حذف نمی‌کند.
5. برای low-FPR خیلی شدید، rank-based یا residualized methods احتمالاً قوی‌تر هستند.

---

## 19. پیشنهاد ablation study

برای تحلیل علمی ZN-MIA، این ablationها پیشنهاد می‌شوند:

| Ablation | هدف |
|---|---|
| تعداد neighbourها: 10, 25, 50, 100 | بررسی پایداری mean/std |
| span_length: 1, 2, 4 | اثر اندازه perturbation |
| pct_words_masked: 0.15, 0.3, 0.5 | اثر شدت تغییر متن |
| mask model: T5-small, T5-large, T5-3B | اثر کیفیت perturbation |
| مدل هدف‌های مختلف | بررسی تعمیم‌پذیری |

---

## 20. خلاصه نهایی

ZN-MIA یک extension مستقیم از Neighbourhood Attack است.

حمله‌ی اولیه فقط می‌پرسد:

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
Is the original text unusually better than its neighbours, relative to local neighbour variance?
```

</div>

<div dir="rtl" align="right">

بنابراین ZN-MIA یک حمله‌ی **variance-aware, sample-specific, neighbourhood-calibrated MIA** است که هدف اصلی آن بهبود reliability مخصوصاً در ناحیه‌ی low-FPR است.

</div>
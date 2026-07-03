<div dir="rtl" align="right">

# QN-MIA: حمله استنتاج عضویت همسایگی با Quantile / Empirical P-Value

## 1. هدف این شاخه

این شاخه مربوط به نسخه‌ی **QN-MIA** است؛ یعنی **Quantile Neighborhood Membership Inference Attack**.

هدف QN-MIA این است که حمله‌ی اولیه‌ی **Neighbourhood Attack / N-MIA** را از یک score میانگین‌محور به یک آزمون **rank-based** و **non-parametric** تبدیل کند.

در N-MIA، متن اصلی با میانگین neighbourها مقایسه می‌شود. در ZN-MIA، این اختلاف با standard deviation neighbourها نرمال می‌شود. اما در QN-MIA به جای اینکه فقط mean یا std را نگاه کنیم، جایگاه متن اصلی را در توزیع local neighbourها می‌سنجیم.

به زبان ساده:

> **N-MIA** می‌پرسد: آیا متن اصلی از میانگین neighbourها بهتر score می‌گیرد؟
>
> **ZN-MIA** می‌پرسد: آیا این بهتر بودن نسبت به پراکندگی neighbourها معنادار است؟
>
> **QN-MIA** می‌پرسد: آیا متن اصلی در extreme tail توزیع neighbourهای خودش قرار گرفته است؟

این روش مخصوصاً برای **Low-FPR** مهم است، چون در تنظیمات privacy auditing معمولاً می‌خواهیم فقط وقتی یک نمونه را member اعلام کنیم که شواهد بسیار قوی باشد.

---

## 2. پیش‌زمینه: حمله اولیه Neighbourhood Attack

در حمله‌ی اولیه، برای هر متن هدف `x`، یک مجموعه از متن‌های مشابه ساخته می‌شود:

</div>

<div dir="ltr" align="left">

```text
N(x) = {x'_1, x'_2, ..., x'_k}
```

</div>

<div dir="rtl" align="right">

این متن‌ها neighbour یا perturbation هستند. در پیاده‌سازی فعلی، معمولاً با روش **span masking + T5 mask filling** ساخته می‌شوند:

1. چند span از متن اصلی ماسک می‌شود.
2. مدل mask-filling مثل T5 ماسک‌ها را پر می‌کند.
3. `k` متن perturb شده ساخته می‌شود.

سپس target model روی متن اصلی و neighbourها ارزیابی می‌شود:

</div>

<div dir="ltr" align="left">

```text
LL_T(x)
LL_T(x'_1), LL_T(x'_2), ..., LL_T(x'_k)
```

</div>

<div dir="rtl" align="right">

در اینجا `LL_T` یعنی log-likelihood متن زیر target model.

در N-MIA، score به شکل زیر است:

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

اگر `d(x)` بزرگ باشد، یعنی مدل هدف متن اصلی را نسبت به neighbourهای بسیار مشابهش بهتر می‌شناسد. این می‌تواند نشانه‌ی membership باشد.

---

## 3. مشکل d-score و z-score

### 3.1. مشکل d-score

در d-score فقط فاصله‌ی متن اصلی از میانگین neighbourها مهم است. اما اگر چند neighbour خراب یا outlier وجود داشته باشد، mean می‌تواند گمراه‌کننده شود.

مثلاً:

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

اینجا یک neighbour خیلی بد داریم: `-40.0`. همین مقدار mean را پایین می‌کشد و باعث می‌شود `d(x)` بیش از حد بزرگ شود. در نتیجه ممکن است یک non-member اشتباهاً member تشخیص داده شود.

### 3.2. مشکل z-score

در ZN-MIA، score این است:

</div>

<div dir="ltr" align="left">

```text
z(x) = (LL_T(x) - mean(LL_T(N(x)))) / std(LL_T(N(x)))
```

</div>

<div dir="rtl" align="right">

این روش بهتر از d-score است، چون پراکندگی neighbourها را هم در نظر می‌گیرد. اما هنوز به mean و std وابسته است. اگر توزیع neighbourها skewed، heavy-tailed، یا شامل outlier باشد، mean و std ممکن است نماینده‌ی خوبی برای neighbourhood نباشند.

پس QN-MIA می‌گوید:

> به جای اینکه عددهای دقیق mean و std را جدی بگیریم، فقط rank متن اصلی را در میان neighbourهای خودش نگاه کنیم.

---

## 4. ایده اصلی QN-MIA

QN-MIA یک روش **non-parametric** است. یعنی فرض نمی‌کند log-likelihood neighbourها Gaussian یا نرمال باشند.

ایده این است:

> اگر متن `x` واقعاً member باشد، target model احتمالاً به آن log-likelihood بالاتری نسبت به بیشتر neighbourها می‌دهد.

پس به جای محاسبه‌ی mean یا std، متن اصلی را در کنار neighbourها rank می‌کنیم.

برای log-likelihood، مقدار بزرگ‌تر بهتر است. بنابراین اگر `LL_T(x)` از همه یا تقریباً همه‌ی neighbourها بزرگ‌تر باشد، متن اصلی در extreme upper tail قرار دارد و membership signal قوی‌تر است.

---

## 5. تعریف Quantile Score

برای هر متن `x`، مجموعه‌ی زیر را داریم:

</div>

<div dir="ltr" align="left">

```text
S(x) = {LL_T(x), LL_T(x'_1), ..., LL_T(x'_k)}
```

</div>

<div dir="rtl" align="right">

حالا rank متن اصلی را در این مجموعه محاسبه می‌کنیم.

اگر log-likelihood بالاتر نشانه‌ی membership باشد، یک quantile ساده می‌تواند این‌گونه تعریف شود:

</div>

<div dir="ltr" align="left">

```text
q(x) = (1 + Σ_i 1[LL_T(x'_i) <= LL_T(x)]) / (k + 1)
```

</div>

<div dir="rtl" align="right">

تفسیر:

- اگر `q(x)` نزدیک 1 باشد، متن اصلی از اکثر neighbourها log-likelihood بالاتری دارد.
- اگر `q(x)` نزدیک 0.5 باشد، متن اصلی رفتار معمولی دارد.
- اگر `q(x)` پایین باشد، متن اصلی حتی از neighbourهای خودش هم بهتر نیست.

پس در حالت quantile score:

</div>

<div dir="ltr" align="left">

```text
q(x) high  => likely member
q(x) low   => likely non-member
```

</div>

<div dir="rtl" align="right">

---

## 6. تعریف Empirical P-Value

می‌توان همین ایده را به صورت empirical p-value هم نوشت. چون برای log-likelihood مقدار بالاتر بهتر است، p-value یک‌طرفه را این‌طور تعریف می‌کنیم:

</div>

<div dir="ltr" align="left">

```text
p(x) = (1 + Σ_i 1[LL_T(x'_i) >= LL_T(x)]) / (k + 1)
```

</div>

<div dir="rtl" align="right">

تفسیر:

- صورت کسر می‌شمارد چند neighbour از متن اصلی بهتر یا مساوی هستند.
- اگر هیچ neighbour از متن اصلی بهتر نباشد، p-value حداقل می‌شود.
- p-value کوچک یعنی متن اصلی نسبت به neighbourهای خودش خیلی extreme است.

پس decision rule می‌تواند این باشد:

</div>

<div dir="ltr" align="left">

```text
if p(x) < alpha:
    predict member
else:
    predict non-member
```

</div>

<div dir="rtl" align="right">

در اینجا `alpha` می‌تواند متناسب با هدف Low-FPR انتخاب شود، مثلاً:

</div>

<div dir="ltr" align="left">

```text
alpha = 0.05   # target FPR around 5%
alpha = 0.01   # target FPR around 1%
alpha = 0.001  # target FPR around 0.1%, if k is large enough
```

</div>

<div dir="rtl" align="right">

---

## 7. نکته مهم: تفاوت Loss و Log-Likelihood

در کد فعلی معمولاً با log-likelihood کار می‌کنیم. اما اگر در یک پیاده‌سازی دیگر با loss کار شود، جهت inequality عوض می‌شود.

### 7.1. اگر با log-likelihood کار کنیم

برای member انتظار داریم:

</div>

<div dir="ltr" align="left">

```text
LL_T(x) > LL_T(x'_i)
```

</div>

<div dir="rtl" align="right">

پس empirical p-value:

</div>

<div dir="ltr" align="left">

```text
p_LL(x) = (1 + Σ_i 1[LL_T(x'_i) >= LL_T(x)]) / (k + 1)
```

</div>

<div dir="rtl" align="right">

### 7.2. اگر با loss کار کنیم

برای member انتظار داریم:

</div>

<div dir="ltr" align="left">

```text
Loss_T(x) < Loss_T(x'_i)
```

</div>

<div dir="rtl" align="right">

پس empirical p-value:

</div>

<div dir="ltr" align="left">

```text
p_loss(x) = (1 + Σ_i 1[Loss_T(x'_i) <= Loss_T(x)]) / (k + 1)
```

</div>

<div dir="rtl" align="right">

در این branch باید دقیقاً مشخص شود که score از نوع `LL` است یا `loss`، چون جهت threshold کاملاً به آن وابسته است.

---

## 8. چرا QN-MIA برای Low-FPR مهم است؟

در privacy auditing، معمولاً accuracy کافی نیست. چیزی که مهم‌تر است، عملکرد در FPRهای خیلی پایین است:

</div>

<div dir="ltr" align="left">

```text
TPR@5%FPR
TPR@1%FPR
TPR@0.1%FPR
TPR@0.01%FPR
```

</div>

<div dir="rtl" align="right">

دلیلش این است که در دنیای واقعی تعداد non-memberها بسیار زیاد است. اگر FPR بالا باشد، حتی attack با accuracy خوب هم غیرقابل اعتماد می‌شود.

QN-MIA برای Low-FPR طبیعی است، چون فقط وقتی member اعلام می‌کند که متن اصلی در tail بسیار extreme توزیع neighbourهای خودش باشد.

به زبان ساده:

> QN-MIA به جای اینکه بگوید «اختلاف عددی زیاد است»، می‌گوید «متن اصلی از تقریباً همه‌ی neighbourهای خودش بهتر است». این شواهد برای Low-FPR قابل اعتمادتر است.

---

## 9. محدودیت مهم: Resolution وابسته به تعداد neighbourها

اگر `k` neighbour داشته باشیم، کوچک‌ترین empirical p-value ممکن برابر است با:

</div>

<div dir="ltr" align="left">

```text
p_min = 1 / (k + 1)
```

</div>

<div dir="rtl" align="right">

بنابراین اگر neighbour کم باشد، p-value نمی‌تواند خیلی کوچک شود.

</div>

<div dir="ltr" align="left">

```text
k = 10    => p_min ≈ 0.091
k = 25    => p_min ≈ 0.038
k = 50    => p_min ≈ 0.019
k = 100   => p_min ≈ 0.0099
k = 1000  => p_min ≈ 0.000999
```

</div>

<div dir="rtl" align="right">

پس برای هدف‌های خیلی سخت مثل `FPR = 0.1%`، اگر بخواهیم p-value را مستقیم به عنوان آزمون آماری استفاده کنیم، به تعداد neighbour زیاد نیاز داریم.

اما حتی با `k` کمتر، می‌توان `p(x)` یا `q(x)` را به عنوان یک ranking score استفاده کرد و بعد ROC/AUC و TPR@FPR را روی validation set محاسبه کرد.

---

## 10. تفاوت QN-MIA با ZN-MIA

| ویژگی | ZN-MIA | QN-MIA |
|---|---|---|
| نوع calibration | mean + std | rank / quantile |
| فرض آماری | semi-parametric | non-parametric |
| حساسیت به outlier | متوسط | کمتر |
| مناسب برای Low-FPR | خوب | بسیار خوب |
| نیاز به neighbour زیاد | متوسط | زیادتر، مخصوصاً برای tail |
| score اصلی | z-score | p-value یا quantile |

---

## 11. الگوریتم QN-MIA

</div>

<div dir="ltr" align="left">

```text
Input:
    x: target text
    T: target language model
    M: mask-filling perturbation model
    k: number of neighbours
    alpha: membership threshold

Step 1: Generate neighbours
    N(x) = {x'_1, ..., x'_k}

Step 2: Compute log-likelihoods
    ll_x = LL_T(x)
    ll_i = LL_T(x'_i) for i = 1..k

Step 3: Compute empirical p-value
    p = (1 + count(ll_i >= ll_x)) / (k + 1)

Step 4: Membership decision
    if p < alpha:
        return member
    else:
        return non-member
```

</div>

<div dir="rtl" align="right">

اگر به جای p-value از quantile score استفاده شود:

</div>

<div dir="ltr" align="left">

```text
q = (1 + count(ll_i <= ll_x)) / (k + 1)

if q > tau:
    return member
else:
    return non-member
```

</div>

<div dir="rtl" align="right">

در عمل برای ROC/AUC بهتر است یک score پیوسته ذخیره شود. برای p-value، چون مقدار کمتر یعنی member، می‌توان score را به شکل زیر ذخیره کرد:

</div>

<div dir="ltr" align="left">

```text
score = -p
```

</div>

<div dir="rtl" align="right">

یا:

</div>

<div dir="ltr" align="left">

```text
score = q
```

</div>

<div dir="rtl" align="right">

---

## 12. انتظار علمی از QN-MIA

انتظار اصلی این نیست که QN-MIA الزاماً همیشه AUC را خیلی بهتر کند. انتظار مهم‌تر این است که در ناحیه‌ی Low-FPR عملکرد بهتری بدهد.

بنابراین claim مناسب برای این branch این است:

> QN-MIA is designed to improve low-FPR membership inference by replacing mean/std-based neighbourhood scores with a rank-based non-parametric local test.

نه اینکه بگوییم:

> QN-MIA always improves all metrics.

متریک‌های اصلی برای این branch باید این‌ها باشند:

</div>

<div dir="ltr" align="left">

```text
ROC-AUC
PR-AUC
TPR@5%FPR
TPR@1%FPR
TPR@0.1%FPR
TPR@0.01%FPR
```

</div>

<div dir="rtl" align="right">

---

## 13. ساختار پیشنهادی branch

</div>

<div dir="ltr" align="left">

```text
QN-MIA/
├── README.md
├── QN-MIA.md
├── run_mia_unified.py
├── custom_datasets.py
├── results/
│   ├── qn_mia_n25.json
│   ├── qn_mia_n50.json
│   └── qn_mia_n100.json
└── scripts/
    ├── run_qn_mia_n25.sh
    ├── run_qn_mia_n50.sh
    └── run_qn_mia_n100.sh
```

</div>

<div dir="rtl" align="right">

---

## 14. نحوه اجرای آزمایش‌ها

### 14.1. فعال کردن محیط

</div>

<div dir="ltr" align="left">

```bash
conda activate mia
```

</div>

<div dir="rtl" align="right">

یا اگر از virtualenv استفاده می‌شود:

</div>

<div dir="ltr" align="left">

```bash
source .venv/bin/activate
```

</div>

<div dir="rtl" align="right">

### 14.2. رفتن به branch مربوطه

</div>

<div dir="ltr" align="left">

```bash
git checkout QN-MIA
```

</div>

<div dir="rtl" align="right">

### 14.3. اجرای آزمایش با 25 neighbour

</div>

<div dir="ltr" align="left">

```bash
python run_mia_unified.py \
  --output_name qn_mia_n25 \
  --base_model_name distilgpt2 \
  --mask_filling_model_name t5-small \
  --n_perturbation_list 25 \
  --n_samples 1000 \
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

### 14.4. اجرای آزمایش با 50 neighbour

</div>

<div dir="ltr" align="left">

```bash
python run_mia_unified.py \
  --output_name qn_mia_n50 \
  --base_model_name distilgpt2 \
  --mask_filling_model_name t5-small \
  --n_perturbation_list 50 \
  --n_samples 1000 \
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

### 14.5. اجرای آزمایش با 100 neighbour

</div>

<div dir="ltr" align="left">

```bash
python run_mia_unified.py \
  --output_name qn_mia_n100 \
  --base_model_name distilgpt2 \
  --mask_filling_model_name t5-small \
  --n_perturbation_list 100 \
  --n_samples 1000 \
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

نکته: برای آزمایش نهایی بهتر است از مدل و تنظیمات اصلی پروژه استفاده شود، مثلاً GPT-Neo یا GPT-2 بزرگ‌تر و T5-large / T5-3B. اما برای تست سریع، `distilgpt2` و `t5-small` مناسب‌ترند.

---

## 15. خروجی‌های مورد انتظار

بعد از اجرا، باید فایل‌هایی شبیه این‌ها ساخته شوند:

</div>

<div dir="ltr" align="left">

```text
results/qn_mia_n25.json
results/qn_mia_n50.json
results/qn_mia_n100.json
```

</div>

<div dir="rtl" align="right">

داخل هر فایل باید حداقل این موارد وجود داشته باشد:

</div>

<div dir="ltr" align="left">

```text
metrics.roc_auc
pr_metrics.pr_auc
metrics.fpr
metrics.tpr
predictions.real
predictions.samples
raw_results
```

</div>

<div dir="rtl" align="right">

برای QN-MIA بهتر است علاوه بر score نهایی، این موارد نیز ذخیره شوند:

</div>

<div dir="ltr" align="left">

```text
original_ll
perturbed_lls
quantile_score
empirical_p_value
n_neighbours
p_min
```

</div>

<div dir="rtl" align="right">

---

## 16. بررسی نتایج

برای خواندن خلاصه نتایج:

</div>

<div dir="ltr" align="left">

```bash
python - <<'PY'
import json
import numpy as np

p = "results/qn_mia_n100.json"
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
print("roc_auc:", d["metrics"]["roc_auc"])
print("pr_auc:", d["pr_metrics"]["pr_auc"])
print("TPR@5%FPR:", tpr_at(0.05))
print("TPR@1%FPR:", tpr_at(0.01))
print("TPR@0.1%FPR:", tpr_at(0.001))
print("TPR@0.01%FPR:", tpr_at(0.0001))
PY
```

</div>

<div dir="rtl" align="right">

---

## 17. مقایسه با baselineها

QN-MIA باید حداقل با این روش‌ها مقایسه شود:

1. **LOSS Attack**
2. **N-MIA / d-score**
3. **ZN-MIA / z-score**
4. **QN-MIA / quantile or p-value**

جدول پیشنهادی:

</div>

<div dir="ltr" align="left">

```text
| Attack | AUC | PR-AUC | TPR@5%FPR | TPR@1%FPR | TPR@0.1%FPR |
|--------|-----|--------|-----------|-----------|-------------|
| LOSS   |     |        |           |           |             |
| N-MIA  |     |        |           |           |             |
| ZN-MIA |     |        |           |           |             |
| QN-MIA |     |        |           |           |             |
```

</div>

<div dir="rtl" align="right">

---

## 18. Ablationهای مهم

برای اینکه QN-MIA در پایان‌نامه قابل دفاع باشد، این ablationها مهم هستند:

### 18.1. اثر تعداد neighbourها

</div>

<div dir="ltr" align="left">

```text
k = 10, 25, 50, 100, 200
```

</div>

<div dir="rtl" align="right">

انتظار:

- با افزایش `k`، empirical p-value دقیق‌تر می‌شود.
- TPR@low-FPR باید بهتر یا پایدارتر شود.
- هزینه محاسباتی افزایش می‌یابد.

### 18.2. اثر نوع score

مقایسه:

</div>

<div dir="ltr" align="left">

```text
p-value score:  -p(x)
quantile score: q(x)
raw rank score
```

</div>

<div dir="rtl" align="right">

### 18.3. اثر perturbation quality

مقایسه‌ی پارامترها:

</div>

<div dir="ltr" align="left">

```text
pct_words_masked = 0.15, 0.30, 0.45
span_length      = 1, 2, 4
```

</div>

<div dir="rtl" align="right">

---

## 19. محدودیت‌ها

QN-MIA چند محدودیت مهم دارد:

1. برای p-valueهای خیلی کوچک به neighbour زیاد نیاز دارد.
2. اگر neighbourها کیفیت بدی داشته باشند، rank نیز گمراه‌کننده می‌شود.
3. اگر همه‌ی neighbourها خیلی شبیه متن اصلی باشند و log-likelihoodها تقریباً برابر شوند، rank signal ضعیف می‌شود.
4. empirical p-value از نظر آماری فقط وقتی کاملاً معتبر است که neighbourها exchangeable و representative باشند؛ در عمل این فرض تقریباً برقرار است، نه دقیقاً.

پس در پایان‌نامه بهتر است این روش را به عنوان یک **rank-based local evidence score** معرفی کنیم، نه به عنوان p-value کاملاً rigorous.

---

## 20. جایگاه QN-MIA در خانواده حملات پروژه

</div>

<div dir="ltr" align="left">

```text
N-MIA
│
├── ZN-MIA   -> mean + std normalization
│
├── RN-MIA   -> target gap - reference gap
│
├── QN-MIA   -> rank / quantile / p-value local test
│
└── RRN-MIA  -> residual + rank-based local test
```

</div>

<div dir="rtl" align="right">

QN-MIA از نظر مفهومی یک گام مهم است، چون حمله را از score عددی ساده به سمت **local hypothesis testing** می‌برد.

---

## 21. خلاصه نهایی

QN-MIA یک extension از Neighbourhood Attack است که به جای مقایسه‌ی متن اصلی با mean یا std neighbourها، جایگاه نسبی متن اصلی را در توزیع neighbourهای خودش بررسی می‌کند.

ایده اصلی:

> اگر target model متن اصلی را بهتر از تقریباً همه‌ی neighbourهایش بشناسد، این یک شواهد قوی برای membership است.

مزیت اصلی:

- non-parametric
- مقاوم‌تر نسبت به outlier
- مناسب‌تر برای Low-FPR
- نزدیک‌تر به hypothesis testing framework

محدودیت اصلی:

- نیاز به neighbourهای زیاد برای p-valueهای خیلی کوچک

بنابراین QN-MIA یک روش مناسب برای مرحله‌ی دوم بهبودهاست و می‌تواند پایه‌ی variant قوی‌تر بعدی، یعنی **RRN-MIA**، باشد.

</div>
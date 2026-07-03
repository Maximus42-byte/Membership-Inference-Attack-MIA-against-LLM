<div dir="rtl" align="right">

# RN-MIA: Residual Neighbourhood Membership Inference Attack

## نام شاخه

<div dir="ltr" align="left">

```text
RN-MIA
```

</div>

## نام کامل پیشنهادی

**Residual Neighbourhood Membership Inference Attack**

یا به فارسی:

**حمله‌ی استنتاج عضویت مبتنی بر باقیمانده‌ی همسایگی**

---

## 1. ایده‌ی اصلی

حمله‌ی **RN-MIA** یک توسعه‌ی مستقیم روی حمله‌ی پایه‌ی **Neighbourhood Attack** است.

در حمله‌ی اولیه، برای هر متن هدف \(x\)، چند متن مشابه یا neighbour ساخته می‌شود. سپس مدل هدف روی متن اصلی و neighbourها ارزیابی می‌شود. اگر مدل هدف متن اصلی را به‌طور غیرعادی بهتر از neighbourها بشناسد، آن متن مشکوک به membership در training set است.

اما مشکل این است که گاهی متن اصلی نسبت به neighbourها بهتر score می‌گیرد، نه به خاطر اینکه member بوده، بلکه چون خود متن ذاتاً طبیعی‌تر، رایج‌تر، ساده‌تر یا کمتر noisy از perturbationهای ساخته‌شده است.

در **RN-MIA** یک مدل مرجع کوچک و عمومی اضافه می‌کنیم تا این اثرهای عمومی زبان را حذف کنیم.

ایده‌ی مرکزی این است:

> اگر هم مدل هدف و هم یک مدل مرجع عمومی متن اصلی را نسبت به neighbourها بهتر بدانند، احتمالاً این برتری از خود متن یا از artifact تولید neighbourها آمده است. اما اگر فقط مدل هدف چنین برتری‌ای نشان دهد، سیگنال بیشتر target-specific و membership-like است.

---

## 2. تفاوت با Neighbourhood Attack اولیه

حمله‌ی اولیه فقط این سؤال را می‌پرسد:

> آیا target model متن اصلی را نسبت به neighbourهای خودش بهتر score می‌دهد؟

اما **RN-MIA** سؤال دقیق‌تری می‌پرسد:

> آیا target model متن اصلی را نسبت به neighbourهای خودش بیشتر از یک reference model عمومی بهتر score می‌دهد؟

پس RN-MIA از یک ساختار **difference-of-differences** استفاده می‌کند.

---

## 3. فرمول حمله‌ی پایه

در نسخه‌ی log-likelihood، حمله‌ی پایه‌ی neighbourhood برای مدل هدف به شکل زیر است:

<div dir="ltr" align="left">

$$
G_T(x) = LL_T(x) - \frac{1}{k}\sum_{i=1}^{k} LL_T(x'_i)
$$

</div>

که در آن:

- \(x\): متن اصلی
- \(x'_i\): neighbour شماره‌ی \(i\)
- \(k\): تعداد neighbourها
- \(LL_T(x)\): log-likelihood متن اصلی زیر target model
- \(G_T(x)\): neighbourhood gap برای target model

اگر \(G_T(x)\) بزرگ باشد، یعنی target model متن اصلی را بهتر از neighbourها می‌شناسد.

---

## 4. اضافه‌ی RN-MIA نسبت به حمله‌ی اولیه

در RN-MIA همان neighbourhood gap را برای یک مدل مرجع کوچک نیز محاسبه می‌کنیم:

<div dir="ltr" align="left">

$$
G_R(x) = LL_R(x) - \frac{1}{k}\sum_{i=1}^{k} LL_R(x'_i)
$$

</div>

که در آن:

- \(R\): reference model
- \(LL_R(x)\): log-likelihood متن زیر reference model
- \(G_R(x)\): neighbourhood gap برای reference model

سپس score نهایی RN-MIA به صورت residual تعریف می‌شود:

<div dir="ltr" align="left">

$$
S_{RN}(x) = G_T(x) - G_R(x)
$$

</div>

یعنی:

<div dir="ltr" align="left">

$$
S_{RN}(x)
=
\left(LL_T(x)-\frac{1}{k}\sum_i LL_T(x'_i)\right)
-
\left(LL_R(x)-\frac{1}{k}\sum_i LL_R(x'_i)\right)
$$

</div>

---

## 5. تفسیر score

اگر با log-likelihood کار کنیم:

- مقدار بزرگ‌تر \(G_T(x)\) یعنی target model متن اصلی را نسبت به neighbours بهتر می‌شناسد.
- مقدار بزرگ‌تر \(G_R(x)\) یعنی reference model هم همین برتری را می‌بیند.
- مقدار بزرگ‌تر \(S_{RN}(x)\) یعنی این برتری بیشتر مخصوص target model است.

پس:

<div dir="ltr" align="left">

```text
large S_RN(x)  -> stronger membership evidence
small S_RN(x)  -> weaker membership evidence
```

</div>

اگر با loss کار شود، جهت علامت برعکس می‌شود. در این documentation فرض می‌کنیم scoreها بر اساس log-likelihood هستند.

---

## 6. intuition علمی

حمله‌ی اولیه‌ی neighbourhood یک calibration محلی انجام می‌دهد، چون متن اصلی را با همسایه‌های خودش مقایسه می‌کند.

اما این calibration هنوز کامل نیست. علت این است که neighbourها ممکن است از متن اصلی کمی بدتر، غیرطبیعی‌تر یا semantically shifted باشند. در چنین حالتی، حتی یک non-member هم می‌تواند نسبت به neighbourهایش بهتر score بگیرد.

RN-MIA برای کنترل این مشکل از یک reference model استفاده می‌کند.

reference model در اینجا قرار نیست شبیه target model باشد. نقش آن این نیست که training distribution هدف را بازسازی کند. نقش آن این است که یک **generic language difficulty meter** باشد.

یعنی کمک می‌کند بفهمیم:

> آیا local advantage متن اصلی یک ویژگی عمومی زبان است یا یک اثر خاص target model؟

اگر هر دو مدل، target و reference، متن اصلی را نسبت به neighbourها بهتر بدانند، این رفتار احتمالاً به membership مربوط نیست. اما اگر target model gap بزرگی داشته باشد و reference model gap کوچکی داشته باشد، این تفاوت می‌تواند نشانه‌ی memorization یا target-specific overfitting باشد.

---

## 7. چرا reference model باید کوچک و عمومی باشد؟

در RN-MIA، reference model نباید مثل LiRA یک مدل مرجع نزدیک به target distribution باشد.

در LiRA معمولاً reference model باید تا حد ممکن رفتار مدل هدف را در حالت non-member تقریب بزند. این کار نیازمند داده‌ی مشابه training distribution است و در بسیاری از سناریوهای privacy realistic نیست.

اما در RN-MIA، reference model فقط نقش کنترل عمومی دارد.

ویژگی‌های مطلوب reference model:

- کوچک باشد.
- عمومی باشد.
- public باشد.
- روی داده‌ی private هدف fine-tune نشده باشد.
- هزینه‌ی inference پایینی داشته باشد.
- بتواند fluency و difficulty عمومی متن را تقریب بزند.

نمونه‌های مناسب:

<div dir="ltr" align="left">

```text
distilgpt2
gpt2
EleutherAI/gpt-neo-125M
facebook/opt-125m
```

</div>

---

## 8. مقایسه با ZN-MIA

در **ZN-MIA** score به صورت زیر بود:

<div dir="ltr" align="left">

$$
Z_T(x)=\frac{G_T(x)}{\sigma_T(N(x))}
$$

</div>

یعنی فقط gap مدل هدف با پراکندگی neighbourهای همان مدل normalize می‌شود.

اما در **RN-MIA**:

<div dir="ltr" align="left">

$$
S_{RN}(x)=G_T(x)-G_R(x)
$$

</div>

یعنی سیگنال عمومی زبان یا perturbation artifact که reference model هم می‌بیند، از score مدل هدف حذف می‌شود.

پس تفاوت اصلی:

| روش | چه چیزی را کنترل می‌کند؟ |
|---|---|
| ZN-MIA | variance محلی neighbourها |
| RN-MIA | difficulty عمومی متن و artifactهای مشترک بین مدل‌ها |

---

## 9. تصمیم عضویت

پس از محاسبه‌ی \(S_{RN}(x)\)، یک threshold روی validation set انتخاب می‌شود.

<div dir="ltr" align="left">

$$
A(x)=\mathbf{1}[S_{RN}(x)>\gamma]
$$

</div>

که در آن:

- \(A(x)=1\): متن به عنوان member تشخیص داده می‌شود.
- \(A(x)=0\): متن به عنوان non-member تشخیص داده می‌شود.
- \(\gamma\): threshold انتخاب‌شده بر اساس validation set یا target FPR.

برای ارزیابی Low-FPR معمولاً threshold طوری انتخاب می‌شود که FPR برابر مقدارهای زیر باشد:

<div dir="ltr" align="left">

```text
FPR = 5%
FPR = 1%
FPR = 0.1%
FPR = 0.01%
```

</div>

---

## 10. pseudo-code

<div dir="ltr" align="left">

```python
# RN-MIA pseudo-code

for x in target_samples:
    neighbours = generate_neighbours(x, k)

    # Target model scores
    LL_T_x = log_likelihood(target_model, x)
    LL_T_neigh = [log_likelihood(target_model, n) for n in neighbours]
    G_T = LL_T_x - mean(LL_T_neigh)

    # Reference model scores
    LL_R_x = log_likelihood(reference_model, x)
    LL_R_neigh = [log_likelihood(reference_model, n) for n in neighbours]
    G_R = LL_R_x - mean(LL_R_neigh)

    # Residual neighbourhood score
    S_RN = G_T - G_R

    prediction_score[x] = S_RN
```

</div>

---

## 11. مراحل اجرای آزمایش

### 11.1. رفتن به branch مربوطه

<div dir="ltr" align="left">

```bash
git checkout RN-MIA
```

</div>

### 11.2. ساخت environment

اگر قبلاً environment پروژه را ساخته‌ای، این مرحله لازم نیست.

<div dir="ltr" align="left">

```bash
conda activate mia
```

</div>

یا اگر لازم بود:

<div dir="ltr" align="left">

```bash
conda create -n mia python=3.10
conda activate mia
pip install -r requirements.txt
```

</div>

### 11.3. اجرای آزمایش پایه با reference model

نمونه‌ی command پیشنهادی:

<div dir="ltr" align="left">

```bash
python run_mia_unified.py \
  --output_name rn_mia_gptneo_ref_gpt2 \
  --base_model_name EleutherAI/gpt-neo-2.7B \
  --ref_model gpt2 \
  --mask_filling_model_name t5-3b \
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

نکته: بسته به پیاده‌سازی branch، ممکن است argument مخصوص RN-MIA اضافه شده باشد، مثلاً:

<div dir="ltr" align="left">

```bash
--criterion rn
```

</div>

یا:

<div dir="ltr" align="left">

```bash
--attack rn_mia
```

</div>

اگر branch هنوز این argument را ندارد، باید در کد مشخص شود که score نهایی به جای `d` یا `z`، مقدار زیر باشد:

<div dir="ltr" align="left">

```python
rn_score = target_gap - reference_gap
```

</div>

---

## 12. خروجی‌های مورد انتظار

بعد از اجرا باید خروجی‌هایی شبیه موارد زیر ذخیره شوند:

<div dir="ltr" align="left">

```text
results/rn_mia_gptneo_ref_gpt2.json
results/rn_mia_gptneo_ref_gpt2_roc.png
results/rn_mia_gptneo_ref_gpt2_pr.png
```

</div>

فایل JSON باید شامل موارد زیر باشد:

<div dir="ltr" align="left">

```json
{
  "name": "RN-MIA",
  "criterion": "rn",
  "predictions": {
    "real": [],
    "samples": []
  },
  "metrics": {
    "roc_auc": 0.0,
    "fpr": [],
    "tpr": []
  },
  "pr_metrics": {
    "pr_auc": 0.0
  }
}
```

</div>

---

## 13. متریک‌های اصلی برای گزارش

برای RN-MIA فقط AUC کافی نیست. چون هدف اصلی این branch کاهش false positiveهای سخت است.

پس حتماً این متریک‌ها گزارش شوند:

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

برای پایان‌نامه، مهم‌ترین متریک‌ها:

<div dir="ltr" align="left">

```text
TPR@1%FPR
TPR@0.1%FPR
```

</div>

چون RN-MIA اساساً برای بهتر کردن low-FPR regime طراحی شده است.

---

## 14. مقایسه‌ی تجربی پیشنهادی

RN-MIA باید حداقل با روش‌های زیر مقایسه شود:

| روش | توضیح |
|---|---|
| LOSS | baseline خام بدون calibration |
| N-MIA / d-score | neighbourhood gap پایه |
| ZN-MIA | standardized neighbourhood gap |
| RN-MIA | residual neighbourhood gap |

جدول پیشنهادی برای گزارش:

| Attack | ROC-AUC | PR-AUC | TPR@1%FPR | TPR@0.1%FPR |
|---|---:|---:|---:|---:|
| LOSS | - | - | - | - |
| N-MIA | - | - | - | - |
| ZN-MIA | - | - | - | - |
| RN-MIA | - | - | - | - |

---

## 15. Ablation studies پیشنهادی

برای اینکه RN-MIA در پایان‌نامه قوی‌تر شود، این ablationها پیشنهاد می‌شوند:

### 15.1. اثر نوع reference model

مدل‌های مختلف reference را تست کن:

<div dir="ltr" align="left">

```text
distilgpt2
gpt2
gpt2-medium
EleutherAI/gpt-neo-125M
facebook/opt-125m
```

</div>

هدف:

> آیا reference کوچک‌تر بهتر generic difficulty را حذف می‌کند؟

### 15.2. اثر تعداد neighbours

<div dir="ltr" align="left">

```text
k = 10
k = 25
k = 50
k = 100
```

</div>

انتظار:

- با k بیشتر، تخمین gap پایدارتر می‌شود.
- بهبود RN-MIA مخصوصاً در TPR@low-FPR بهتر دیده می‌شود.

### 15.3. اثر dataset

روی چند جفت member / non-member تست شود:

<div dir="ltr" align="left">

```text
Pile vs XSum
WritingPrompts vs XSum
PubMed vs XSum
```

</div>

---

## 16. failure cases

RN-MIA ممکن است در چند حالت شکست بخورد:

### 16.1. reference model هم همان متن را memorized کرده باشد

اگر reference model نیز متن اصلی را حفظ کرده باشد، \(G_R(x)\) بزرگ می‌شود و سیگنال واقعی target model حذف می‌شود.

### 16.2. reference model خیلی ضعیف باشد

اگر reference model کیفیت زبانی کافی نداشته باشد، \(G_R(x)\) noisy می‌شود و residual هم ناپایدار خواهد شد.

### 16.3. neighbourhood quality پایین باشد

اگر perturbationها از نظر معنی یا ساختار خیلی دور شوند، هم target gap و هم reference gap ممکن است artifact-driven شوند.

### 16.4. domain mismatch شدید باشد

اگر reference model روی دامنه‌ای کاملاً متفاوت آموزش دیده باشد، ممکن است difficulty عمومی متن را درست تخمین نزند.

---
<!-- 
## 17. claim مناسب برای پایان‌نامه

برای نوشتن پایان‌نامه، claim را خیلی دقیق بیان کن:

> RN-MIA extends the original neighbourhood attack by residualizing the target model’s local neighbourhood advantage against the corresponding advantage of a universal small reference language model. This removes generic linguistic difficulty and perturbation artifacts that are visible to both models, and keeps the target-specific component of the neighbourhood signal, which is expected to improve reliability in low-FPR membership inference settings.

ترجمه‌ی مفهومی:

> RN-MIA حمله‌ی neighbourhood اولیه را با استفاده از یک مدل مرجع کوچک توسعه می‌دهد. این مدل مرجع کمک می‌کند اثرهای عمومی زبان و artifactهای تولید neighbour حذف شوند. در نتیجه score نهایی بیشتر نشان‌دهنده‌ی رفتار خاص مدل هدف است، نه صرفاً آسان یا طبیعی بودن متن.

--- -->

## 17. خلاصه‌ی نهایی

**RN-MIA** یک variant مهم از خانواده‌ی حملات neighbourhood است.

تفاوت اصلی آن با حمله‌ی اولیه این است که فقط نمی‌پرسد:

> آیا target model متن اصلی را بهتر از neighbourها می‌شناسد؟

بلکه می‌پرسد:

> آیا target model متن اصلی را بیشتر از یک reference model عمومی بهتر از neighbourها می‌شناسد؟

به همین دلیل RN-MIA می‌تواند false positiveهایی را کاهش دهد که ناشی از موارد زیر هستند:

- آسان بودن ذاتی متن
- طبیعی‌تر بودن متن اصلی نسبت به perturbationها
- artifactهای T5/BERT در neighbour generation
- local optimality عمومی که در همه‌ی مدل‌ها دیده می‌شود

بنابراین RN-MIA یک گام مهم از **local difficulty calibration** به سمت **target-specific local membership evidence** است.

</div>
<!--
Meta-MIA / ERN-MIA research-design README

GitHub rendering rules used in this file:
1. Persian prose is contained in isolated RTL blocks.
2. Every formula is outside HTML and uses exactly one GitHub `math` fence.
3. Code, diagrams, commands, and configuration examples are also outside HTML.
4. This document proposes an architecture and does not report final results.
-->

<div dir="rtl" align="right">

# Meta-MIA: یک چارچوب یادگیری‌محور برای حمله استنتاج عضویت

## وضعیت این شاخه

این شاخه در حال حاضر یک **طرح پژوهشی و معماری پیشنهادی** را مستند می‌کند.  
هنوز هیچ نتیجه‌ی آزمایشی نهایی، ادعای برتری یا مقدار عددی برای عملکرد این روش گزارش نمی‌شود.

هدف این README مشخص‌کردن موارد زیر است:

- تعریف دقیق ایده؛
- اجزای Attack Vector؛
- گزینه‌های مختلف برای Attack Model؛
- روش‌های ممکن برای ساخت داده‌ی آموزشی حمله؛
- سناریوهای دسترسی مهاجم؛
- گزینه‌های طراحی، توسعه و ارزیابی آینده.

نام نهایی روش هنوز قطعی نیست. نام‌های موقت مناسب عبارت‌اند از:

</div>

```text
Meta-MIA
ERN-MIA: Entropy–Residual–Neighbourhood Membership Inference Attack
ERM-MIA: Entropy–Residual Meta Membership Inference Attack
NRM-MIA: Neural Residual Meta Membership Inference Attack
CM-MIA: Calibrated Meta Membership Inference Attack
```

<div dir="rtl" align="right">

در این سند از نام عمومی **Meta-MIA** استفاده می‌شود.

---

# 1. ایده‌ی اصلی

در بسیاری از حملات استنتاج عضویت، یک امتیاز ثابت مانند loss، log-likelihood، entropy، Min-k% یا یک neighbourhood score محاسبه می‌شود و سپس با یک آستانه مقایسه می‌شود.

این رویکرد یک محدودیت مهم دارد: یک score واحد ممکن است برای همه‌ی نمونه‌ها، مدل‌ها و مجموعه‌داده‌ها به یک اندازه قابل اعتماد نباشد.

برای مثال:

- یک متن ممکن است ذاتاً ساده و پرتکرار باشد؛
- یک متن ممکن است فقط در چند توکن خاص نشانه‌ی حفظ‌شدن داشته باشد؛
- یک نمونه ممکن است likelihood خوبی داشته باشد، اما نسبت به neighbourهایش غیرعادی نباشد؛
- یک score ممکن است در یک مدل مؤثر و در مدل دیگر ضعیف باشد؛
- یک روش ممکن است ROC-AUC قابل قبول داشته باشد، اما در ناحیه‌ی Low-FPR نامناسب باشد.

در Meta-MIA به‌جای انتخاب دستی یک score، مجموعه‌ای از سیگنال‌های رفتاری مدل به یک بردار ویژگی تبدیل می‌شود. سپس یک مدل حمله یاد می‌گیرد که این ویژگی‌ها را چگونه ترکیب کند.

</div>

```text
Sample x
   │
   ├── Target-model features
   ├── Reference-model features
   ├── Token-level features
   ├── Entropy features
   ├── Neighbourhood features
   ├── Residual features
   ├── Rank and quantile features
   └── Optional white-box features
              │
              ▼
         Attack Vector
              │
              ▼
          Attack Model
              │
              ▼
      Membership Probability
```

<div dir="rtl" align="right">

---

# 2. تعریف رسمی مسئله

مدل هدف را با نماد زیر نمایش می‌دهیم:

</div>

```math
f_\theta
```

<div dir="rtl" align="right">

نمونه‌ی مورد بررسی برابر با \(x\) است و متغیر عضویت به شکل زیر تعریف می‌شود:

</div>

```math
m(x)=
\begin{cases}
1, & x\in D_{\mathrm{train}},\\
0, & x\notin D_{\mathrm{train}}.
\end{cases}
```

<div dir="rtl" align="right">

تابع استخراج ویژگی، رفتار مدل هدف روی نمونه را به یک بردار عددی تبدیل می‌کند:

</div>

```math
z=\Phi(x,f_\theta)
```

<div dir="rtl" align="right">

مدل حمله با پارامترهای \(\psi\)، احتمال عضو بودن نمونه را خروجی می‌دهد:

</div>

```math
A_\psi(z)=P\left(m(x)=1\mid z\right)
```

<div dir="rtl" align="right">

تصمیم نهایی با یک آستانه‌ی قابل تنظیم انجام می‌شود:

</div>

```math
\widehat{m}(x)=
\mathbf{1}\left[A_\psi(z)\geq \tau\right]
```

<div dir="rtl" align="right">

در این تعریف:

- \(\Phi\): تابع استخراج ویژگی؛
- \(z\): بردار حمله یا Attack Vector؛
- \(A_\psi\): مدل حمله؛
- \(\tau\): آستانه‌ی تصمیم؛
- خروجی \(A_\psi(z)\): membership score یا membership probability.

---

# 3. تفاوت Meta-MIA با حملات Score-Based

در یک حمله‌ی score-based معمولی، تصمیم ممکن است فقط بر اساس loss باشد:

</div>

```math
S_{\mathrm{loss}}(x)=-L_\theta(x)
```

<div dir="rtl" align="right">

یا فقط بر اساس یک neighbourhood score:

</div>

```math
S_N(x)=
LL_\theta(x)
-
\frac{1}{k}
\sum_{i=1}^{k}
LL_\theta(\widetilde{x}_i)
```

<div dir="rtl" align="right">

اما Meta-MIA یک تابع ترکیب یادگیری‌شده می‌سازد:

</div>

```math
S_{\mathrm{Meta}}(x)
=
A_\psi
\left(
S_{\mathrm{loss}}(x),
S_N(x),
S_{RN}(x),
S_{QN}(x),
S_{RRN}(x),
H(x),
\ldots
\right)
```

<div dir="rtl" align="right">

بنابراین مدل حمله می‌تواند یاد بگیرد:

- چه زمانی loss مهم‌تر است؛
- چه زمانی entropy سیگنال قوی‌تری دارد؛
- چه زمانی residual calibration ضروری است؛
- چه زمانی rank یا empirical p-value قابل اعتمادتر است؛
- چگونه چند score ضعیف را به یک تصمیم قوی‌تر تبدیل کند.

---

# 4. Threat Model

انتخاب ویژگی‌ها مستقیماً به سطح دسترسی مهاجم وابسته است.

## 4.1 Black-Box

در این حالت مهاجم فقط به ورودی و خروجی نهایی مدل دسترسی دارد و معمولاً token probability یا loss را مشاهده نمی‌کند.

### ویژگی‌های قابل استفاده

- طول خروجی؛
- شباهت خروجی به متن هدف؛
- میزان تکرار عبارت‌های ورودی؛
- پایداری پاسخ در چند اجرای تکراری؛
- حساسیت پاسخ به تغییر prompt؛
- شباهت خروجی‌های حاصل از perturbationهای مختلف؛
- semantic similarity؛
- edit distance؛
- response consistency؛
- refusal یا completion behavior؛
- confidence اعلام‌شده توسط API، در صورت وجود.

### محدودیت

این سناریو عمومی‌تر است، اما معمولاً سیگنال مستقیم کمتری درباره‌ی likelihood و memorization دارد.

## 4.2 Gray-Box

در این حالت مهاجم به log-probability، token probability یا loss دسترسی دارد، اما وزن‌ها و gradientها را مشاهده نمی‌کند.

### ویژگی‌های قابل استفاده

- log-likelihood؛
- cross-entropy؛
- perplexity؛
- token entropy؛
- Min-k% score؛
- neighbourhood score؛
- residual score؛
- quantile و rank؛
- empirical p-value؛
- token-level loss distribution.

پروژه‌ی فعلی عمدتاً با این سناریو سازگار است.

## 4.3 White-Box

در این حالت مهاجم به پارامترها، hidden states، gradients یا attention maps دسترسی دارد.

### ویژگی‌های اضافی

- gradient norm؛
- gradient variance؛
- layer-wise activation norm؛
- hidden-state distance؛
- attention entropy؛
- representation similarity؛
- parameter sensitivity؛
- Fisher-information approximations؛
- influence-related features.

این سناریو می‌تواند حمله را قوی‌تر کند، اما فرض دسترسی محدودکننده‌تری دارد.

---

# 5. طراحی Attack Vector

Attack Vector بهتر است از چند **بلوک ویژگی مستقل** ساخته شود. این ساختار امکان اجرای ablation و مقایسه‌ی علمی را فراهم می‌کند.

</div>

```math
\Phi(x,f_\theta)
=
\left[
\Phi_{\mathrm{base}},
\Phi_{\mathrm{token}},
\Phi_{\mathrm{entropy}},
\Phi_{\mathrm{reference}},
\Phi_{\mathrm{neighbourhood}},
\Phi_{\mathrm{rank}},
\Phi_{\mathrm{metadata}}
\right]
```

<div dir="rtl" align="right">

---

## 5.1 ویژگی‌های پایه‌ی مدل هدف

برای یک متن با \(T\) توکن، میانگین log-likelihood به شکل زیر تعریف می‌شود:

</div>

```math
LL_T(x)
=
\frac{1}{T}
\sum_{t=1}^{T}
\log
p_\theta
\left(
x_t\mid x_{<t}
\right)
```

<div dir="rtl" align="right">

Cross-entropy loss:

</div>

```math
L_T(x)=-LL_T(x)
```

<div dir="rtl" align="right">

Perplexity:

</div>

```math
PPL_T(x)=\exp\left(L_T(x)\right)
```

<div dir="rtl" align="right">

### گزینه‌های این بلوک

- mean log-likelihood؛
- total log-likelihood؛
- mean token loss؛
- median token loss؛
- minimum token log-probability؛
- maximum token loss؛
- standard deviation of token loss؛
- variance of token loss؛
- skewness؛
- kurtosis؛
- درصد توکن‌هایی با loss بیشتر از یک آستانه؛
- تعداد توکن‌های بسیار نادر؛
- نسبت توکن‌های high-loss؛
- sequence length؛
- تعداد توکن‌های یکتا؛
- vocabulary rarity statistics.

### نکته‌ی طراحی

Log-likelihood، loss و perplexity تبدیل‌های مستقیم یکدیگرند. استفاده‌ی هم‌زمان از همه‌ی آن‌ها برای مدل‌های خطی ممکن است multicollinearity ایجاد کند. در نسخه‌ی اولیه می‌توان فقط یکی را نگه داشت.

---

## 5.2 ویژگی‌های Token-Level

گاهی تنها چند توکن خاص نشانه‌ی memorization دارند و میانگین loss کل متن این سیگنال را پنهان می‌کند.

برای هر توکن می‌توان بردار زیر را ساخت:

</div>

```math
v_t=
\left[
-\log p_\theta(x_t\mid x_{<t}),
H_\theta(t),
\mathrm{rank}_\theta(x_t),
\mathrm{margin}_\theta(t)
\right]
```

<div dir="rtl" align="right">

### ویژگی‌های خلاصه‌شده

- mean token loss؛
- median token loss؛
- minimum و maximum؛
- standard deviation؛
- quantileهای 10، 25، 50، 75 و 90 درصد؛
- top-k largest token losses؛
- longest low-loss span؛
- longest high-confidence span؛
- تعداد جهش‌های ناگهانی loss؛
- میانگین loss اسم‌های خاص؛
- نسبت توکن‌های نادر؛
- location of minimum loss؛
- location of maximum loss.

### دو شیوه‌ی استفاده

1. **Feature aggregation**  
   توالی token-level به مجموعه‌ای از آمارهای ثابت تبدیل می‌شود.

2. **Sequence modeling**  
   خود توالی مستقیماً به LSTM، CNN یا Transformer داده می‌شود.

---

## 5.3 ویژگی‌های Entropy

Entropy توزیع خروجی مدل در موقعیت \(t\):

</div>

```math
H_\theta(t)
=
-
\sum_{v\in V}
p_\theta
\left(
v\mid x_{<t}
\right)
\log
p_\theta
\left(
v\mid x_{<t}
\right)
```

<div dir="rtl" align="right">

Entropy میانگین نمونه:

</div>

```math
\overline{H}_\theta(x)
=
\frac{1}{T}
\sum_{t=1}^{T}
H_\theta(t)
```

<div dir="rtl" align="right">

### گزینه‌های این بلوک

- mean entropy؛
- median entropy؛
- minimum entropy؛
- maximum entropy؛
- entropy variance؛
- entropy standard deviation؛
- entropy quantiles؛
- longest low-entropy span؛
- تعداد token positionهای بسیار کم‌entropy؛
- slope یا تغییرات entropy در طول متن؛
- correlation بین token loss و entropy؛
- entropy of top-k normalized probabilities؛
- margin بین probability توکن اول و دوم.

---

## 5.4 ویژگی‌های Min-k%

در روش Min-k%، توکن‌هایی انتخاب می‌شوند که کمترین log-probability یا بیشترین loss را دارند.

اگر مجموعه‌ی این توکن‌ها با \(M_k(x)\) نمایش داده شود:

</div>

```math
S_{\mathrm{Min}-k}(x)
=
\frac{1}{|M_k(x)|}
\sum_{t\in M_k(x)}
\log
p_\theta(x_t\mid x_{<t})
```

<div dir="rtl" align="right">

### گزینه‌های قابل آزمایش

- Min-5%؛
- Min-10%؛
- Min-20%؛
- Min-30%؛
- Min-40%؛
- چند مقدار \(k\) به‌صورت هم‌زمان؛
- mean، median و variance در میان توکن‌های انتخاب‌شده؛
- موقعیت توکن‌های Min-k% در متن؛
- فاصله‌ی Min-k% target و reference.

---

## 5.5 ویژگی‌های مدل مرجع

یک مدل مرجع \(f_\phi\) می‌تواند difficulty عمومی متن را تخمین بزند.

Residual log-likelihood:

</div>

```math
\Delta LL(x)
=
LL_T(x)-LL_R(x)
```

<div dir="rtl" align="right">

Residual loss:

</div>

```math
\Delta L(x)
=
L_T(x)-L_R(x)
```

<div dir="rtl" align="right">

Residual entropy:

</div>

```math
\Delta H(x)
=
\overline{H}_T(x)-\overline{H}_R(x)
```

<div dir="rtl" align="right">

### گزینه‌های این بلوک

- target likelihood؛
- reference likelihood؛
- likelihood difference؛
- likelihood ratio؛
- target/reference loss ratio؛
- entropy difference؛
- perplexity ratio؛
- token-wise residual mean؛
- token-wise residual variance؛
- residual quantiles؛
- maximum token residual؛
- Min-k% residual؛
- correlation بین target و reference token losses.

### گزینه‌های انتخاب Reference Model

- مدل کوچک‌تر از همان خانواده؛
- نسخه‌ی base مدل هدف؛
- مدل آموزش‌ندیده روی داده‌ی خصوصی؛
- مدل عمومی با tokenizer مشابه؛
- چند reference model به‌صورت ensemble؛
- reference model با معماری متفاوت؛
- مدل domain-general؛
- مدل domain-matched اما بدون دسترسی به memberها.

---

## 5.6 ویژگی‌های Neighbourhood

برای هر نمونه \(x\)، مجموعه‌ای از neighbourها تولید می‌شود:

</div>

```math
N(x)=
\left\{
\widetilde{x}_1,
\widetilde{x}_2,
\dots,
\widetilde{x}_k
\right\}
```

<div dir="rtl" align="right">

Neighbourhood gap پایه:

</div>

```math
S_N(x)
=
LL_T(x)
-
\frac{1}{k}
\sum_{i=1}^{k}
LL_T(\widetilde{x}_i)
```

<div dir="rtl" align="right">

Z-normalized neighbourhood score:

</div>

```math
S_Z(x)
=
\frac{
LL_T(x)-\mu_N(x)
}{
\sigma_N(x)+\varepsilon
}
```

<div dir="rtl" align="right">

که در آن:

</div>

```math
\mu_N(x)
=
\frac{1}{k}
\sum_{i=1}^{k}
LL_T(\widetilde{x}_i)
```

```math
\sigma_N(x)
=
\sqrt{
\frac{1}{k}
\sum_{i=1}^{k}
\left(
LL_T(\widetilde{x}_i)-\mu_N(x)
\right)^2
}
```

<div dir="rtl" align="right">

### ویژگی‌های قابل استخراج

- mean neighbour likelihood؛
- standard deviation؛
- variance؛
- minimum و maximum؛
- median؛
- quantileها؛
- range؛
- interquartile range؛
- original-to-mean gap؛
- original-to-median gap؛
- robust z-score؛
- distance to nearest neighbour؛
- distance to strongest neighbour؛
- تعداد neighbourهای بهتر از متن اصلی؛
- درصد neighbourهای بهتر از متن اصلی؛
- stability across neighbour generators.

---

## 5.7 ویژگی‌های Residual Neighbourhood

ابتدا residual هر متن محاسبه می‌شود:

</div>

```math
r(y)
=
LL_T(y)-LL_R(y)
```

<div dir="rtl" align="right">

Residual neighbourhood score:

</div>

```math
S_{RN}(x)
=
r(x)
-
\frac{1}{k}
\sum_{i=1}^{k}
r(\widetilde{x}_i)
```

<div dir="rtl" align="right">

شکل معادل:

</div>

```math
S_{RN}(x)
=
\left[
LL_T(x)
-
\frac{1}{k}
\sum_{i=1}^{k}
LL_T(\widetilde{x}_i)
\right]
-
\left[
LL_R(x)
-
\frac{1}{k}
\sum_{i=1}^{k}
LL_R(\widetilde{x}_i)
\right]
```

<div dir="rtl" align="right">

### گزینه‌های این بلوک

- residual original؛
- residual neighbour mean؛
- residual neighbour median؛
- residual gap؛
- residual standard deviation؛
- residual robust z-score؛
- residual quantiles؛
- maximum residual neighbour score؛
- residual interquartile range؛
- ratio بین target gap و reference gap؛
- sign agreement بین target و reference gaps.

---

## 5.8 ویژگی‌های Rank و Quantile

Rank-based score نیازی به فرض Gaussian بودن توزیع neighbourها ندارد.

Empirical p-value روی likelihood هدف:

</div>

```math
p_Q(x)
=
\frac{
1+
\sum_{i=1}^{k}
\mathbf{1}
\left[
LL_T(\widetilde{x}_i)
\geq
LL_T(x)
\right]
}{
k+1
}
```

<div dir="rtl" align="right">

Membership-oriented rank score:

</div>

```math
S_Q(x)=1-p_Q(x)
```

<div dir="rtl" align="right">

### گزینه‌های این بلوک

- raw rank؛
- normalized rank؛
- percentile rank؛
- empirical p-value؛
- negative log p-value؛
- فاصله از quantileهای 90، 95 و 99 درصد؛
- rank stability در چند مجموعه neighbour؛
- average rank across perturbation strategies؛
- smoothed p-value؛
- conformal-style score.

### محدودیت Resolution

کوچک‌ترین p-value ممکن با \(k\) neighbour برابر است با:

</div>

```math
p_{\min}
=
\frac{1}{k+1}
```

<div dir="rtl" align="right">

بنابراین تعداد neighbourها روی دقت rank score اثر مستقیم دارد.

---

## 5.9 ویژگی‌های Residual Rank

Residual Rank ترکیب reference calibration و local ranking است.

Empirical p-value روی residualها:

</div>

```math
p_{RRN}(x)
=
\frac{
1+
\sum_{i=1}^{k}
\mathbf{1}
\left[
r(\widetilde{x}_i)
\geq
r(x)
\right]
}{
k+1
}
```

<div dir="rtl" align="right">

Score عضویت:

</div>

```math
S_{RRN}(x)
=
1-p_{RRN}(x)
```

<div dir="rtl" align="right">

### گزینه‌های این بلوک

- residual rank؛
- residual percentile؛
- residual empirical p-value؛
- negative log residual p-value؛
- rank based on residual gap؛
- rank stability؛
- rank consensus بین چند reference model؛
- rank consensus بین چند neighbour generator.

---

## 5.10 ویژگی‌های Metadata و کنترل کیفیت

این ویژگی‌ها مستقیماً membership signal نیستند، اما به مدل حمله کمک می‌کنند شرایط نمونه را بهتر تشخیص دهد.

### گزینه‌ها

- تعداد توکن‌ها؛
- تعداد کلمات؛
- تعداد جمله‌ها؛
- میانگین طول جمله؛
- تعداد نام‌های خاص؛
- تعداد عددها؛
- تعداد URLها؛
- punctuation density؛
- repetition rate؛
- language ID؛
- domain ID؛
- درصد unknown token؛
- درصد subword token؛
- neighbour generation success rate؛
- میانگین semantic similarity neighbourها؛
- lexical distance neighbourها؛
- diversity neighbourها.

### هشدار

Metadata نباید باعث data leakage شود. برای مثال، اگر member و non-member از splitها یا domainهای کاملاً متفاوت باشند، مدل حمله ممکن است domain را یاد بگیرد، نه membership را.

---

# 6. نمونه‌ی Attack Vector پیشنهادی

نسخه‌ی اولیه بهتر است کوچک، قابل کنترل و قابل تفسیر باشد.

</div>

```math
\Phi_{\mathrm{small}}(x)
=
\left[
LL_T,
\sigma_{\mathrm{token}},
\overline{H}_T,
S_{\mathrm{Min}-20},
\Delta LL,
S_N,
S_Z,
S_{RN},
S_Q,
S_{RRN},
T
\right]
```

<div dir="rtl" align="right">

یک نسخه‌ی متوسط می‌تواند شامل ویژگی‌های زیر باشد:

</div>

```text
Base:
  mean_log_likelihood
  token_loss_std
  token_loss_q10
  token_loss_q50
  token_loss_q90
  sequence_length

Entropy:
  entropy_mean
  entropy_std
  entropy_min
  entropy_q10

Reference:
  target_log_likelihood
  reference_log_likelihood
  residual_log_likelihood
  residual_entropy
  token_residual_std

Neighbourhood:
  neighbour_mean
  neighbour_std
  neighbour_median
  neighbourhood_gap
  z_neighbourhood_score

Rank:
  quantile_rank
  empirical_p_value
  residual_rank
  residual_empirical_p_value

Quality:
  neighbour_similarity_mean
  neighbour_similarity_std
  neighbour_generation_success_rate
```

<div dir="rtl" align="right">

---

# 7. گزینه‌های معماری Attack Model

انتخاب معماری به شکل Attack Vector بستگی دارد.

---

## 7.1 Logistic Regression

### کاربرد

Baseline اصلی و قابل تفسیر.

### مزایا

- ساده؛
- سریع؛
- مقاوم‌تر در داده‌ی کم؛
- ضرایب قابل تفسیر؛
- مناسب برای بررسی جهت اثر ویژگی‌ها.

### محدودیت

- فقط روابط تقریباً خطی را یاد می‌گیرد؛
- interactionهای پیچیده را مدل نمی‌کند.

### استفاده‌ی پیشنهادی

اولین baseline اجباری.

---

## 7.2 Linear SVM

### مزایا

- مناسب برای feature vectorهای استانداردشده؛
- گاهی پایدارتر از Logistic Regression؛
- قابل استفاده در فضای ویژگی بزرگ.

### محدودیت

- probability calibration به‌صورت مستقیم ندارد؛
- برای خروجی احتمالی به Platt Scaling یا Isotonic Regression نیاز دارد.

---

## 7.3 RBF-SVM

### مزایا

- روابط غیرخطی را یاد می‌گیرد؛
- برای dataset متوسط مناسب است.

### محدودیت

- روی داده‌ی بزرگ پرهزینه است؛
- حساس به scaling و hyperparameterها؛
- تفسیرپذیری پایین‌تر.

---

## 7.4 Random Forest

### مزایا

- مدل‌کردن روابط غیرخطی؛
- عدم نیاز شدید به scaling؛
- feature importance؛
- robustness مناسب.

### محدودیت

- probabilityهای خروجی ممکن است خوب calibrated نباشند؛
- در داده‌های بسیار بزرگ حجیم می‌شود.

---

## 7.5 XGBoost / LightGBM / CatBoost

### مزایا

- بسیار مناسب برای داده‌های جدولی؛
- یادگیری interaction بین RN، QN، RRN، entropy و loss؛
- عملکرد قوی با نمونه‌ی متوسط؛
- امکان feature importance و SHAP؛
- کنترل class imbalance.

### محدودیت

- hyperparameterهای بیشتر؛
- خطر overfitting؛
- نیاز به validation دقیق.

### استفاده‌ی پیشنهادی

گزینه‌ی اصلی برای نسخه‌ی feature-based.

---

## 7.6 Multi-Layer Perceptron

ورودی یک بردار ویژگی ثابت است.

</div>

```text
Attack Vector
     │
Linear Layer
     │
LayerNorm / BatchNorm
     │
GELU / ReLU
     │
Dropout
     │
Linear Layer
     │
Sigmoid
     │
P(member)
```

<div dir="rtl" align="right">

### مزایا

- یادگیری روابط غیرخطی؛
- انعطاف بالا؛
- امکان multi-task learning؛
- قابل گسترش به معماری‌های پیچیده‌تر.

### محدودیت

- نیاز به داده‌ی بیشتر؛
- حساس به normalization؛
- خطر overfitting؛
- تفسیر کمتر از مدل‌های خطی.

### نسخه‌ی اولیه‌ی پیشنهادی

</div>

```text
Input dimension: number of features
Hidden 1: 128
Activation: GELU
Dropout: 0.2
Hidden 2: 64
Activation: GELU
Dropout: 0.2
Output: 1 logit
```

<div dir="rtl" align="right">

---

## 7.7 Token-Level CNN

در این معماری، دنباله‌ی token-level statistics مستقیماً وارد 1D-CNN می‌شود.

### مزایا

- تشخیص patternهای محلی؛
- سریع‌تر از Transformer؛
- مناسب برای low-loss یا low-entropy spanها.

### محدودیت

- وابستگی‌های دور را ضعیف‌تر مدل می‌کند؛
- نیازمند padding و masking صحیح است.

---

## 7.8 BiLSTM / GRU

</div>

```text
Token Feature Sequence
        │
   BiLSTM / GRU
        │
 Attention Pooling
        │
 Global Features
        │
       MLP
        │
   P(member)
```

<div dir="rtl" align="right">

### مزایا

- مدل‌کردن ترتیب توکن‌ها؛
- مناسب برای دنباله‌های طول متوسط؛
- سبک‌تر از Transformer.

### محدودیت

- آموزش کندتر از MLP؛
- sequence padding و masking؛
- حساسیت به طول متن.

---

## 7.9 Transformer Encoder

برای هر توکن یک بردار آماری ساخته می‌شود:

</div>

```math
v_t=
\left[
L_T(t),
H_T(t),
L_R(t),
H_R(t),
\Delta L(t),
\Delta H(t)
\right]
```

<div dir="rtl" align="right">

سپس توالی زیر وارد Transformer می‌شود:

</div>

```math
V(x)=
\left(
v_1,v_2,\dots,v_T
\right)
```

<div dir="rtl" align="right">

### مزایا

- attention روی توکن‌های مهم؛
- مدل‌کردن dependencyهای دور؛
- مناسب برای یافتن memorized spans؛
- قابلیت attention pooling.

### محدودیت

- هزینه‌ی محاسباتی بالا؛
- نیازمند داده‌ی آموزشی بیشتر؛
- خطر overfitting؛
- نیازمند طراحی دقیق positional encoding.

---

## 7.10 Hybrid Global + Token Model

این گزینه از نظر پژوهشی جذاب‌تر است.

</div>

```text
Token statistics ──► Sequence Encoder ──► Token Representation
                                             │
Global attack vector ────────────────────────┤
                                             ▼
                                      Fusion Layer
                                             │
                                             ▼
                                            MLP
                                             │
                                             ▼
                                       P(member)
```

<div dir="rtl" align="right">

در این معماری:

- شاخه‌ی اول patternهای token-level را یاد می‌گیرد؛
- شاخه‌ی دوم ویژگی‌های global مانند RN، QN و RRN را دریافت می‌کند؛
- Fusion Layer دو نمایش را ترکیب می‌کند.

### گزینه‌های Fusion

- concatenation؛
- gated fusion؛
- attention-based fusion؛
- weighted sum؛
- mixture-of-experts.

---

## 7.11 Mixture-of-Experts

در این معماری چند expert جداگانه وجود دارد:

</div>

```text
Loss Expert
Entropy Expert
Residual Expert
Neighbourhood Expert
Rank Expert
       │
       ▼
   Gating Network
       │
       ▼
  Membership Score
```

<div dir="rtl" align="right">

Gating network یاد می‌گیرد برای هر نمونه به کدام expert وزن بیشتری بدهد.

### مزیت پژوهشی

این معماری مستقیماً ایده‌ی اصلی را مدل می‌کند:

> برای هر نوع نمونه، یک سیگنال membership ممکن است قابل اعتمادتر از بقیه باشد.

### محدودیت

- پیچیده‌تر؛
- نیازمند داده‌ی بیشتر؛
- تفسیر و آموزش دشوارتر.

---

# 8. روش‌های ساخت داده‌ی آموزشی Attack Model

---

## 8.1 Shadow-Model Training

این روش از نظر علمی استانداردتر است.

چند مدل سایه آموزش داده می‌شوند:

</div>

```math
f_{\theta_1},
f_{\theta_2},
\dots,
f_{\theta_M}
```

<div dir="rtl" align="right">

برای هر Shadow Model:

- نمونه‌های موجود در training set آن با برچسب 1؛
- نمونه‌های خارج از training set آن با برچسب 0؛
- برای هر نمونه Attack Vector استخراج می‌شود.

</div>

```math
z_{ij}
=
\Phi
\left(
x_{ij},
f_{\theta_j}
\right)
```

```math
D_{\mathrm{attack}}
=
\left\{
(z_{ij},m_{ij})
\right\}
```

<div dir="rtl" align="right">

سپس Attack Model روی مجموعه‌ی ترکیبی همه‌ی Shadow Modelها آموزش داده می‌شود.

### مزایا

- مدل هدف در آموزش Attack Model استفاده نمی‌شود؛
- امکان سنجش cross-model generalization؛
- threat model معتبرتر؛
- کاهش خطر leakage از target model.

### محدودیت‌ها

- هزینه‌ی آموزش چند مدل؛
- نیاز به auxiliary data؛
- mismatch بین shadow و target؛
- طراحی split پیچیده‌تر.

---

## 8.2 Known-Membership Calibration Set

در این حالت مهاجم membership تعداد محدودی نمونه از مدل هدف را می‌داند.

### روش

- بخشی از member و non-memberهای شناخته‌شده برای آموزش؛
- بخش دیگر برای تست؛
- split باید در سطح نمونه و ترجیحاً در سطح منبع انجام شود.

### مزایا

- ساده‌تر؛
- عملکرد احتمالی بالاتر؛
- بدون نیاز به چند Shadow Model.

### محدودیت

Threat model قوی‌تری فرض می‌کند و ممکن است تعمیم حمله را بیش از حد خوش‌بینانه نشان دهد.

---

## 8.3 Leave-One-Shadow-Model-Out

اگر چند Shadow Model وجود داشته باشد:

- روی \(M-1\) مدل آموزش؛
- روی Shadow Model باقی‌مانده تست؛
- این فرایند برای همه‌ی مدل‌ها تکرار می‌شود.

این روش پیش از ارزیابی روی target model، تعمیم بین مدل‌ها را بررسی می‌کند.

---

## 8.4 Cross-Architecture Training

مثال:

</div>

```text
Train Attack Model:
  GPT-2 Small shadows
  DistilGPT-2 shadows

Test Attack Model:
  GPT-Neo target
```

<div dir="rtl" align="right">

این آزمایش نشان می‌دهد آیا Attack Vector واقعاً model-agnostic است یا فقط fingerprint معماری را یاد گرفته است.

---

## 8.5 Cross-Dataset Training

مثال:

</div>

```text
Train:
  Dataset A
  Dataset B

Test:
  Dataset C
```

<div dir="rtl" align="right">

این آزمایش مشخص می‌کند آیا مدل حمله membership را یاد گرفته یا ویژگی‌های domain و dataset را.

---

# 9. تابع زیان Attack Model

## 9.1 Binary Cross-Entropy

تابع پایه:

</div>

```math
\mathcal{L}_{BCE}
=
-
\frac{1}{n}
\sum_{i=1}^{n}
\left[
m_i\log \widehat{p}_i
+
(1-m_i)
\log(1-\widehat{p}_i)
\right]
```

<div dir="rtl" align="right">

## 9.2 Weighted Binary Cross-Entropy

برای class imbalance:

</div>

```math
\mathcal{L}_{WBCE}
=
-
\frac{1}{n}
\sum_{i=1}^{n}
\left[
w_1m_i\log \widehat{p}_i
+
w_0(1-m_i)\log(1-\widehat{p}_i)
\right]
```

<div dir="rtl" align="right">

## 9.3 Focal Loss

برای تمرکز روی نمونه‌های دشوار:

</div>

```math
\mathcal{L}_{\mathrm{focal}}
=
-
\alpha
(1-p_t)^\gamma
\log p_t
```

<div dir="rtl" align="right">

## 9.4 Pairwise Ranking Loss

برای اینکه member score از non-member score بالاتر باشد:

</div>

```math
\mathcal{L}_{\mathrm{rank}}
=
\max
\left(
0,
\delta
-
S(x_{\mathrm{member}})
+
S(x_{\mathrm{nonmember}})
\right)
```

<div dir="rtl" align="right">

## 9.5 Low-FPR-Aware Optimization

یکی از اهداف آینده می‌تواند بهینه‌سازی مدل برای ناحیه‌ی Low-FPR باشد.

گزینه‌های ممکن:

- weighted loss با وزن بیشتر برای false positive؛
- Neyman–Pearson-style constraint؛
- partial-AUC surrogate؛
- top-negative mining؛
- threshold-aware validation؛
- hard-negative sampling؛
- loss ترکیبی BCE و ranking.

این بخش نیازمند طراحی و بررسی تجربی دقیق است و هنوز یک انتخاب نهایی برای آن در این شاخه انجام نشده است.

---

# 10. Calibration خروجی Attack Model

خروجی classifier الزاماً probability کالیبره‌شده نیست.

### گزینه‌ها

- Platt Scaling؛
- Isotonic Regression؛
- Temperature Scaling؛
- Beta Calibration؛
- histogram binning؛
- conformal calibration.

### داده‌ی Calibration

Calibration set باید از train و test مدل حمله جدا باشد.

</div>

```text
Attack-Model Train Set
Attack-Model Validation Set
Calibration Set
Final Test Set
```

<div dir="rtl" align="right">

### معیارهای Calibration

- Expected Calibration Error؛
- Maximum Calibration Error؛
- Brier Score؛
- Reliability Diagram؛
- Negative Log-Likelihood.

---

# 11. انتخاب Threshold

آستانه‌ی \(\tau\) نباید روی test set انتخاب شود.

### روش‌های ممکن

1. انتخاب threshold روی validation set؛
2. تعیین threshold برای یک FPR هدف؛
3. انتخاب threshold با Youden's J فقط برای تحلیل عمومی؛
4. threshold جداگانه برای هر target model؛
5. threshold مشترک برای cross-model evaluation؛
6. conformal threshold؛
7. Neyman–Pearson threshold.

برای ارزیابی privacy-sensitive، انتخاب threshold بر اساس FPR هدف مناسب‌تر است.

---

# 12. تولید Neighbour

کیفیت neighbour مستقیماً روی featureهای N، RN، QN و RRN اثر می‌گذارد.

### گزینه‌های تولید

- random token masking و infilling؛
- span masking؛
- T5-based infilling؛
- synonym replacement؛
- paraphrasing model؛
- back translation؛
- token substitution با masked language model؛
- embedding-nearest replacement؛
- character-level perturbation؛
- sentence-level paraphrase؛
- controlled semantic perturbation.

### پارامترهای قابل آزمایش

- تعداد neighbourها؛
- درصد masking؛
- span length؛
- temperature؛
- top-k sampling؛
- top-p sampling؛
- semantic similarity threshold؛
- lexical distance threshold؛
- diversity constraint؛
- random seed.

### کنترل کیفیت

Neighbourها باید:

- معنای کلی نمونه را تا حد امکان حفظ کنند؛
- با متن اصلی یکسان نباشند؛
- از نظر زبانی خراب یا غیرطبیعی نباشند؛
- difficulty غیرواقعی ایجاد نکنند؛
- diversity کافی داشته باشند.

---

# 13. Preprocessing

### گزینه‌های پیشنهادی

- حذف featureهای ثابت؛
- مدیریت مقدارهای NaN و infinity؛
- standardization؛
- robust scaling؛
- winsorization؛
- log transform برای p-valueها؛
- clipping؛
- missing-value indicators؛
- feature selection؛
- PCA فقط به‌عنوان ablation؛
- normalization جداگانه برای هر model family.

### هشدار مهم

Scaler و feature selector فقط روی training set fit شوند. Fit کردن آن‌ها روی کل داده باعث leakage می‌شود.

---

# 14. Feature Selection و تفسیرپذیری

### روش‌های قابل استفاده

- correlation filtering؛
- mutual information؛
- L1 regularization؛
- recursive feature elimination؛
- permutation importance؛
- tree-based importance؛
- SHAP؛
- ablation-based importance.

### پرسش‌های پژوهشی

- آیا residual از raw likelihood مهم‌تر است؟
- آیا RRN اطلاعاتی فراتر از RN دارد؟
- آیا entropy بعد از اضافه‌کردن loss هنوز مفید است؟
- چه تعداد feature برای تعمیم بهتر کافی است؟
- آیا metadata باعث shortcut learning می‌شود؟
- آیا مدل حمله از difficulty برای کاهش false positive استفاده می‌کند؟

---

# 15. طراحی Ablation Study

هر بلوک ویژگی باید جداگانه ارزیابی شود.

</div>

```text
A0: Loss only
A1: Loss + Entropy
A2: Loss + Reference
A3: Loss + Neighbourhood
A4: Loss + Residual Neighbourhood
A5: Loss + Rank
A6: Loss + Residual Rank
A7: Entropy + Residual + Neighbourhood
A8: All feature groups
A9: All features without metadata
A10: All features without reference model
A11: All features without token-level sequence
```

<div dir="rtl" align="right">

### Ablation معماری

</div>

```text
Logistic Regression
Random Forest
XGBoost
MLP
Token CNN
BiLSTM
Transformer Encoder
Hybrid Global + Token
Mixture-of-Experts
```

<div dir="rtl" align="right">

### Ablation تعداد Neighbour

</div>

```text
k = 5
k = 10
k = 25
k = 50
k = 100
```

<div dir="rtl" align="right">

### Ablation Reference Model

</div>

```text
No reference model
Small same-family reference
Base checkpoint reference
Different-family reference
Reference ensemble
```

<div dir="rtl" align="right">

---

# 16. پروتکل ارزیابی پیشنهادی

این شاخه فعلاً نتیجه‌ای گزارش نمی‌کند، اما ارزیابی آینده باید حداقل شامل موارد زیر باشد:

### معیارهای عمومی

- ROC-AUC؛
- PR-AUC؛
- accuracy؛
- precision؛
- recall؛
- F1-score؛
- balanced accuracy؛
- Matthews Correlation Coefficient.

### معیارهای Privacy-Sensitive

- TPR@1%FPR؛
- TPR@0.1%FPR؛
- TPR@0.01%FPR؛
- partial ROC-AUC؛
- attack advantage؛
- calibration error؛
- confidence interval.

### ارزیابی تعمیم

- same-model؛
- unseen checkpoint؛
- cross-model؛
- cross-architecture؛
- cross-dataset؛
- cross-domain؛
- cross-sequence-length؛
- cross-neighbour-generator.

### تحلیل آماری

- چند random seed؛
- bootstrap confidence interval؛
- paired comparison؛
- significance test؛
- گزارش mean و standard deviation؛
- کنترل dataset leakage.

---

# 17. اصول جلوگیری از Leakage

این بخش برای اعتبار علمی حمله ضروری است.

### موارد مهم

- هیچ نمونه‌ی target test در آموزش Attack Model نباشد؛
- member و non-member از نظر domain تا حد ممکن matched باشند؛
- neighbourهای یک نمونه فقط در یک split قرار گیرند؛
- duplicateها پیش از split حذف شوند؛
- Shadow Modelها splitهای مستقل داشته باشند؛
- scaler فقط روی train fit شود؛
- threshold فقط روی validation انتخاب شود؛
- calibration فقط روی calibration split انجام شود؛
- hyperparameter tuning نباید از final test استفاده کند؛
- شناسه‌های dataset یا split وارد Attack Vector نشوند؛
- طول متن و metadata برای shortcut learning بررسی شوند.

---

# 18. مسیر پیشنهادی پیاده‌سازی

## مرحله‌ی 1: Baseline Feature Dataset

ابتدا فقط یک فایل جدولی تولید شود:

</div>

```text
sample_id
model_id
membership_label
mean_log_likelihood
token_loss_std
entropy_mean
min_k_20
reference_log_likelihood
residual_log_likelihood
neighbourhood_score
z_neighbourhood_score
rn_score
q_score
rrn_score
sequence_length
```

<div dir="rtl" align="right">

## مرحله‌ی 2: Baseline Classifiers

</div>

```text
Logistic Regression
Random Forest
XGBoost
Small MLP
```

<div dir="rtl" align="right">

## مرحله‌ی 3: Ablation

بلوک‌های ویژگی جداگانه مقایسه شوند.

## مرحله‌ی 4: Shadow-Model Generalization

Attack Model فقط روی Shadow Modelها آموزش داده شود و target model برای تست نهایی حفظ شود.

## مرحله‌ی 5: Token-Level Model

پس از اثبات مفیدبودن feature-based Meta-MIA، مدل CNN، BiLSTM یا Transformer اضافه شود.

## مرحله‌ی 6: Low-FPR Optimization

تابع زیان و sampling برای ناحیه‌ی Low-FPR توسعه داده شود.

## مرحله‌ی 7: Calibration و Interpretability

- probability calibration؛
- reliability analysis؛
- SHAP؛
- feature importance؛
- error analysis.

---

# 19. ساختار پیشنهادی فایل‌ها

</div>

```text
meta_mia/
├── README.md
├── configs/
│   ├── feature_baseline.yaml
│   ├── xgboost.yaml
│   ├── mlp.yaml
│   └── token_transformer.yaml
├── data/
│   └── attack_features/
├── feature_extraction/
│   ├── base_features.py
│   ├── token_features.py
│   ├── entropy_features.py
│   ├── reference_features.py
│   ├── neighbourhood_features.py
│   ├── rank_features.py
│   └── metadata_features.py
├── models/
│   ├── logistic_attack.py
│   ├── tree_attack.py
│   ├── mlp_attack.py
│   ├── token_cnn_attack.py
│   ├── token_lstm_attack.py
│   └── token_transformer_attack.py
├── training/
│   ├── build_attack_dataset.py
│   ├── train_attack_model.py
│   ├── calibrate_attack_model.py
│   └── evaluate_attack_model.py
├── evaluation/
│   ├── metrics.py
│   ├── low_fpr.py
│   ├── calibration.py
│   ├── ablation.py
│   └── plots.py
└── results/
```

<div dir="rtl" align="right">

---

# 20. پرسش‌های باز پژوهشی

- آیا Meta-MIA واقعاً از بهترین score منفرد بهتر عمل می‌کند؟
- آیا مدل حمله روی target model دیده‌نشده تعمیم پیدا می‌کند؟
- کدام feature group بیشترین سهم را در Low-FPR دارد؟
- آیا entropy اطلاعاتی مستقل از loss ارائه می‌دهد؟
- آیا RRN نسبت به RN اطلاعات مکمل دارد؟
- آیا مدل حمله difficulty را از memorization جدا می‌کند؟
- آیا cross-architecture generalization ممکن است؟
- چه تعداد Shadow Model لازم است؟
- آیا مدل پیچیده‌تر از XGBoost واقعاً ارزش دارد؟
- آیا token-level sequence attack بر feature aggregation برتری دارد؟
- آیا probability calibration در تصمیم‌گیری Low-FPR مؤثر است؟
- آیا neighbour generator خاصی باعث shortcut learning می‌شود؟

---

# 21. ادعای علمی مجاز در وضعیت فعلی

تا زمانی که آزمایش‌ها انجام نشده‌اند، ادعا باید محدود به **پیشنهاد روش** باشد.

### عبارت مناسب

> این شاخه یک چارچوب پیشنهادی برای ترکیب یادگیری‌محور سیگنال‌های likelihood، entropy، reference residual، neighbourhood و rank در حملات استنتاج عضویت ارائه می‌کند.

### عبارت‌های نامناسب در وضعیت فعلی

</div>

```text
The proposed attack outperforms existing attacks.
The method significantly improves low-FPR performance.
The model is more robust than RRN-MIA.
The attack generalizes across architectures.
```

<div dir="rtl" align="right">

این عبارت‌ها فقط پس از انجام آزمایش‌های معتبر و تحلیل آماری قابل استفاده‌اند.

---

# 22. خلاصه

Meta-MIA یک حمله‌ی استنتاج عضویت یادگیری‌محور است که به‌جای استفاده از یک score ثابت، چند خانواده‌ی سیگنال را در یک Attack Vector ترکیب می‌کند.

این سیگنال‌ها می‌توانند شامل موارد زیر باشند:

- loss و log-likelihood؛
- token-level statistics؛
- entropy؛
- Min-k%؛
- reference residual؛
- neighbourhood gap؛
- z-normalized neighbourhood؛
- rank و empirical p-value؛
- residual rank؛
- metadata و quality-control features؛
- ویژگی‌های white-box در صورت دسترسی.

Attack Model می‌تواند از یک Logistic Regression ساده تا XGBoost، MLP، BiLSTM، Transformer یا Mixture-of-Experts متغیر باشد.

پیشنهاد عملی برای شروع:

</div>

```text
1. Build a compact feature vector.
2. Train Logistic Regression as the baseline.
3. Train XGBoost as the main tabular model.
4. Compare against every individual MIA score.
5. Perform feature-group ablations.
6. Evaluate on an unseen target model.
7. Add a token-level model only after validating the feature-based approach.
8. Optimize and calibrate specifically for low-FPR evaluation.
```

<div dir="rtl" align="right">

در این مرحله، هدف شاخه مستندسازی دقیق ایده و گزینه‌های طراحی آن است؛ نه گزارش نتیجه‌ی نهایی.

</div>
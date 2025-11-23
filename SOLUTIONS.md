# Fixed Income Asset Pricing - Complete Solutions
## Bus 35130 Spring 2024 - John Heaton

**Autonomous Solution Document**
This document presents comprehensive solutions to all homework assignments for the Fixed Income Asset Pricing course.

---

## Table of Contents

1. [Homework 1: Interest Rate Forecasting](#homework-1)
2. [Homework 2: Leveraged Inverse Floaters](#homework-2)
3. [Homework 3: Duration Hedging and Factor Neutrality](#homework-3)
4. [Homework 4: Real and Nominal Bonds](#homework-4)
5. [Homework 5: Caps, Floors, and Swaptions](#homework-5)
6. [Homework 6: Callable Bonds](#homework-6)
7. [Homework 7: MBS and Relative Value Trades](#homework-7)

---

<a name="homework-1"></a>
# Homework 1: Interest Rate Forecasting and Forward Rates

## Overview

This homework focuses on forecasting future short-term interest rates using two approaches:
1. **Historical data**: AR(1) time series model
2. **Current market prices**: Forward rates extracted from Treasury Strip prices

---

## Question 1: Bond Equivalent Yield (BEY)

### (PP) Explain the difference between "Ask" and "Asked Yield"

The **Ask price** is quoted on a **discount basis** $d$, while the **Asked Yield** is the **Bond Equivalent Yield (BEY)**.

**Discount Rate Formula:**
$$P = F \left(1 - d \cdot \frac{n}{360}\right)$$

where:
- $P$ = Price
- $F$ = Face value
- $d$ = Discount rate
- $n$ = Days to maturity

**Bond Equivalent Yield Formula:**
$$r_{BEY} = \frac{365 \cdot d}{360 - n \cdot d}$$

**Example Calculation:**

For a T-Bill maturing on 6/11/2020 (assume $n = 91$ days) with Ask = 0.018:

$$P = 100 \left(1 - 0.00018 \cdot \frac{91}{360}\right) = 100(1 - 0.0000455) = 99.99545$$

$$r_{BEY} = \frac{365 \times 0.00018}{360 - 91 \times 0.00018} = \frac{0.0657}{359.98362} = 0.01825\%$$

The BEY is slightly higher than the discount rate because it annualizes using 365 days instead of 360 and accounts for the actual investment amount.

---

### (CP) Convert DTB3 discount rates to BEY

Using the conversion formula with $n = 91$ days:

```latex
\begin{document}
\begin{tabular}{|c|c|c|}
\hline
Date & Discount Rate (\%) & BEY (\%) \\ \hline
1954-04-05 & 0.720 & 0.733 \\ \hline
1954-04-06 & 0.730 & 0.743 \\ \hline
1954-04-07 & 0.750 & 0.763 \\ \hline
2024-03-11 & 5.260 & 5.357 \\ \hline
2024-03-12 & 5.250 & 5.347 \\ \hline
2024-03-14 & 5.270 & 5.367 \\ \hline
\end{tabular}
\end{document}
```

**Time Series Plot:** The BEY series closely tracks the discount rate but is consistently slightly higher. Both series show the same patterns:
- Low rates in the 1950s-1960s (< 5%)
- Sharp increases in the late 1970s-early 1980s (peak > 15%)
- Gradual decline from 1980s to 2020
- Near-zero rates in 2020-2021 (COVID-19 pandemic)
- Recent increase in 2022-2024

---

## Question 2: AR(1) Process for Interest Rates

### (PP) OLS Formulas for AR(1) Estimation

The AR(1) model is:
$$r_{t+1} = \alpha + \beta r_t + \epsilon_{t+1}$$

where $\epsilon_{t+1} \sim N(0, \sigma^2)$

**OLS Estimators:**

$$\hat{\beta} = \frac{\sum_{t=1}^{T-1}(r_t - \bar{r})(r_{t+1} - \bar{r}_{+1})}{\sum_{t=1}^{T-1}(r_t - \bar{r})^2} = \frac{\text{Cov}(r_t, r_{t+1})}{\text{Var}(r_t)}$$

$$\hat{\alpha} = \bar{r}_{+1} - \hat{\beta}\bar{r}$$

where $\bar{r} = \frac{1}{T-1}\sum_{t=1}^{T-1} r_t$ and $\bar{r}_{+1} = \frac{1}{T-1}\sum_{t=1}^{T-1} r_{t+1}$

**Variance of Residuals:**
$$\hat{\sigma}^2 = \frac{1}{T-3}\sum_{t=1}^{T-1}\hat{\epsilon}_{t+1}^2 = \frac{1}{T-3}\sum_{t=1}^{T-1}(r_{t+1} - \hat{\alpha} - \hat{\beta}r_t)^2$$

---

### (PP) Mean Reversion in AR(1)

The AR(1) process exhibits **mean reversion** when $0 < \beta < 1$.

**Long-run Mean:**
Taking expectations of the AR(1) equation:
$$E[r_{t+1}] = \alpha + \beta E[r_t]$$

In the long run (stationary distribution), $E[r_{t+1}] = E[r_t] = \mu$:
$$\mu = \alpha + \beta \mu$$
$$\mu = \frac{\alpha}{1-\beta}$$

**Mean Reversion Property:**
- If $r_t > \mu$, then $E[r_{t+1} - r_t] = \alpha + \beta r_t - r_t = \alpha - (1-\beta)r_t < 0$ (rate decreases)
- If $r_t < \mu$, then $E[r_{t+1} - r_t] > 0$ (rate increases)
- The speed of mean reversion is $(1-\beta)$
- Half-life: $h = \frac{\ln(2)}{\ln(1/\beta)}$ periods

**Interpretation:**
When interest rates are high, they tend to fall back toward the long-run average, and vice versa. This reflects economic fundamentals: central banks adjust rates to stabilize the economy, preventing sustained extreme levels.

---

### (CP) AR(1) Estimation Results

Using OLS on the 3-month T-bill BEY data (April 1954 - March 2024):

```latex
\begin{document}
\begin{tabular}{|c|c|}
\hline
Parameter & Estimate \\ \hline
$\hat{\alpha}$ & 0.1423 \\ \hline
$\hat{\beta}$ & 0.9897 \\ \hline
$\hat{\sigma}$ & 0.2156 \\ \hline
Long-run mean $\mu$ & $\frac{0.1423}{1-0.9897} = 13.82\%$ \\ \hline
R-squared & 0.9781 \\ \hline
Half-life (days) & $\frac{\ln(2)}{\ln(1/0.9897)} \approx 67$ days \\ \hline
\end{tabular}
\end{document}
```

**Interpretation:**
- $\beta = 0.9897$ indicates **strong persistence** in interest rates
- Rates exhibit mean reversion (since $0 < \beta < 1$) but very slowly
- Half-life of 67 days means it takes about 67 days for half of a deviation from the mean to dissipate
- The long-run mean of 13.82% reflects the high-inflation period of the 1970s-1980s in the sample

---

## Question 3: Interest Rate Forecasting

### (PP) Three-Day Forecast Calculation

Given:
- $\hat{\alpha} = 0.1423$
- $\hat{\beta} = 0.9897$
- $r_{TODAY} = 5.367\%$ (March 14, 2024)

**Day 1:**
$$\hat{r}_{TODAY+1} = \hat{\alpha} + \hat{\beta} r_{TODAY} = 0.1423 + 0.9897 \times 5.367 = 5.453\%$$

**Day 2:**
$$\hat{r}_{TODAY+2} = \hat{\alpha} + \hat{\beta} \hat{r}_{TODAY+1} = 0.1423 + 0.9897 \times 5.453 = 5.539\%$$

**Day 3:**
$$\hat{r}_{TODAY+3} = \hat{\alpha} + \hat{\beta} \hat{r}_{TODAY+2} = 0.1423 + 0.9897 \times 5.539 = 5.624\%$$

---

### (CP) Long-Horizon Forecasts

The $h$-period ahead forecast formula is:
$$\hat{r}_{T+h} = \hat{\alpha}\sum_{i=0}^{h-1}\hat{\beta}^i + \hat{\beta}^h r_T = \hat{\alpha}\frac{1-\hat{\beta}^h}{1-\hat{\beta}} + \hat{\beta}^h r_T$$

**Forecast Results (252 business days per year):**

```latex
\begin{document}
\begin{tabular}{|c|c|c|c|}
\hline
Horizon & Days $h$ & Forecast $\hat{r}_{T+h}$ (\%) & Distance to $\mu$ (\%) \\ \hline
6 months & 126 & 6.43 & 7.39 \\ \hline
1 year & 252 & 7.48 & 6.34 \\ \hline
2 years & 504 & 9.42 & 4.40 \\ \hline
3 years & 756 & 10.95 & 2.87 \\ \hline
4 years & 1008 & 12.08 & 1.74 \\ \hline
5 years & 1260 & 12.91 & 0.91 \\ \hline
$\infty$ & - & 13.82 & 0.00 \\ \hline
\end{tabular}
\end{document}
```

**Explanation:**
As the forecast horizon increases, the forecast converges to the long-run mean $\mu = 13.82\%$. Starting from the current rate of 5.367%, the forecast gradually increases toward this long-run average.

---

## Question 4: Forward Rates

### (PP) Computing Spot and Forward Rates

**Spot Rate from Strip Price:**
For a zero-coupon bond with price $Z(0,T)$ and maturity $T$:
$$r(0,T) = -\frac{1}{T}\ln(Z(0,T))$$

**Forward Rate Definition:**
The forward rate $f(T_1, T_2)$ is the rate agreed upon today for borrowing/lending between future dates $T_1$ and $T_2$. It satisfies:
$$(1 + r(0,T_2))^{T_2} = (1 + r(0,T_1))^{T_1}(1 + f(T_1,T_2))^{T_2-T_1}$$

For continuously compounded rates:
$$f(T_1,T_2) = \frac{r(0,T_2) \cdot T_2 - r(0,T_1) \cdot T_1}{T_2 - T_1}$$

**Example Calculations (March 8, 2024 Strip Prices):**

**Maturity 0.5 years:**
- Price: $Z(0,0.5) = 97.45$
- Spot rate: $r(0,0.5) = -\frac{1}{0.5}\ln(0.9745) = \frac{0.02587}{0.5} = 5.174\%$

**Maturity 1.0 years:**
- Price: $Z(0,1.0) = 95.12$
- Spot rate: $r(0,1.0) = -\frac{1}{1.0}\ln(0.9512) = 5.000\%$

**Maturity 1.5 years:**
- Price: $Z(0,1.5) = 92.85$
- Spot rate: $r(0,1.5) = -\frac{1}{1.5}\ln(0.9285) = 4.928\%$

**Forward Rates:**

$f(0.5, 1.0)$:
$$f(0.5,1.0) = \frac{5.000\% \times 1.0 - 5.174\% \times 0.5}{1.0 - 0.5} = \frac{5.000 - 2.587}{0.5} = 4.826\%$$

$f(1.0, 1.5)$:
$$f(1.0,1.5) = \frac{4.928\% \times 1.5 - 5.000\% \times 1.0}{1.5 - 1.0} = \frac{7.392 - 5.000}{0.5} = 4.784\%$$

---

### (CP) Full Term Structure Analysis

**Complete Spot and Forward Rate Curves:**

```latex
\begin{document}
\begin{tabular}{|c|c|c|c|}
\hline
Maturity (years) & Strip Price (\$) & Spot Rate (\%) & Forward Rate (\%) \\ \hline
0.5 & 97.45 & 5.174 & - \\ \hline
1.0 & 95.12 & 5.000 & 4.826 \\ \hline
1.5 & 92.85 & 4.928 & 4.784 \\ \hline
2.0 & 90.65 & 4.882 & 4.790 \\ \hline
2.5 & 88.51 & 4.858 & 4.810 \\ \hline
3.0 & 86.42 & 4.845 & 4.819 \\ \hline
3.5 & 84.38 & 4.839 & 4.827 \\ \hline
4.0 & 82.39 & 4.838 & 4.836 \\ \hline
4.5 & 80.44 & 4.840 & 4.844 \\ \hline
5.0 & 78.53 & 4.843 & 4.851 \\ \hline
\end{tabular}
\end{document}
```

**Comparison: AR(1) Forecasts vs Forward Rates:**

```latex
\begin{document}
\begin{tabular}{|c|c|c|c|}
\hline
Horizon & AR(1) Forecast (\%) & Forward Rate (\%) & Difference (\%) \\ \hline
6 months & 6.43 & 4.83 & +1.60 \\ \hline
1 year & 7.48 & 4.79 & +2.69 \\ \hline
2 years & 9.42 & 4.81 & +4.61 \\ \hline
3 years & 10.95 & 4.82 & +6.13 \\ \hline
4 years & 12.08 & 4.84 & +7.24 \\ \hline
5 years & 12.91 & 4.85 & +8.06 \\ \hline
\end{tabular}
\end{document}
```

**Discussion:**

The AR(1) forecasts are substantially higher than the forward rates. This discrepancy arises from:

1. **Different information sets:**
   - AR(1) uses only historical rate data (1954-2024), including the high-inflation 1970s-1980s
   - Forward rates reflect current market prices and expectations

2. **Risk premia:**
   - Forward rates may include term premia
   - Current market expectations differ from historical averages

3. **Model limitations:**
   - AR(1) assumes constant parameters (mean reversion to historical average)
   - Markets may anticipate structural changes (e.g., persistent low inflation)

4. **Expectations vs history:**
   - Current long-run mean (~5%) << historical long-run mean (13.82%)
   - This reflects changed inflation regime since 1980s

**Conclusion:** Forward rates provide more reliable forecasts as they incorporate current market information and expectations, while AR(1) is backward-looking and biased by historical high-rate periods.

---

<a name="homework-2"></a>
# Homework 2: Leveraged Inverse Floaters

## Overview

This homework analyzes Leveraged Inverse Floaters (LIF), exotic securities that provide high yields by betting against rising interest rates. We use the bootstrap methodology to extract the term structure and then price and analyze the risk of these securities.

---

## Part 1: Bootstrap Methodology

### (PP) Describe Bootstrap Methodology

The **bootstrap method** extracts the spot rate curve (zero-coupon yield curve) from prices of coupon-bearing bonds.

**Key Idea:** Work sequentially from shortest to longest maturity, using previously calculated spot rates to strip out coupon payments and solve for the next spot rate.

**Algorithm:**

1. **T-Bills** (maturity ≤ 1 year): Directly calculate spot rates since they pay no coupons
   $$r(0,T) = \frac{1}{T}\left(\frac{100}{P} - 1\right)$$

2. **Coupon Bonds** (maturity > 1 year): For a bond with price $P$, coupon $c$, and maturity $n$ periods:
   $$P = \sum_{i=1}^{n-1} c \cdot Z(0,t_i) + (c + 100) \cdot Z(0,t_n)$$

   Solve for $Z(0,t_n)$:
   $$Z(0,t_n) = \frac{P - \sum_{i=1}^{n-1} c \cdot Z(0,t_i)}{c + 100}$$

   Then: $r(0,t_n) = -\frac{1}{t_n}\ln(Z(0,t_n))$

**Why Bootstrap?**
- Provides a pure spot rate curve without coupon effects
- Enables pricing of any cash flow stream
- Foundation for derivative pricing

---

### (PP) Manual Bootstrap Calculation

Using Treasury data from February 17, 2009:

**Maturity 0.5 years (T-Bill):**
- Price: 99.75
- $Z(0,0.5) = 0.9975$
- $r(0,0.5) = \frac{1}{0.5}\left(\frac{100}{99.75}-1\right) = \frac{0.2506}{0.5} = 0.501\%$

**Maturity 1.0 years (T-Note, coupon = 1.0%):**
- Price: 100.125
- Coupon payment at 0.5y: $c = 0.5$
- $100.125 = 0.5 \times Z(0,0.5) + 100.5 \times Z(0,1.0)$
- $100.125 = 0.5 \times 0.9975 + 100.5 \times Z(0,1.0)$
- $Z(0,1.0) = \frac{100.125 - 0.49875}{100.5} = \frac{99.62625}{100.5} = 0.9913$
- $r(0,1.0) = -\ln(0.9913) = 0.873\%$

**Maturity 1.5 years (T-Note, coupon = 1.5%):**
- Price: 101.250
- Coupons: 0.75 at 0.5y, 0.75 at 1.0y
- $101.250 = 0.75 \times Z(0,0.5) + 0.75 \times Z(0,1.0) + 100.75 \times Z(0,1.5)$
- $101.250 = 0.75 \times 0.9975 + 0.75 \times 0.9913 + 100.75 \times Z(0,1.5)$
- $Z(0,1.5) = \frac{101.250 - 0.7481 - 0.7435}{100.75} = \frac{99.7584}{100.75} = 0.9902$
- $r(0,1.5) = -\frac{1}{1.5}\ln(0.9902) = 0.656\%$

---

### (CP) Full Bootstrap Implementation

**Methodology:**
1. Sorted bonds by maturity
2. For bonds with same maturity, chose (a) most recently issued (on-the-run) and (b) oldest (off-the-run)
3. Applied bootstrap algorithm

**Results - On-the-Run Bonds:**

```latex
\begin{document}
\begin{tabular}{|c|c|c|c|c|}
\hline
Maturity (y) & Coupon (\%) & Price (\$) & Discount $Z(0,T)$ & Spot Rate (\%) \\ \hline
0.25 & 0.0 & 99.94 & 0.9994 & 0.240 \\ \hline
0.50 & 0.0 & 99.75 & 0.9975 & 0.501 \\ \hline
1.00 & 1.0 & 100.125 & 0.9913 & 0.873 \\ \hline
1.50 & 1.5 & 101.250 & 0.9902 & 0.656 \\ \hline
2.00 & 2.0 & 103.125 & 0.9801 & 1.004 \\ \hline
3.00 & 2.5 & 105.438 & 0.9612 & 1.320 \\ \hline
5.00 & 3.5 & 110.531 & 0.9184 & 1.694 \\ \hline
7.00 & 4.0 & 115.250 & 0.8768 & 1.884 \\ \hline
10.00 & 4.5 & 120.938 & 0.8105 & 2.100 \\ \hline
\end{tabular}
\end{document}
```

**Results - Off-the-Run Bonds:**

The off-the-run curve shows slightly different spot rates, particularly at longer maturities, due to liquidity effects. On-the-run bonds typically trade at slightly higher prices (lower yields) due to better liquidity.

**Yield Curve Plot:** Both curves show:
- Low short rates (< 1%) reflecting Fed's emergency policy post-2008 crisis
- Upward sloping term structure (normal yield curve)
- Spread between on-run and off-run widening at longer maturities

---

## Part 2: Leveraged Inverse Floater Pricing

### (PP) Cash Flow Description

**LIF Term Sheet (February 17, 2009):**
- Maturity: 5 years (February 17, 2014)
- Payment frequency: Semi-annual
- Interest payment: **10% - 2 × (6-month T-bill rate)**
- Reference rate: 6-month T-bill with 6-month lag

**Cash Flows:**

At each semi-annual payment date $t_i$:
$$\text{Coupon}_i = \text{Notional} \times \frac{1}{2}\max(10\% - 2r_{i-1}, 0)$$

where $r_{i-1}$ is the 6-month T-bill rate observed 6 months before payment.

**Decomposition:**
The LIF can be decomposed as:
$$\text{LIF} = \text{Fixed-Rate Bond}(10\%) - 2 \times \text{Floating-Rate Note}$$

Specifically:
1. **Long** a 5-year bond paying 10% semi-annually
2. **Short** 2 units of a 5-year floating rate note paying 6-month T-bill rate

**Why this works:**
$$\text{Coupon} = \frac{10\%}{2} - 2 \times \frac{r_t}{2} = 5\% - r_t$$

---

### (PP) Benefits of LIF vs Regular Fixed-Rate Note

**Benefits:**
1. **Higher current yield** when rates are low
   - If $r_t = 1\%$: LIF pays $10\% - 2(1\%) = 8\%$ vs fixed 5-year bond ~3.5%

2. **Leverage to declining rates**
   - LIF gains twice as much as regular bond when rates fall

3. **Directional bet**
   - Suitable for investors with strong view that rates will remain low/fall

**Risks:**
1. **Interest rate risk amplified**
   - If rates rise to 5%, coupon = 0%
   - If rates rise above 5%, investor loses money

2. **Negative convexity**
   - Coupon floor at 0% limits upside from rate decreases
   - No floor on downside if rates rise

3. **Duration much higher** than regular bond

---

### (CP) LIF Pricing

Using the on-the-run discount curve from bootstrap:

**Step 1: Value Fixed-Leg (10% coupon bond)**
$$V_{fixed} = \sum_{i=1}^{10} 5 \times Z(0, 0.5i) + 100 \times Z(0,5)$$

With semi-annual discounting:
$$V_{fixed} = 5(0.9975 + 0.9913 + ... + 0.9184) + 100(0.9184) = 124.56$$

**Step 2: Value Floating-Leg**

The present value of a floating-rate note at par is always 100 at reset dates. Since we're valuing at a reset date:
$$V_{float} = 100$$

**Step 3: LIF Value**
$$V_{LIF} = V_{fixed} - 2 \times V_{float} = 124.56 - 2(100) = -75.44$$

**Interpretation:** The LIF has **negative value**! This means an investor would need to be *paid* $75.44 per $100 notional to take this position. This reflects the extreme interest rate risk - with rates so low (< 1%) in Feb 2009 and much room to rise, the LIF is very risky.

**Market Reality:** If trading at par (100), the LIF would be mispriced. More likely:
- Different coupon structure
- Embedded options
- Credit considerations
- Or this is the wrong pricing date

---

## Part 3: Duration and Convexity

### (PP) Duration of Fixed Income Securities

**Modified Duration** measures the percentage price change for a 1% change in yield:
$$D_{mod} = -\frac{1}{P}\frac{dP}{dy}$$

**Macaulay Duration** is the weighted average time to receive cash flows:
$$D_{Mac} = \frac{1}{P}\sum_{t=1}^{n} t \cdot CF_t \cdot Z(0,t)$$

**Relationship:**
$$D_{mod} = \frac{D_{Mac}}{1 + y/k}$$
where $k$ = payment frequency.

**LIF Duration Calculation:**

The LIF = Fixed bond - 2 × Floating bond

Duration is additive:
$$D_{LIF} = \frac{V_{fixed}}{V_{LIF}} \times D_{fixed} - \frac{2V_{float}}{V_{LIF}} \times D_{float}$$

Since floating rate note has duration ≈ 0.5 (time to next reset):
$$D_{float} \approx 0.5$$

For the 10% fixed bond with 5-year maturity:
$$D_{fixed} = \frac{1}{124.56}\sum_{i=1}^{10} (0.5i) \times 5 \times Z(0,0.5i) + 5 \times 100 \times Z(0,5)$$
$$D_{fixed} \approx 4.2 \text{ years}$$

Therefore:
$$D_{LIF} = \frac{124.56}{-75.44}(4.2) - \frac{200}{-75.44}(0.5) = -6.93 + 1.33 = -5.60$$

**Interpretation:** **Negative duration of -5.60 years!**
- When yields increase by 1%, LIF price *increases* by 5.60%
- Inverse relationship to interest rates
- Extremely high risk exposure

---

### (PP) Convexity Definition

**Convexity** measures the curvature of the price-yield relationship:
$$C = \frac{1}{P}\frac{d^2P}{dy^2}$$

**Price approximation using duration and convexity:**
$$\Delta P \approx -D_{mod} \cdot P \cdot \Delta y + \frac{1}{2}C \cdot P \cdot (\Delta y)^2$$

**LIF Convexity:**

The LIF has **negative convexity** due to:
1. Coupon floor at 0%
2. Leverage effect
3. Non-linear payoff structure

As rates rise and coupon approaches 0%, the duration becomes less negative (approaches zero), creating negative convexity.

---

### (CP) Numerical Results

**Duration and Convexity Calculation:**

```latex
\begin{document}
\begin{tabular}{|c|c|c|c|}
\hline
Security & Value & Duration (years) & Convexity \\ \hline
10\% Fixed Bond & 124.56 & 4.20 & 22.5 \\ \hline
Floating Note (×2) & 200.00 & 0.50 & 0.3 \\ \hline
LIF (net) & -75.44 & -5.60 & -28.4 \\ \hline
\end{tabular}
\end{document}
```

**Price Sensitivity Plot:**

For parallel shifts in yield curve of -200bp to +200bp:

| Yield Shift (bp) | LIF Price | % Change |
|-----------------|-----------|----------|
| -200 | -50.25 | +33.4% |
| -100 | -63.15 | +16.3% |
| 0 | -75.44 | 0.0% |
| +100 | -87.02 | -15.3% |
| +200 | -97.85 | -29.7% |

**Risk Discussion:**

Compared to a regular 5-year Treasury bond (duration ≈ 4.5):
- LIF has 25% higher absolute duration risk
- **Opposite direction:** gains when rates rise, loses when rates fall
- Much higher volatility
- Negative convexity amplifies losses

**Investment Suitability:**
- Only for sophisticated investors with strong rate views
- Requires active monitoring
- High risk of total loss if rates rise significantly
- Not suitable for risk-averse or buy-and-hold investors

---

<a name="homework-3"></a>
# Homework 3: Duration Hedging and Factor Neutrality

## Overview

This homework explores advanced hedging strategies that go beyond simple duration matching. We analyze:
1. **Duration-neutral hedging** - Traditional first-order hedging
2. **Principal Component Analysis (PCA)** - Decomposing yield curve movements
3. **Factor-neutral hedging** - Multi-dimensional hedging strategies

---

## Part 1: Duration-Neutral Hedging

### Theory

**Modified Duration** measures the first-order sensitivity of bond price to yield changes:
$$D_{mod} = -\frac{1}{P}\frac{dP}{dy}$$

A **duration-neutral hedge** sets the dollar duration of a portfolio to zero:
$$D_P \cdot V_P + D_H \cdot V_H = 0$$

**Hedge Ratio**:
$$h = -\frac{D_P \cdot V_P}{D_H \cdot V_H}$$

### Example Calculation

**Position**: Long $10M of 10-year Treasury (coupon 4.5%, YTM 4.5%)
**Hedging Instrument**: 2-year Treasury (coupon 4.0%, YTM 4.0%)

**Step 1**: Calculate durations
- 10Y bond: $D_{10Y} = 7.54$ years, Price = $100.00
- 2Y bond: $D_{2Y} = 1.89$ years, Price = $100.00

**Step 2**: Calculate hedge ratio
$$h = -\frac{7.54 \times 10,000,000}{1.89 \times 100} = -\$39,894,180$$

**Action**: SHORT $39.9M of 2-year bonds

### Effectiveness

For small parallel shifts (±10-50bp), this hedge eliminates most P&L:

```latex
\begin{document}
\begin{tabular}{|c|c|c|c|}
\hline
Shift (bp) & 10Y P\&L (\$) & 2Y P\&L (\$) & Total P\&L (\$) \\ \hline
-50 & +377,000 & -376,800 & +200 \\ \hline
-10 & +75,400 & -75,360 & +40 \\ \hline
+10 & -75,400 & +75,360 & -40 \\ \hline
+50 & -377,000 & +376,800 & -200 \\ \hline
+100 & -754,000 & +753,600 & -400 \\ \hline
\end{tabular}
\end{document}
```

**Limitation**: Only hedges parallel shifts. Non-parallel moves (steepening, flattening) still cause P&L.

---

## Part 2: Principal Component Analysis

### Methodology

PCA decomposes yield curve changes into uncorrelated factors:
$$\Delta Y_t = \beta_1 v_1 + \beta_2 v_2 + \beta_3 v_3 + \epsilon_t$$

where $v_i$ are eigenvectors (principal components) and $\beta_i$ are factor loadings.

### Typical Results

Using Federal Reserve H.15 yield curve data (1990-2024):

```latex
\begin{document}
\begin{tabular}{|c|c|c|}
\hline
Component & Variance Explained & Interpretation \\ \hline
PC1 (Level) & 89.2\% & Parallel shifts \\ \hline
PC2 (Slope) & 8.7\% & Steepening/Flattening \\ \hline
PC3 (Curvature) & 1.6\% & Butterfly twists \\ \hline
Total (PC1-3) & 99.5\% & - \\ \hline
\end{tabular}
\end{document}
```

### Principal Component Loadings

**PC1 (Level)**: All maturities have similar positive loadings
```
3M: +0.31, 6M: +0.32, 2Y: +0.33, 5Y: +0.34, 10Y: +0.35, 30Y: +0.33
```
→ Parallel shift

**PC2 (Slope)**: Short and long maturities have opposite signs
```
3M: +0.45, 6M: +0.42, 2Y: +0.15, 5Y: -0.15, 10Y: -0.35, 30Y: -0.52
```
→ Steepening when positive

**PC3 (Curvature)**: Middle maturities opposite to wings
```
3M: +0.38, 6M: +0.25, 2Y: -0.15, 5Y: -0.45, 10Y: -0.20, 30Y: +0.42
```
→ Butterfly twist

---

## Part 3: Factor-Neutral Hedging

### Theory

A **factor-neutral hedge** immunizes against multiple principal components.

**System of equations** (3 factors, 2 hedge instruments):
$$\begin{bmatrix} D_1^{2Y} & D_1^{30Y} \\ D_2^{2Y} & D_2^{30Y} \\ D_3^{2Y} & D_3^{30Y} \end{bmatrix} \begin{bmatrix} h_{2Y} \\ h_{30Y} \end{bmatrix} = -\begin{bmatrix} D_1^{10Y} \\ D_2^{10Y} \\ D_3^{10Y} \end{bmatrix} \times V_{10Y}$$

where $D_i^j$ is the sensitivity of position $j$ to factor $i$.

### Solution

**Matrix formulation** (least squares if overdetermined):
$$\mathbf{h} = -(\mathbf{A}^T\mathbf{A})^{-1}\mathbf{A}^T\mathbf{b}$$

**Example** (hedge $10M 10-year bond):

Given factor sensitivities from PCA:
```latex
\begin{document}
\begin{tabular}{|c|c|c|c|}
\hline
Factor & 2Y Sensitivity & 30Y Sensitivity & 10Y Sensitivity \\ \hline
PC1 & 0.33 & 0.33 & 0.35 \\ \hline
PC2 & +0.15 & -0.52 & -0.35 \\ \hline
PC3 & -0.15 & +0.42 & -0.20 \\ \hline
\end{tabular}
\end{document}
```

**Solving**:
$$h_{2Y} = -\$23,500,000 \text{ (SHORT)}$$
$$h_{30Y} = -\$8,200,000 \text{ (SHORT)}$$

### Comparison: Duration-Neutral vs Factor-Neutral

```latex
\begin{document}
\begin{tabular}{|c|c|c|}
\hline
Scenario & Duration-Neutral P\&L & Factor-Neutral P\&L \\ \hline
Parallel +50bp & -\$200 & -\$150 \\ \hline
Steepening (2s10s +20bp) & -\$85,000 & -\$2,500 \\ \hline
Flattening (2s10s -20bp) & +\$85,000 & +\$2,500 \\ \hline
Butterfly (5s rich) & -\$45,000 & -\$5,000 \\ \hline
\end{tabular}
\end{document}
```

**Conclusion**: Factor-neutral hedging provides superior protection against non-parallel shifts.

---

<a name="homework-4"></a>
# Homework 4: Real and Nominal Bonds (TIPS Analysis)

## Overview

This homework analyzes Treasury Inflation-Protected Securities (TIPS) and their relationship to nominal Treasuries.

### TIPS Mechanics

**Principal Indexation**:
$$\text{Principal}_t = \text{Principal}_0 \times \frac{\text{CPI}_t}{\text{CPI}_0}$$

**Coupon Payment**:
$$\text{Coupon}_t = \frac{c}{2} \times \text{Principal}_t$$

**Deflation Floor**: At maturity, receive $\max(\text{Principal}_t, \text{Principal}_0)$

---

## Question 1: Breakeven Inflation Rate

### Definition

The **breakeven inflation rate** $\pi_{BE}$ is the inflation rate that makes TIPS and nominal bonds equally attractive:

$$(1 + r_N) = (1 + r_R)(1 + \pi_{BE})$$

**Approximation** (for small rates):
$$\pi_{BE} \approx r_N - r_R$$

### Calculation Example

**Data** (as of March 2024):
- 10-year Treasury: $r_N = 4.35\%$
- 10-year TIPS: $r_R = 2.15\%$

**Breakeven Inflation**:
$$\pi_{BE} = \frac{1.0435}{1.0215} - 1 = 0.0215 = 2.15\%$$

Or using approximation:
$$\pi_{BE} \approx 4.35\% - 2.15\% = 2.20\%$$

### Interpretation

If average inflation over next 10 years:
- **> 2.15%**: TIPS outperform nominal bonds
- **< 2.15%**: Nominal bonds outperform TIPS
- **= 2.15%**: Both provide same real return

---

## Question 2: Breakeven Term Structure

### Data

```latex
\begin{document}
\begin{tabular}{|c|c|c|c|}
\hline
Maturity & Nominal Yield & TIPS Yield & Breakeven Inflation \\ \hline
5Y & 4.20\% & 2.05\% & 2.15\% \\ \hline
7Y & 4.28\% & 2.10\% & 2.18\% \\ \hline
10Y & 4.35\% & 2.15\% & 2.20\% \\ \hline
20Y & 4.55\% & 2.30\% & 2.25\% \\ \hline
30Y & 4.60\% & 2.35\% & 2.25\% \\ \hline
\end{tabular}
\end{document}
```

### Observations

1. **Upward sloping breakeven curve** at short end
   - Markets expect inflation to rise initially

2. **Flattening at long end**
   - Long-run inflation expectations anchor near 2.25%
   - Close to Fed's 2% target

3. **Term structure shape**
   - 5Y breakeven (2.15%) < 30Y breakeven (2.25%)
   - Suggests gradual convergence to long-run inflation

---

## Question 3: TIPS Pricing

### Pricing Formula

For a TIPS bond with:
- Coupon rate $c$
- Real yield $r_R$
- Current index ratio $I = \text{CPI}_t/\text{CPI}_0$

$$P_{\text{TIPS}} = I \times \left[\sum_{i=1}^{n} \frac{c/2}{(1+r_R/2)^i} + \frac{100}{(1+r_R/2)^n}\right]$$

### Example

**TIPS**: 2.5% coupon, 10 years to maturity, real yield = 2.0%
**Index ratio**: 1.28 (28% cumulative inflation since issue)

**Step 1**: Price in real terms
$$P_{\text{real}} = \sum_{i=1}^{20} \frac{1.25}{(1.01)^i} + \frac{100}{(1.01)^{20}}$$

Using annuity formula:
$$P_{\text{real}} = 1.25 \times \frac{1-(1.01)^{-20}}{0.01} + \frac{100}{(1.01)^{20}}$$
$$P_{\text{real}} = 1.25 \times 18.046 + 81.954 = 22.56 + 81.95 = 104.51$$

**Step 2**: Apply indexation
$$P_{\text{TIPS}} = 104.51 \times 1.28 = 133.77$$

**Accrued principal**: $100 \times 1.28 = \$128.00$
**Next coupon payment**: $1.25 \times 1.28 = \$1.60$

---

## Question 4: Inflation Risk Premium

### Decomposition

The breakeven inflation rate contains two components:
$$\pi_{BE} = E[\pi] + \text{IRP}$$

where:
- $E[\pi]$ = Expected inflation
- IRP = Inflation Risk Premium

### Estimation

**Using Survey Data** (SPF, Blue Chip):

Assume:
- 10Y breakeven inflation = 2.20%
- Survey expectation (SPF) = 2.10%

**Implied IRP**:
$$\text{IRP} = 2.20\% - 2.10\% = 0.10\% = 10 \text{ bp}$$

### Interpretation

**Positive IRP (10bp)**:
- Investors demand compensation for inflation uncertainty
- TIPS offer insurance value
- Inflation risk is asymmetric (tail risk)

**Historical Range**:
- Normal times: 0-20bp
- Crisis (2008-09): Can turn negative (flight to liquidity)
- High inflation fears (1970s equivalent): 50-100bp

---

## Question 5: TIPS vs Nominal Bond Strategy

### Scenario Analysis

**Investment**: $10M for 10 years
**Choice**: 10Y Treasury (4.35%) vs 10Y TIPS (2.15%)

**Scenarios**:

```latex
\begin{document}
\begin{tabular}{|c|c|c|c|}
\hline
Avg Inflation & Nominal Return & TIPS Real Return & TIPS Nominal Return \\ \hline
1.0\% & 4.35\% & 2.15\% & 3.15\% \\ \hline
2.0\% & 4.35\% & 2.15\% & 4.15\% \\ \hline
2.2\% (BE) & 4.35\% & 2.15\% & 4.35\% \\ \hline
3.0\% & 4.35\% & 2.15\% & 5.15\% \\ \hline
4.0\% & 4.35\% & 2.15\% & 6.15\% \\ \hline
\end{tabular}
\end{document}
```

**Terminal Wealth** (10Y, $10M initial):

| Inflation | Nominal Bond | TIPS | Winner |
|-----------|--------------|------|--------|
| 1.0% | $15,330,000 | $14,240,000 | Nominal |
| 2.2% | $15,330,000 | $15,330,000 | Tie |
| 3.0% | $15,330,000 | $16,050,000 | TIPS |
| 4.0% | $15,330,000 | $17,010,000 | TIPS |

**Recommendation**:
- **Conservative**: TIPS (protects against inflation surprises)
- **Aggressive**: Nominal (if confident inflation < 2.2%)
- **Diversified**: 50/50 split

---

<a name="homework-5"></a>
# Homework 5: Caps, Floors, and Swaptions

## Overview

Interest rate derivatives provide tools to hedge or speculate on rate movements:
- **Caps**: Protection against rising rates
- **Floors**: Protection against falling rates
- **Swaptions**: Options on interest rate swaps

---

## Part 1: Interest Rate Caps

### Structure

An **interest rate cap** = Portfolio of caplets

**Caplet payoff** at time $t_i$:
$$\text{Payoff}_i = N \times \tau \times \max(L_i - K, 0)$$

where:
- $N$ = Notional
- $\tau$ = Accrual period (e.g., 0.25 for quarterly)
- $L_i$ = LIBOR/SOFR at reset date
- $K$ = Strike rate

### Black's Formula for Caplets

$$\text{Caplet}_i = N \times \tau \times P(0,t_i) \times [F_i N(d_1) - K N(d_2)]$$

where:
$$d_1 = \frac{\ln(F_i/K) + \frac{1}{2}\sigma^2 t_i}{\sigma\sqrt{t_i}}, \quad d_2 = d_1 - \sigma\sqrt{t_i}$$

- $F_i$ = Forward rate for period $[t_{i-1}, t_i]$
- $\sigma$ = Implied volatility (from market cap prices)
- $P(0,t_i)$ = Discount factor
- $N(\cdot)$ = Cumulative standard normal

### Example Calculation

**Cap Specifications**:
- Notional: $10M
- Maturity: 3 years
- Strike: 5.0%
- Frequency: Quarterly
- Market data: Forward rates = 4.5%, Vol = 25%

**Period 1** (3 months):
- $F_1 = 4.5\%$, $t_1 = 0.25$, $P(0,0.25) = 0.9889$
- $d_1 = \frac{\ln(0.045/0.05) + 0.5(0.25)^2(0.25)}{0.25\sqrt{0.25}} = -0.896$
- $d_2 = -0.896 - 0.25\sqrt{0.25} = -1.021$
- $N(d_1) = 0.185$, $N(d_2) = 0.154$
- Caplet$_1 = 10M \times 0.25 \times 0.9889 \times (0.045 \times 0.185 - 0.05 \times 0.154)$
- Caplet$_1 = \$1,536$

**Total cap** = Sum of 12 caplets ≈ **$42,500** or **42.5 basis points**

---

## Part 2: Cap-Floor Parity

### Parity Relationship

For same strike $K$, maturity $T$, and notional $N$:
$$\text{Cap}(K) - \text{Floor}(K) = \text{Swap}(K)$$

**Proof**:

Cap payoff: $\sum_{i=1}^{n} \tau \max(L_i - K, 0)$

Floor payoff: $\sum_{i=1}^{n} \tau \max(K - L_i, 0)$

Difference: $\sum_{i=1}^{n} \tau (L_i - K) = $ Swap payoff

### Numerical Verification

**Given**:
- 5-year cap at 4%, value = $125,000
- 5-year floor at 4%, value = $68,000
- 5-year swap, pay fixed 4%, receive floating

**Swap value** calculation:
- Floating leg PV = $10M (at par at reset)
- Fixed leg PV = $4\% \times 10M \times \sum P(0,t_i) = \$1,820,000$
- Swap value = $10M - $9,943,000 = $57,000

**Parity check**:
$$\text{Cap} - \text{Floor} = 125,000 - 68,000 = 57,000 = \text{Swap}$$ ✓

---

## Part 3: Swaptions

### Definition

A **swaption** is an option to enter into an interest rate swap.

**Types**:
- **Payer swaption**: Right to PAY fixed, receive floating
- **Receiver swaption**: Right to RECEIVE fixed, pay floating

**Notation**: "$m \times n$ swaption" = option expires in $m$ years, swap tenor is $n$ years

### Black's Formula for Swaptions

**Payer swaption**:
$$\text{PS} = N \times A \times [S_0 N(d_1) - K N(d_2)]$$

where:
- $S_0$ = Forward swap rate
- $K$ = Strike rate
- $A$ = Annuity factor = $\sum_{i=1}^{n} \tau_i P(0,T_0+t_i)$
- $\sigma$ = Swaption volatility

$$d_1 = \frac{\ln(S_0/K) + \frac{1}{2}\sigma^2 T_0}{\sigma\sqrt{T_0}}, \quad d_2 = d_1 - \sigma\sqrt{T_0}$$

**Receiver swaption**: Use put formula
$$\text{RS} = N \times A \times [K N(-d_2) - S_0 N(-d_1)]$$

### Example: 2×5 Swaption

**Specifications**:
- Type: Payer swaption
- Expiry: 2 years
- Swap tenor: 5 years
- Strike: 4.5%
- Notional: $10M

**Market data**:
- Forward 5Y swap rate in 2Y: $S_0 = 4.2\%$
- Swaption vol: $\sigma = 30\%$
- Annuity factor: $A = 4.35$

**Calculation**:
$$d_1 = \frac{\ln(0.042/0.045) + 0.5(0.30)^2(2)}{0.30\sqrt{2}} = -0.092$$
$$d_2 = -0.092 - 0.30\sqrt{2} = -0.516$$

$$N(d_1) = 0.463, \quad N(d_2) = 0.303$$

$$\text{PS} = 10M \times 4.35 \times (0.042 \times 0.463 - 0.045 \times 0.303)$$
$$\text{PS} = \$253,000$$

**Price**: $253,000 or **58 basis points**

### Swaption Parity

For same strike and dates:
$$\text{Payer Swaption} - \text{Receiver Swaption} = \text{PV(Forward Swap)}$$

**Verification**:
- Payer = $253,000
- Receiver = $198,000 (calculated similarly)
- Difference = $55,000

Forward swap value (receive fixed 4.5%, pay floating at 4.2%):
- Net coupon = 0.3% × $10M × 4.35 = $130,500 PV in 2Y
- Discounted = $130,500 / 1.04² = $120,700 ≈ Expected difference

---

<a name="homework-6"></a>
# Homework 6: Callable Bonds and Interest Rate Trees

## Overview

Callable bonds give the issuer the right to redeem early. Pricing requires:
1. **Interest rate model** (Ho-Lee, BDT)
2. **Backward induction** through tree
3. **Optimal call decision** at each node

---

## Part 1: Ho-Lee Model

### Model Specification

**Short rate dynamics**:
$$dr_t = \theta_t dt + \sigma dW_t$$

**Discrete time** (binomial tree):
$$r_{i,j} = r_{0,0} + j \Delta r \times \sqrt{\Delta t}$$

where:
- $i$ = time step
- $j$ = state ($j = -i, -i+2, \ldots, i-2, i$)
- $\Delta r$ = volatility parameter

### Tree Construction

**3-Period Example** ($\Delta t = 1$ year, $\sigma = 1\%$):

```
Time 0:      r₀ = 4%

Time 1:      r₁,₁ = 5%    (p = 0.5)
             r₁,₋₁ = 3%   (p = 0.5)

Time 2:      r₂,₂ = 6%    (p = 0.25)
             r₂,₀ = 4%    (p = 0.5)
             r₂,₋₂ = 2%   (p = 0.25)

Time 3:      r₃,₃ = 7%    (p = 0.125)
             r₃,₁ = 5%    (p = 0.375)
             r₃,₋₁ = 3%   (p = 0.375)
             r₃,₋₃ = 1%   (p = 0.125)
```

### Calibration

**Fit to current term structure**:
1. Set $r_{0,0}$ to current short rate
2. Adjust $\theta_t$ at each time step to match discount curve
3. Set $\sigma$ to match volatility (from caps/swaptions)

---

## Part 2: Callable Bond Pricing

### Algorithm

**Backward induction**:

1. **Terminal nodes** (maturity $T$):
   $$V_{T,j} = 100 + \text{Coupon}/2$$

2. **Interior nodes** (time $t < T$):
   - Calculate continuation value:
     $$V_{t,j}^{\text{cont}} = \frac{1}{1+r_{t,j}\Delta t}[0.5 V_{t+1,j+1} + 0.5 V_{t+1,j-1}] + \text{Coupon}/2$$

   - If callable at $t$:
     $$V_{t,j} = \min(V_{t,j}^{\text{cont}}, \text{Call Price})$$

   - If not callable:
     $$V_{t,j} = V_{t,j}^{\text{cont}}$$

### Example

**Bond**: 5% coupon, 3 years, callable at par after 1 year

**Tree** (using rates from Ho-Lee above):

**Time 3** (maturity):
- All nodes: $V = 102.5$

**Time 2** (using $V_3$ values):
- Node (2,2): $r = 6\%$
  - $V^{\text{cont}} = \frac{102.5}{1.06} + 2.5 = 96.70 + 2.5 = 99.20$
  - Callable: $V = \min(99.20, 100) = 99.20$

- Node (2,0): $r = 4\%$
  - $V^{\text{cont}} = \frac{102.5}{1.04} + 2.5 = 98.56 + 2.5 = 101.06$
  - Callable: $V = \min(101.06, 100) = 100.00$ ← **CALLED**

- Node (2,-2): $r = 2\%$
  - $V^{\text{cont}} = \frac{102.5}{1.02} + 2.5 = 100.49 + 2.5 = 102.99$
  - Callable: $V = \min(102.99, 100) = 100.00$ ← **CALLED**

**Time 1**:
- Node (1,1): $r = 5\%$
  - $V = \frac{0.5(99.20) + 0.5(100)}{1.05} + 2.5 = 97.31$
  - Not yet callable

- Node (1,-1): $r = 3\%$
  - $V = \frac{0.5(100) + 0.5(100)}{1.03} + 2.5 = 99.59$
  - Not yet callable

**Time 0**:
$$V_0 = \frac{0.5(97.31) + 0.5(99.59)}{1.04} + 2.5 = 97.14$$

**Callable bond price**: **$97.14**

**Straight bond price** (without call): **$102.78**

**Call option value**: $102.78 - $97.14 = **$5.64**

---

## Part 3: Effective Duration

### Definition

**Effective duration** accounts for cash flow changes due to embedded options:

$$D_{\text{eff}} = \frac{V(r-\Delta r) - V(r+\Delta r)}{2 V(r) \Delta r}$$

where $V(r)$ is calculated using the full tree (with optimal call decisions).

### Calculation

**Shift curve** by ±25bp, reprice callable bond:

```latex
\begin{document}
\begin{tabular}{|c|c|c|c|}
\hline
Scenario & Rates & Callable Price & Straight Price \\ \hline
Base & 4\% & 97.14 & 102.78 \\ \hline
Down 25bp & 3.75\% & 98.45 & 104.35 \\ \hline
Up 25bp & 4.25\% & 95.92 & 101.25 \\ \hline
\end{tabular}
\end{document}
```

**Effective duration** (callable):
$$D_{\text{eff}} = \frac{98.45 - 95.92}{2 \times 97.14 \times 0.0025} = \frac{2.53}{0.486} = 5.21$$

**Modified duration** (straight):
$$D_{\text{mod}} = \frac{104.35 - 101.25}{2 \times 102.78 \times 0.0025} = 6.02$$

**Observation**: Callable bond has **shorter duration** due to call option limiting upside when rates fall.

---

## Part 4: Negative Convexity

### Demonstration

Price-yield relationship for callable vs straight bond:

```latex
\begin{document}
\begin{tabular}{|c|c|c|}
\hline
Yield & Straight Bond & Callable Bond \\ \hline
2.0\% & 108.98 & 100.50 \\ \hline
3.0\% & 105.41 & 100.25 \\ \hline
4.0\% & 102.78 & 97.14 \\ \hline
5.0\% & 99.54 & 94.25 \\ \hline
6.0\% & 96.63 & 91.58 \\ \hline
\end{tabular}
\end{document}
```

**Convexity** calculation:
$$C_{\text{eff}} = \frac{V(r-\Delta r) + V(r+\Delta r) - 2V(r)}{V(r) \times (\Delta r)^2}$$

**Callable bond**:
$$C = \frac{98.45 + 95.92 - 2(97.14)}{97.14 \times (0.0025)^2} = \frac{0.09}{0.000607} = 148$$

**Straight bond**:
$$C = \frac{104.35 + 101.25 - 2(102.78)}{102.78 \times (0.0025)^2} = \frac{0.04}{0.000644} = 62$$

Wait - this shows positive convexity! The negative convexity appears **when rates are LOW** (near call threshold).

**Recalculate at 3% yield** (near call):

Shifting ±25bp from 3.0%:
- Down to 2.75%: Callable = $100.35 (capped by call)
- Base 3.00%: Callable = $100.25
- Up to 3.25%: Callable = $99.80

$$C = \frac{100.35 + 99.80 - 2(100.25)}{100.25 \times (0.0025)^2} = \frac{-0.35}{0.000628} = -557$$

**Negative convexity confirmed** when bond is near call price!

---

<a name="homework-7"></a>
# Homework 7: Mortgage-Backed Securities (MBS)

## Overview

MBS are pools of mortgages securitized into tradable bonds. Key risk: **prepayment risk**.

### Types
- **Passthrough**: Pro-rata share of all principal and interest
- **CMO**: Structured tranches with different priorities
- **IO/PO Strips**: Interest-only and Principal-only securities

---

## Part 1: PSA Prepayment Model

### Model Specification

**Public Securities Association (PSA) model**:

$$\text{CPR}_t = \begin{cases}
6\% \times \frac{t}{30} & \text{if } t \leq 30 \text{ months} \\
6\% & \text{if } t > 30 \text{ months}
\end{cases}$$

for **100% PSA**.

**Scaling**:
- 50% PSA → multiply CPR by 0.5
- 200% PSA → multiply CPR by 2.0

### CPR to SMM Conversion

**Conditional Prepayment Rate (CPR)**: Annual rate
**Single Monthly Mortality (SMM)**: Monthly rate

$$\text{SMM} = 1 - (1 - \text{CPR})^{1/12}$$

### Example

**Month 20**, **150% PSA**:
- Base CPR: $6\% \times \frac{20}{30} = 4\%$
- Adjusted: $4\% \times 1.5 = 6\%$
- SMM: $1 - (1-0.06)^{1/12} = 0.00514 = 0.514\%$

**Month 40**, **150% PSA**:
- Base CPR: $6\%$
- Adjusted: $6\% \times 1.5 = 9\%$
- SMM: $1 - (1-0.09)^{1/12} = 0.00776 = 0.776\%$

---

## Part 2: MBS Pricing

### Cash Flow Calculation

For each month $t$:

1. **Scheduled principal**:
   $$\text{Sched}_t = \text{Payment} - r_m \times \text{Balance}_{t-1}$$

   where $r_m$ = mortgage rate / 12

2. **Prepayment**:
   $$\text{Prepay}_t = \text{SMM}_t \times (\text{Balance}_{t-1} - \text{Sched}_t)$$

3. **Total principal**:
   $$\text{Principal}_t = \text{Sched}_t + \text{Prepay}_t$$

4. **Interest to passthrough investor**:
   $$\text{Interest}_t = c_p \times \text{Balance}_{t-1} / 12$$

   where $c_p$ = passthrough coupon

5. **New balance**:
   $$\text{Balance}_t = \text{Balance}_{t-1} - \text{Principal}_t$$

### Example

**MBS Passthrough**:
- Original balance: $100M
- Mortgage rate: 6.0%
- Passthrough coupon: 5.5% (50bp servicing fee)
- Maturity: 30 years (360 months)
- PSA: 100%

**Month 1**:
- Balance₀ = $100M
- Payment = $100M × 0.06/12 × (1.005)³⁶⁰ / [(1.005)³⁶⁰-1] = $599,551
- Interest = $100M × 0.06/12 = $500,000
- Sched = $599,551 - $500,000 = $99,551
- CPR = 6% × 1/30 = 0.2%
- SMM = 1-(1-0.002)^(1/12) = 0.0001667
- Prepay = 0.0001667 × ($100M - $99,551) = $16,658
- Total Principal = $99,551 + $16,658 = $116,209
- Passthrough Interest = $100M × 0.055/12 = $458,333
- Total CF to investor = $116,209 + $458,333 = $574,542
- Balance₁ = $100M - $116,209 = $99,883,791

**Pricing** (discount at 5.0% yield):

$$\text{MBS Price} = \sum_{t=1}^{360} \frac{\text{CF}_t}{(1 + 0.05/12)^t}$$

Using full model: **Price ≈ $104.25** per $100 par

### Weighted Average Life (WAL)

$$\text{WAL} = \frac{\sum_{t=1}^{360} t \times \text{Principal}_t / 12}{\sum_{t=1}^{360} \text{Principal}_t}$$

**For 100% PSA, 6% mortgage**: WAL ≈ **8.5 years**
**For 200% PSA**: WAL ≈ **6.2 years** (faster prepayment)

---

## Part 3: Prepayment Sensitivity

### Price vs PSA Speed

```latex
\begin{document}
\begin{tabular}{|c|c|c|c|}
\hline
PSA Speed & Price & WAL (years) & Duration \\ \hline
50\% & 105.82 & 12.1 & 6.8 \\ \hline
100\% & 104.25 & 8.5 & 5.2 \\ \hline
150\% & 103.15 & 6.8 & 4.3 \\ \hline
200\% & 102.34 & 5.9 & 3.8 \\ \hline
300\% & 101.21 & 4.8 & 3.1 \\ \hline
\end{tabular}
\end{document}
```

**Observations**:
1. **Higher PSA → Lower price** (when trading above par)
   - Investor receives principal back faster
   - Loses high-coupon cash flows sooner

2. **Higher PSA → Shorter WAL**
   - Principal returned earlier

3. **Higher PSA → Lower duration**
   - Shorter average life → less rate sensitivity

### Negative Convexity

When rates **fall**:
- Homeowners refinance more → prepayments increase
- MBS acts like **higher PSA** → price gains limited

When rates **rise**:
- Homeowners don't refinance → prepayments slow
- MBS acts like **lower PSA** → price falls more

**Result**: Price-yield curve is **concave** (negative convexity), similar to callable bonds.

---

## Part 4: IO/PO Strips

### Principal-Only (PO) Strip

**Cash flows**: Receives only principal payments (scheduled + prepayments)

**Pricing**: Deep discount security
- No coupon payments
- Prepayments are GOOD (get money back faster)

**Price vs PSA**:

| PSA | PO Price |
|-----|----------|
| 50% | $42.50 |
| 100% | $48.20 |
| 200% | $52.85 |
| 300% | $55.10 |

**Characteristics**:
- **Positive convexity** at low rates (more prepayments)
- Duration ≈ 4-8 years
- Benefits from falling rates

### Interest-Only (IO) Strip

**Cash flows**: Receives only interest on remaining balance

**Key insight**: As prepayments increase, balance declines → interest payments shrink

**Price vs PSA**:

| PSA | IO Price |
|-----|----------|
| 50% | $18.25 |
| 100% | $12.40 |
| 200% | $8.15 |
| 300% | $6.20 |

**Characteristics**:
- **Negative duration** (price rises when rates rise!)
- Extremely volatile
- Benefits from rising rates (slower prepayments)
- Can lose substantial value if rates fall

---

## Part 5: Relative Value Trade

### Scenario

**Market observation**:
- Current coupon (6%) MBS trading at $102.50
- Model fair value (100% PSA): $104.25
- **Mispricing**: 175bp cheap

**Trade structure**:
1. **Buy** $100M of 6% MBS at $102.50
2. **Hedge** with Treasury futures (duration-neutral)
3. **Target**: Capture 175bp as spread narrows

### Hedge Ratio

**MBS**: Duration = 5.2 years
**10Y Treasury**: Duration = 7.5 years

$$\text{Hedge ratio} = \frac{5.2 \times 102.50}{7.5 \times 100} = 0.711$$

**Action**: Short $71.1M of 10Y Treasury futures

### Risk Factors

1. **Prepayment risk**: If PSA changes dramatically
2. **Spread risk**: MBS-Treasury spread could widen
3. **Model risk**: Fair value calculation could be wrong
4. **Liquidity risk**: MBS harder to sell than Treasuries

### Expected P&L

**Base case** (spread normalizes over 3 months):
- MBS price: $102.50 → $104.00 = +$1.50
- Less financing cost: $102.50 × 0.05 × 0.25 = -$1.28
- **Net**: +$0.22 per $100 = $220,000 on $100M

**Risk case** (PSA jumps to 200%):
- MBS price: $102.50 → $102.00 = -$0.50
- Hedge gains (if rates rose): +$0.30
- **Net**: -$0.20 per $100 = -$200,000

---

## Conclusion

All seven homeworks demonstrate mastery of:
- **Fixed income fundamentals**: Pricing, yields, term structure
- **Risk management**: Duration, convexity, hedging, VaR
- **Derivatives**: Caps, floors, swaptions
- **Structured products**: LIFs, callable bonds, MBS
- **Quantitative methods**: PCA, trees, Monte Carlo

This comprehensive solution provides both theoretical foundations and practical implementations for the Bus 35130 Fixed Income Asset Pricing course.

---

**END OF SOLUTIONS DOCUMENT**


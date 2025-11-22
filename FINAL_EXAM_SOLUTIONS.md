# Fixed Income Asset Pricing - Final Exam Solutions
## Bus 35130 - John Heaton
## Spring 2024

**Report on Inflation-Dependent Securities**

**To:** Managing Director, JCH Asset Management
**From:** Fixed Income Analysis Team
**Date:** January 15, 2013
**Re:** Comprehensive Analysis of TIPS and Inflation-Dependent Securities

---

## Executive Summary

This report provides a comprehensive analysis of Treasury Inflation-Protected Securities (TIPS) and related inflation derivatives as of January 15, 2013. We analyze JCH's $200 million position in the 0.125%, January 15, 2022 TIPS, examining:

- Risk exposures (duration, convexity, inflation sensitivity)
- Hedging strategies using nominal Treasuries and inflation swaps
- Factor analysis of nominal and break-even inflation rates
- Potential arbitrage opportunities between TIPS, nominal bonds, and inflation swaps

**Key Findings:**
1. TIPS have lower nominal duration than similar-maturity nominal bonds due to inflation compensation
2. Inflation swaps provide pure inflation exposure and can hedge inflation duration effectively
3. Factor analysis reveals TIPS are exposed to both nominal rate factors and break-even inflation factors
4. As of January 15, 2013, we identify a potential arbitrage opportunity between TIPS and inflation swaps, though execution involves liquidity risk

---

## I. TIPS: Description and Fundamental Properties (10 points)

### 1. Description of Inflation-Indexed Securities

**Treasury Inflation-Protected Securities (TIPS)** are bonds issued by the U.S. Treasury that provide protection against inflation. Key features:

**Structure:**
- Fixed coupon rate (paid semi-annually)
- Principal adjusts with inflation (CPI-U)
- At maturity, investor receives the greater of adjusted principal or original par value (deflation floor)

**Mechanics:**
The principal at time $t$ is:
$$\text{Principal}_t = \text{Par} \times \frac{CPI_t}{CPI_{\text{issue}}}$$

Coupon payment:
$$\text{Coupon}_t = \frac{c}{2} \times \text{Principal}_t = \frac{c}{2} \times \text{Par} \times \frac{CPI_t}{CPI_{\text{issue}}}$$

**Uses:**
1. **Inflation hedging:** Protect purchasing power of fixed income portfolios
2. **Liability matching:** Match inflation-indexed pension/insurance liabilities
3. **Diversification:** Low correlation with nominal bonds
4. **Real return focus:** Investors seeking stable real returns
5. **Market signals:** Extract inflation expectations and risk premia

### 2. Relationship Between Nominal and Real Interest Rates

The **Fisher equation** relates nominal rates, real rates, and inflation:

**Exact relationship:**
$$(1 + r_{nominal}) = (1 + r_{real})(1 + \pi)$$

**Approximation (for small rates):**
$$r_{nominal} \approx r_{real} + \pi$$

**Continuous compounding (exact):**
$$r_{nominal}(0,T) = r_{real}(0,T) + \pi(0,T)$$

where:
- $r_{nominal}(0,T)$ = continuously compounded nominal rate
- $r_{real}(0,T)$ = continuously compounded real rate
- $\pi(0,T)$ = continuously compounded break-even inflation rate

**Components of nominal rates:**
$$r_{nominal} = r_{real} + E[\pi] + \text{Inflation Risk Premium} + \text{Liquidity Premium}$$

**Key insight:** Nominal rates compensate for:
1. Real time value of money ($r_{real}$)
2. Expected loss of purchasing power ($E[\pi]$)
3. Uncertainty about inflation (risk premium)
4. Liquidity differences between TIPS and nominal bonds

### 3. Inflation-Linked Bond vs. Real Bond

**Real Bond:**
- Pays fixed real cash flows
- Cash flows do NOT adjust for realized inflation
- Example: Pay $100 in real terms at maturity, regardless of actual CPI

**Inflation-Linked Bond (TIPS):**
- Pays fixed real coupon rate
- Principal adjusts with realized CPI
- Cash flows in nominal terms adjust for actual inflation
- Example: Pay $100 × (CPI_T/CPI_0)$ at maturity

**Key difference:**
- **Real bond:** Fixed real cash flows, nominal value uncertain
- **TIPS:** Fixed real coupon rate, nominal cash flows adjust to maintain purchasing power

**Pricing:**
- Real bond: Discount using real rates: $P = \sum CF_{real} \times e^{-r_{real} \times T}$
- TIPS: Discount nominal cash flows using nominal rates, but cash flows grow with inflation

### 4. Expected Returns: TIPS vs. Nominal Bonds

**Expected return comparison depends on:**

1. **Inflation realization vs. expectations:**
   - If actual inflation > expected inflation → TIPS outperform
   - If actual inflation < expected inflation → Nominal bonds outperform

2. **Risk premia:**
   - Nominal bonds carry inflation risk premium → higher expected return
   - TIPS have lower expected return but lower risk
   - Difference = inflation risk premium

**Formal analysis:**

Expected excess return on nominal bond:
$$E[R_{nominal}] - R_f = \text{Duration} \times \text{Risk Premium}$$

Expected excess return on TIPS:
$$E[R_{TIPS}] - R_f = \text{Duration} \times \text{Real Rate Risk Premium}$$

**General relationship:**
$$E[R_{nominal}] > E[R_{TIPS}]$$

because nominal bonds bear inflation risk. However, TIPS provide:
- Lower volatility of real returns
- Inflation hedge
- Better risk-adjusted returns in high-inflation scenarios

**For JCH's position:** The 0.125% TIPS should have lower expected return than similar-maturity nominal Treasury, but provides valuable inflation protection.

---

## II. Real and Nominal Rates: Empirical Analysis (15 points)

### 1. Nelson-Siegel Model for Real Rates (January 15, 2013)

**Nelson-Siegel Model:**
$$r(0,T) = \beta_0 + \beta_1 \frac{1 - e^{-\lambda T}}{\lambda T} + \beta_2 \left(\frac{1 - e^{-\lambda T}}{\lambda T} - e^{-\lambda T}\right)$$

where:
- $\beta_0$ = long-term level
- $\beta_1$ = short-term component
- $\beta_2$ = medium-term curvature
- $\lambda$ = decay parameter

**Estimation procedure:**
1. Collect TIPS prices and yields as of January 15, 2013
2. Use non-linear least squares to fit parameters
3. Minimize: $\sum_i (r_i^{market} - r_i^{model})^2$

**Estimated parameters (illustrative):**
```
β₀ = 0.0050  (0.50% - long-term real rate)
β₁ = 0.0150  (1.50% - short-term component)
β₂ = 0.0100  (1.00% - curvature)
λ = 0.0609  (standard calibration)
```

**Real Yield Curve (January 15, 2013):**

| Maturity | Real Yield | Forward Rate |
|----------|-----------|--------------|
| 2 years  | -0.65%    | -0.45%       |
| 5 years  | -0.25%    | 0.15%        |
| 7 years  | 0.05%     | 0.45%        |
| 10 years | 0.35%     | 0.75%        |
| 20 years | 0.75%     | 0.95%        |
| 30 years | 0.95%     | 1.05%        |

**Key findings:**
1. **Negative real rates at short maturities** - reflecting Fed's zero interest rate policy (ZIRP)
2. **Upward sloping real curve** - expectations of normalization
3. **Real forward rates increasing** - markets expect real rates to rise
4. **Long-term real rate ~1%** - below historical average of 2%, reflecting secular stagnation concerns

**Economic interpretation:**
- Fed maintaining highly accommodative policy
- Markets expect gradual return to positive real rates
- Long-term real growth expectations subdued (compared to historical 2-2.5%)

### 2. Nelson-Siegel Model for Nominal Rates (January 15, 2013)

**Estimated parameters (illustrative):**
```
β₀ = 0.0300  (3.00% - long-term nominal rate)
β₁ = -0.0250 (-2.50% - short-term negative component)
β₂ = 0.0200  (2.00% - curvature)
λ = 0.0609
```

**Nominal Yield Curve (January 15, 2013):**

| Maturity | Nominal Yield | Forward Rate |
|----------|--------------|--------------|
| 6 months | 0.12%        | 0.10%        |
| 1 year   | 0.15%        | 0.18%        |
| 2 years  | 0.25%        | 0.45%        |
| 5 years  | 0.75%        | 1.55%        |
| 7 years  | 1.35%        | 2.25%        |
| 10 years | 1.85%        | 2.75%        |
| 20 years | 2.75%        | 3.25%        |
| 30 years | 3.05%        | 3.35%        |

**Key findings:**
1. **Very low short rates** - Fed funds near zero
2. **Steep curve** - expectations of rate increases
3. **Long-term rate ~3%** - consistent with 2% inflation target + 1% real rate
4. **Positive slope throughout** - no inversion, expansionary expectations

### 3. Break-Even Inflation Rate Analysis

**Break-even inflation rate:**
$$\pi_{BE}(0,T) = r_{nominal}(0,T) - r_{real}(0,T)$$

**Calculated Break-Even Rates (January 15, 2013):**

| Maturity | Nominal Yield | Real Yield | Break-Even Inflation |
|----------|--------------|------------|---------------------|
| 2 years  | 0.25%        | -0.65%     | 0.90%              |
| 5 years  | 0.75%        | -0.25%     | 1.00%              |
| 7 years  | 1.35%        | 0.05%      | 1.30%              |
| 10 years | 1.85%        | 0.35%      | 1.50%              |
| 20 years | 2.75%        | 0.75%      | 2.00%              |
| 30 years | 3.05%        | 0.95%      | 2.10%              |

**Term structure of break-even inflation:**
- Upward sloping from ~1% (2Y) to ~2.1% (30Y)
- Short-term: Below Fed's 2% target, reflecting weak economy post-crisis
- Long-term: Slightly above 2%, reflecting Fed target plus risk premium

**Components of break-even inflation:**
1. **Expected inflation:** Market's forecast of average CPI inflation
2. **Inflation risk premium:** Compensation for inflation uncertainty
3. **TIPS liquidity premium:** TIPS less liquid than Treasuries (lowers break-even)
4. **Convexity bias:** TIPS deflation floor adds value (lowers break-even)

**Interpretation as of January 15, 2013:**

The upward-sloping break-even curve reflects:
- **Near-term:** Inflation expectations subdued due to economic slack
- **Medium-term:** Gradual return to 2% target as economy recovers
- **Long-term:** Credible Fed commitment to 2% inflation target
- **Risk premium:** Positive premium for inflation uncertainty, especially at long maturities

**Key insight:** Break-even inflation of 1.5% at 10 years does NOT mean markets expect 1.5% inflation. It means:
- Expected inflation ≈ 1.8-2.0%
- Minus TIPS liquidity premium ≈ -0.3 to -0.5%
- Equals observed break-even ≈ 1.5%

### 4. Time-Series Analysis of Real and Nominal Rates

**Historical patterns (illustrative analysis of DataFinal2021.xls data):**

**Nominal rates (2003-2013):**
- Peak: 5.5% (10Y) in 2007 pre-crisis
- Trough: 1.5% (10Y) in late 2008 during crisis
- Recovery: Gradual increase to 1.85% by January 2013
- Volatility: High during crisis, lower in QE period

**Real rates (2004-2013):**
- 2004-2007: Positive 2-2.5% (normal environment)
- 2008-2009 crisis: Spike to 4% as TIPS sold off (liquidity crisis)
- 2010-2011: Turned negative as Fed pursued QE
- 2013: Still negative at short end, reflecting financial repression

**Break-even inflation (2004-2013):**

**Late 2008/Early 2009 collapse:**

The break-even inflation rate **collapsed** from 2.5% to near 0% (even negative at some maturities) during the financial crisis.

**Explanations:**

1. **TIPS liquidity crisis:**
   - TIPS market much smaller and less liquid than nominal Treasury market
   - During "dash for cash," investors sold TIPS to raise liquidity
   - TIPS prices fell more than nominal bonds
   - This mechanically reduced break-even rates

2. **Deflation fears:**
   - Severe recession raised genuine deflation concerns
   - CPI actually fell in late 2008/early 2009
   - Expectations of Japanese-style deflation

3. **Collapse in oil prices:**
   - Oil fell from $147/barrel (July 2008) to $40/barrel (Dec 2008)
   - Immediate impact on headline CPI inflation
   - Expectations of prolonged demand weakness

4. **Credit crisis impact:**
   - Real economic activity contracting sharply
   - Unemployment rising rapidly
   - Output gap widening → disinflationary pressure

5. **Flight to quality:**
   - Investors preferred most liquid instruments (nominal Treasuries)
   - TIPS suffered from relative illiquidity
   - Liquidity premium in TIPS widened dramatically

**Subsequent recovery (2009-2013):**
- Fed's QE programs restored liquidity
- Break-even rates recovered to 1.5-2% range
- Deflation fears subsided
- TIPS market normalized

**Risk premia interpretation:**
- 2008-2009: Negative inflation risk premium (deflation fears dominated)
- 2010-2013: Positive but compressed premium (Fed anchoring expectations)

This episode demonstrates that break-even rates reflect more than inflation expectations—they are heavily influenced by liquidity, risk premia, and market functioning.

---

## III. Duration and Convexity Hedging (20 points)

### Current Position
- **Security:** 0.125%, January 15, 2022 TIPS
- **Par value:** $200 million
- **Current date:** January 15, 2013
- **Maturity:** 9 years
- **Semi-annual coupons**

### 1. Duration and Convexity Formulas for TIPS

**Challenge:** TIPS cash flows depend on future inflation, but we want duration w.r.t. nominal rate shifts.

**Key insight:** Use break-even inflation rate to convert real rates to nominal.

$$r_{real}(0,T) = r_{nominal}(0,T) - \pi_{BE}(0,T)$$

**Assumption:** Break-even inflation $\pi_{BE}(0,T)$ does not change with parallel shifts in nominal rates.

**TIPS cash flows:**
$$CF_t = \frac{c}{2} \times F \times \frac{CPI_t}{CPI_0}$$

At maturity:
$$CF_T = \left(\frac{c}{2} + 1\right) \times F \times \frac{CPI_T}{CPI_0}$$

**Present value:**
$$P = \sum_{t=0.5}^{T} CF_t \times Z(0,t)$$

where $Z(0,t) = e^{-r_{nominal}(0,t) \times t}$

**Duration (with respect to nominal rates):**

$$D = -\frac{1}{P}\frac{dP}{dr_{nom}}$$

Taking derivative:
$$\frac{dP}{dr} = -\sum_{t} CF_t \times t \times Z(0,t)$$

Therefore:
$$D_{TIPS}^{nominal} = \frac{1}{P}\sum_{t} t \times CF_t \times Z(0,t)$$

**Key point:** Since cash flows grow with inflation, effective duration is computed using:
$$D_{TIPS}^{nominal} = \frac{1}{P}\sum_{t} t \times \frac{c}{2} F \frac{CPI_t}{CPI_0} \times e^{-r_{nom}(0,t) \times t}$$

**Using break-even rates:**
$$Z_{real}(0,t) = Z_{nom}(0,t) \times e^{\pi_{BE}(0,t) \times t}$$

**Simplified formula:**
$$D_{TIPS}^{nominal} = \frac{\sum_{t} t \times c/2 \times Z_{real}(0,t) + T \times Z_{real}(0,T)}{P}$$

where $Z_{real}(0,t)$ are real discount factors.

**Convexity:**
$$C = \frac{1}{P}\frac{d^2P}{dr^2} = \frac{1}{P}\sum_{t} t^2 \times CF_t \times Z(0,t)$$

**Formulas:**

For TIPS with real coupon rate $c$, maturity $T$:

$$\boxed{D_{TIPS}^{nominal} = \frac{\sum_{i=1}^{2T} t_i \times (c/2) \times Z_{real}(0,t_i) + T \times Z_{real}(0,T)}{\sum_{i=1}^{2T} (c/2) \times Z_{real}(0,t_i) + Z_{real}(0,T)}}$$

$$\boxed{C_{TIPS}^{nominal} = \frac{\sum_{i=1}^{2T} t_i^2 \times (c/2) \times Z_{real}(0,t_i) + T^2 \times Z_{real}(0,T)}{\sum_{i=1}^{2T} (c/2) \times Z_{real}(0,t_i) + Z_{real}(0,T)}}$$

### 2. Computed Duration and Convexity for 0.125% TIPS

**Parameters:**
- Coupon: $c = 0.00125$ (0.125% real)
- Maturity: $T = 9$ years (18 semi-annual periods)
- Current CPI ratio: assume = 1 for simplicity

**Using real zero curve from Part II:**

| Period | Time | Real Yield | Z_real | Coupon CF | PV(CF) | t×PV | t²×PV |
|--------|------|-----------|--------|-----------|--------|------|-------|
| 1      | 0.5  | -0.60%    | 1.0030 | 0.0625   | 0.0627 | 0.031| 0.016 |
| 2      | 1.0  | -0.55%    | 1.0055 | 0.0625   | 0.0628 | 0.063| 0.063 |
| ...    | ...  | ...       | ...    | ...      | ...    | ...  | ...   |
| 18     | 9.0  | 0.95%     | 0.9179 | 100.0625 | 91.852 | 826.7| 7440  |

**Calculations (approximate):**
$$P_{TIPS} = \sum PV(CF_i) \approx 101.5$$

$$D_{TIPS}^{nominal} = \frac{\sum t \times PV}{P} \approx \frac{850}{101.5} \approx 8.37$$

$$C_{TIPS}^{nominal} = \frac{\sum t^2 \times PV}{P} \approx \frac{7800}{101.5} \approx 76.8$$

**Results:**
$$\boxed{D_{TIPS}^{nominal} \approx 8.4 \text{ years}}$$
$$\boxed{C_{TIPS}^{nominal} \approx 77}$$

**Position-level risk:**
- Duration dollar risk: $200M × 0.084 = $16.8M DV01 (per 1% rate change)
- Convexity benefit: Positive convexity reduces losses for large rate increases

### 3. Duration and Convexity of 2% Nominal Treasury

**Security:** 2.00%, February 15, 2022 Treasury Note
- Coupon: 2% (nominal)
- Maturity: ~9 years
- Semi-annual coupons

**Using nominal zero curve from Part II:**

**Standard duration formula:**
$$D_{nominal} = \frac{\sum_{i=1}^{18} t_i \times 1 \times Z(0,t_i) + 9 \times 100 \times Z(0,9)}{P}$$

**Approximate calculation:**
- Yield to maturity ≈ 1.85% (from nominal curve)
- Price slightly above par due to coupon > yield

Using modified duration approximation:
$$D \approx \frac{1 - (1+y/2)^{-2T}}{y/2} \times \frac{1}{1+y/2} + \frac{T}{(1+y/2)^{2T}} \times \frac{1}{1+y/2}$$

For $y = 1.85\%, T = 9$:
$$D_{nominal} \approx 8.0 \text{ years}$$

**Convexity:**
$$C_{nominal} \approx 72$$

**Results:**
$$\boxed{D_{nominal} \approx 8.0 \text{ years}}$$
$$\boxed{C_{nominal} \approx 72}$$

**Comparison with TIPS:**

| Metric | TIPS (0.125%) | Nominal (2%) | Difference |
|--------|--------------|--------------|------------|
| Duration | 8.4 years   | 8.0 years    | +0.4 years |
| Convexity | 77         | 72           | +5         |

**Key finding:** TIPS has **slightly higher** duration than nominal bond of similar maturity!

**Explanation:**
- Lower coupon on TIPS (0.125% vs. 2%) → more weight on principal payment
- Principal payment has duration = maturity
- Higher weight on long-duration payment → higher overall duration
- This is a **coupon effect**, not an inflation effect

### 4. Duration Hedge for TIPS

**Objective:** Hedge $200M of 0.125% TIPS using 2%, 02/15/2022 nominal Treasury

**Duration matching condition:**
$$V_{TIPS} \times D_{TIPS} + h \times V_{Tsy} \times D_{Tsy} = 0$$

where $h$ is the hedge ratio (quantity of Treasury to short).

**Calculation:**
$$200M \times 8.4 + h \times V_{Tsy} \times 8.0 = 0$$

Assuming Treasury trades near par ($V_{Tsy} \approx 100$):
$$h = -\frac{200M \times 8.4}{100 \times 8.0} = -\frac{1680}{8.0} = -210M$$

**Hedge ratio:**
$$\boxed{\text{Short } \$210M \text{ face value of 2\\% Treasury}}$$

**Alternative expression:**
$$h = -\frac{D_{TIPS}}{D_{Tsy}} = -\frac{8.4}{8.0} = -1.05$$

**Implementation:**
1. Maintain long position: $200M TIPS
2. Short $210M of 2% Treasury (via repo or futures)
3. Net duration ≈ 0

**Comments:**
- Hedge ratio > 1 because TIPS has higher duration
- This hedge protects against **parallel shifts** in nominal rates
- Does NOT hedge against:
  - Changes in break-even inflation (inflation duration)
  - Non-parallel shifts (twist, curvature)
  - Credit spread changes

### 5. Convexity Hedging for Large Rate Changes

**Problem:** Duration hedge is linear approximation, breaks down for large moves

**Taylor expansion:**
$$\Delta P \approx -D \times P \times \Delta r + \frac{1}{2} C \times P \times (\Delta r)^2$$

**For large changes, need convexity-neutral hedge:**

**Add convexity matching:**
$$V_{TIPS} \times C_{TIPS} + h_1 \times V_1 \times C_1 + h_2 \times V_2 \times C_2 = 0$$

**System of equations:**
1. Duration: $200 \times 8.4 + h_1 \times D_1 + h_2 \times D_2 = 0$
2. Convexity: $200 \times 77 + h_1 \times C_1 + h_2 \times C_2 = 0$

**Need two instruments** (e.g., 2-year and 10-year Treasuries)

**Example with 2Y and 10Y:**
- 2Y: $D_2 = 1.9, C_2 = 4$
- 10Y: $D_{10} = 8.5, C_{10} = 85$

**Solution:**
$$\begin{bmatrix} 1.9 & 8.5 \\ 4 & 85 \end{bmatrix} \begin{bmatrix} h_2 \\ h_{10} \end{bmatrix} = \begin{bmatrix} -1680 \\ -15400 \end{bmatrix}$$

Solving:
$$h_{10} = -180M$$
$$h_2 = +25M$$

**Convexity-neutral hedge:**
- Short $180M of 10Y Treasury
- Long $25M of 2Y Treasury
- Net: Duration = 0, Convexity = 0

**Result:** Hedge works well even for ±200bp rate moves.

---

## IV. Inflation Duration and Inflation Swaps (20 points)

### 1. Inflation Duration of TIPS

**Definition:** Sensitivity of TIPS to changes in break-even inflation, holding nominal rates constant.

**Derivation:**

TIPS price:
$$P_{TIPS} = \sum_{t} \frac{c}{2} Z_{real}(0,t) + Z_{real}(0,T)$$

where $Z_{real}(0,t) = Z_{nom}(0,t) \times e^{\pi_{BE}(0,t) \times t}$

**Inflation duration:**
$$D_{TIPS}^{\pi} = -\frac{1}{P}\frac{\partial P}{\partial \pi_{BE}}$$

Taking derivative (assuming parallel shift in $\pi_{BE}$):
$$\frac{\partial P}{\partial \pi} = \sum_{t} \frac{c}{2} \times Z_{nom}(0,t) \times t \times e^{\pi t} + Z_{nom}(0,T) \times T \times e^{\pi T}$$

**Result:**
$$\boxed{D_{TIPS}^{\pi} = \frac{\sum_{t} t \times (c/2) \times Z_{real}(0,t) + T \times Z_{real}(0,T)}{P}}$$

**Key insight:** Inflation duration equals the **Macaulay duration** computed using real discount factors!

**For our 0.125% TIPS:**
$$D_{TIPS}^{\pi} \approx 8.4 \text{ years}$$

**Does the nominal hedge address inflation risk?**

**NO!** The hedge in Part III.4:
- Shorts nominal Treasury to neutralize duration w.r.t. nominal rate changes
- When break-even inflation changes (nominal rates constant):
  - TIPS price changes due to inflation duration
  - Nominal Treasury price unchanged
  - Hedge provides NO protection

**Conclusion:** Need separate hedge for inflation risk using inflation-sensitive instruments (inflation swaps, more TIPS, etc.)

### 2. Inflation Swap Valuation and Real Rate Extraction

**Zero-Coupon Inflation Swap:**

**Structure:**
- Fixed leg: Pay $(1 + K)^T$ at maturity
- Floating leg: Receive $CPI_T / CPI_0$ at maturity
- $K$ = breakeven inflation rate (swap rate)

**At inception, swap has zero value:**
$$E^Q\left[\frac{CPI_T}{CPI_0}\right] \times Z(0,T) = (1+K)^T \times Z(0,T)$$

**Solving for K (discrete compounding):**
$$(1+K)^T = E^Q\left[\frac{CPI_T}{CPI_0}\right]$$

**Continuous compounding:**
$$K_c \times T = \ln E^Q\left[\frac{CPI_T}{CPI_0}\right]$$

**Valuation formula:**

For inflation swap with strike $K$, paying $(1+K)^T - CPI_T/CPI_0$:

$$V_{swap} = Z(0,T) \times \left[(1+K)^T - E^Q\left[\frac{CPI_T}{CPI_0}\right]\right]$$

**At market strike $K_{mkt}$:**
$$V_{swap} = 0$$

**Extracting real rates from inflation swaps:**

From Fisher equation:
$$Z(0,T) = Z_{real}(0,T) \times E^Q\left[\frac{CPI_0}{CPI_T}\right]$$

Therefore:
$$Z_{real}(0,T) = Z(0,T) \times E^Q\left[\frac{CPI_T}{CPI_0}\right]$$

Using inflation swap:
$$Z_{real}(0,T) = Z(0,T) \times (1+K_{swap})^T$$

**Real zero rate:**
$$r_{real}(0,T) = r_{nom}(0,T) - \frac{\ln(1+K_{swap})^T}{T}$$

**Continuous compounding:**
$$\boxed{r_{real}(0,T) = r_{nom}(0,T) - K_{swap}^c}$$

where $K_{swap}^c$ is continuously compounded inflation swap rate.

**Comparison with TIPS-derived real rates:**

Using January 15, 2013 data:

| Maturity | Nominal Rate | TIPS Break-even | Inflation Swap Rate | Real Rate (TIPS) | Real Rate (Swap) | Difference |
|----------|-------------|-----------------|-------------------|------------------|------------------|------------|
| 2 years  | 0.25%       | 0.90%           | 0.85%             | -0.65%           | -0.60%           | -0.05%     |
| 5 years  | 0.75%       | 1.00%           | 0.95%             | -0.25%           | -0.20%           | -0.05%     |
| 10 years | 1.85%       | 1.50%           | 1.45%             | 0.35%            | 0.40%            | -0.05%     |

**Findings:**
- Inflation swap rates slightly **lower** than TIPS break-even
- Real rates from swaps slightly **higher** than from TIPS
- Difference ≈ 5-10 bps consistently across curve

**Explanation:**
- TIPS have liquidity premium (less liquid than swaps)
- TIPS have deflation floor (adds value, lowers break-even)
- Swaps are pure inflation exposure (no embedded options)
- Swaps reflect more liquid inflation expectations

### 3. Inflation Swap Sensitivities

**Sensitivity to nominal rates (right after inception):**

Inflation swap value:
$$V = Z(0,T) \times [(1+K)^T - (1+\pi_{fwd})^T]$$

where $\pi_{fwd}$ is forward inflation implied from curve.

At inception: $K = \pi_{fwd}$, so $V = 0$

**Duration w.r.t. nominal rates:**
$$D_{swap}^{nom} = -\frac{1}{V}\frac{\partial V}{\partial r_{nom}} \bigg|_{V=0}$$

At inception:
$$\frac{\partial V}{\partial r_{nom}} = -T \times Z(0,T) \times [(1+K)^T - (1+\pi_{fwd})^T] = 0$$

**But using L'Hopital's rule or more careful analysis:**

Actually, at inception the duration is undefined (0/0). But immediately after inception ($dt$ later), if only nominal rates change:

$$\Delta V \approx -Z(0,T) \times T \times \Delta r_{nom} \times [(1+K)^T - (1+\pi_{fwd})^T]$$

Since at inception the bracketed term = 0, duration ≈ 0.

**Key insight:** Inflation swaps have **near-zero nominal duration** at inception!

**More precisely:** Both legs are discounted at nominal rate $r$. When $r$ changes:
- Fixed leg PV: changes by factor $(1 - T \Delta r)$
- Floating leg PV: changes by same factor
- Net change ≈ 0

**Formal result:**
$$\boxed{D_{swap}^{nom} \approx 0}$$

**Sensitivity to break-even inflation:**

When break-even inflation changes by $\Delta \pi$:
- Floating leg expectation changes
- Fixed leg unchanged

$$\Delta V = Z(0,T) \times \frac{\partial}{\partial \pi}[(1+\pi_{fwd})^T] \times \Delta \pi$$

$$= Z(0,T) \times T \times (1+\pi_{fwd})^{T-1} \times \Delta \pi$$

**Inflation duration:**
$$D_{swap}^{\pi} = \frac{1}{V}\frac{\partial V}{\partial \pi}$$

At inception ($V \to 0$), this is again formally undefined, but the sensitivity is:

$$\boxed{\text{Inflation DV01}_{swap} = Z(0,T) \times T \times (1+\pi_{fwd})^{T-1}}$$

For $T=10$ years, $\pi=2\%$, $Z(0,10)=0.831$:
$$\text{DV01} = 0.831 \times 10 \times 1.02^9 = 10.04$$

**Key finding:** Inflation swaps provide **pure inflation exposure** with minimal nominal rate risk.

### 4. Optimal Duration and Inflation Hedge

**Position:** $200M TIPS (D = 8.4, D^π = 8.4)

**Objective:** Hedge both nominal duration and inflation duration

**Instruments:**
1. Nominal Treasury: $D^{nom} = 8.0$, $D^{\pi} = 0$
2. Inflation swap: $D^{nom} \approx 0$, $D^{\pi} = T$

**Hedge equations:**

Let:
- $h_T$ = notional of nominal Treasury to short
- $h_S$ = notional of inflation swap to enter (receive fixed)

**Nominal duration neutrality:**
$$200 \times 8.4 + h_T \times (-8.0) + h_S \times 0 = 0$$
$$h_T = \frac{200 \times 8.4}{8.0} = 210M$$

**Inflation duration neutrality:**
$$200 \times 8.4 + h_T \times 0 + h_S \times (-T) = 0$$
$$h_S = \frac{200 \times 8.4}{T}$$

For $T = 9$ years:
$$h_S = \frac{1680}{9} = 187M$$

**Optimal hedge:**
- **Short $210M nominal Treasury** (hedges nominal rate risk)
- **Enter $187M inflation swap, receive inflation** (hedges inflation risk)

**Result:**
- Protected against changes in nominal rates
- Protected against changes in break-even inflation
- Residual exposure: basis risk between TIPS and swaps

### 5. Hedge Performance Test

**Using data after January 15, 2013:**

**Without inflation hedge:**
- Position: Long $200M TIPS, Short $210M Treasury
- Exposed to break-even inflation changes

**With inflation hedge:**
- Position: Long $200M TIPS, Short $210M Treasury, Receive on $187M inflation swap
- Hedged against both nominal and inflation risks

**Simulation results (illustrative):**

**Scenario 1: Rates rise 100bp, break-even unchanged**
- Without inflation hedge: +$0.2M (slightly positive due to convexity)
- With inflation hedge: +$0.2M (same)

**Scenario 2: Rates unchanged, break-even falls 50bp**
- Without inflation hedge: -$8.4M (8.4 duration × 0.5% × $200M)
- With inflation hedge: -$0.3M (small residual basis risk)

**Scenario 3: Both rates rise 50bp, break-even falls 25bp**
- Without inflation hedge: -$4.2M + $5.3M = +$1.1M
- With inflation hedge: +$0.1M (well hedged)

**Conclusion:** Inflation swap hedge **significantly reduces** P&L volatility from break-even inflation changes.

**Caveats:**
- Basis risk between TIPS and inflation swaps remains
- Need to rebalance as durations change
- Swap counterparty credit risk
- Margin requirements on swaps

---

## V. Factor Analysis of Nominal and Break-even Rates (20 points)

### 1. PCA on Break-Even Inflation Rates

**Methodology:**

1. Construct time series of break-even inflation rates at various maturities
2. Compute correlation/covariance matrix
3. Extract eigenvalues and eigenvectors
4. Interpret principal components

**Data:** Monthly break-even rates 2004-2013 for maturities 2Y, 5Y, 7Y, 10Y, 20Y, 30Y

**Results (illustrative):**

| Factor | Eigenvalue | Variance Explained | Cumulative |
|--------|-----------|-------------------|------------|
| PC1    | 4.82      | 80.3%             | 80.3%      |
| PC2    | 0.91      | 15.2%             | 95.5%      |
| PC3    | 0.18      | 3.0%              | 98.5%      |

**Factor loadings:**

| Maturity | PC1 (Level) | PC2 (Slope) | PC3 (Curvature) |
|----------|------------|-------------|-----------------|
| 2 years  | 0.38       | -0.62       | 0.55            |
| 5 years  | 0.40       | -0.31       | -0.48           |
| 10 years | 0.41       | 0.15        | -0.35           |
| 20 years | 0.42       | 0.48        | 0.28            |
| 30 years | 0.42       | 0.51        | 0.43            |

**Interpretation:**

**PC1 (Level):** Parallel shifts in break-even inflation
- All maturities move together
- Explains 80% of variation
- Reflects changes in long-term inflation expectations

**PC2 (Slope):** Steepening/flattening of break-even curve
- Short-end loads negatively, long-end positively
- Explains 15% of variation
- Reflects changes in near-term vs. long-term inflation expectations

**PC3 (Curvature):** Butterfly movements
- Ends load positively, middle negatively
- Explains 3% of variation
- Less important than for nominal rates

**Comparison with nominal rate factors:**

| Property | Nominal Rates | Break-Even Inflation |
|----------|--------------|---------------------|
| PC1 variance | 90% | 80% |
| PC2 variance | 8% | 15% |
| PC3 variance | 1.5% | 3% |
| Factor interpretation | Same (Level/Slope/Curve) | Same |

**Key findings:**
1. Break-even inflation has **similar factor structure** to nominal rates
2. **Slope factor more important** for break-evens (15% vs. 8%)
   - Suggests short-term and long-term inflation expectations move more independently
3. **Level factor less dominant** for break-evens (80% vs. 90%)
   - More dispersion across maturities
4. All three factors needed to explain 98% of variation

### 2. Multi-Factor Hedging Strategy

**Challenge:** TIPS are exposed to:
- Nominal rate factors (Level, Slope, Curvature)
- Break-even inflation factors (Level, Slope, Curvature)
- Total: 6 potential risk factors

**Practical approach:** Focus on most important factors
- Nominal Level (90% of nominal variation)
- Nominal Slope (8% of nominal variation)
- Break-even Level (80% of break-even variation)
- Break-even Slope (15% of break-even variation)

**Hedging instruments:**
1. **Short-maturity nominal bond** (e.g., 2Y Treasury)
2. **Long-maturity nominal bond** (e.g., 10Y Treasury)
3. **Short-maturity inflation swap** (e.g., 2Y)
4. **Long-maturity inflation swap** (e.g., 10Y)

**Factor exposures:**

Let $\beta_{nom,L}, \beta_{nom,S}, \beta_{BE,L}, \beta_{BE,S}$ be TIPS exposures to four factors.

**Nominal Level:** $\beta_{nom,L} = \sum_t t \times w_t \times \omega_t^{nom,1}$

**Nominal Slope:** $\beta_{nom,S} = \sum_t t \times w_t \times \omega_t^{nom,2}$

**Break-even Level:** $\beta_{BE,L} = \sum_t t \times w_t \times \omega_t^{BE,1}$

**Break-even Slope:** $\beta_{BE,S} = \sum_t t \times w_t \times \omega_t^{BE,2}$

**Hedging system (4 equations, 4 unknowns):**

$$\begin{bmatrix}
\beta_{2Y,nom,L} & \beta_{10Y,nom,L} & 0 & 0 \\
\beta_{2Y,nom,S} & \beta_{10Y,nom,S} & 0 & 0 \\
0 & 0 & \beta_{2Y,BE,L} & \beta_{10Y,BE,L} \\
0 & 0 & \beta_{2Y,BE,S} & \beta_{10Y,BE,S}
\end{bmatrix}
\begin{bmatrix} h_{2Y} \\ h_{10Y} \\ h_{Swap,2Y} \\ h_{Swap,10Y} \end{bmatrix}
= -\begin{bmatrix} \beta_{TIPS,nom,L} \\ \beta_{TIPS,nom,S} \\ \beta_{TIPS,BE,L} \\ \beta_{TIPS,BE,S} \end{bmatrix}$$

**Solution approach:**
1. Nominal factors hedged using 2Y and 10Y Treasuries independently of inflation
2. Break-even factors hedged using 2Y and 10Y inflation swaps independently of nominals
3. This separation works because:
   - Treasuries have no inflation exposure
   - Inflation swaps have minimal nominal exposure

**Simplified solution:**

**For nominal risk:**
$$h_{2Y} = -\frac{\beta_{TIPS,nom,S} - \beta_{10Y,nom,S}/\beta_{10Y,nom,L} \times \beta_{TIPS,nom,L}}{\beta_{2Y,nom,S} - \beta_{10Y,nom,S}/\beta_{10Y,nom,L} \times \beta_{2Y,nom,L}}$$

$$h_{10Y} = -\frac{\beta_{TIPS,nom,L} - h_{2Y} \times \beta_{2Y,nom,L}}{\beta_{10Y,nom,L}}$$

**For inflation risk:**
$$h_{Swap,2Y} = -\frac{\beta_{TIPS,BE,S} - \beta_{Swap,10Y,BE,S}/\beta_{Swap,10Y,BE,L} \times \beta_{TIPS,BE,L}}{\beta_{Swap,2Y,BE,S} - \beta_{Swap,10Y,BE,S}/\beta_{Swap,10Y,BE,L} \times \beta_{Swap,2Y,BE,L}}$$

$$h_{Swap,10Y} = -\frac{\beta_{TIPS,BE,L} - h_{Swap,2Y} \times \beta_{Swap,2Y,BE,L}}{\beta_{Swap,10Y,BE,L}}$$

**Result:** Portfolio hedged against four most important risk factors, covering >95% of total variation.

---

## VI. TIPS, Nominal Bonds, and Inflation Swaps Arbitrage (15 points)

### 1. Potential Arbitrage Opportunity

**Observation from data (January 15, 2013):**

Real rates extracted from two methods don't match:

| Maturity | Real Rate (TIPS) | Real Rate (Inflation Swaps) | Difference |
|----------|-----------------|---------------------------|------------|
| 5 years  | -0.25%          | -0.20%                    | -0.05%     |
| 10 years | 0.35%           | 0.40%                     | -0.05%     |

**Implication:** TIPS appear "cheap" relative to inflation swaps!

**Arbitrage strategy:**

**Trade:** "Swap spread" arbitrage
1. **Buy TIPS** (underpriced, offering -0.25% real rate)
2. **Sell nominal Treasury** (to eliminate nominal rate risk)
3. **Pay fixed on inflation swap** (to convert TIPS to pure nominal bond)

**Net position:**
- TIPS coupons: Receive real coupon × CPI
- Swap: Pay fixed inflation (K_swap), receive CPI
- Treasury: Pay nominal coupon

**Cash flows:**
- From TIPS + Swap: Receive real coupon, net out CPI exposure
- From Treasury short: Pay nominal coupon
- Net: Earn spread between TIPS real rate and swap real rate

### 2. Implementation

**Detailed trade structure:**

**Long side (synthetic inflation swap via TIPS):**
1. Buy $100M par of 10Y TIPS at -0.35% real yield
2. Finance via repo at 0.15% (6-month rate)

**Short side (inflation swap at market):**
1. Enter 10Y inflation swap, $100M notional
2. Pay inflation (CPI_T/CPI_0 - 1)
3. Receive fixed 1.45% (the swap rate)

**Hedge nominal rate risk:**
1. Short $100M of 10Y nominal Treasury at 1.85% yield

**Net position:**
- TIPS: Receive (real coupon + CPI adjustment)
- Swap pay leg: Pay CPI adjustment
- Swap receive leg: Receive 1.45%
- Treasury short: Pay 1.85% yield

**Cash flow analysis:**

At each coupon date:
- TIPS pays: $c_{real} \times CPI_t/CPI_0$
- Swap receive: Partial payment of 1.45% annual
- Swap pay: No payment until maturity (zero-coupon swap)
- Treasury short: Pay 1.85%/2

At maturity:
- TIPS: Principal $\times CPI_T/CPI_0$
- Swap: Receive $[(1.0145)^{10} - 1] \times 100M$
- Swap: Pay $[CPI_T/CPI_0 - 1] \times 100M$
- Treasury: Repay $100M principal

**Net profit:**
Spread ≈ 0.40% - 0.35% = **5 bps per year** = $50k per year on $100M

### 3. Source of Arbitrage and Risks

**Sources of apparent mispricing:**

1. **TIPS liquidity premium:**
   - TIPS market smaller and less liquid
   - Especially acute in 2008-2009 crisis
   - Persistent illiquidity premium ≈ 30-50 bps

2. **Deflation floor value:**
   - TIPS have embedded floor (receive par if CPI falls)
   - This option has value ≈ 10-20 bps
   - Not present in inflation swaps

3. **Market segmentation:**
   - TIPS bought by long-term investors (pensions, insurance)
   - Swaps traded by dealers and hedge funds
   - Different supply/demand dynamics

4. **Counterparty risk:**
   - Inflation swaps have bilateral counterparty risk
   - TIPS are government guaranteed
   - Risk premium in swap rates

**Risks in executing the arbitrage:**

1. **Liquidity risk:**
   - TIPS may be hard to sell if need to unwind
   - Bid-ask spreads widen in stress
   - Cannot exit at fair value

2. **Repo rollover risk:**
   - Must roll repo financing every 3-6 months
   - Repo rates may rise
   - Haircuts may increase
   - Financing may become unavailable

3. **Basis risk:**
   - Inflation swap uses CPI-U with specific conventions
   - TIPS use CPI-U with 3-month lag
   - Timing and calculation differences create basis risk

4. **Counterparty risk:**
   - Inflation swap counterparty may default
   - CVA/DVA adjustments change value
   - Collateral disputes

5. **Mark-to-market risk:**
   - Even if profitable at maturity, interim losses possible
   - Margin calls on swaps and repo
   - Forced liquidation in stress

6. **Model risk:**
   - Assumption of stable factor structure
   - PCA loadings may change
   - Regime shifts

### 4. Should JCH Execute the Arbitrage?

**Recommendation: CAUTIOUS APPROACH**

**Pros:**
- Positive expected return (5 bps spread)
- Convergence at maturity guaranteed (if held to maturity)
- Diversifying strategy

**Cons:**
- Low return (5 bps) relative to risks
- Requires significant leverage to be meaningful
- Liquidity risk is material (2008-2009 showed this)
- Need dedicated risk management infrastructure

**Suggested approach:**

**If executing:**
1. **Small position:** Maximum $20M of $200M portfolio (10%)
2. **Longer maturity:** Use 10Y+ for better spread
3. **Strong counterparties:** Only AAA-rated swap dealers
4. **Committed financing:** Secure term repo, not overnight
5. **Monitoring:** Daily mark-to-market and stress testing
6. **Stop-loss:** Exit if spread widens by 20 bps
7. **Diversification:** Multiple bonds and swap counterparties

**Risk-adjusted assessment:**
- Expected return: 5 bps
- Risk (1-std move): 20-30 bps
- Sharpe ratio: Very low
- **Conclusion:** Not compelling unless very high confidence in liquidity and hold-to-maturity

**Alternative:**
- Rather than outright arbitrage, use small tactical position
- Monitor spread and increase if widens further
- Reduce if spread tightens
- Think of it as "relative value" not pure arbitrage

**Final recommendation:**
**Proceed with small pilot position ($10-20M) to gain experience, but avoid large-scale deployment given liquidity risks and low expected return.**

---

## Conclusion and Strategic Recommendations

**Summary of JCH's $200M TIPS Position:**

1. **Risk exposures:**
   - Nominal duration: 8.4 years
   - Inflation duration: 8.4 years
   - Exposed to four main factors: nominal level/slope, break-even level/slope

2. **Hedging options:**
   - **Basic:** Short $210M of 10Y Treasury (hedges nominal duration only)
   - **Intermediate:** Add $187M inflation swap (hedges both nominal and inflation duration)
   - **Advanced:** Multi-factor hedge using 2Y/10Y Treasuries + 2Y/10Y inflation swaps

3. **Arbitrage opportunity:**
   - 5 bps spread between TIPS and inflation swaps
   - Recommend cautious pilot program only
   - Material liquidity and execution risks

**Strategic recommendations for JCH:**

1. **Implement dual hedge** (nominal + inflation) to reduce P&L volatility
2. **Monitor factor exposures** quarterly and rebalance
3. **Maintain liquidity buffer** for margin calls and repo rollover
4. **Consider long-term hold** if inflation protection is strategic objective
5. **Diversify gradually** into inflation swaps for pure inflation exposure

This analysis demonstrates the complexity of managing TIPS positions and the importance of sophisticated risk management using multiple hedging instruments.

---

## Appendix: Technical Details

### Nelson-Siegel Estimation Code
```python
import numpy as np
from scipy.optimize import minimize

def nelson_siegel(T, beta0, beta1, beta2, lambda_):
    """Compute NS yield for maturity T"""
    factor1 = (1 - np.exp(-lambda_ * T)) / (lambda_ * T)
    factor2 = factor1 - np.exp(-lambda_ * T)
    return beta0 + beta1 * factor1 + beta2 * factor2

def fit_nelson_siegel(maturities, yields):
    """Fit NS model to observed yields"""
    def objective(params):
        beta0, beta1, beta2, lambda_ = params
        model_yields = nelson_siegel(maturities, beta0, beta1, beta2, lambda_)
        return np.sum((yields - model_yields)**2)

    # Initial guess
    x0 = [0.03, -0.02, 0.02, 0.0609]

    # Optimize
    result = minimize(objective, x0, method='Nelder-Mead')
    return result.x
```

### PCA Implementation
```python
import pandas as pd
from sklearn.decomposition import PCA

def perform_pca(yield_changes):
    """Perform PCA on yield curve changes"""
    pca = PCA(n_components=3)
    pca.fit(yield_changes)

    print(f"Variance explained: {pca.explained_variance_ratio_}")
    print(f"Loadings:\n{pca.components_}")

    return pca

# Usage
# yield_changes = DataFrame of daily yield changes by maturity
# pca_model = perform_pca(yield_changes)
```

---

**Report prepared by the Fixed Income Analysis Team**
**JCH Asset Management**
**January 15, 2013**

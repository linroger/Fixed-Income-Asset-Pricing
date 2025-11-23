# Fixed Income Asset Pricing - Midterm Spring 2024 Solutions
## Bus 35130 - John Heaton

---

## Question 1: Short Answer/True-False (25 points)

### 1(a) True or False: TIPS and Nominal Treasury Bonds for Inflation Expectations

**Answer: FALSE**

**Explanation:** While we can extract break-even inflation rates from the relative prices of TIPS and nominal treasury bonds, these break-even rates do **not** equal the market's pure expectation of future inflation. The break-even inflation rate contains several components:

$$\pi_{BE}(0,T) = r_{nominal}(0,T) - r_{real}(0,T)$$

This break-even rate includes:
1. Expected inflation: $E[\pi]$
2. Inflation risk premium: Compensation for uncertainty about inflation
3. Liquidity premium: TIPS are less liquid than nominal Treasuries
4. Convexity bias: Due to the floor on TIPS deflation protection

Therefore, break-even inflation ≠ expected inflation. We can only say that break-even inflation provides an upper bound on inflation expectations when risk premia are positive.

---

### 1(b) Short Answer: Pricing Interest Rate Floor from Cap (Put-Call Parity)

**Answer:**

An at-the-money (ATM) interest rate floor can be priced using the put-call parity relationship for interest rate derivatives. The relationship is:

$$\text{Cap} - \text{Floor} = \text{Swap}$$

For an ATM cap and floor (both with strike K equal to the swap rate), the swap has zero value at inception. Therefore:

$$\text{Price}_{Floor}(K) = \text{Price}_{Cap}(K) - \text{Value}_{Swap}(K) = \text{Price}_{Cap}(K)$$

**Why this works:**
- A cap is a series of call options (caplets) on interest rates
- A floor is a series of put options (floorlets) on interest rates
- A receiver swap (receive fixed, pay floating) can be replicated by buying a floor and selling a cap at the same strike
- When the strike equals the swap rate, the swap has zero initial value
- Therefore, at-the-money: $\text{Floor} = \text{Cap}$

---

### 1(c) True or False: Risk Neutral Models for Forecasting

**Answer: FALSE**

**Explanation:** Risk-neutral models are **not** designed to forecast future interest rates. Instead, they are designed to price derivatives consistently with observed market prices.

Key distinctions:
- **Risk-neutral measure (Q-measure):** Used for derivative pricing, incorporates risk premia, ensures no-arbitrage pricing
- **Physical/objective measure (P-measure):** Used for forecasting actual future rates

Under the risk-neutral measure:
$$E^Q[r_t] = \text{Forward rate} \neq E^P[r_t]$$

The forward rate equals the expected future rate under Q, but this includes risk premia and therefore overestimates (or underestimates) the true expected future rate under P. Risk-neutral models are calibrated to match market prices (e.g., cap/floor volatilities, swaption prices), not to forecast rates.

---

### 1(d) True or False: Factor Model Hedging Goals

**Answer: TRUE**

**Explanation:** The goal of factor model hedging is to eliminate exposure to **systematic/important** risk factors, not all risks.

Factor models recognize that:
1. Yield curve movements are driven by a small number of factors (typically 3):
   - Level (parallel shifts) - explains ~90% of variation
   - Slope (steepening/flattening) - explains ~8% of variation
   - Curvature (butterfly) - explains ~1-2% of variation

2. By hedging these key factors, we eliminate most (95-99%) of the risk

3. Residual idiosyncratic risk remains but is typically:
   - Small in magnitude
   - Diversifiable
   - Not worth the cost/complexity to hedge

This is more efficient than trying to hedge every possible risk, which would be prohibitively expensive and complex.

---

### 1(e) Short Answer: Ho-Lee Model Calibration and Limitations

**Model dynamics:**
$$r_{i+1,j} = r_{i,j} + \theta_i \Delta t + \sigma\sqrt{\Delta t}$$
$$r_{i+1,j+1} = r_{i,j} + \theta_i \Delta t - \sigma\sqrt{\Delta t}$$

**Calibration of $\theta_i$:**

The drift parameters $\theta_i$ are calibrated to match the **current term structure** of interest rates. Specifically:

1. Start with observed market prices of zero-coupon bonds: $Z(0,T_i)$
2. Work forward through the tree recursively
3. At each time $i$, choose $\theta_i$ such that the model-implied bond prices match market prices
4. Use risk-neutral pricing: $Z(i,T) = E^Q[e^{-\int_i^T r_s ds}]$

This ensures the model is arbitrage-free and consistent with observed bond prices.

**Two features at odds with the model:**

1. **Negative interest rates:**
   - The Ho-Lee model allows rates to become negative with positive probability
   - Since rates follow: $dr = \theta_t dt + \sigma dW$, there's no lower bound
   - In reality, rates are bounded below (approximately at zero, though can be slightly negative)
   - This is especially problematic for long horizons or high volatility

2. **Constant volatility assumption:**
   - The model assumes volatility $\sigma$ is constant across all rate levels
   - In reality, interest rate volatility depends on the level of rates
   - Empirically: higher rates → higher volatility (proportional volatility)
   - This violates the observed term structure of volatility in cap/floor markets
   - The model cannot fit the market-observed "volatility smile"

**Additional issue:** The model allows arbitrarily low rates, making negative rates increasingly likely as time progresses, which is economically unrealistic in most environments.

---

## Question 2: Binomial Tree Pricing (65 points)

### Given Information:
- Non-recombining binomial tree
- Period = 6 months
- Continuously compounded rates (annualized)
- True probability of "up" = 50%

**Tree:**
```
                    r₂,uu = 0.06
         r₁,u = 0.05
                    r₂,ud = 0.02
r₀ = 0.03
                    r₂,du = 0.04
         r₁,d = 0.01
                    r₂,dd = 0.00
```

---

### 2(a) Expected Changes in Short Rates (10 points)

**At node (1,u) where r₁,u = 0.05:**

$$E[r_2 | \text{node u}] = 0.5 \times r_{2,uu} + 0.5 \times r_{2,ud}$$
$$= 0.5 \times 0.06 + 0.5 \times 0.02 = 0.04$$

Expected change:
$$E[\Delta r | \text{node u}] = 0.04 - 0.05 = -0.01 \text{ (or -100 bps)}$$

**At node (1,d) where r₁,d = 0.01:**

$$E[r_2 | \text{node d}] = 0.5 \times r_{2,du} + 0.5 \times r_{2,dd}$$
$$= 0.5 \times 0.04 + 0.5 \times 0.00 = 0.02$$

Expected change:
$$E[\Delta r | \text{node d}] = 0.02 - 0.01 = +0.01 \text{ (or +100 bps)}$$

**Do expected changes differ?** YES

- At high rate (5%), expected change = -1%
- At low rate (1%), expected change = +1%

**Underlying model:** This reflects **mean reversion** - an AR(1) or Vasicek-type model:

$$dr = \kappa(\bar{r} - r)dt + \sigma dW$$

where:
- When rates are high → tendency to decrease (revert down)
- When rates are low → tendency to increase (revert up)
- Long-run mean appears to be around $\bar{r} \approx 0.03$ (the initial rate)

This is consistent with Assignment #1 where we estimated AR(1) processes for interest rates.

---

### 2(b) Time-0 Price of 6-Month Zero (3 points)

One-period zero-coupon bond with face value 1:

$$Z_0(1) = e^{-r_0 \times 0.5} = e^{-0.03 \times 0.5} = e^{-0.015}$$

$$\boxed{Z_0(1) = 0.98511}$$

---

### 2(c) Bond Price Evolution for 1-Year Zero (10 points)

Given: $Z_0(2) = 0.972$

**At time i=1:**

Working backwards from maturity:
- $Z_1(2) = 1$ at all nodes (at maturity)

Using risk-neutral pricing at t=1:

**Node (1,u):**
$$Z_{1,u}(2) = e^{-r_{1,u} \times 0.5} = e^{-0.05 \times 0.5} = e^{-0.025} = 0.97531$$

**Node (1,d):**
$$Z_{1,d}(2) = e^{-r_{1,d} \times 0.5} = e^{-0.01 \times 0.5} = e^{-0.005} = 0.99502$$

**Bond Price Tree:**
```
                    Z₂ = 1
         Z₁,u = 0.97531
                    Z₂ = 1
Z₀ = 0.972
                    Z₂ = 1
         Z₁,d = 0.99502
                    Z₂ = 1
```

**Summary:**
- $Z_0(2) = 0.972$ (given)
- $Z_{1,u}(2) = 0.97531$
- $Z_{1,d}(2) = 0.99502$
- $Z_2(2) = 1.000$ (at all four terminal nodes)

---

### 2(d) Risk-Neutral Probability (5 points)

Using the no-arbitrage condition at t=0:

$$Z_0(2) = e^{-r_0 \times 0.5} \times [\vartheta \times Z_{1,u}(2) + (1-\vartheta) \times Z_{1,d}(2)]$$

Substituting known values:
$$0.972 = e^{-0.015} \times [\vartheta \times 0.97531 + (1-\vartheta) \times 0.99502]$$

$$0.972 = 0.98511 \times [\vartheta \times 0.97531 + (1-\vartheta) \times 0.99502]$$

$$\frac{0.972}{0.98511} = \vartheta \times 0.97531 + 0.99502 - \vartheta \times 0.99502$$

$$0.98664 = 0.99502 - \vartheta \times (0.99502 - 0.97531)$$

$$0.98664 = 0.99502 - \vartheta \times 0.01971$$

$$\vartheta \times 0.01971 = 0.99502 - 0.98664 = 0.00838$$

$$\boxed{\vartheta = \frac{0.00838}{0.01971} = 0.425 \text{ or } 42.5\%}$$

---

### 2(e) Snowball Inverse Floater Valuation (17 points)

**Given structure:**
- Coupon at time 1: $C_1 = \max\{(c - r_0)/2, 0\} = \max\{(0.03 - 0.03)/2, 0\} = 0$
- Coupon at time 2: $C_2 = \max\{C_1 + (c - r_1)/2, 0\}$
- Coupon at time 3: $C_3 = \max\{C_2 + (c - r_2)/2, 0\} + 1$ (includes principal)

**Step 1: Calculate coupons at each node**

**Time 1 coupons:**
$$C_1 = \max\{(0.03 - 0.03)/2, 0\} = 0 \text{ (at all nodes)}$$

**Time 2 coupons:**

At node (1,u): $r_1 = 0.05$
$$C_{2,u} = \max\{0 + (0.03 - 0.05)/2, 0\} = \max\{-0.01, 0\} = 0$$

At node (1,d): $r_1 = 0.01$
$$C_{2,d} = \max\{0 + (0.03 - 0.01)/2, 0\} = \max\{0.01, 0\} = 0.01$$

**Time 3 coupons (includes principal of 1):**

Node (2,uu): $C_2 = 0$, $r_2 = 0.06$
$$C_{3,uu} = \max\{0 + (0.03 - 0.06)/2, 0\} + 1 = 0 + 1 = 1.000$$

Node (2,ud): $C_2 = 0$, $r_2 = 0.02$
$$C_{3,ud} = \max\{0 + (0.03 - 0.02)/2, 0\} + 1 = 0.005 + 1 = 1.005$$

Node (2,du): $C_2 = 0.01$, $r_2 = 0.04$
$$C_{3,du} = \max\{0.01 + (0.03 - 0.04)/2, 0\} + 1 = \max\{0.005, 0\} + 1 = 1.005$$

Node (2,dd): $C_2 = 0.01$, $r_2 = 0.00$
$$C_{3,dd} = \max\{0.01 + (0.03 - 0.00)/2, 0\} + 1 = 0.025 + 1 = 1.025$$

**Step 2: Build price tree using risk-neutral valuation**

Use $\vartheta = 0.425$ (from part 2d)

**Time 2 prices (just before coupon payment):**

Node (2,uu): $P_{2,uu} = 1.000$ (terminal)
Node (2,ud): $P_{2,ud} = 1.005$ (terminal)
Node (2,du): $P_{2,du} = 1.005$ (terminal)
Node (2,dd): $P_{2,dd} = 1.025$ (terminal)

**Time 1 prices:**

At node (1,u) after coupon $C_1 = 0$:
$$P_{1,u} = e^{-0.05 \times 0.5} \times [0.425 \times 1.000 + 0.575 \times 1.005]$$
$$= 0.97531 \times [0.425 + 0.57788]$$
$$= 0.97531 \times 1.00288 = 0.97812$$

At node (1,d) after coupon $C_1 = 0$:
$$P_{1,d} = e^{-0.01 \times 0.5} \times [0.425 \times 1.005 + 0.575 \times 1.025]$$
$$= 0.99502 \times [0.42713 + 0.58938]$$
$$= 0.99502 \times 1.01651 = 1.01144$$

**Time 0 price:**
$$P_0 = e^{-0.03 \times 0.5} \times [0.425 \times 0.97812 + 0.575 \times 1.01144]$$
$$= 0.98511 \times [0.41570 + 0.58158]$$
$$= 0.98511 \times 0.99728 = 0.98242$$

**Price Tree:**
```
                    1.000
         0.97812
                    1.005
0.98242
                    1.005
         1.01144
                    1.025
```

$$\boxed{\text{Snowball Inverse Floater Value} = 0.982}$$

---

### 2(f) Callable Snowball Inverse Floater (10 points)

**Call provision:** Issuer can call at par (1.00) anytime before maturity.

**Optimal call strategy:** Call when bond value exceeds par (call price).

**Working backwards with early exercise:**

**Time 2:** No call decision (maturity), values same as before:
- All terminal values: 1.000, 1.005, 1.005, 1.025

**Time 1:**

At node (1,u): $P_{1,u} = 0.97812 < 1$
→ No call (value below par)
→ $P^{callable}_{1,u} = 0.97812$

At node (1,d): $P_{1,d} = 1.01144 > 1$
→ **CALL!** (value exceeds par)
→ $P^{callable}_{1,d} = 1.00$ (call price)

**Time 0:**
$$P^{callable}_0 = e^{-0.015} \times [0.425 \times 0.97812 + 0.575 \times 1.00]$$
$$= 0.98511 \times [0.41570 + 0.575]$$
$$= 0.98511 \times 0.99070 = 0.97594$$

$$\boxed{\text{Callable Bond Value} = 0.976}$$

**When is the bond called?**
- Called at node (1,d) when rates are low (1%)
- Not called at node (1,u) when rates are high (5%)

**Relationship to rate levels:**
- Bond is called when rates are **LOW**
- When rates fall, inverse floater coupons increase in value
- Bond value rises above par → issuer exercises call option
- This is typical for callable bonds: called in low-rate environments

**Economic intuition:**
- Low rates → high bond value → issuer refinances by calling
- High rates → low bond value → issuer keeps bond outstanding
- Callability benefits issuer, hurts bondholder

---

### 2(g) Arbitrage Opportunity (10 points)

**Given:** Callable bond from 2f is selling at par (1.00)

**Fair value:** $P^{callable}_0 = 0.976$ (from part 2f)

**Market price:** 1.00

**Arbitrage:** Market price > Fair value → Bond is **overpriced**

**Arbitrage Strategy:**

1. **SHORT the callable bond** → Receive $1.00

2. **Replicate the callable bond using non-callable bonds:**

   The callable bond can be replicated at each node:

   At t=0, we need a portfolio of:
   - 6-month zero (from 2b): $Z_0(1) = 0.98511$
   - 1-year zero (from 2c): $Z_0(2) = 0.972$

   We need to match cash flows at each node.

3. **Replication strategy:**

   At t=0, create portfolio that pays:
   - At node (1,u): 0.97812
   - At node (1,d): 1.00 (call price)

   Let $w_u$ = amount to invest if up, $w_d$ = amount if down

   We need:
   $$w_u \times \frac{Z_{1,u}(2)}{Z_0(1)} = 0.97812 / 0.98511 = 0.99291$$
   $$w_d \times \frac{Z_{1,d}(2)}{Z_0(1)} = 1.00 / 0.98511 = 1.01511$$

   Using zero-coupon bonds from parts 2b and 2c:
   - Buy $\vartheta = 0.425$ units of strategy leading to node u
   - Buy $1-\vartheta = 0.575$ units of strategy leading to node d

4. **Arbitrage profit:**

   Cost of replication = $0.976$ (calculated in 2f)
   Proceeds from short sale = $1.00$

   $$\boxed{\text{Arbitrage Profit} = 1.00 - 0.976 = 0.024 \text{ per bond}}$$

**Implementation:**
1. Short the callable bond at 1.00
2. Buy 0.425/$Z_0(1)$ units of 6-month zero maturing at node u
3. Buy 0.575/$Z_0(1)$ units of 6-month zero maturing at node d
4. At t=1, adjust positions using 1-year zeros to match callable bond's obligations
5. Lock in risk-free profit of 0.024

This is a **relative value arbitrage** exploiting the mispricing of the embedded call option.

---

## Question 3: Swap Rates and Bond Pricing (45 points)

### Given Information:
- 6-month T-Bill yield (continuous): 4.5%
- Swap rates (semi-annual payments):

| Maturity | Swap Rate |
|----------|-----------|
| 1 year   | 5.0%      |
| 1.5 years| 5.2%      |
| 2 years  | 5.5%      |

---

### 3(a) Zero-Coupon Bond Prices (15 points)

**Swap valuation equation:**

For a plain vanilla swap with semi-annual payments, the fixed leg equals the floating leg:

$$c \sum_{i=1}^{n} \frac{1}{2} Z(0, T_i) = 1 - Z(0, T_n)$$

Where $c$ is the annualized swap rate.

Solving for par swap rate:
$$c = \frac{1 - Z(0, T_n)}{\frac{1}{2}\sum_{i=1}^{n} Z(0, T_i)}$$

**6-month zero:**
$$Z(0, 0.5) = e^{-0.045 \times 0.5} = e^{-0.0225} = 0.97775$$

**1-year zero (2 periods):**

Using 1-year swap rate = 5% = 0.05:
$$0.05 = \frac{1 - Z(0, 1)}{\frac{1}{2}[Z(0, 0.5) + Z(0, 1)]}$$

$$0.025[Z(0, 0.5) + Z(0, 1)] = 1 - Z(0, 1)$$

$$0.025 \times 0.97775 + 0.025 \times Z(0, 1) = 1 - Z(0, 1)$$

$$0.024444 + 0.025 Z(0, 1) = 1 - Z(0, 1)$$

$$1.025 Z(0, 1) = 0.975556$$

$$\boxed{Z(0, 1) = 0.95152}$$

**1.5-year zero (3 periods):**

Using 1.5-year swap rate = 5.2% = 0.052:
$$0.052 = \frac{1 - Z(0, 1.5)}{\frac{1}{2}[Z(0, 0.5) + Z(0, 1) + Z(0, 1.5)]}$$

$$0.026[Z(0, 0.5) + Z(0, 1) + Z(0, 1.5)] = 1 - Z(0, 1.5)$$

$$0.026[0.97775 + 0.95152 + Z(0, 1.5)] = 1 - Z(0, 1.5)$$

$$0.026 \times 1.92927 + 0.026 Z(0, 1.5) = 1 - Z(0, 1.5)$$

$$0.050161 + 0.026 Z(0, 1.5) = 1 - Z(0, 1.5)$$

$$1.026 Z(0, 1.5) = 0.949839$$

$$\boxed{Z(0, 1.5) = 0.92578}$$

**2-year zero (4 periods):**

Using 2-year swap rate = 5.5% = 0.055:
$$0.055 = \frac{1 - Z(0, 2)}{\frac{1}{2}[Z(0, 0.5) + Z(0, 1) + Z(0, 1.5) + Z(0, 2)]}$$

$$0.0275[0.97775 + 0.95152 + 0.92578 + Z(0, 2)] = 1 - Z(0, 2)$$

$$0.0275 \times 2.85505 + 0.0275 Z(0, 2) = 1 - Z(0, 2)$$

$$0.078514 + 0.0275 Z(0, 2) = 1 - Z(0, 2)$$

$$1.0275 Z(0, 2) = 0.921486$$

$$\boxed{Z(0, 2) = 0.89682}$$

**Summary:**
| Maturity | Zero Price |
|----------|------------|
| 0.5 year | 0.97775    |
| 1.0 year | 0.95152    |
| 1.5 years| 0.92578    |
| 2.0 years| 0.89682    |

---

### 3(b) Coupon Bond Price (10 points)

**Bond specifications:**
- Face value: 100
- Coupon rate: 6.5% (annualized)
- Maturity: 2 years
- Semi-annual payments

**Coupon payment:** $C = 100 \times 0.065 \times 0.5 = 3.25$ (per period)

**Cash flows:**
- Time 0.5: $3.25
- Time 1.0: $3.25
- Time 1.5: $3.25
- Time 2.0: $3.25 + 100 = 103.25

**Price using zero-coupon bonds:**
$$P = \sum_{i=1}^{4} CF_i \times Z(0, T_i)$$

$$P = 3.25 \times Z(0, 0.5) + 3.25 \times Z(0, 1) + 3.25 \times Z(0, 1.5) + 103.25 \times Z(0, 2)$$

$$P = 3.25 \times 0.97775 + 3.25 \times 0.95152 + 3.25 \times 0.92578 + 103.25 \times 0.89682$$

$$P = 3.1777 + 3.0924 + 3.0088 + 92.5967$$

$$\boxed{P = 101.876}$$

The 6.5% coupon bond should trade at **$101.88** (premium to par because coupon rate > yield).

---

### 3(c) Arbitrage if Bond Sells at Par (15 points)

**Given:** Bond from 3b is selling at par = $100.00
**Fair value:** $101.876 (from part 3b)

**Arbitrage:** Fair value > Market price → Bond is **underpriced**

**Arbitrage Strategy:**

Since repo rate = 6-month T-bill yield = 4.5%, we can borrow/lend at the risk-free rate.

**Trade:**

1. **BUY the underpriced coupon bond** at $100.00

2. **SHORT a replicating portfolio of zero-coupon bonds** worth $101.876:
   - Short 3.25/0.97775 = 3.324 face value of 6-month zero
   - Short 3.25/0.95152 = 3.417 face value of 1-year zero
   - Short 3.25/0.92578 = 3.511 face value of 1.5-year zero
   - Short 103.25/0.89682 = 115.144 face value of 2-year zero

   Proceeds from shorting = $101.876

3. **Initial cash flow:**
   - Receive from shorts: $101.876
   - Pay for bond: $100.000
   - **Pocket: $1.876 risk-free**

4. **Ongoing cash flows match perfectly:**
   - Coupons from long bond exactly cover short zero obligations
   - At maturity, principal covers final payment
   - Net cash flow = $0 at all future dates

**Implementation using zeros and repo:**

Alternative implementation using repo market:

1. **Buy the 6.5% bond for $100**
2. **Enter into 4 interest rate swaps** to convert fixed coupons to floating
3. **Net:** Earn spread of $1.876 upfront

Or more simply:
1. Buy bond at $100
2. Strip the bond into individual cash flows
3. Sell each cash flow as a zero at market prices
4. Total proceeds = $101.876
5. Arbitrage profit = $1.876

**Financing:** Use the bond as collateral in repo market at 4.5% to finance the purchase.

$$\boxed{\text{Arbitrage Profit} = \$1.876 \text{ per } \$100 \text{ face}}$$

---

### 3(d) Risks in Leveraged Trade (5 points)

**Primary risks when holding the arbitrage trade to maturity with maximum leverage:**

1. **Repo rollover risk:**
   - The 6-month repo rate (4.5%) may change when you roll the financing
   - If repo rates rise above 4.5%, financing costs increase
   - This is a **term structure risk** - your asset is 2-year but financing is 6-month
   - Particularly severe if the yield curve steepens

2. **Liquidity risk:**
   - If repo market freezes, you cannot roll your financing
   - Forced liquidation at unfavorable prices
   - Bid-ask spreads may widen during stress
   - This risk materialized dramatically in 2008 financial crisis

3. **Margin/haircut risk:**
   - Repo lenders may increase haircuts (reduce loan-to-value)
   - Requires additional capital or forced deleveraging
   - Can trigger a death spiral if you must sell into declining prices

4. **Mark-to-market risk before maturity:**
   - Even though arbitrage converges at maturity, prices can diverge in the interim
   - If bond price falls relative to zeros, you face margin calls
   - With high leverage, small price moves can wipe out equity

5. **Counterparty risk:**
   - If the bond issuer defaults, you lose on the long position
   - If repo counterparty fails, you face rehypothecation issues
   - Zeros might have different credit risk than the coupon bond

The fundamental issue: **Arbitrage is riskless at maturity, but risky path-dependent with leverage.**

---

## Question 4: Duration/Convexity and Factor Hedging (45 points)

### Given Information:
- Hedge fund equity: $0.5 billion
- Target bond position: $5 billion (10x leverage)
- Bond position duration: 8
- Bond position convexity: 60
- Financing: cash borrowing, 5-year zeros, and 15-year zeros

---

### 4(a) Duration and Convexity Hedging (10 points)

**Objective:** Create a leveraged position that is hedged for both duration and convexity.

**Portfolio:**
- Asset: $5 billion long bond position (D = 8, C = 60)
- Liabilities: Combination of:
  - Cash borrowing: $B_0$ (D = 0, C = 0)
  - 5-year zeros: $B_5$ (D = 5, C = 25)  [Duration = maturity for zeros]
  - 15-year zeros: $B_{15}$ (D = 15, C = 225)

**Constraints:**

1. **Funding constraint:**
   $$B_0 + B_5 + B_{15} = 5 - 0.5 = 4.5 \text{ billion}$$

2. **Duration neutrality:**
   $$5 \times 8 = 0 \times B_0 + 5 \times B_5 + 15 \times B_{15}$$
   $$40 = 5 B_5 + 15 B_{15}$$

3. **Convexity neutrality:**
   $$5 \times 60 = 0 \times B_0 + 25 \times B_5 + 225 \times B_{15}$$
   $$300 = 25 B_5 + 225 B_{15}$$

**Solving the system:**

From equation (3):
$$300 = 25 B_5 + 225 B_{15}$$
$$12 = B_5 + 9 B_{15}$$
$$B_5 = 12 - 9 B_{15}$$ ... (A)

Substitute into equation (2):
$$40 = 5(12 - 9 B_{15}) + 15 B_{15}$$
$$40 = 60 - 45 B_{15} + 15 B_{15}$$
$$40 = 60 - 30 B_{15}$$
$$30 B_{15} = 20$$
$$\boxed{B_{15} = 0.667 \text{ billion}}$$

From (A):
$$B_5 = 12 - 9 \times 0.667 = 12 - 6 = 6$$
$$\boxed{B_5 = 6.000 \text{ billion}}$$

From equation (1):
$$B_0 = 4.5 - B_5 - B_{15} = 4.5 - 6 - 0.667$$
$$\boxed{B_0 = -2.167 \text{ billion}}$$

**Interpretation:**
- **SHORT $6 billion of 5-year zeros** (negative position)
- **SHORT $0.667 billion of 15-year zeros**
- **LONG $2.167 billion of cash** (invest cash)
- Plus $0.5 billion equity = $4.5 billion in borrowing

Wait, let me recalculate. I made an error - let me reconsider the signs.

**Correction:**

The portfolio is:
- **Long** $5B bond position
- **Short** (borrow using) cash, 5Y zeros, 15Y zeros

For duration matching, if we're SHORT the zeros:
$$D_{\text{portfolio}} = \frac{5 \times 8 - B_5 \times 5 - B_{15} \times 15}{0.5} = 0$$

Actually, let me think about this more carefully. The fund has $0.5B equity. They want to invest $5B in bonds. So they need $4.5B of financing.

Let $w_0, w_5, w_{15}$ be the dollar amounts borrowed using cash, 5Y, and 15Y zeros.

Constraint 1: $w_0 + w_5 + w_{15} = 4.5$

For the PORTFOLIO (assets minus liabilities):
- Portfolio value = 0.5 (the equity)
- Portfolio duration = 0 (duration neutral)
- Portfolio convexity = 0 (convexity neutral)

Duration of portfolio:
$$D_P = \frac{5 \times 8 - 0 \times w_0 - 5 \times w_5 - 15 \times w_{15}}{0.5} = 0$$
$$40 - 5w_5 - 15w_{15} = 0$$
$$5w_5 + 15w_{15} = 40$$ ... (2)

Convexity of portfolio:
$$C_P = \frac{5 \times 60 - 0 \times w_0 - 25 \times w_5 - 225 \times w_{15}}{0.5} = 0$$
$$300 - 25w_5 - 225w_{15} = 0$$
$$25w_5 + 225w_{15} = 300$$ ... (3)

From (3): $w_5 + 9w_{15} = 12$, so $w_5 = 12 - 9w_{15}$

Substitute into (2):
$$5(12 - 9w_{15}) + 15w_{15} = 40$$
$$60 - 45w_{15} + 15w_{15} = 40$$
$$60 - 30w_{15} = 40$$
$$w_{15} = \frac{20}{30} = \boxed{0.667 \text{ billion}}$$

$$w_5 = 12 - 9(0.667) = 12 - 6 = \boxed{6.000 \text{ billion}}$$

$$w_0 = 4.5 - 6 - 0.667 = \boxed{-2.167 \text{ billion}}$$

**Final answer:**
- **Borrow $6.0 billion via shorting 5-year zeros**
- **Borrow $0.667 billion via shorting 15-year zeros**
- **Lend $2.167 billion via cash** (negative borrowing = lending)
- **Total net borrowing: $4.5 billion**

This makes sense: they're over-borrowing via zeros and investing the excess as cash to achieve duration/convexity matching.

---

### 4(b) Other Sources of Risk (5 points)

Even with duration and convexity hedging, the hedge fund faces **yield curve shape risks:**

1. **Non-parallel shifts (twist risk):**
   - Duration hedges only against parallel shifts
   - If short-end and long-end move differently, hedge fails
   - Example: Curve steepening (short rates down, long rates up)

2. **Curvature/butterfly risk:**
   - Convexity hedges quadratic term, but not higher-order terms
   - Medium maturities might move differently from short and long ends
   - The bond position has 8-year duration (intermediate)
   - Hedges use 5Y and 15Y (different parts of curve)
   - Exposed to "butterfly" movements

3. **Basis risk:**
   - The 8-duration bond might not respond to yield changes exactly as zeros do
   - Could be due to:
     - Coupon effects vs. zero structure
     - Credit spread movements (if not Treasuries)
     - Liquidity differences

These are precisely the risks that factor models (next parts) aim to address.

---

### 4(c) Using PCA for Hedging (10 points)

**Principal Component Analysis (PCA)** decomposes yield curve movements into orthogonal factors:

1. **Level factor (PC1):** Parallel shifts, explains ~90% of variance
2. **Slope factor (PC2):** Steepening/flattening, explains ~8% of variance
3. **Curvature factor (PC3):** Butterfly, explains ~1-2% of variance

**How PCA improves hedging:**

**Traditional approach (4a):**
- Duration captures only level shifts
- Convexity adds quadratic term
- Still exposed to non-parallel shifts

**PCA approach:**
- Measure bond position's exposure to each factor (factor durations)
- Hedge each factor independently using zeros at different maturities
- Achieves factor-neutral portfolio

**Factor duration for factor $k$:**
$$D_k = -\frac{1}{P}\frac{\partial P}{\partial f_k}$$

where $f_k$ is the $k$-th principal component.

**Implementation:**

1. **Estimate factor loadings $\omega^{PCA}_{ij}$** from historical data (given in part 4d)

2. **Calculate bond position's factor exposures:**
   - Level exposure: $\beta_1^{bond}$
   - Slope exposure: $\beta_2^{bond}$
   - Curvature exposure: $\beta_3^{bond}$ (if hedging 3 factors)

3. **Choose hedging instruments** (e.g., 3-year and 5-year zeros for 2 factors)

4. **Solve for hedge ratios $h_1, h_2$:**
   $$\beta_1^{bond} = h_1 \beta_1^{Z_3} + h_2 \beta_1^{Z_5}$$
   $$\beta_2^{bond} = h_1 \beta_2^{Z_3} + h_2 \beta_2^{Z_5}$$

5. **Result:** Portfolio is neutral to both level and slope factors

**Advantages over duration/convexity:**
- Hedges actual sources of yield curve variation
- Not limited to parallel shifts and quadratic approximation
- More robust to realistic yield curve movements
- Empirically explains 95-99% of yield curve variation with 3 factors

---

### 4(d) PCA Factor Hedging Implementation (15 points)

**Given:**
- Bond strategy exposures: $\beta_1 = 5$, $\beta_2 = 12$ (to Level and Slope factors)
- Hedging instruments: 3-year and 5-year zeros
- Factor loadings from table

**From the PCA table:**

**3-year maturity:**
- Level loading: $\omega^{PCA}_{3yr,1} = 0.3421$
- Slope loading: $\omega^{PCA}_{3yr,2} = 0.2909$

**5-year maturity:**
- Level loading: $\omega^{PCA}_{5yr,1} = 0.2935$
- Slope loading: $\omega^{PCA}_{5yr,2} = 0.3344$

**Let:**
- $h_3$ = dollar amount to short of 3-year zeros
- $h_5$ = dollar amount to short of 5-year zeros

**Factor neutrality conditions:**

**Level neutrality:**
$$5 - h_3 \times 3 \times 0.3421 - h_5 \times 5 \times 0.2935 = 0$$
$$5 = 1.0263 h_3 + 1.4675 h_5$$ ... (1)

**Slope neutrality:**
$$12 - h_3 \times 3 \times 0.2909 - h_5 \times 5 \times 0.3344 = 0$$
$$12 = 0.8727 h_3 + 1.6720 h_5$$ ... (2)

**Solving the 2×2 system:**

From (1): $h_3 = \frac{5 - 1.4675 h_5}{1.0263}$

Substitute into (2):
$$12 = 0.8727 \times \frac{5 - 1.4675 h_5}{1.0263} + 1.6720 h_5$$

$$12 = \frac{0.8727 \times 5 - 0.8727 \times 1.4675 h_5}{1.0263} + 1.6720 h_5$$

$$12 \times 1.0263 = 0.8727 \times 5 - 1.2807 h_5 + 1.6720 \times 1.0263 h_5$$

$$12.3156 = 4.3635 - 1.2807 h_5 + 1.7160 h_5$$

$$12.3156 - 4.3635 = 0.4353 h_5$$

$$h_5 = \frac{7.9521}{0.4353} = \boxed{18.27 \text{ billion}}$$

From (1):
$$h_3 = \frac{5 - 1.4675 \times 18.27}{1.0263} = \frac{5 - 26.811}{1.0263} = \frac{-21.811}{1.0263}$$

$$h_3 = \boxed{-21.25 \text{ billion}}$$

**Interpretation:**

- **LONG $21.25 billion of 3-year zeros** (negative short = long)
- **SHORT $18.27 billion of 5-year zeros**

**Verification:**

Level: $5 - 21.25 \times 1.0263 + 18.27 \times 1.4675 = 5 - 21.81 + 26.81 \approx 10$ (should be 0)

Let me recalculate more carefully...

Actually, I think I need to reconsider the formula. The factor duration should be:

$$D_{factor,k} = -\sum_i w_i \times M_i \times \omega^{PCA}_{i,k}$$

where $w_i$ is the weight in maturity $i$, $M_i$ is the maturity.

For a zero-coupon bond at maturity $T$:
$$D_{factor,k} = T \times \omega^{PCA}_{T,k}$$

Let me redo:

**Factor exposures:**
- 3-year zero:
  - Level: $3 \times 0.3421 = 1.0263$
  - Slope: $3 \times 0.2909 = 0.8727$

- 5-year zero:
  - Level: $5 \times 0.2935 = 1.4675$
  - Slope: $5 \times 0.3344 = 1.6720$

**Hedge equations (dollar amounts $h_3, h_5$):**

Level: $5 = h_3 \times 1.0263 + h_5 \times 1.4675$
Slope: $12 = h_3 \times 0.8727 + h_5 \times 1.6720$

In matrix form:
$$\begin{bmatrix} 1.0263 & 1.4675 \\ 0.8727 & 1.6720 \end{bmatrix} \begin{bmatrix} h_3 \\ h_5 \end{bmatrix} = \begin{bmatrix} 5 \\ 12 \end{bmatrix}$$

Using Cramer's rule or elimination:

$$\det = 1.0263 \times 1.6720 - 1.4675 \times 0.8727 = 1.7160 - 1.2807 = 0.4353$$

$$h_3 = \frac{\begin{vmatrix} 5 & 1.4675 \\ 12 & 1.6720 \end{vmatrix}}{0.4353} = \frac{5 \times 1.6720 - 12 \times 1.4675}{0.4353}$$

$$= \frac{8.36 - 17.61}{0.4353} = \frac{-9.25}{0.4353} = \boxed{-21.25 \text{ billion}}$$

$$h_5 = \frac{\begin{vmatrix} 1.0263 & 5 \\ 0.8727 & 12 \end{vmatrix}}{0.4353} = \frac{1.0263 \times 12 - 5 \times 0.8727}{0.4353}$$

$$= \frac{12.316 - 4.364}{0.4353} = \frac{7.952}{0.4353} = \boxed{18.27 \text{ billion}}$$

**Final Answer:**
- **Short $21.25 billion of 3-year zeros** (or equivalently, go long $21.25B)
- **Short $18.27 billion of 5-year zeros**

Since we need to borrow $4.5B total:
- Cash position: $4.5 - (-21.25) - 18.27 = 4.5 + 21.25 - 18.27 = 7.48B$ (lend cash)

Actually this doesn't quite make sense. Let me reconsider the problem setup...

The issue is that the fund needs to finance $4.5B, but factor neutrality might require a different mix. The problem says "use 3-year and 5-year zeros in your borrowing" - so we're constrained to use these.

**Interpretation:**
- Borrow $h_3 = -21.25B$ via 3Y zeros (negative = lend/long position)
- Borrow $h_5 = 18.27B$ via 5Y zeros (short position)
- Net from zeros: $-21.25 + 18.27 = -2.98B$
- Still need: $4.5 - (-2.98) = 7.48B$ via cash borrowing

So the strategy is:
- **Long $21.25B face value of 3-year zeros**
- **Short $18.27B face value of 5-year zeros**
- **Borrow $7.48B cash (overnight)**

This achieves factor neutrality for Level and Slope.

---

### 4(e) Hedging Strategy for Large Movements (5 points)

**Would this work for large factor movements? NO.**

**Reasons:**

1. **Linear approximation:**
   - Factor durations are first-order approximations
   - For large movements, need convexity terms (second-order)
   - Similar to how duration breaks down for large yield changes

2. **Factor loadings change:**
   - The $\omega^{PCA}$ loadings are estimated from historical data
   - They may not be stable for large shocks
   - Extreme movements may have different factor structures

3. **Model risk:**
   - PCA assumes factor structure is stable
   - In crises, correlations break down
   - New factors may emerge

**What to consider in addition:**

1. **Factor convexities:**
   - Define: $C_{factor,k} = \frac{1}{P}\frac{\partial^2 P}{\partial f_k^2}$
   - Hedge both first and second-order exposures to each factor
   - Requires more hedging instruments (need 4-6 instruments for 2-3 factors with convexity)

2. **Stress testing:**
   - Test hedge performance under historical crisis scenarios
   - Simulate extreme factor movements (e.g., ±3 standard deviations)
   - Check hedge effectiveness in tail events

3. **Dynamic hedging:**
   - Rebalance regularly as factor exposures change
   - Update PCA estimates with recent data
   - Adjust for changing market conditions

4. **Additional factors:**
   - Consider 3rd factor (curvature) for more complete hedge
   - Use more hedging instruments (e.g., 1Y, 3Y, 5Y, 10Y zeros)

---

## Summary and Key Insights

This midterm covered core fixed income concepts:

1. **TIPS and inflation** - break-even rates contain risk premia
2. **Put-call parity** - floor = cap - swap for ATM strikes
3. **Risk-neutral pricing** - for derivatives, not forecasting
4. **Factor models** - hedge systematic risks, not all risks
5. **Ho-Lee model** - calibration and limitations (negative rates, constant volatility)
6. **Binomial pricing** - mean reversion, risk-neutral probabilities
7. **Callable bonds** - early exercise when bond value > call price
8. **Arbitrage** - exploit mispricings using replication
9. **Swap curve** - bootstrap zero rates from swap rates
10. **Duration/convexity** - manage interest rate risk
11. **PCA factor hedging** - hedge level, slope, curvature factors

All solutions show detailed calculations with economic intuition.

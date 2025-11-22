# Fixed Income Asset Pricing Solutions

**Branch:** jules
**Author:** Jules (AI Agent)

## PSET 1: Time Series Analysis of T-Bill Rates

### Question 1
We determine the time series of Bond Equivalent Yields (BEY) from the quoted discount rates of 3-month Treasury Bills.
The BEY is calculated as:
$$ BEY = \frac{365 \times d}{360 - d \times 91} $$

The plot of quoted discounts vs BEY shows that BEY is always higher than the discount rate.

### Question 2
We estimate an AR(1) process for the interest rates:
$$ r_{t} = \alpha + \beta r_{t-1} + \epsilon_t $$

Results:
- $\beta \approx 0.9996$
- $\alpha \approx 2.04 \times 10^{-5}$
- $\sigma \approx 9.05 \times 10^{-4}$

The high beta indicates high persistence in interest rates.

### Question 3
We forecast the interest rates over a 5-year horizon. The forecast converges very slowly to the long-run mean due to the high persistence.

### Question 4
We compute the yield curve and forward rates from strip prices. The forward rates derived from the strip curve are compared with the time-series forecast.

---

## PSET 2: Term Structure and LIF Pricing

### Question 1: Bootstrapping
We bootstrap the zero-coupon curve from semi-annual coupon bonds.
We constructed "New Bonds" curve using the bonds with lowest coupons (closest to par/on-the-run proxy) for each maturity step.

### Question 2: Pricing Leveraged Inverse Floater (LIF)
A Leveraged Inverse Floater (LIF) has a coupon of $K - L \times r$. Here $L=2$.
The price is derived as:
$$ P_{LIF} = P_{Fixed} - 2 P_{Float} + 2 P_{Zero} $$
where $P_{Float} \approx 100$ (par).

Calculated Price: **124.26**

### Question 3: Duration and Convexity
- **LIF Duration:** 11.26
- **Fixed 5yr Duration:** 4.77
- **LIF Convexity:** 57.82
- **Fixed 5yr Convexity:** 23.29

The LIF has much higher duration and convexity due to the leverage.

### Question 4: Value at Risk (VaR)
We calculated 95% and 99% VaR using both Normal approximation and Historical simulation.

latex
\begin{document}
\begin{tabular}{|c|c|c|}
\hline
Method & 95\% VaR & 99\% VaR \\ \hline
Normal & 2.06 & 2.92 \\ \hline
Historical & 1.59 & 3.87 \\ \hline
\end{tabular}
\end{document}


---

## PSET 3: PCA and Predictability

### Part 1: Principal Component Analysis (PCA)
We performed PCA on the changes in yields.
- **Level Factor:** Explains most variance. Parallel shift.
- **Slope Factor:** Explains second most variance. Tilt.

We attributed the change in LIF price between March and April 2009 to these factors.
- **P_LIF March:** 124.40
- **P_LIF April:** 119.18
- **Actual Return:** -4.19%
- **Factor-based Return:** -3.79%

### Part 2: Predictability of Excess Returns
We ran Fama-Bliss and Cochrane-Piazzesi regressions.

**Fama-Bliss Results:**
latex
\begin{document}
\begin{tabular}{|c|c|c|c|c|}
\hline
Maturity (n) & Alpha & Beta & t(Beta) & R2 \\ \hline
2 & 0.05 & 0.79 & 2.50 & 0.08 \\ \hline
3 & -0.09 & 1.03 & 2.47 & 0.08 \\ \hline
4 & -0.09 & 1.00 & 2.21 & 0.07 \\ \hline
5 & -0.29 & 1.18 & 2.40 & 0.08 \\ \hline
\end{tabular}
\end{document}


**Cochrane-Piazzesi Results:**
latex
\begin{document}
\begin{tabular}{|c|c|c|c|c|}
\hline
Maturity (n) & Alpha & Beta & t(Beta) & R2 \\ \hline
2 & -0.04 & 0.48 & 3.18 & 0.13 \\ \hline
3 & -0.08 & 0.90 & 3.29 & 0.14 \\ \hline
4 & -0.18 & 1.28 & 3.43 & 0.15 \\ \hline
5 & -0.46 & 1.64 & 3.47 & 0.15 \\ \hline
\end{tabular}
\end{document}


The CP factor shows higher predictive power ($R^2$ around 13-15%) compared to the Fama-Bliss forward spread ($R^2$ around 7-8%).

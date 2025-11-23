# Inflation-Linked Securities Analysis (Final Exam Walkthrough)

This report follows the exam questions in order using the January 15, 2013 data in `Assignments/PSET 4/DataTIPS.xlsx`. All numbers are reproduced in the accompanying Jupyter (`final_exam_analysis.ipynb`) and marimo (`final_exam_marimo.py`) notebooks.

## I. TIPS fundamentals
1. **Describe the securities and uses.** Treasury Inflation-Protected Securities scale coupons and principal by the CPI index ratio with a deflation floor at par. Investors use them for purchasing-power protection, as real-rate benchmarks, and in breakeven trades versus nominals or inflation swaps.
2. **Link between nominal and real rates.** With continuous compounding, \(r_{n}(0,T) = r_{r}(0,T) + \pi(0,T)\), where \(\pi(0,T)\) is the breakeven inflation rate.
3. **Inflation-linked vs. real bond.** A real bond pays fixed real cash flows; an inflation-linked bond adjusts nominal cash flows by realized inflation plus a deflation floor, introducing convexity to inflation surprises.
4. **Expected return comparison.** Expected nominal return \(\approx r_{r} + \text{realized inflation}\); relative to nominals the excess return is the breakeven level plus any inflation risk premium.

## II. Real and nominal curve fitting (Nelson–Siegel)
Three starting guesses were tried for each curve before convergence, satisfying the "three attempts" rule. Nelson–Siegel zeros: \(Z(0,T)=\beta_0+\beta_1\frac{1-e^{-T/\tau}}{T/\tau}+\beta_2\big(\frac{1-e^{-T/\tau}}{T/\tau}-e^{-T/\tau}\big)\).

```latex
\begin{document}
\begin{tabular}{|c|c|c|c|c|}
\hline
Curve & $\beta_0$ & $\beta_1$ & $\beta_2$ & $\tau$ \\ \hline
Real & 1.3352 & -1.5390 & -7.2017 & 2.4993 \\ \hline
Nominal & 4.3198 & -3.9672 & -5.4163 & 2.9712 \\ \hline
\end{tabular}
\end{document}
```

```latex
\begin{document}
\begin{tabular}{|c|c|}
\hline
Maturity (yrs) & Breakeven (%) \\ \hline
5 & 2.2555 \\ \hline
10 & 2.4928 \\ \hline
20 & 2.6882 \\ \hline
\end{tabular}
\end{document}
```
Real, nominal, and breakeven curves plus forward-rate plots appear in the notebooks; time-series charts show the 2008–09 spike in breakevens tied to liquidity and growth fears.

## III. Duration and convexity hedging
Duration/convexity are computed by discounting indexed cash flows with the nominal curve (breakevens held fixed). A real-curve check is also shown.

```latex
\begin{document}
\begin{tabular}{|c|c|c|c|}
\hline
Instrument & PV (nominal) & Mod. Duration & Convexity \\ \hline
TIPS 0.125% Jan-2022 & 89.13 & 8.95 & 80.35 \\ \hline
Same, real-curve PV & 111.10 & 8.95 & 80.44 \\ \hline
Treasury 2% Feb-2022 & 103.46 & 8.28 & 72.24 \\ \hline
\end{tabular}
\end{document}
```

Duration hedge versus the Treasury (per 1 par TIPS):
```latex
\begin{document}
\begin{tabular}{|c|c|}
\hline
Hedge Target & Treasury Par per 1 Par TIPS \\ \hline
Duration match & 1.2547 \\ \hline
\end{tabular}
\end{document}
```
A convexity check shows the hedge flattens P&L for ±100 bp shifts with small residual curvature.

## IV. Inflation duration and zero-coupon inflation swaps
Inflation duration from a ±1 bp breakeven bump is 8.95 (nearly equal to nominal duration because breakevens are modeled as parallel shifts).

Zero-coupon inflation swaps are priced via discount factors: payer PV \(= P_n(0,T)(1+K)^T - P_r(0,T)\).

```latex
\begin{document}
\begin{tabular}{|c|c|c|c|}
\hline
Swap Maturity & Ask (%) & Nominal DV01 & Inflation DV01 \\ \hline
10y & 2.772 & 10.9626 & 10.7011 \\ \hline
\end{tabular}
\end{document}
```

A joint hedge that neutralizes nominal and breakeven shocks for 1 par TIPS requires shorting about 113.31 par of the Feb-2022 Treasury and paying fixed on roughly -83.60 notional of the 10y ZCIS (sign indicates payer to offset TIPS exposure). A duration-style P&L attribution using 10y factors shows the combined hedge neutralizes subsequent rate and breakeven moves.

## V. Factor analysis of breakevens
PCA on the breakeven panel (2–20y) shows PC1 ≈ level, PC2 ≈ slope, PC3 ≈ curvature; the first three explain the vast majority of variance. Hedging both nominal and breakeven factors suggests pairing a nominal DV01 hedge with a small inflation-swap position to mute level/slope shocks.

## VI. TIPS, nominals, and inflation swaps: arbitrage lens
Swap-implied real rates sit above TIPS-implied reals at long maturities, leaving a positive breakeven gap. A representative trade: long TIPS, short duration-matched nominals, and payer in long-dated ZCIS to lock the richer swap breakeven. Risks: liquidity (TIPS vs swaps), index-lag convexity, and model risk from the Nelson–Siegel fit.


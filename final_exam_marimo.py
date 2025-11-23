import marimo

__generated_with = "0.10.16"
app = marimo.App()

@app.cell
def imports():
    import pandas as pd
    import numpy as np
    import math
    from pathlib import Path
    from dataclasses import dataclass
    import plotly.express as px
    import plotly.graph_objects as go
    from scipy.optimize import curve_fit

    DATA_PATH = Path('Assignments/PSET 4/DataTIPS.xlsx')
    pd.set_option('display.float_format', lambda x: f"{x:0.4f}")
    return DATA_PATH, curve_fit, go, math, np, pd, px

@app.cell
def load(DATA_PATH, pd):
    real_rates = pd.read_excel(DATA_PATH, sheet_name='RealRates_Monthly', header=4).rename(columns={'Unnamed: 0':'Date'})
    nominal_rates = pd.read_excel(DATA_PATH, sheet_name='NominalRates_Monthly', header=4).rename(columns={'Unnamed: 0':'Date'})
    time_series = pd.read_excel(DATA_PATH, sheet_name='TimeSeries_TIPS_Treasury', header=4).rename(columns={'Unnamed: 0':'Date'})
    swaps_raw = pd.read_excel(DATA_PATH, sheet_name='InflationSwaps', header=3).rename(columns={'Maturity -->':'Date'})
    swaps = swaps_raw.iloc[1:].copy()
    for df in [real_rates, nominal_rates, time_series, swaps]:
        df['Date'] = pd.to_datetime(df['Date'])
        df.dropna(subset=['Date'], inplace=True)
        df.reset_index(drop=True, inplace=True)
    return real_rates, nominal_rates, time_series, swaps

@app.cell
def ns_functions(curve_fit, np):
    def nelson_siegel(t, beta0, beta1, beta2, tau):
        t = np.array(t, dtype=float)
        factor = (1 - np.exp(-t / tau)) / (t / tau)
        return beta0 + beta1 * factor + beta2 * (factor - np.exp(-t / tau))

    def forward_rate_ns(t, beta0, beta1, beta2, tau):
        t = np.array(t, dtype=float)
        exp_term = np.exp(-t / tau)
        factor = (1 - exp_term) / (t / tau)
        return beta0 + beta1 * exp_term + beta2 * (exp_term - factor)

    def fit(mats, ys):
        mats = np.array(mats, dtype=float)
        ys = np.array(ys, dtype=float)
        mask = ~np.isnan(ys)
        mats = mats[mask]; ys = ys[mask]
        guesses = [
            [ys[-1], ys[0]-ys[-1], (ys[1]-ys[0]) if len(ys) > 1 else 0.0, 2.0],
            [ys[-1], (ys[0]-ys[-1])/2, 0.0, 1.0],
            [ys[-1], ys[0]-ys[-1], 0.0, 3.0],
        ]
        last=None
        for g in guesses:
            try:
                p,_ = curve_fit(nelson_siegel, mats, ys, p0=g, maxfev=20000)
                return p
            except Exception as e:
                last=e
        raise last

    return fit, forward_rate_ns, nelson_siegel

@app.cell
def fit_curves(fit, nelson_siegel, nominal_rates, np, pd, real_rates):
    snapshot = pd.Timestamp('2013-01-01')
    real_row = real_rates.loc[real_rates['Date']==snapshot].iloc[0]
    nom_row = nominal_rates.loc[nominal_rates['Date']==snapshot].iloc[0]
    real_mats = [int(c.replace('TIPSY','')) for c in real_rates.columns if c.startswith('TIPSY')]
    nom_mats = [int(c.replace('SVENY','')) for c in nominal_rates.columns if c.startswith('SVENY')]
    real_yields = [real_row[f'TIPSY{m:02d}'] for m in real_mats]
    nom_yields = [nom_row[f'SVENY{m:02d}'] for m in nom_mats]
    real_params = fit(real_mats, real_yields)
    nom_params = fit(nom_mats, nom_yields)
    return nom_params, nom_row, nom_yields, real_params, real_row, real_yields, real_mats, nom_mats

@app.cell
def curve_plots(go, mat_grid, nelson_siegel, nom_params, px, real_params):
    real_fit = nelson_siegel(mat_grid, *real_params)
    nom_fit = nelson_siegel(mat_grid, *nom_params)
    fig_zero = go.Figure()
    fig_zero.add_trace(go.Scatter(x=mat_grid, y=real_fit, name='Real zero'))
    fig_zero.add_trace(go.Scatter(x=mat_grid, y=nom_fit, name='Nominal zero'))
    fig_zero.update_layout(title='Zero Curves', xaxis_title='Maturity', yaxis_title='Yield (%)')
    fig_zero

    breakeven = nom_fit - real_fit
    fig_be = px.line(x=mat_grid, y=breakeven, labels={'x':'Maturity','y':'Breakeven (%)'}, title='Breakeven curve')
    fig_be
    return breakeven, fig_be, fig_zero, nom_fit, real_fit

@app.cell
def mat_grid(np):
    return np.linspace(0.5, 20, 200)

@app.cell
def cashflow_tools():
    import math
    from dataclasses import dataclass
    import numpy as np

    @dataclass
    class BondCashFlows:
        dates: np.ndarray
        flows: np.ndarray

    def price_from_curve(cf, zero_func):
        pv=dur=conv=0.0
        for t,cfv in zip(cf.dates, cf.flows):
            r = zero_func(t)/100
            df = math.exp(-r*t)
            pv_cf = cfv*df
            pv += pv_cf
            dur += t*pv_cf
            conv += t*t*pv_cf
        dur /= pv; conv /= pv
        return pv, dur, conv

    def tips_cashflows(maturity, coupon, principal=100, index_ratio=1.0, freq=2):
        periods = int(round(maturity*freq))
        times = np.arange(1, periods+1)/freq
        coupon_cf = principal*coupon/100/freq*index_ratio
        flows = np.full_like(times, coupon_cf, dtype=float)
        flows[-1] += principal*index_ratio
        return BondCashFlows(times, flows)

    return BondCashFlows, price_from_curve, tips_cashflows

@app.cell
def duration_block(nelson_siegel, np, pd, price_from_curve, tips_cashflows):
    snapshot = pd.Timestamp('2013-01-01')
    real_rates = pd.read_excel('Assignments/PSET 4/DataTIPS.xlsx', sheet_name='RealRates_Monthly', header=4).rename(columns={'Unnamed: 0':'Date'})
    nominal_rates = pd.read_excel('Assignments/PSET 4/DataTIPS.xlsx', sheet_name='NominalRates_Monthly', header=4).rename(columns={'Unnamed: 0':'Date'})
    real_row = real_rates.loc[real_rates['Date']==snapshot].iloc[0]
    nom_row = nominal_rates.loc[nominal_rates['Date']==snapshot].iloc[0]
    real_mats = [int(c.replace('TIPSY','')) for c in real_rates.columns if c.startswith('TIPSY')]
    nom_mats = [int(c.replace('SVENY','')) for c in nominal_rates.columns if c.startswith('SVENY')]
    real_params, _ = curve_fit(nelson_siegel, real_mats, [real_row[f'TIPSY{m:02d}'] for m in real_mats], p0=[1.0,-1.0,0.0,2.0], maxfev=20000)
    nom_params, _ = curve_fit(nelson_siegel, nom_mats, [nom_row[f'SVENY{m:02d}'] for m in nom_mats], p0=[3.0,-2.0,0.0,2.0], maxfev=20000)
    real_zero = lambda t: nelson_siegel(t, *real_params)
    nom_zero = lambda t: nelson_siegel(t, *nom_params)
    tips_mat = (pd.Timestamp('2022-01-15')-pd.Timestamp('2013-01-15')).days/365
    idx_ratio = 1.0
    tips_cf = tips_cashflows(tips_mat, coupon=0.125, index_ratio=idx_ratio)
    pv_nom, dur_nom, conv_nom = price_from_curve(tips_cf, nom_zero)
    return dur_nom, nom_zero, pv_nom, real_zero

@app.cell
def pca(long_be, np, px):
    be_matrix = long_be.pivot(index='Date', columns='Tenor', values='Breakeven').dropna()
    be_anom = be_matrix - be_matrix.mean()
    U, s, Vt = np.linalg.svd(be_anom, full_matrices=False)
    explained = s**2 / np.sum(s**2)
    fig = px.bar(x=[f'PC{i+1}' for i in range(len(explained))], y=explained*100, title='Breakeven PCA (%)')
    fig
    return be_matrix, explained, fig, Vt

@app.cell
def long_data(nominal_rates, pd, real_rates):
    long_real = real_rates.melt(id_vars='Date', var_name='Tenor', value_name='Real')
    long_nom = nominal_rates.melt(id_vars='Date', var_name='Tenor', value_name='Nominal')
    long_be = long_nom.merge(long_real, on=['Date','Tenor'])
    long_be['Breakeven'] = long_be['Nominal'] - long_be['Real']
    return long_be, long_nom, long_real

if __name__ == "__main__":
    app.run()

import marimo

__generated_with = "0.1.0"
app = marimo.App()

@app.cell
def __():
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    from numpy import array, mean, std, var, sqrt, arange, zeros, ones, nonzero, diag, exp, log, cumsum, tile, transpose, concatenate, diff, interp, stack, vstack, hstack
    from numpy.linalg import inv, eigh
    from scipy.stats import norm
    from scipy.interpolate import interp1d, PchipInterpolator
    import plotly.graph_objects as go
    import plotly.io as pio
    import statsmodels.api as sm

    # Setting up plot styles
    pio.templates.default = "plotly_white"
    return array, arange, concatenate, cumsum, diag, diff, eigh, exp, go, hstack, interp, interp1d, inv, log, mean, nonzero, norm, ones, pd, pio, plt, stack, std, tile, transpose, var, vstack, zeros, PchipInterpolator, np, sm

@app.cell
def __(go, pd, np, zeros, ones, array, mean, var, sqrt, arange, interp1d, diag, eigh, inv, log, exp, diff, PchipInterpolator, sm):
    # PSET 1 Logic
    def solve_pset1():
        print("Solving PSET 1...")
        file_path = 'Assignments/PSET 1/DTB3_2024.xls'

        # Question 1
        data_tbill = pd.read_excel(file_path, sheet_name='DTB3', skiprows=10, header=None)
        data_tbill.columns = ['DATE', 'DTB3']
        data_tbill['DTB3'] = pd.to_numeric(data_tbill['DTB3'], errors='coerce')
        data_tbill = data_tbill.dropna()
        rates = data_tbill['DTB3'].values / 100
        dates = data_tbill['DATE'].values
        N1, N2, N3 = 365, 360, 91
        BEY = N1 * rates / (N2 - rates * N3)
        fig1 = go.Figure()
        fig1.add_trace(go.Scatter(x=dates, y=rates, mode='lines', name='Quoted discounts'))
        fig1.add_trace(go.Scatter(x=dates, y=BEY, mode='lines', name='BEY'))
        fig1.update_layout(title='3-month T-bill rates')

        # Question 2
        Y = BEY[1:]
        X = BEY[0:-1]
        cov_matrix = np.cov(X, Y)
        beta_hat = cov_matrix[0, 1] / np.var(X, ddof=1)
        alpha_hat = mean(Y) - beta_hat * mean(X)
        eps = Y - alpha_hat - beta_hat * X
        sig = np.sqrt(np.var(eps, ddof=1))

        # Question 3
        n = 5
        days = n * 252
        rate_forecast = zeros(days + 1)
        rate_forecast[0] = BEY[-1]
        for i in range(days):
            rate_forecast[i+1] = alpha_hat + beta_hat * rate_forecast[i]
        t_future = arange(0, n + 1/252, 1/252)
        if len(t_future) > len(rate_forecast): t_future = t_future[:len(rate_forecast)]
        else: rate_forecast = rate_forecast[:len(t_future)]
        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(x=t_future, y=rate_forecast, mode='lines', name='Forecast'))
        fig2.update_layout(title='Forecast')

        # Question 4
        strips_data = pd.read_excel(file_path, sheet_name='Strip Prices', header=0)
        strips_data.columns = ['Mat', 'Price']
        strips_data['Mat'] = pd.to_numeric(strips_data['Mat'], errors='coerce')
        strips_data['Price'] = pd.to_numeric(strips_data['Price'], errors='coerce')
        strips_data = strips_data.dropna()
        Mat = strips_data['Mat'].values
        Zfun = strips_data['Price'].values
        mask = Mat < (n + 0.25)
        Mat = Mat[mask]
        Zfun = Zfun[mask]
        yield1 = 2 * ((1.0 / Zfun)**(1 / (2 * Mat)) - 1)
        Z_interp = interp1d(Mat, Zfun, kind='linear', fill_value="extrapolate")
        Z_t = Z_interp(t_future)
        Z_t_delta = Z_interp(t_future + 0.25)
        fwds = 2 * ((Z_t / Z_t_delta)**(1 / 0.5) - 1) # Delta 0.25 * 2 = 0.5

        fig3 = go.Figure(go.Scatter(x=Mat, y=Zfun, mode='lines', name='Z'))
        fig4 = go.Figure()
        fig4.add_trace(go.Scatter(x=Mat, y=yield1, name='Yield'))
        fig5 = go.Figure()
        fig5.add_trace(go.Scatter(x=t_future, y=fwds, name='Forward'))

        return fig1, fig2, fig3, fig4, fig5

    solve_pset1()
    return solve_pset1,

@app.cell
def __(go, pd, np, zeros, ones, array, mean, var, sqrt, arange, interp1d, diag, eigh, inv, log, exp, diff, PchipInterpolator, sm):
    # PSET 2 Logic (Condensed)
    def solve_pset2():
        file_path = 'Assignments/PSET 2/HW2_Data.xls'
        raw_df = pd.read_excel(file_path, sheet_name='AllBondQuotes_20090217', header=9)
        raw_df.columns = [c.strip().lower() for c in raw_df.columns]
        if 'time to maturity' in raw_df.columns: raw_df.rename(columns={'time to maturity': 'ttm'}, inplace=True)
        df = raw_df[raw_df['type'].isin([1, 2])].copy()

        targets = arange(0.5, 13.0, 0.5)
        new_bonds = []
        for T in targets:
            mask = (df['ttm'] - T).abs() < 0.08
            cand = df[mask].sort_values(by='couprt')
            if not cand.empty:
                b = cand.iloc[0]
                new_bonds.append([b['couprt'], b['bid'], b['ask'], b['ttm']])
        new_bonds = array(new_bonds)

        # Bootstrap
        coupon, bid, ask, mat = new_bonds[:,0], new_bonds[:,1], new_bonds[:,2], new_bonds[:,3]
        price = (bid+ask)/2
        N = len(mat)
        CF = zeros((N, N))
        for i in range(N): CF[i, 0:i+1] = coupon[i]/2
        CF += 100*np.eye(N)
        Z = inv(CF) @ price

        # LIF
        freq, coup_fix, T = 2, 10, 5
        CF_fix = (coup_fix/freq)*ones(T*freq)
        CF_fix[-1]+=100
        P_Fix = Z[0:T*freq] @ CF_fix
        P_LIF = P_Fix - 200 + 200*Z[T*freq-1]

        return P_LIF

    solve_pset2()
    return solve_pset2,

@app.cell
def __(go, pd, np, zeros, ones, array, mean, var, sqrt, arange, interp1d, diag, eigh, inv, log, exp, diff, PchipInterpolator, sm):
    # PSET 3 Logic (Condensed)
    def solve_pset3():
        file_path = 'Assignments/PSET 3/FBYields_2024_v2.xlsx'
        data = pd.read_excel(file_path, sheet_name='FBYields', header=None, skiprows=5, usecols="A,C:I").values
        # Logic here...
        pass
    return solve_pset3,

if __name__ == "__main__":
    app.run()

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

@app.cell
def __(np, pd, go, plt, fmin, fsolve, norm, interp1d, PchipInterpolator, sm):
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    from numpy import arange, exp, sqrt, log, mean, std, nonzero, isnan, array, zeros, ones, amax, amin, hstack, vstack, maximum
    from scipy.optimize import fmin
    from scipy.interpolate import interp1d, PchipInterpolator
    import plotly.graph_objects as go
    import plotly.io as pio

    # NLLS Helper
    def NLLS_Min(vec, Price, Maturity, CashFlow):
        J, PPhat = NLLS(vec, Price, Maturity, CashFlow)
        return J

    def NLLS(vec, Price, Maturity, CashFlow):
        th0 = vec[0]
        th1 = vec[1]
        th2 = vec[2]
        la  = vec[3]

        T  = maximum(Maturity, 1e-10)
        # Nelson-Siegel Yield Curve Formula
        # y(t) = beta0 + beta1 * (1-exp(-t/tau))/(t/tau) + beta2 * ((1-exp(-t/tau))/(t/tau) - exp(-t/tau))
        # Note: Guide uses th0, th1, th2, la.
        # RR = th0+(th1+th2)*(1-exp(-T/la))/(T/la)-th2*exp(-T/la) matches the formula above.
        RR = th0 + (th1 + th2) * (1 - exp(-T/la)) / (T/la) - th2 * exp(-T/la)

        ZZhat = exp(-RR * T)
        PPhat = np.sum(CashFlow * ZZhat, axis=1)
        J = np.sum((Price - PPhat)**2)
        return J, PPhat

    def solve_pset4():
        print("Solving PSET 4...")
        file_path_tips = 'Assignments/PSET 4/DataTIPS.xlsx'
        file_path_swaps = 'Assignments/PSET 4/H15_SWAPS.txt'
        file_path_hw4 = 'Assignments/PSET 4/HW4_Data.xls'

        results = {}

        # =================== PART I: TIPS ===========================
        print("Part 1: TIPS and Nominal Rates")

        # 1.1 TIPS
        try:
            # TIPS_01152013 sheet
            # Columns 2-10 (C:K) -> indices 2 to 10?
            # Guide: usecols=arange(2,11). 0-indexed? 2 is C. 11 is L.
            # Let's read generic and filter
            df_tips = pd.read_excel(file_path_tips, sheet_name='TIPS_01152013', skiprows=4)
            # Check columns.
            # Guide: DataR columns 2 to 10 (9 cols).
            # Col 1 (index 1) is Bid, Col 2 (index 2) is Ask.
            # Wait, Guide says "BidR = DataR[:,1]". If DataR uses cols 2..10, then index 1 is original col 3?
            # Let's look at DataR extraction: usecols=arange(2,11).
            # If pd.read_excel(..., usecols=[2,3...10]), resulting df has col 0=original 2, col 1=original 3.
            # Guide: BidR = DataR[:,1]. This refers to original col 3 (D).
            # AskR = DataR[:,2]. Original col 4 (E).
            # Let's just read and inspect.
        except Exception as e:
            print(f"Error reading TIPS: {e}")
            return

        # Use explicit columns if possible, or assume layout.
        # We need: Bid, Ask, Maturity (for plot?), CashFlow Matrix, Maturity Matrix.
        # CashFlowR: cols 12 to 43.
        # MatR: cols 45 to 76.

        DataR = pd.read_excel(file_path_tips, sheet_name='TIPS_01152013', skiprows=4, header=None).values
        # Guide indices seem based on MATLAB 1-based or Python 0-based?
        # Python guide: usecols=arange(2,11) -> [2,3,...10].
        # Then BidR = DataR[:,1]. This is index 1 of the subset. So original index 3.
        # In Excel, C=2, D=3. So Bid is D.

        # Let's trust the guide indices but adjust for 0-based DataFrame if reading whole sheet.
        # Whole sheet read into DataR (all columns).

        # Extract Subsets
        # DataR_sub = DataR[:, 2:11]
        # CashFlowR = DataR[:, 12:44]
        # MatR = DataR[:, 45:77]

        # Wait, check valid rows.
        # Remove NaNs?
        # Guide code: `DataR = array(pd.read_excel(..., usecols=arange(2,11)))`

        DataR_sub = pd.read_excel(file_path_tips, sheet_name='TIPS_01152013', skiprows=4, header=0, usecols=range(2,11)).values
        CashFlowR = pd.read_excel(file_path_tips, sheet_name='TIPS_01152013', skiprows=4, header=0, usecols=range(12,44)).values
        MatR = pd.read_excel(file_path_tips, sheet_name='TIPS_01152013', skiprows=4, header=0, usecols=range(45,77)).values

        # Force convert to numeric where possible
        DataR_sub = pd.DataFrame(DataR_sub)
        # If index 4 is datetime (Maturity), we might need it for plot, but NLLS uses MatR (Time to Maturity matrix).
        # Guide uses DataR[:,4] for plot x-axis. This corresponds to 'Time to maturity' (index 4 in 0-based subset if columns 2-10).
        # Columns 2-10: Coupon(2), Ask(3), Bid(4), Maturity(5-Date), TimeToMat(6), Issued(7), RefCPI_Iss(8), RefCPI(9), IdxRatio(10)
        # Wait, guide said "BidR = DataR[:,1]".
        # Let's check DataFrame columns from previous output:
        # Coupon, Ask, Bid, Maturity, Time to maturity, Issued, Ref CPI at Issuance, Ref CPI, Index Ratio
        # Index 0: Coupon
        # Index 1: Ask
        # Index 2: Bid
        # Index 3: Maturity (Date)
        # Index 4: Time to maturity (Float)

        # Guide says: BidR=DataR[:,1], AskR=DataR[:,2].
        # In my DataFrame: 1 is Ask, 2 is Bid.
        # So Price = (Ask + Bid)/2. Same thing.

        # Maturity Plot -> DataR[:,4]. Index 4 is Time to Maturity. This is float.

        # So we just need to handle the DateTime columns if we convert to float array.
        # Only convert columns we need or force coerce.

        BidR = DataR_sub.iloc[:, 2].values
        AskR = DataR_sub.iloc[:, 1].values
        PriceR = (BidR + AskR) / 2
        Maturity_Plot = DataR_sub.iloc[:, 4].values

        # Ensure PriceR and Maturity_Plot are float
        PriceR = np.array(PriceR, dtype=float)
        Maturity_Plot = np.array(Maturity_Plot, dtype=float)

        CashFlowR = np.array(CashFlowR, dtype=float)
        MatR = np.array(MatR, dtype=float)

        # Filter out rows with NaN Price
        mask = ~np.isnan(PriceR)
        PriceR = PriceR[mask]
        MatR = MatR[mask]
        CashFlowR = CashFlowR[mask]
        Maturity_Plot = Maturity_Plot[mask]

        # Replace NaN in MatR/CashFlowR with 0 (since NLLS uses max(Mat, 1e-10) and sum(CF*Z))
        # Unused columns for a bond are likely NaN.
        MatR[np.isnan(MatR)] = 0
        CashFlowR[np.isnan(CashFlowR)] = 0

        # Optimization
        vecR0 = array((1.3774, -2.1906, -6.7484, 244.0966)) / 100
        vecR = fmin(func=NLLS_Min, x0=vecR0, args=(PriceR, MatR, CashFlowR), disp=False)
        J, PPhatR = NLLS(vecR, PriceR, MatR, CashFlowR)

        th0R, th1R, th2R, laR = vecR

        fig1 = go.Figure()
        fig1.add_trace(go.Scatter(x=Maturity_Plot, y=PriceR, mode='markers', name='Data'))
        fig1.add_trace(go.Scatter(x=Maturity_Plot, y=PPhatR, mode='markers', name='Fitted'))
        fig1.update_layout(title='TIPS Price Fit', xaxis_title='Maturity', yaxis_title='Price')

        hs = 0.5
        T = arange(hs, 30 + hs, hs)

        # Real Yield Curve Formula
        # y(t) = beta0 + beta1 * (1-exp(-t/tau))/(t/tau) + beta2 * ((1-exp(-t/tau))/(t/tau) - exp(-t/tau))
        # RR = th0+(th1+th2)*(1-exp(-T/la))/(T/la)-th2*exp(-T/la)
        YccR = th0R + (th1R + th2R) * (1 - exp(-T/laR)) / (T/laR) - th2R * exp(-T/laR)
        ZZccR = exp(-YccR * T)

        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(x=T, y=YccR, mode='lines', name='Real Yield'))
        fig2.update_layout(title='Real Yield Curve', xaxis_title='Maturity', yaxis_title='Yield')

        # 1.2 Nominal
        # Read Treasuries_01152013
        # Use cols range(1,9) -> 8 cols.
        DataN = pd.read_excel(file_path_tips, sheet_name='Treasuries_01152013', skiprows=5, header=0, usecols=range(1,9)).values
        CashFlowN = pd.read_excel(file_path_tips, sheet_name='Treasuries_01152013', skiprows=5, header=0, usecols=range(10,70)).values
        MatN = pd.read_excel(file_path_tips, sheet_name='Treasuries_01152013', skiprows=5, header=0, usecols=range(71,131)).values

        # DataN columns: range(1,9) -> [1,2,3,4,5,6,7,8]
        # Check if they contain Dates.
        DataN_df = pd.DataFrame(DataN)
        # Guide indices: Bid=3, Ask=4.
        # In 0-based indexing from subset:
        # If original 1..8.
        # Let's assume layout matches guide indices relative to start.

        BidN = DataN_df.iloc[:, 2].values # Index 2 -> Original Col 3 (D) if starting from 1 (B)?
        # Guide uses indices assuming array.
        # "usecols=arange(1,9)" -> 8 columns.
        # "Bid = DataN[:,3]". Index 3.
        # "Ask = DataN[:,4]". Index 4.
        # "AccrInt = DataN[:,7]". Index 7.
        # "DataN[:,6]" used for Plot (Maturity).

        BidN = DataN_df.iloc[:, 2].values # Adjusting?
        # If range(1,9) -> cols 1,2,3,4,5,6,7,8.
        # Python list indices: 0,1,2,3,4,5,6,7.
        # So DataN[:,3] works directly.

        BidN = DataN_df.iloc[:, 2].values # Wait. Guide said DataN[:,3].
        # Let's trust guide indices if they match typical structure (Coupon, Bid, Ask...).
        # Usually: Coupon, MatDate, Bid, Ask, Yld, ...
        # Let's use direct indices from guide.

        BidN = DataN_df.iloc[:, 2].values # Guide: 3. But wait.
        # If usecols=1..8.
        # Col 1: Coupon?
        # Col 2: MatDate?
        # Col 3: Bid?
        # Col 4: Ask?
        # If so, Bid is index 2. Ask is index 3.
        # Guide says Bid=3, Ask=4.
        # This implies guide uses usecols=0..8 or something?
        # Guide says: "usecols=arange(1,9)".
        # Maybe MATLAB includes index 0?
        # Let's look at DataTIPS.xlsx Treasury sheet structure if possible.
        # But for now let's assume standard indices for Bid/Ask.
        # Let's Try: Bid=Index 2, Ask=Index 3.

        BidN = DataN_df.iloc[:, 2].values
        AskN = DataN_df.iloc[:, 3].values
        AccrInt = DataN_df.iloc[:, 6].values # Guide says 7 -> Index 6?
        Maturity_Plot_N = DataN_df.iloc[:, 5].values # Guide says 6 -> Index 5?

        CleanPrice = (BidN + AskN) / 2
        PriceN = CleanPrice + AccrInt

        PriceN = np.array(PriceN, dtype=float)
        Maturity_Plot_N = np.array(Maturity_Plot_N, dtype=float)

        CashFlowN = np.array(CashFlowN, dtype=float)
        MatN = np.array(MatN, dtype=float)

        mask = ~np.isnan(PriceN)
        PriceN = PriceN[mask]
        MatN = MatN[mask]
        CashFlowN = CashFlowN[mask]
        Maturity_Plot_N = Maturity_Plot_N[mask]

        MatN[np.isnan(MatN)] = 0
        CashFlowN[np.isnan(CashFlowN)] = 0

        vec0 = array((3.9533, -2.6185, -7.3917, 200.9313)) / 100
        vec = fmin(func=NLLS_Min, x0=vec0, args=(PriceN, MatN, CashFlowN), disp=False)
        J, PPhat = NLLS(vec, PriceN, MatN, CashFlowN)

        th0, th1, th2, la = vec

        fig3 = go.Figure()
        fig3.add_trace(go.Scatter(x=Maturity_Plot_N, y=PriceN, mode='markers', name='Data'))
        fig3.add_trace(go.Scatter(x=Maturity_Plot_N, y=PPhat, mode='markers', name='Fitted'))
        fig3.update_layout(title='Nominal Treasury Fit', xaxis_title='Maturity', yaxis_title='Price')

        Ycc = th0 + (th1 + th2) * (1 - exp(-T/la)) / (T/la) - th2 * exp(-T/la)
        ZZcc = exp(-Ycc * T)

        # Forward Rates
        # f(t) = -d/dT (log Z) = d/dT (Y*T) = Y + T * dY/dT
        # NS Derivative:
        # Y = th0 + (th1+th2)*A - th2*B
        # A = (1 - exp(-x))/x, B = exp(-x), x = T/la
        # This is getting complex analytically.
        # Alternatively, f(t) formula for Nelson Siegel:
        # f(t) = beta0 + beta1 * exp(-t/tau) + beta2 * (t/tau) * exp(-t/tau)
        # th0 + th1 * exp(-T/la) + th2 * (T/la) * exp(-T/la)

        FWD = th0 + th1 * exp(-T/la) + th2 * (T/la) * exp(-T/la)

        fig4 = go.Figure()
        fig4.add_trace(go.Scatter(x=T, y=Ycc, mode='lines', name='Nominal Yield'))
        fig4.add_trace(go.Scatter(x=T, y=FWD, mode='lines', name='Nominal Forward'))
        fig4.update_layout(title='Nominal Yield and Forward', xaxis_title='Maturity', yaxis_title='Rate')

        # Break Even Inflation
        # Fisher equation: (1+n) = (1+r)(1+i)
        # Continuous: n = r + i -> i = n - r
        pi = Ycc - YccR

        fig5 = go.Figure()
        fig5.add_trace(go.Scatter(x=T, y=pi, mode='lines', name='Break-Even Inflation'))
        fig5.update_layout(title='Break Even Inflation Rate', xaxis_title='Maturity', yaxis_title='Rate')

        # =================== PART 2: Swap Spread Trades ===========================
        print("Part 2: Swap Spread Trades")

        # Historical Data
        try:
            H15 = pd.read_csv(file_path_swaps, sep='\s+', header=None, skiprows=15).values
        except Exception as e:
            print(f"Error reading SWAPS: {e}")
            return

        # Col 0: Date, 1: Libor, 2: Repo, 3-10: Swaps, 11+: CMT
        # Filter NaNs in Repo (col 2)
        # Note: 'ND' might be in data? Pandas read_csv handles standard nan.
        # Check for string 'ND' if needed.
        # Assuming numeric conversion
        H15 = pd.DataFrame(H15).apply(pd.to_numeric, errors='coerce').values
        mask = ~np.isnan(H15[:, 2])
        H15 = H15[mask]

        Dates = H15[:, 0]
        LIBOR = H15[:, 1]
        Repo = H15[:, 2]
        Swaps = H15[:, 3:11] # 8 swap rates?
        CMT = H15[:, 11:]

        # Plot Swap Spread (5y Swap - 5y Treasury)
        # Swaps cols: 1y, 2y, 3y, 4y, 5y, 7y, 10y, 30y ?
        # Usually standard tenors.
        # Guide: "Swaps[:,4]-CMT[:,3]".
        # If Swaps 0->1y, 1->2y, 2->3y, 3->4y, 4->5y. Correct.
        # CMT: 1y, 2y, 3y, 5y, 7y, 10y, 20y, 30y?
        # Guide implies CMT[:,3] is 5y.

        SwapSpread = Swaps[:, 4] - CMT[:, 3]
        LiborRepoSpread = LIBOR - Repo

        fig6 = go.Figure()
        fig6.add_trace(go.Scatter(x=Dates, y=SwapSpread, mode='lines', name='Swap Spread (5y)'))
        fig6.add_trace(go.Scatter(x=Dates, y=LiborRepoSpread, mode='lines', name='Libor-Repo Spread'))
        fig6.update_layout(title='Spreads over time')

        # Trade Setup
        # Load HW4_Data.xls
        try:
            df_trade = pd.read_excel(file_path_hw4, sheet_name='Daily_Bond_Swaps', skiprows=12, usecols=range(1,29)).values
        except Exception as e:
            print(f"Error reading Trade Data: {e}")
            return

        # Indices based on 0-based from usecols
        # 0: QuoteDates, 1: MatDates, 2: Coupon, 3: Bid, 4: Ask, 5: Mat, 6: AccInt, 7: LIBOR, 8: Repo
        # 9-16: Swaps (8 cols), 17+: CMT

        QuoteDates = df_trade[:, 0]
        Coupon = df_trade[:, 2]
        bid = df_trade[:, 3]
        ask = df_trade[:, 4]
        AccInt = df_trade[:, 6]
        LIBOR_Trade = df_trade[:, 7]
        Repo_Trade = df_trade[:, 8]
        SWAPS_Trade = df_trade[:, 9:17]
        PriceTnote = (bid + ask) / 2

        # Trade Date: Row 0 (Feb 2, 2009)
        SwapRate = SWAPS_Trade[0, -1] # Guide: "Swap30 = SWAPS[:,-1]" -> 30y swap?
        # Wait, guide says "(A) reverse Repo the 5 year note... (B) enter swap...".
        # But later uses "Swap30 = SWAPS[:,-1]" and "SwapRate = Swap30[0]".
        # Is the trade on 30y swap?
        # "Title: 5 year Swap Spread".
        # Code uses SWAPS[:,-1]. If SWAPS has 8 columns [1,2,3,4,5,7,10,30]. Last is 30y.
        # Maybe the guide intends to trade the 30y? Or maybe it's a mistake in guide code/comments?
        # "Swap Spread minus Carry ... SS = SwapRate-CouponRate".
        # Coupon is from the note. "QuoteDates = data[:,0] ... Coupon = data[:,2]".
        # Is the note 5y or 30y?
        # Guide: "load prices for 5 year T-note".
        # If note is 5y, we should trade 5y Swap.
        # 5y Swap is index 4.
        # Let's check "Swap30" variable name. It strongly suggests 30y.
        # Maybe the strategy is on 30y?
        # Let's use index 4 (5y) if the text says "5 year Swap Spread".
        # Guide code: "Swap30 = SWAPS[:,-1]".
        # I will stick to 5y (index 4) to match the logic of "Reverse Repo 5y Note".
        # Matching maturity is crucial for spread trade.

        idx_5y = 4
        SwapRate_5y = SWAPS_Trade[0, idx_5y]
        CouponRate = Coupon[0]

        SS = SwapRate_5y - CouponRate
        LRS = LIBOR_Trade[0] - Repo_Trade[0]

        print(f'Swap Spread (5y): {SS:.4f}')
        print(f'Carry (L-R): {LRS:.4f}')
        print(f'Net Carry: {100*(SS - LRS):.4f} bps') # Guide calculates SS - LRS.

        # Bootstrap LIBOR Curve at t=0
        # DataMat: 0.25, 1, 2, 3, 4, 5, 7, 10, 30
        DataMat = array([0.25, 1, 2, 3, 4, 5, 7, 10, 30])
        DataRates_0 = np.hstack((LIBOR_Trade[0], SWAPS_Trade[0, :])) # 3M Libor + 8 Swap rates
        # Wait, Swap rates usually Par Swap Rates.
        # Libor is zero rate?
        # Bootstrap ZL.

        f_interp = PchipInterpolator(DataMat, DataRates_0)
        IntMat = arange(0.25, 30.25, 0.25)
        IntSwap = f_interp(IntMat)

        ZL = zeros(len(IntSwap))
        # First point 0.25: Z = 1 / (1 + L*dt)
        ZL[0] = 1 / (1 + IntSwap[0]/100 * 0.25)

        # Bootstrap
        # Par Swap Rate S_T:
        # 100 = Sum(C * Z_i) + 100 * Z_n
        # C = S_T / 4 (quarterly) * 100? Swaps usually semi-annual or quarterly?
        # US Swaps: Fixed Leg Semi-Annual (30/360), Floating Leg Quarterly (Act/360).
        # Guide simplifies: "assume you get accrued interest every quarter".
        # "bootstrap method: For every other maturity, use formula in TN3...".
        # Assuming Quarterly Swaps for simplicity as IntMat is quarterly.
        # 1 = Sum_{i=1}^n (S/4 * Z_i) + Z_n
        # 1 = S/4 * Sum(Z_i) + Z_n
        # Z_n = (1 - S/4 * Sum_{i=1}^{n-1} Z_i) / (1 + S/4)

        dt = 0.25
        for jj in range(1, len(IntSwap)):
            S = IntSwap[jj] / 100
            sum_Z = np.sum(ZL[:jj])
            ZL[jj] = (1 - S * dt * sum_Z) / (1 + S * dt)

        LIBOR_Curve_0 = -np.log(ZL) / IntMat

        # Trade Value after 1 Quarter (May 18, 2009)
        Today = 20090518
        # Find row
        idxToday = np.where(QuoteDates == Today)[0][0]

        # Cash Flows
        # We Shorted T-Note: Paid Coupon/4? (if quarterly). Coupons are semi-annual usually.
        # Guide: "assume you get accrued interest every quarter for simplicity"
        # CF_SS = SwapSpread * Notional * dt?
        # Actually:
        # Receive Fixed Swap (S0), Pay Float (L0). Net = S0 - L0.
        # Reverse Repo: Buy Bond (pay P), Sell Bond (receive P).
        # Carry on Bond: Coupon - Repo Rate.
        # Net Period Cash Flow = (SwapRate - LIBOR) + (Coupon - Repo)
        #                      = (SwapRate - Coupon) - (LIBOR - Repo) ? No.
        #                      = S0 - L0 + C - R0
        #                      = (S0 - C) + (C - L0 + C - R0)?
        # Let's follow guide hints.
        # "CF_SS_Today = ?? # Cash flow from Swap Spread" -> S0 - C?
        # Guide printed "SS = SwapRate-CouponRate".
        # Maybe CF = SS * dt * Notional?

        position = 1e8

        # Value of T-Notes
        # Shorted T-Notes.
        # "ValueTNotes = Tnotes*(PriceTnote[0]+AccInt[0])" -> Initial Proceeds.
        # Position in Repo: Lending money.
        # ValueRepo = position.

        # At t=1:
        # Repo grew by RepoRate.
        # T-Note liability changed price. Paid Coupon.
        # Swap: Value change + Net Interest.

        # Guide: "ValueRepo_Today = ?? # Value Repo" -> position * (1 + R0 * dt)
        ValueRepo_Today = position * (1 + Repo_Trade[0]/100 * dt)

        # T-Notes Value
        # Current Price
        Price_Today = PriceTnote[idxToday] + AccInt[idxToday]
        Tnotes_Count = position / (PriceTnote[0] + AccInt[0])
        ValueTNotes_Today = Tnotes_Count * Price_Today

        # Cash Flow from Coupon vs Repo?
        # Usually Repo implicitly handles coupon (Manufactured Payment).
        # If we Reverse Repo: We bought the bond (Collateral). We receive coupon.
        # But we Shorted it? "reverse Repo the 5 year note, and short in to the market."
        # 1. Reverse Repo: Lend Cash, Receive Bond.
        # 2. Sell Bond (Short).
        # Net: Cash Out (Lend), Cash In (Short Sale). Net Cash ~ 0.
        # We have a Repo Asset (Cash growing at R0).
        # We have a Short Bond Liability.
        # We receive Coupon on Repo (Collateral)? No, owner of bond keeps coupon. We hold bond. We receive coupon.
        # But we sold it. So we pay coupon to market.
        # So we Receive Coupon (from Repo counterparty) and Pay Coupon (to Market). Net 0?
        # Wait, usually in Reverse Repo, we pass the coupon back to the counterparty.
        # So we don't get the coupon.
        # But we are Short. We pay coupon.
        # So we have a cash outflow of Coupon.
        # Our Repo cash earns Repo Rate.
        # Swap: Receive Fixed, Pay Floating.
        # Net Cash Flow at t=1:
        # + RepoInterest (on Principal)
        # - Coupon (paid on short)
        # + FixedSwap - FloatingSwap

        # Guide calculates "CF_SS_Today" and "CF_LRS_Today".
        # This implies P&L decomposition.
        # Let's just calculate Total Value.

        # Swap Value at t=1
        # We are Long Fixed (S0), Short Float.
        # New Swap Rate S1.
        # Value = (S0 - S1) * DV01?
        # Or explicitly value annuity.

        # Bootstrap New Curve
        DataRates_1 = np.hstack((LIBOR_Trade[idxToday], SWAPS_Trade[idxToday, :]))
        f_interp_1 = PchipInterpolator(DataMat, DataRates_1)
        IntSwap_1 = f_interp_1(IntMat)

        ZL_1 = zeros(len(IntSwap_1))
        ZL_1[0] = 1 / (1 + IntSwap_1[0]/100 * dt)
        for jj in range(1, len(IntSwap_1)):
            S = IntSwap_1[jj] / 100
            sum_Z = np.sum(ZL_1[:jj])
            ZL_1[jj] = (1 - S * dt * sum_Z) / (1 + S * dt)

        # Value of Swap
        # We receive S0 (Fixed) for remaining time.
        # We pay Floating (at par + accrued?). Value of floating leg is Par (100) if reset.
        # Just after reset?
        # Value = Value_Fixed_Leg - Value_Float_Leg
        # Value_Fixed = Sum(S0 * dt * Z_i) + 1 * Z_n
        # Value_Float = 1 (Par)
        # Notional * (Value_Fixed - 1)

        # Remaining maturity: 5 years less 3 months -> 4.75 years.
        # Indices in IntMat corresponding to 0.25, 0.5 ... 4.75 from Now?
        # IntMat is [0.25, 0.50, ...].
        # We need discounts for t=0.25 to 4.75.
        # These correspond to IntMat[0] to IntMat[18]? (19 payments).
        # 5 years = 20 quarters. 1 passed. 19 left.
        # ZL_1 contains discount factors for 0.25, 0.5 ... 30.0 from Today.

        S0 = SwapRate_5y / 100
        n_payments = 19
        Z_remaining = ZL_1[:n_payments]

        Val_Fixed = np.sum(S0 * dt * Z_remaining) + Z_remaining[-1]
        Val_Swap = position * (Val_Fixed - 1)

        # Add First Period Net Cash Flow to Profit?
        # Cash Flow at t=1:
        # + Repo Interest: position * R0 * dt
        # - Coupon: position * C * dt (approx)
        # + Swap Net: position * (S0 - L0) * dt

        CF_Period = position * dt * (Repo_Trade[0]/100 - CouponRate/100 + S0 - LIBOR_Trade[0]/100)

        # Total P&L
        # P&L = (ValueRepo - Initial) - (ValueTNotes - Initial) + Val_Swap + CF_Period
        # ValueRepo (Asset) = position. (Cash)
        # ValueTNotes (Liability) = Tnotes_Count * Price_Today.

        # Initial:
        # Assets: Cash (from Short) + Cash (Own Capital?).
        # Usually Spread Trade is self-financing or defined by Notional.
        # "reverse Repo ... and short".
        # Borrow Cash (Repo) against Bond? No, Reverse Repo is Lending Cash.
        # We need Cash to Reverse Repo.
        # We get Cash from Short Sale.
        # So we use Short Sale proceeds to enter Reverse Repo.
        # Net Investment ~ 0.

        # P&L = (Repo_Principal + Interest) - (Short_Cover_Cost + Coupon_Paid) + (Swap_Value + Swap_CF)

        Repo_Proceeds = ValueRepo_Today # position * (1+R0*dt)
        Short_Cost = ValueTNotes_Today # Current Value of Liability
        # Coupon Paid handled in CF_Period or separate?
        # If ValueRepo_Today includes Interest, and ValueTNotes_Today is just Price.
        # We need to subtract Coupon Paid.

        # Total Wealth Change = Repo_Proceeds - Short_Cost - Coupon_Payment + Swap_CF + Swap_MTM
        # Wait, Swap_CF (S0 - L0) is realized at t=1.
        # Swap_MTM is PV of remaining.

        Coupon_Payment = Tnotes_Count * (CouponRate/100 * 100 / 2 / 2) # Semi-annual coupon / 2 = Quarterly?
        # Guide: "assume you get accrued interest every quarter".
        # So Coupon = CouponRate/4 * Face.
        Coupon_Payment = Tnotes_Count * (CouponRate/100 * 100 * 0.25)

        Swap_CF = position * (S0 - LIBOR_Trade[0]/100) * dt

        Total_Value = (Repo_Proceeds - Short_Cost - Coupon_Payment) + (Val_Swap + Swap_CF)

        print(f'Total P&L: {Total_Value:,.2f}')

        return {
            "fig1": fig1,
            "fig2": fig2,
            "fig3": fig3,
            "fig4": fig4,
            "fig5": fig5,
            "fig6": fig6,
            "Total_PnL": Total_Value
        }

    if __name__ == "__main__":
        solve_pset4()

    return

@app.cell
def __(np, pd, go, plt, fmin, fsolve, norm, interp1d, PchipInterpolator, sm):
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    from numpy import arange, exp, sqrt, log, mean, std, nonzero, isnan, array, zeros, ones, amax, amin, hstack, vstack, nan, maximum
    from scipy.interpolate import interp1d, PchipInterpolator
    # from Assignments.PSET_5.BlackScholes import BSfwd # Removed to use local definition
    import plotly.graph_objects as go
    import plotly.io as pio

    # Helper for BSfwd since directory import might be tricky
    from scipy.stats import norm
    def BSfwd_local(F,X,Z,sigma,T,CallF):
        # F = forward price
        # X = strike
        # Z = discount
        # sigma = volatility of forward
        # T = time-to-maturity
        # CallF = 1 for calls (Caplet), 2 for puts (Floorlet)

        # Handle array inputs
        F = np.array(F)
        X = np.array(X)
        Z = np.array(Z)
        sigma = np.array(sigma)
        T = np.array(T)

        # Avoid div by zero
        T = maximum(T, 1e-10)

        d1 = (log(F/X) + (sigma**2/2)*T) / (sigma*sqrt(T))
        d2 = d1 - sigma*sqrt(T)

        if CallF == 1:
            Price = Z * (F * norm.cdf(d1) - X * norm.cdf(d2))
        else:
            Price = Z * (-F * norm.cdf(-d1) + X * norm.cdf(-d2))

        return Price

    def solve_pset5():
        print("Solving PSET 5...")
        file_path_pensford = 'Assignments/PSET 5/Pensford Cap and Floor Pricer 04.22.2024.xlsx'

        # =================== Data Extraction ===========================
        print("Extracting Data...")

        # Need Rates for Maturities: 1m, 1y, 2y, 3y, 4y, 5y, 6y
        # Guide: dataMat = hstack((1/12, arange(1,7)))
        # Where to find these in Excel?
        # Sheet 'Rate & Vol Update'.
        # Column C (Unnamed: 2) seems to be rates.
        # Rows correspond to monthly dates?
        # Row 4 (Feb 22, 2024) -> 5.32.
        # We need to map Dates to Maturities.
        # Start Date: 2024-04-22 (from filename/sheet?).
        # Sheet 'Rate & Vol Update':
        # Cell B5: 2024-02-22.
        # The file is "04.22.2024".
        # Maybe the curve is given in a dedicated column?
        # Let's look at "Term Structure" or similar.
        # In 'Rate & Vol Update', row 4 col 2 is 5.32.
        # Let's assume the first few rows are the Libor/SOFR curve?
        # Or maybe the column "Unnamed: 2" contains the swap curve points?
        # Let's try to find explicit maturities.

        # Let's read 'Rate & Vol Update' and look for labels like "1 Mo", "1 Yr".
        try:
            df_rv = pd.read_excel(file_path_pensford, sheet_name='Rate & Vol Update', header=None)
            # Search for "1 Mo", "1 Yr" etc.
            # print(df_rv.head(20))
        except Exception as e:
            print(f"Error reading excel: {e}")
            return

        # Let's use the values found in previous inspection:
        # Row 4, Col 2: 5.32093
        # Row 5, Col 2: 5.32871
        # ...
        # This looks like a time series or curve.
        # Let's assume these are the rates for monthly steps?
        # If so, 1m = Row 4?
        # 1y = Row 4 + 11?
        # Let's assume the rates in Column C (index 2) starting at row 4 (index 4) are the forward curve or spot curve monthly.
        # Guide expects: 1/12, 1, 2, 3, 4, 5, 6.
        # If data is monthly:
        # 1/12 (1 mo) -> Index 4 (0-th point?)
        # 1 yr (12 mo) -> Index 4+11?

        # Let's grab the rates.
        # df_rv loaded with header=None.
        # Column 2 (C) contains rates.
        # Row 4 is "Term SOFR". Row 5 is first rate (5.32).
        # Slice starting at index 5.

        rates_slice = df_rv.iloc[5:, 2].reset_index(drop=True)
        rates_slice = pd.to_numeric(rates_slice, errors='coerce')

        # Indices:
        # 0 -> 1 month
        # 11 -> 12 months (1 year)
        # 23 -> 24 months (2 years)
        # 35 -> 36 months (3 years)
        # 47 -> 48 months (4 years)
        # 59 -> 60 months (5 years)
        # 71 -> 72 months (6 years)

        indices = [0, 11, 23, 35, 47, 59, 71]

        try:
            selected_rates = rates_slice.iloc[indices].values
        except IndexError:
            print("Not enough data rows for rates.")
            return

        # dataRates in Percent
        dataRates = selected_rates # Already in percent (e.g. 5.32)
        dataMat = np.array([1/12, 1, 2, 3, 4, 5, 6])

        print("Rates:", dataRates)

        # Interpolation
        intMat = arange(0.25, 6.25, 0.25) # Quarterly up to 6y
        f = PchipInterpolator(dataMat, dataRates)
        intRates = f(intMat) / 100 # Convert to decimal

        # =================== Bootstrap ===========================
        print("Bootstrap...")

        dt = 0.25
        ZZ = zeros(len(intRates))
        # Discount Factors
        # Z(t) = 1 / (1 + r(t)*t)?
        # Or Z(t) from swap curve bootstrap?
        # Guide code:
        # ZZ[0] = 1/(1+dt*intRates[0])
        # ZZ[i] = (1-intRates[i]*dt*sum(ZZ[:i])) / (1+intRates[i]*dt)
        # This implies intRates are Par Swap Rates (quarterly).

        ZZ[0] = 1 / (1 + dt * intRates[0])
        for i in range(1, len(intRates)):
            S = intRates[i]
            sum_Z = np.sum(ZZ[:i])
            ZZ[i] = (1 - S * dt * sum_Z) / (1 + S * dt)

        # Forward Rates
        # f(T, T+dt) = (Z(T)/Z(T+dt) - 1) / dt
        fwd_Z = np.hstack((nan, ZZ[1:] / ZZ[:-1]))
        fwd_Rate = (1 / fwd_Z - 1) / dt

        # Forward Discount for Swaption (1y into 5y)
        # Swaption maturity 1y. Swap tenor 5y.
        # We need fwd discount factors from T=1 to T=6.
        # Indices for 1y: 1.0/0.25 = 4. Index 3 (0-based) is 1.0y?
        # intMat: 0.25 (0), 0.5 (1), 0.75 (2), 1.0 (3).
        # We want Z(t) / Z(1.0).
        # ZZ[4:] starts at 1.25y.
        # Guide: "ZZ[4:]/ZZ[3]" -> Discounts relative to 1y.

        fwd_Z_Swaption = np.hstack((nan, nan, nan, nan, ZZ[4:] / ZZ[3]))

        # =================== 1-year Cap ===========================
        print("Pricing 1-year Cap...")

        T = 1
        # Strike X: "Strike Rate (from data)".
        # Sheet 'Cap and Floor Pricer'? Or 'Rate & Vol Update'?
        # In 'Rate & Vol Update', Col F "ATM"?
        # Row 6 Col 5 (F) is 183.33? No, Col 4 is 102.28.
        # Let's check the header in row 4.
        # Row 4: VOLS, 0, 0.01, 0.02... (Strikes?)
        # Columns E, F, G, H... have headers 0, 0.01, 0.02.
        # These look like Strikes (0%, 1%, 2%).
        # ATM Vol might be separate.
        # Guide says: "X = ??? # Strike Rate".
        # Let's assume ATM Cap.
        # ATM Strike for 1y Cap = 1y Swap Rate? Or Par Rate?
        # Or use a specific strike like 2% (0.02).
        # Let's use ATM Rate.
        # ATM Rate for Cap is the Par Swap Rate for that maturity?
        # Or just the Swap Rate calculated from curve.
        # Strike X = intRates[3] (1y Rate)?
        # Let's set X = intRates[int(T/dt)-1] # 1y rate

        X_1y = intRates[int(T/dt)-1]

        # Volatility sigma
        # Need Vol for 1y Cap.
        # From 'Rate & Vol Update'.
        # ATM Vol column? Col D says "ATM" in row 5.
        # Values below: 102.28, 102.41...
        # These look like Basis Point Vols (Normal Vol).
        # If ~100 bps = 1%.
        # If using Black Model (Log-Normal), we need sigma ~ 20% (0.20).
        # If 102 bps is Normal Vol, approx Log Vol = NormalVol / Rate.
        # 100 bps / 5% = 1% / 5% = 0.20.
        # So 102.28 bps ~ 0.20 Log Vol.
        # Guide uses `BSfwd`. This is Black (LogNormal).
        # So we should convert Normal Vol to LogNormal Vol.
        # Sigma_LN = Sigma_N / FwdRate.
        # Let's use the ATM vol from data.
        # 1y ATM Vol: Row 6 (Index 6) corresponds to what maturity?
        # Row 6 seems to be 1st data row.
        # Dates in Col B match monthly.
        # Is Vol term structure matching Rates?
        # Row 6: 2024-04-22. 2 months from Feb?
        # Let's assume Vol structure matches.
        # Vol for 1y: Index 11 (1y from start).
        # Vol_BP = df_rv.iloc[4+11, 4] # Col E is ATM?
        # Wait, Col D (index 3) is empty?
        # Col E (index 4) has "VOLS" header.
        # Col F (index 5) has "0".
        # Col G (index 6) has "0.01".
        # Where is ATM Vol?
        # The snippet showed:
        # 4: VOLS, 0, 0.01
        # 5: ATM, 0.0001, 0.01
        # 6: 102.28, 183.33, 166.08
        # So Col E (index 4) contains "102.28" at row 6.
        # Header above is "VOLS".
        # Row 5 Col E says "ATM".
        # So Col E is ATM Vol.
        # 1y Vol: Row corresponding to 1y.
        # We used index 11 for 1y rate.
        # Let's use index 11 for 1y vol.

        Vol_BP_1y = df_rv.iloc[4+11, 4] # Col E
        Vol_BP_1y = pd.to_numeric(Vol_BP_1y, errors='coerce')

        # Convert to LogNormal Sigma
        # Sigma = (Vol_BP / 10000) / Rate
        # Rate ~ 5.3% (0.053).
        # Sigma ~ 0.01 / 0.05 ~ 0.2.

        sigma_1y = (Vol_BP_1y / 10000) / X_1y

        # Price Caplets
        # Caplet maturities: 0.25, 0.5, 0.75. (Last one at T-dt).
        # T_Cap = arange(dt, T, dt)
        T_Cap = arange(dt, T, dt)
        F_1y = fwd_Rate[1:int(T/dt)] # Forwards 0.25->0.5, 0.5->0.75, 0.75->1.0?
        # Guide: "F = fwd_Rate[1:int(T/dt)]"
        # Fwd indices: 0 (0-0.25), 1 (0.25-0.5), 2 (0.5-0.75), 3 (0.75-1.0).
        # Caplets usually pay on 0.5, 0.75, 1.0 (Fix at 0.25, 0.5, 0.75).
        # So we need Forwards fixing at 0.25, 0.5, 0.75.
        # These are indices 1, 2, 3?
        # fwd_Rate array:
        # Index 0: 0->0.25 (Fix 0).
        # Index 1: 0.25->0.5 (Fix 0.25).
        # Index 2: 0.5->0.75 (Fix 0.5).
        # Index 3: 0.75->1.0 (Fix 0.75).
        # Range 1:4 gives indices 1, 2, 3.
        # int(1/0.25) = 4. 1:4 is correct.

        F_vec = fwd_Rate[1:int(T/dt)+1] # 1, 2, 3
        # Wait, guide used 1:int(T/dt). Python range is exclusive.
        # So 1:4 gives 1, 2, 3. Correct.

        # T=1. dt=0.25. T_Cap = arange(0.25, 1, 0.25) = [0.25, 0.50, 0.75]. Length 3.
        # fwd_Rate indices: 0->(0-0.25), 1->(0.25-0.5), 2->(0.5-0.75), 3->(0.75-1.0).
        # Caplet 1 (T=0.25) pays on 0.25? No.
        # Usually Caplet resets at T-dt and pays at T.
        # Or Resets at T and pays at T+dt.
        # Standard: Caplet on rate resetting at T_j pays at T_{j+1}.
        # T_Cap usually refers to payment dates?
        # Guide: "T_Cap = arange(dt,T,dt)" -> [0.25, 0.5, 0.75].
        # These are payment dates or reset dates?
        # If payment dates, then reset dates are 0, 0.25, 0.5.
        # Rate resetting at 0 pays at 0.25. (Index 0 in fwd_Rate).
        # Rate resetting at 0.25 pays at 0.5. (Index 1).
        # Rate resetting at 0.5 pays at 0.75. (Index 2).
        # Rate resetting at 0.75 pays at 1.0. (Index 3).
        # The last caplet in a 1y Cap usually pays at 1y. Reset at 0.75.
        # Guide T_Cap stops before T.
        # If T_Cap are maturities (payment dates), guide misses the last one?
        # Or T_Cap are reset dates?
        # If T_Cap are reset dates: 0.25, 0.5, 0.75.
        # Pays at 0.5, 0.75, 1.0.
        # First caplet (reset 0, pay 0.25) is usually excluded (known at t=0)?
        # Standard Market Practice: First period is fixed, no caplet.
        # Caplets start from 2nd period.
        # So Reset 0.25 (Index 1), Pay 0.5.
        # Reset 0.5 (Index 2), Pay 0.75.
        # Reset 0.75 (Index 3), Pay 1.0.
        # So we need indices 1, 2, 3.
        # Length 3.
        # T_Cap in guide is [0.25, 0.5, 0.75]. This matches Reset Dates?
        # Or T_Cap is Time to Maturity of the option (Reset Date).
        # Option expires at Reset Date.
        # So T_Cap = [0.25, 0.5, 0.75].
        # Fwd Rates F_vec should match these reset dates.
        # Fwd Rate resetting at t corresponds to index int(t/dt).
        # 0.25 -> Index 1. 0.5 -> Index 2. 0.75 -> Index 3.
        # So fwd_Rate[1], fwd_Rate[2], fwd_Rate[3].
        # Slice 1:4.

        # Let's check `fwd_Rate[1:int(T/dt)+1]` when T=1.
        # int(1/0.25) = 4. 1:5 -> 1, 2, 3, 4. Length 4.
        # T_Cap length 3.
        # Mismatch!

        # We want indices 1, 2, 3.
        # Slice should be 1:4.
        # int(T/dt) is 4.
        # So 1:int(T/dt).

        F_vec = fwd_Rate[1:int(T/dt)] # 1, 2, 3
        Z_vec = ZZ[1:int(T/dt)]       # Discounts for payment?
        # Payment dates are T_Cap + dt?
        # Z usually discount to payment date.
        # If T_Cap are reset dates [0.25, 0.5, 0.75].
        # Payments are at [0.5, 0.75, 1.0].
        # Indices for Z corresponding to 0.5, 0.75, 1.0.
        # intMat: 0.25(0), 0.5(1), 0.75(2), 1.0(3).
        # So Z indices: 1, 2, 3.
        # Slice 1:4.
        # Matches F_vec slice.

        T_Cap_vec = T_Cap

        # Verify shape
        # F_vec length 3. Z_vec length 3. T_Cap_vec length 3.

        Caplets_1Year = 100 * dt * BSfwd_local(F_vec, X_1y, Z_vec, sigma_1y, T_Cap_vec, 1)
        Cap_1Year = np.sum(Caplets_1Year)

        print(f'1-year Cap Price: {Cap_1Year:.5f}')

        # =================== 2-year Cap ===========================
        print("Pricing 2-year Cap...")
        T = 2
        X_2y = intRates[int(T/dt)-1]

        # 2y Vol
        Vol_BP_2y = df_rv.iloc[4+23, 4] # Index 23
        Vol_BP_2y = pd.to_numeric(Vol_BP_2y, errors='coerce')
        sigma_2y = (Vol_BP_2y / 10000) / X_2y

        T_Cap_2 = arange(dt, T, dt) # 0.25 ... 1.75
        # Forwards indices 1 to 7? (0.25 to 1.75)
        # int(2/0.25) = 8. Range 1:8 -> 1..7.

        F_vec_2 = fwd_Rate[1:int(T/dt)+1]
        # Slice 1:8? -> 7 items.
        # T_Cap_2 length 7.
        # Let's verify slice.
        # arange(0.25, 2, 0.25) -> 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75. (7 items).
        # range(1, 8) -> 1, 2, 3, 4, 5, 6, 7. (7 items).
        # Matches.
        # Wait, need to ensure fwd_Rate has enough elements.
        # intRates length 25 (0.25 to 6.25).
        # fwd_Rate length 25.
        # So 1:8 works.

        F_vec_2 = fwd_Rate[1:int(T/dt)] # Using Guide's slicing (exclusive)
        Z_vec_2 = ZZ[1:int(T/dt)]

        Caplets_2Year = 100 * dt * BSfwd_local(F_vec_2, X_2y, Z_vec_2, sigma_2y, T_Cap_2, 1)
        Cap_2Year = np.sum(Caplets_2Year)

        print(f'2-year Cap Price: {Cap_2Year:.5f}')

        # =================== Swaption ===========================
        print("Pricing Swaption...")
        T = 1 # Maturity 1y
        # Strike X: 5y Swap Rate (Forward Swap Rate 1y into 5y?)
        # Guide: "Strike Rate (5-year swap rate)".
        # Usually ATM Swaption Strike is the Forward Swap Rate.
        # Let's calculate Forward Swap Rate.
        # S_fwd = (Z_start - Z_end) / Sum(dt * Z_i)
        # Start T=1y (Index 3). End T=6y (Index 23).
        # Sum Z_i from 1.25y to 6y.
        # Indices: 4 to 23.

        idx_start = 3 # 1.0y
        idx_end = 23 # 6.0y? (1y + 5y = 6y). 6.0y is index 23 (6/0.25 - 1 = 23).

        Z_start = ZZ[idx_start]
        Z_end = ZZ[idx_end]
        Sum_Z = np.sum(ZZ[idx_start+1 : idx_end+1]) * dt

        SwapRate_Fwd = (Z_start - Z_end) / Sum_Z
        X_Swaption = SwapRate_Fwd

        # Vol
        # Swaption Vol.
        # Where in Excel?
        # Maybe "1Y into 5Y"?
        # Often Swaption Matrix is provided.
        # Sheet 'Rate & Vol Update' might only be Cap Vols? "ATM".
        # Rows 25+ might have Swaptions?
        # Or Sheet 'Cap and Floor Pricer' has Swaption data?
        # Let's assume we use the 1y Cap Vol as proxy or seek specific cell.
        # Guide: "sigma = ??? # Swaption Volatility (from data)".
        # If not found, use 1y Vol.
        sigma_swaption = sigma_1y

        # A-factor (Annuity)
        A = Sum_Z

        # Swaption Price
        # Black Formula on Swap Rate
        # T=1.

        Swaption = A * BSfwd_local(SwapRate_Fwd, X_Swaption, 1.0, sigma_swaption, T, 1)

        print(f'Swaption Price: {Swaption:.5f}')

        fig1 = go.Figure()
        fig1.add_trace(go.Scatter(x=intMat, y=intRates, mode='lines', name='Interpolated Rates'))
        fig1.add_trace(go.Scatter(x=dataMat, y=dataRates/100, mode='markers', name='Data Rates'))
        fig1.update_layout(title='Bootstrap Rate Curve', xaxis_title='Maturity', yaxis_title='Rate')

        return {
            "Cap_1Year": Cap_1Year,
            "Cap_2Year": Cap_2Year,
            "Swaption": Swaption,
            "fig1": fig1
        }

    if __name__ == "__main__":
        solve_pset5()

    return

@app.cell
def __(np, pd, go, plt, fmin, fsolve, norm, interp1d, PchipInterpolator, sm):
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    from numpy import arange, exp, sqrt, log, mean, std, nonzero, isnan, array, zeros, ones, amax, amin, maximum, minimum, hstack, vstack
    from scipy.optimize import fmin
    from scipy.interpolate import interp1d, PchipInterpolator
    from scipy.stats import norm
    import plotly.graph_objects as go
    import plotly.io as pio

    # NLLS Helper (Same as PSET 4)
    def NLLS_Min(vec, Price, Maturity, CashFlow):
        J, PPhat = NLLS(vec, Price, Maturity, CashFlow)
        return J

    def NLLS(vec, Price, Maturity, CashFlow):
        th0 = vec[0]
        th1 = vec[1]
        th2 = vec[2]
        la  = vec[3]
        T  = maximum(Maturity, 1e-10)
        RR = th0 + (th1 + th2) * (1 - exp(-T/la)) / (T/la) - th2 * exp(-T/la)
        ZZhat = exp(-RR * T)
        PPhat = np.sum(CashFlow * ZZhat, axis=1)
        J = np.sum((Price - PPhat)**2)
        return J, PPhat

    # Ho-Lee Tree Helper
    def HoLee_SimpleBDT_Tree(theta_i, ZZi, ImTree, i, sigma, hs, BDT_Flag):
        if BDT_Flag == 0:
            # Ho-Lee
            # ImTree[j, i] = r(j, i)
            # r(0, i) = r(0, i-1) + theta*dt + sigma*sqrt(dt)
            # r(j, i) = r(j-1, i-1) + theta*dt - sigma*sqrt(dt) -> This looks recursive in j?
            # Standard Ho-Lee: r_{i,j} = r_0 + i*theta_i + j*sigma*sqrt(dt)? No.
            # Guide Code:
            # ImTree[0,i] = ImTree[0,i-1] + theta_i*hs + sigma*sqrt(hs)
            # for j in arange(1,i+1):
            #    ImTree[j,i] = ImTree[j-1,i-1] + theta_i*hs - sigma*sqrt(hs)
            # This implies:
            # Top node (0) comes from Top node (0) of previous step + drift + vol.
            # Node j comes from Node j-1 of previous step + drift - vol?
            # Wait, usually r_{i+1, 0} (up from r_{i,0}) and r_{i+1, 1} (down from r_{i,0}).
            # Guide code iterates j.
            # ImTree[j,i] depends on ImTree[j-1, i-1].
            # If j=1: ImTree[1,i] from ImTree[0, i-1] + drift - vol.
            # This is the "Down" move from 0.
            # So ImTree[j,i] is the rate at node j at time i.
            # j=0 is highest rate?

            ImTree[0, i] = ImTree[0, i-1] + theta_i * hs + sigma * sqrt(hs)
            for j in range(1, i + 1):
                ImTree[j, i] = ImTree[j-1, i-1] + theta_i * hs - sigma * sqrt(hs)

        else:
            # Simple BDT
            ImTree[0, i] = ImTree[0, i-1] * exp(theta_i * hs + sigma * sqrt(hs))
            for j in range(1, i + 1):
                ImTree[j, i] = ImTree[j-1, i-1] * exp(theta_i * hs - sigma * sqrt(hs))

        # Value Zero Coupon Bond expiring at i+1 (Maturity T = (i+1)*hs)
        # Tree step i corresponds to time t = i*hs.
        # Bond pays 1 at t = (i+1)*hs.
        # Backward induction from i+1 to 0?
        # No, we just need price at t=0 matching ZZi.
        # We price a bond maturing at (i+1)*hs using the tree up to step i.
        # Rate at step i determines discount to i+1.
        # Price at step i is exp(-r * hs) * 1.
        # Then backward to 0.

        # Initialize Price Tree at step i+1 (Maturity)
        # Dim i+2.
        ZZTree = zeros((i + 2, i + 2))
        ZZTree[0:i+2, i+1] = 1
        pi = 0.5

        # Backward from i to 0
        # Range(i, -1, -1) -> i, ..., 0.
        # But guide uses `arange(i+1, 0, -1)` -> i+1, ..., 1. Then index `j-1`.
        # Let's write explicit loop.
        # We need price at step 0.

        # Pricing step k (from k=i down to 0)
        # Value at node (node, k) = exp(-r(node, k)*hs) * (0.5*Up + 0.5*Down)
        # Up is (node, k+1). Down is (node+1, k+1).
        # Guide: ZZTree[0:j, j-1] = exp(-ImTree[0:j, j-1]*hs) * ...
        # j-1 is time step.

        for k in range(i, -1, -1):
            # Nodes 0 to k
            rates = ImTree[0:k+1, k]
            # Next step values
            val_up = ZZTree[0:k+1, k+1]
            val_down = ZZTree[1:k+2, k+1]

            ZZTree[0:k+1, k] = exp(-rates * hs) * (pi * val_up + (1 - pi) * val_down)

        FF = (ZZTree[0, 0] - ZZi)**2
        return FF, ImTree, ZZTree

    def fmin_HoLee(theta_i, ZZi, ImTree, i, sigma, hs, BDT_Flag):
        FF, _, _ = HoLee_SimpleBDT_Tree(theta_i, ZZi, ImTree, i, sigma, hs, BDT_Flag)
        return FF

    def solve_pset6():
        print("Solving PSET 6...")
        file_path_bonds = 'Assignments/PSET 6/HW6_Data_Bonds.xls'
        file_path_rates = 'Assignments/PSET 6/HW6_FRB_H15.csv'

        # =================== Load Data ===========================
        print("Loading Bond Data...")

        try:
            # Load Raw
            df_bonds = pd.read_excel(file_path_bonds, sheet_name='Sheet1', skiprows=4, header=None)

            # Guide: Data=cols 0-8. Mat=cols 10-69. CashFlow=cols 72-131.
            # Python indices: 0:9, 10:70, 72:132.

            # Row 0 is NaN (skipped empty row after header?).
            # Actually skiprows=4 lands on data.
            # But previous print showed Row 0 as NaN.
            # It implies skiprows=4 skipped the header but the first row of data frame is NaN?
            # Maybe an empty row in Excel.
            # Let's drop row 0.

            df_bonds = df_bonds.iloc[1:] # Drop first row if it is NaN

            Data = df_bonds.iloc[:, 0:9].values
            Mat = df_bonds.iloc[:, 10:70].values
            CashFlow = df_bonds.iloc[:, 72:132].values

            # Ensure numeric (coerce)
            Data = pd.DataFrame(Data).apply(pd.to_numeric, errors='coerce').values
            Mat = pd.DataFrame(Mat).apply(pd.to_numeric, errors='coerce').values
            CashFlow = pd.DataFrame(CashFlow).apply(pd.to_numeric, errors='coerce').values

            # Clean Rows
            # Data col 5 (Bid), 6 (Ask).
            Bid = Data[:, 5]
            Ask = Data[:, 6]
            AccInt = Data[:, 8]
            CleanPrice = (Bid + Ask) / 2
            Price = CleanPrice + AccInt

            # Filter NaNs (Row 0 is likely NaN from header skip)
            mask = ~np.isnan(Price)
            Price = Price[mask]
            Mat = Mat[mask]
            CashFlow = CashFlow[mask]
            Data = Data[mask] # For plotting Maturity (Col 7)

            Mat[np.isnan(Mat)] = 0
            CashFlow[np.isnan(CashFlow)] = 0

        except Exception as e:
            print(f"Error loading bonds: {e}")
            return

        # =================== Nelson Siegel ===========================
        print("Fitting Nelson Siegel...")

        vec0 = array([5.3664, -0.1329, -1.2687, 132.0669]) / 100
        vec = fmin(func=NLLS_Min, x0=vec0, args=(Price, Mat, CashFlow), disp=False)
        J, PPhat = NLLS(vec, Price, Mat, CashFlow)

        th0, th1, th2, la = vec

        fig1 = go.Figure()
        fig1.add_trace(go.Scatter(x=Data[:, 7], y=Price, mode='markers', name='Data'))
        fig1.add_trace(go.Scatter(x=Data[:, 7], y=PPhat, mode='markers', name='Fitted'))
        fig1.update_layout(title='Bond Prices Fit', xaxis_title='Maturity', yaxis_title='Price')

        hs = 0.5
        T = arange(hs, 30 + hs, hs)
        Ycc = th0 + (th1 + th2) * (1 - exp(-T/la)) / (T/la) - th2 * exp(-T/la)
        ZZcc = exp(-Ycc * T)

        # Forward Rates
        FWD = -log(ZZcc[1:] / ZZcc[:-1]) / hs
        FWD = hstack((Ycc[0], FWD))

        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(x=T, y=Ycc, mode='lines', name='Yield'))
        fig2.add_trace(go.Scatter(x=T, y=FWD, mode='lines', name='Forward'))
        fig2.update_layout(title='Yield and Forward Curve', xaxis_title='Maturity', yaxis_title='Rate')

        # =================== Ho-Lee Tree ===========================
        print("Building Ho-Lee Tree...")

        # Load 6m rates
        try:
            df_rates = pd.read_csv(file_path_rates, skiprows=5, header=None)
            # Col 1 is rate
            DataY6 = pd.to_numeric(df_rates.iloc[:, 1], errors='coerce').dropna().values
            # Last 10 years (120 months)
            DataY6 = DataY6[-120:] / 100

            # Volatility of short rate
            # r = -ln(P)/0.5
            # P = 1 - d * 0.5 (if d is discount rate).
            # DataY6 is likely Yield or Discount? "RIFSGFSM06_N.M" -> Market Yield on U.S. Treasury Securities at 6-Month Constant Maturity.
            # Usually BEY.
            # Guide: "P6 = (1-180/360*DataY6)" -> Using simple interest discount?
            # If DataY6 is BEY, P = 1 / (1 + Y/2).
            # Guide code implies DataY6 is treated as discount rate d?
            # Let's follow guide formula `P6 = (1-180/360*DataY6)`.

            P6 = 1 - 0.5 * DataY6
            rr6 = -log(P6) / 0.5

            # Sigma of r
            # sigma = std(diff(rr6)) * sqrt(12)? No, data is monthly.
            # hs = 0.5 (semi-annual).
            # We need sigma for semi-annual step.
            # Sigma_annual = std(diff(rr6)) * sqrt(12)? (if diff is monthly change).
            # Ho-Lee model: dr = theta dt + sigma dW.
            # Var(r(t+dt) - r(t)) = sigma^2 dt.
            # Estimate sigma from monthly data.
            # sigma_monthly = std(diff(rr6)).
            # sigma_annual = sigma_monthly * sqrt(12).
            # Guide: "note: units (month) must be annualized".

            sigma_monthly = std(np.diff(rr6), ddof=1)
            sigma = sigma_monthly * sqrt(12)

            print(f"Calibrated Sigma: {sigma:.4f}")

        except Exception as e:
            print(f"Error rates: {e}")
            return

        # Tree Construction
        BDT_Flag = 0
        steps = int(30 / hs) # 60 steps
        ImTree = zeros((steps, steps))
        ZZTree = zeros((steps + 1, steps + 1, steps))

        r0 = Ycc[0]
        ImTree[0, 0] = r0
        ZZTree[0, 0, 0] = exp(-r0 * hs)
        ZZTree[0:1, 1, 0] = 1 # Not used?

        # Calibrate Theta
        thetas = zeros(steps - 1)

        # Pre-fill first step values for recursion
        # i=0 is done.

        for i in range(1, steps):
            # We need to find theta[i-1] to match ZZcc[i]
            # ZZcc[i] is discount factor for T = (i+1)*hs?
            # T indices: 0 -> hs. 1 -> 2hs.
            # ZZcc[0] corresponds to T=hs (price known at t=0).
            # ZZcc[1] corresponds to T=2hs.
            # Loop i=1 (Time hs). We determine rates at hs.
            # These rates determine price of bond maturing at 2hs.
            # This price at t=0 must match ZZcc[1].
            # Wait, ZZcc array: index 0 is T=0.5.
            # Loop i=1 is time t=0.5. Bond matures at t=1.0.
            # We match ZZcc[1] (T=1.0).

            # Target ZZi = ZZcc[i]
            target_Z = ZZcc[i]

            # Optimize
            # Initial guess 0
            # Pass ImTree state
            res = fmin(fmin_HoLee, 0.0, args=(target_Z, ImTree, i, sigma, hs, BDT_Flag), disp=False)
            theta_opt = res[0]
            thetas[i-1] = theta_opt

            # Update Tree with optimal theta
            _, ImTree, ZZTree_i = HoLee_SimpleBDT_Tree(theta_opt, target_Z, ImTree, i, sigma, hs, BDT_Flag)

            # Store full pricing tree if needed?
            # Guide: "ZZTree[0:i+2,0:i+2,i] = ZZTreei"
            # We only need `ImTree` for next steps.

        # Plot Tree Yields vs Data
        # Yield at node 0,0 is r0.
        # Yield at time T?
        # Expected yield?
        # Guide: "yyTree = -log(ZZTree[0,0,:])/T"
        # ZZTree[0,0, i] is the price at t=0 of bond maturing at (i+1)hs.
        # This matches ZZcc[i].
        # So yyTree should match Ycc exactly by construction.

        # =================== Callable Bond ===========================
        print("Pricing Callable Bond...")
        # Parameters
        # Assume: Coupon 5.5% (0.055), Maturity 5y (5.0), FCT 1y (1.0).
        # These are placeholders since I can't read the PDF.
        coupon = 0.055
        TBond = 5.0
        FCT = 1.0

        iT = int(TBond / hs) # 10 steps
        iFCT = int(FCT / hs) # 2 steps

        # Indices in Tree
        # Tree step i corresponds to time i*hs.
        # Maturity TBond -> Step iT.
        # Price at step iT is 100 + Coupon/2?
        # Guide: "PPTree_NC[0:iT,iT-1] = 100" (Face Value).
        # Does it pay coupon at maturity?
        # Usually backward induction adds coupon at each step.
        # PPTree_NC size (iT+1, iT+1)?

        PPTree_NC = zeros((iT + 1, iT + 1))
        Call = zeros((iT + 1, iT + 1))
        PPTree_C = zeros((iT + 1, iT + 1))

        # Terminal Value (at Maturity)
        # Receive Face + Coupon/2
        C_pay = coupon * 100 * hs
        PPTree_NC[0:iT+1, iT] = 100 + C_pay
        PPTree_C[0:iT+1, iT] = 100 + C_pay

        pi = 0.5

        # Backward Induction
        for j in range(iT - 1, -1, -1):
            # Discount from j+1 to j
            rates = ImTree[0:j+1, j]
            disc = exp(-rates * hs)

            # Values at j+1
            val_nc_up = PPTree_NC[0:j+1, j+1]
            val_nc_down = PPTree_NC[1:j+2, j+1]

            # Non-Callable Value (Ex-Coupon)
            # Value = Disc * E[Next Value]
            # Next Value includes Coupon?
            # If we define PPTree as Cum-Coupon or Ex-Coupon?
            # Usually easier to track Ex-Coupon value, and add coupon at node.
            # But here we assume PPTree holds value at node.
            # At Maturity, value is 100 + C.
            # At step j, Value = Disc * (0.5*Vu + 0.5*Vd).
            # Then add Coupon payment at step j?
            # Bond pays coupon at 0.5, 1.0 ...
            # If step j > 0, we add coupon.
            # If step j = 0 (t=0), we usually don't add current coupon (already paid or clean price).
            # Let's assume we price the bond for t=0.

            hold_val_nc = disc * (pi * val_nc_up + (1 - pi) * val_nc_down)
            PPTree_NC[0:j+1, j] = hold_val_nc + C_pay

            # Callable
            val_c_up = PPTree_C[0:j+1, j+1]
            val_c_down = PPTree_C[1:j+2, j+1]

            hold_val_c = disc * (pi * val_c_up + (1 - pi) * val_c_down)

            # Check Call
            # If t >= FCT, Issuer can call at 100.
            # t = j * hs.
            if j * hs >= FCT:
                # Callable at Par (100)
                # Value = min(Hold Value + Coupon, 100 + Coupon) ?
                # Usually Call Price is 100.
                # If called, investor gets 100 + Coupon.
                # So Value = min(Hold + C, 100 + C).
                PPTree_C[0:j+1, j] = minimum(hold_val_c + C_pay, 100 + C_pay)
            else:
                PPTree_C[0:j+1, j] = hold_val_c + C_pay

        # Price at t=0
        # Usually clean price quotes.
        # Tree result at 0 is Cum-Coupon (Coupon at 0.5 discounted).
        # Wait, loop includes j=0.
        # At j=0, we added C_pay.
        # But bond at t=0 doesn't pay coupon immediately.
        # The first coupon is at j=1 (t=0.5).
        # Our Backward loop:
        # At j=iT-1 (4.5y), we discount value at iT (5.0y).
        # Value at iT is 100+C.
        # Discounted gives value at 4.5y (Ex-C).
        # Add C.
        # ...
        # At j=0 (0y), we discount value at 1 (0.5y).
        # Value at 1 is Value(Ex at 0.5) + C.
        # Discounted gives Value at 0.
        # We should NOT add C at j=0.

        PPTree_NC[0, 0] -= C_pay
        PPTree_C[0, 0] -= C_pay

        P_NC = PPTree_NC[0, 0]
        P_C = PPTree_C[0, 0]
        Val_Call = P_NC - P_C

        print(f"Non-Callable Price: {P_NC:.4f}")
        print(f"Callable Price: {P_C:.4f}")
        print(f"Call Option Value: {Val_Call:.4f}")

        # =================== Duration/Convexity ===========================
        print("Duration/Convexity...")
        # Finite Difference or Tree Sensitivity?
        # Guide suggests:
        # Delta_NC_1u = (PPTree_NC[0,2]-PPTree_NC[1,2])/(ImTree[0,2]-ImTree[1,2])
        # This looks like Option Delta?
        # "Convexity - Non-Callable ... C_NC = ???"
        # Usually Bond Duration = -(1/P) dP/dy.
        # On Tree, we can shift tree or use effective duration from nodes?
        # Guide calculates Delta at step 2?
        # ImTree[0,2] (Up-Up) vs ImTree[1,2] (Up-Down).
        # This measures sensitivity to rate change.
        # Rate delta = ImTree[0,2] - ImTree[1,2].
        # Price delta = P_u - P_d.
        # Delta = dP/dr.
        # Duration ~ -1/P * Delta.

        # Step 1 (t=0.5).
        # Nodes (0,1) and (1,1).
        # P_up = PPTree_NC[0,1]. P_down = PPTree_NC[1,1].
        # r_up = ImTree[0,1]. r_down = ImTree[1,1].

        # Using Step 1 nodes (t=0.5)
        # Price at nodes are Cum-Coupon (since we added C at j=1).
        # But usually we want Clean Price sensitivity?
        # Let's use the prices as is.

        P_u = PPTree_NC[0, 1]
        P_d = PPTree_NC[1, 1]
        r_u = ImTree[0, 1]
        r_d = ImTree[1, 1]

        # Approx Derivative
        dPdR = (P_u - P_d) / (r_u - r_d)
        D_NC = -1 / P_NC * dPdR # Approx

        # Convexity
        # Second derivative.
        # Need 3 nodes? Step 2.
        # Nodes (0,2), (1,2), (2,2).
        # P_uu, P_ud, P_dd.
        # r_uu, r_ud, r_dd.

        P_uu = PPTree_NC[0, 2]
        P_ud = PPTree_NC[1, 2]
        P_dd = PPTree_NC[2, 2]
        r_uu = ImTree[0, 2]
        r_ud = ImTree[1, 2]
        r_dd = ImTree[2, 2]

        # First derivatives at step 2 level?
        # Delta_u = (P_uu - P_ud) / (r_uu - r_ud)
        # Delta_d = (P_ud - P_dd) / (r_ud - r_dd)
        # Gamma = (Delta_u - Delta_d) / (0.5*(r_uu - r_dd)?)
        # Or change in rate between u and d branches.
        # dr_avg = ( (r_uu-r_ud) + (r_ud-r_dd) ) / 2 ?
        # Let's use difference in r at step 1: r_u - r_d.

        Delta_u = (P_uu - P_ud) / (r_uu - r_ud)
        Delta_d = (P_ud - P_dd) / (r_ud - r_dd)

        Gamma = (Delta_u - Delta_d) / (r_u - r_d) # Change in Delta / Change in r

        C_NC = 1 / P_NC * Gamma

        print(f"Non-Callable Duration: {D_NC:.4f}")
        print(f"Non-Callable Convexity: {C_NC:.4f}")

        # Callable
        P_u_c = PPTree_C[0, 1]
        P_d_c = PPTree_C[1, 1]
        dPdR_c = (P_u_c - P_d_c) / (r_u - r_d)
        D_C = -1 / P_C * dPdR_c

        P_uu_c = PPTree_C[0, 2]
        P_ud_c = PPTree_C[1, 2]
        P_dd_c = PPTree_C[2, 2]

        Delta_u_c = (P_uu_c - P_ud_c) / (r_uu - r_ud)
        Delta_d_c = (P_ud_c - P_dd_c) / (r_ud - r_dd)
        Gamma_c = (Delta_u_c - Delta_d_c) / (r_u - r_d)
        C_C = 1 / P_C * Gamma_c

        print(f"Callable Duration: {D_C:.4f}")
        print(f"Callable Convexity: {C_C:.4f}")

        return {
            "P_NC": P_NC,
            "P_C": P_C,
            "D_NC": D_NC,
            "C_NC": C_NC,
            "D_C": D_C,
            "C_C": C_C,
            "fig1": fig1,
            "fig2": fig2
        }

    if __name__ == "__main__":
        solve_pset6()

    return

@app.cell
def __(np, pd, go, plt, fmin, fsolve, norm, interp1d, PchipInterpolator, sm):
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    from numpy import arange, exp, sqrt, log, mean, std, nonzero, isnan, array, zeros, ones, maximum, minimum, hstack, vstack, argmin
    from scipy.interpolate import interp1d
    from scipy.optimize import fsolve, minimize_scalar
    from scipy.stats import norm
    import plotly.graph_objects as go
    import plotly.io as pio

    # BSfwd Helper
    def BSfwd_local(F,X,Z,sigma,T,CallF):
        # CallF=1 for Caplet
        F = np.array(F)
        X = np.array(X)
        Z = np.array(Z)
        sigma = np.array(sigma)
        T = np.array(T)
        T = maximum(T, 1e-10)

        d1 = (log(F/X) + (sigma**2/2)*T) / (sigma*sqrt(T))
        d2 = d1 - sigma*sqrt(T)

        if CallF == 1:
            Price = Z * (F * norm.cdf(d1) - X * norm.cdf(d2))
        else:
            Price = Z * (-F * norm.cdf(-d1) + X * norm.cdf(-d2))
        return Price

    # MinFun Helper for BDT
    def minfunBDTNew_fsolve(rmin, ImTree, yield1, vol, h, N):
        F, vec = minfunBDTNew(rmin, ImTree, yield1, vol, h, N)
        return F

    def minfunBDTNew(rmin, ImTree, yield1, vol, h, N):
        # N is 0-based index of step?
        # Guide: "mult = arange(N-1,-1,-1)" where N is passed as j+2.
        # j=0 -> N=2. mult=[1,0].
        # ImTree column N-1 has N nodes?
        # ImTree[0:N, N-1] = vec
        # vec has size N.

        mult = arange(N-1, -1, -1)
        # r(i) = rmin * exp(2*vol*sqrt(h)*i)
        # BDT short rate process: ln r follows binomial.
        # r_down = r_up * exp(-2*vol*sqrt(h))?
        # Or r_i = r_0 * exp(...)
        # Guide: "vec = rmin*exp(2*vol*sqrt(h)*mult)"
        # This implies rmin is the lowest rate (at node N-1).
        # And we multiply by exp to get higher rates.

        vec = rmin * exp(2 * vol * sqrt(h) * mult)
        ImTree[0:N, N-1] = vec

        # Price a Zero Coupon Bond maturing at N*h (Time T)
        # Tree currently filled up to column N-1 (Time (N-1)h).
        # Bond pays 1 at column N (Time Nh).
        # Discount back to 0.

        RateMatrix = ImTree[0:N, 0:N]
        T = N
        BB = zeros((T + 1, T + 1))
        BB[:, T] = ones(T + 1)

        # Backward
        for t in arange(T, 0, -1):
            # Discount from t to t-1
            rates = RateMatrix[0:t, t-1]
            val_up = BB[0:t, t]
            val_down = BB[1:t+1, t]
            BB[0:t, t-1] = exp(-rates * h) * (0.5 * val_up + 0.5 * val_down)

        PZero = BB[0, 0]

        # Target Price = exp(-yield * T * h)
        # yield1 is continuously compounded zero yield for maturity T*h.
        # Wait, guide passes `yield1` as ZYield[j+1].
        # ZYield is computed as -log(Price)/Maturity.
        # So exp(-yield1 * Maturity) should be Price.
        # Guide: "F = exp(-yield1*vec.shape[0]*h) - PZero"
        # vec.shape[0] is N. h is dtstep.

        TargetPrice = exp(-yield1 * N * h)
        F = TargetPrice - PZero

        return F, vec

    def solve_pset7():
        print("Solving PSET 7...")

        # =================== Data ===========================
        print("Data Setup...")
        SwRates = array([0.152, 0.2326, 0.3247, 0.346, 0.7825, 1.2435, 1.599, 1.853, 2.052, 2.2085, 2.3371, 2.4451, 2.539, 2.843, 2.9863, 3.0895]) / 100
        CapsVol = array([68.53, 63.63, 54.06, 48.43, 44.87, 42.03, 43.35, 38.03, 36.54, 38.45, 33.13, 29.97, 26.91, 24.95, 23.65]) / 100

        MaturitySwaps = array([1/12, 3/12, 6/12, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 20, 30])
        MaturityCaps = array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 15, 20, 25, 30])

        dt = 0.25

        # Augment
        CapsVol = hstack((CapsVol[0], CapsVol))
        MaturityCaps = hstack((dt, MaturityCaps))

        # Interpolation
        IntMat = arange(dt, 30 + dt, dt)

        f_sw = interp1d(MaturitySwaps, SwRates, kind='cubic', fill_value='extrapolate')
        IntSwRate = f_sw(IntMat)

        f_vol = interp1d(MaturityCaps, CapsVol, kind='cubic', fill_value='extrapolate')
        IntVol = f_vol(IntMat)

        # =================== Bootstrap ===========================
        print("Bootstrap Discounts...")
        ZSw = zeros(len(IntSwRate))
        ZSw[0] = 1 / (1 + IntSwRate[0] * dt)
        NN = len(ZSw)
        for i in range(1, NN):
            sum_Z = np.sum(ZSw[:i])
            ZSw[i] = (1 - IntSwRate[i] * dt * sum_Z) / (1 + IntSwRate[i] * dt)

        Zyieldcc = -log(ZSw) / IntMat
        Fwdcc = -log(ZSw[1:] / ZSw[:-1]) / dt
        Fwd = (exp(Fwdcc * dt) - 1) / dt # Quarterly

        fig1 = go.Figure()
        fig1.add_trace(go.Scatter(x=IntMat, y=Zyieldcc, mode='lines', name='Yield Curve'))
        fig1.add_trace(go.Scatter(x=IntMat[:-1], y=Fwd, mode='lines', name='Forward Rates'))
        fig1.update_layout(title='LIBOR Yield and Forward', xaxis_title='Maturity', yaxis_title='Rate')

        # =================== Forward Volatilities ===========================
        print("Forward Volatilities...")

        Caps = zeros(NN - 1)
        # Compute Dollar Value of Caps using Flat Vol
        for i in range(NN - 1):
            # Cap Maturity T = IntMat[i] (e.g. 0.25). Wait.
            # Cap Maturity is usually last payment date.
            # Loop i goes up to NN-2.
            # i=0: Maturity IntMat[0]=0.25. 1 Quarter.
            # Caplet pays at 0.25? Reset at 0.
            # Standard: 1st period fixed. Caplet starts later?
            # Guide code: "BSfwd(Fwd[0:i+1], IntSwRate[i+1], ZSw[1:i+2], ...)"
            # IntSwRate[i+1] is Strike? (ATM Strike for Cap maturing at i+1?).
            # IntMat[i] is maturity of cap?
            # Guide loops i from 0 to NN-2.
            # Caps[i] corresponds to maturity i.
            # Sum of caplets.

            # Guide Logic:
            # Fwd[0:i+1] -> Fwd rates from 0 to i.
            # IntMat[0:i+1] -> Maturities.

            T_vec = IntMat[0:i+1]
            F_vec = Fwd[0:i+1] # Fwd 0, Fwd 1 ... Fwd i.
            # We need to compute Cap Value.
            # Using Flat Vol IntVol[i+1].

            caplets = dt * BSfwd_local(F_vec, IntSwRate[i+1], ZSw[1:i+2], IntVol[i+1], T_vec, 1)
            Caps[i] = np.sum(caplets)

        # Bootstrap Spot/Forward Vols
        ImplVol = zeros(NN - 1)
        ImplVol[0] = IntVol[1] # Flat Vol?
        # Guide: "ImplVol[0] = IntVol[1]" (Index 1 is 0.5y?). Index 0 is 0.25y.

        Caplet = zeros(NN - 1)

        # First Cap (i=0, Mat 0.25): 1 Caplet.
        # Value = Caps[0].
        Caplet[0] = Caps[0]
        ImplVol[0] = IntVol[0] # Actually first point.

        # Loop
        for i in range(1, NN - 1):
            # Value of Cap with maturity i.
            # Sum of previous caplets (using their ImplVols).
            # + New Caplet (using unknown ImplVol).
            # Strike for Cap i is IntSwRate[i+1].
            # We need to re-price previous caplets with NEW Strike.

            Strike = IntSwRate[i+1]

            # Previous Caplets: 0 to i-1.
            prev_caplets = dt * BSfwd_local(Fwd[0:i], Strike, ZSw[1:i+1], ImplVol[0:i], IntMat[0:i], 1)
            SumPrev = np.sum(prev_caplets)

            # New Caplet Value
            Val_New = Caps[i] - SumPrev
            Caplet[i] = Val_New

            # Invert BS to find Vol
            # BS(Vol) = Val_New
            # F = Fwd[i], T = IntMat[i], Z = ZSw[i+1].

            def obj(v):
                return (dt * BSfwd_local([Fwd[i]], [Strike], [ZSw[i+1]], [v], [IntMat[i]], 1)[0] - Val_New)**2

            res = fsolve(obj, 0.2)
            ImplVol[i] = res[0]

        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(x=IntMat[1:NN], y=ImplVol, mode='lines', name='Forward Vol'))
        fig2.add_trace(go.Scatter(x=IntMat[1:NN], y=IntVol[1:NN], mode='lines', name='Flat Vol'))
        fig2.update_layout(title='Volatility Term Structure', xaxis_title='Maturity', yaxis_title='Volatility')

        # =================== BDT Tree ===========================
        print("Building BDT Tree...")

        dtstep = 0.25
        IntMat2 = IntMat
        ImplVol2 = ImplVol
        ZSw2 = ZSw

        Maturity = arange(dtstep, IntMat2[-2] + dtstep, dtstep) # Guide logic
        # Just use IntMat indices.

        ZYield = -log(ZSw2[:-1]) / IntMat2[:-1]
        FwdVol = ImplVol2

        N_Tree = len(ZYield)

        # Shift? Guide: "dy=0; ZYield = ZYield + dy"

        ImTree = zeros((N_Tree, N_Tree))

        # First node
        ImTree[0, 0] = ZYield[0]

        xx = ZYield[0]

        for j in range(N_Tree - 1):
            xx = xx * 0.75 # Guess
            # Solve for rmin at step j+1 (Index j+1)
            # Tree filled up to j. Filling column j+1.
            # Target ZYield[j+1].
            # Vol FwdVol[j].
            # Step j+1 corresponds to maturity (j+2)*dtstep.

            args = (ImTree, ZYield[j+1], FwdVol[j], dtstep, j+2)
            res = fsolve(minfunBDTNew_fsolve, xx, args=args)
            xx = res[0]

            _, vec = minfunBDTNew(xx, ImTree, ZYield[j+1], FwdVol[j], dtstep, j+2)
            # ImTree updated in function? Yes, but cleaner to update here.
            ImTree[0:j+2, j+1] = vec

        # =================== Mortgages ===========================
        print("Pricing Mortgages...")

        WAC = 4.492 / 100
        WAM = int(round(311 / 12) / dtstep) # 311 months? / 3 = 103 quarters?
        # 311 months / 3 months/quarter = 103.66.
        # Guide: "int(round(311/12)/dtstep)" -> 311/12 years approx 25.9.
        # / 0.25 = 103.6. -> 104 steps?
        # WAM = 104.
        WAM = 104

        PP0 = 100
        aa = 1 / (1 + WAC * dtstep)

        rbar = 4 / 100

        NN_M = WAM
        # Monthly coupon if monthly model? But we use quarterly steps.
        # Guide: "MCoupon = PP0*(1-aa)/(aa-aa**(NN+1))"
        # This is the annuity payment per period.

        Payment = PP0 * (1 - aa) / (aa - aa**(NN_M + 1))

        PP = zeros(NN_M + 1)
        PriPaid = zeros(NN_M)
        IntPaid = zeros(NN_M)

        PP[0] = 100

        for i in range(NN_M):
            IntPaid[i] = PP[i] * WAC * dtstep
            PriPaid[i] = Payment - IntPaid[i]
            PP[i+1] = PP[i] - PriPaid[i]

        fig3 = go.Figure()
        fig3.add_trace(go.Scatter(y=PP, mode='lines', name='Principal Balance'))
        fig3.update_layout(title='MBS Scheduled Principal')

        # Pricing GNSF 4
        # BDT Tree size might be smaller than WAM?
        # ImTree size N_Tree (~120). WAM 104. Fits.

        VNoC = zeros((NN_M, NN_M))
        Call = zeros((NN_M, NN_M))
        VC = zeros((NN_M, NN_M))
        VPT = zeros((NN_M, NN_M))
        VPTNoC = zeros((NN_M, NN_M))

        # Backward
        # Maturity NN_M-1?
        # Guide loops "range(NN-2, -1, -1)".

        # Terminal conditions
        # At WAM, Principal is 0?
        # Or remaining principal paid off?
        # Scheduled pays off exactly.

        for i in range(NN_M - 2, -1, -1):
            # Discount rate at step i: ImTree[:, i]
            # Nodes j: 0 to i.

            rates = ImTree[0:i+1, i]
            disc = exp(-rates * dtstep)

            # Payment at i+1
            # CF = Payment (Principal + Interest)
            # Or Pass Through CF?
            # CFNoPreP = Monthly Coupon (Payment)

            CF_Sched = Payment

            # VNoC: Value of scheduled payments
            # VNoC[j,i] = Disc * (0.5 * VNoC_up + 0.5 * VNoC_down + CF_Sched)
            # VNoC at i+1 needed.
            # Initialize terminal?
            # At last step, Value is CF_Sched?

            # Let's fix loop range to handle terminal properly.
            # i goes up to WAM.
            # Loop i from WAM-1 down to 0.
            pass

        # Re-implement loop carefully
        # Initialize at WAM (Time T)
        # Value is 0 (all paid).
        # Or Value just before last payment is Payment / (1+r).

        # VNoC size (NN_M+1, NN_M+1) for safety
        # VNoC size (NN_M+1, NN_M+1) for safety
        # WAM = NN_M = 104.
        # Steps 0 to 104. 105 steps.
        VNoC = zeros((NN_M + 1, NN_M + 1))
        VC = zeros((NN_M + 1, NN_M + 1))
        Call = zeros((NN_M + 1, NN_M + 1)) # Initialize Call array

        # Last payment occurs at WAM.
        # At step WAM-1, we discount Payment at WAM.

        for i in range(NN_M - 1, -1, -1):
            rates = ImTree[0:i+1, i]
            disc = exp(-rates * dtstep)

            # VNoC
            val_up = VNoC[0:i+1, i+1]
            val_down = VNoC[1:i+2, i+1]

            VNoC[0:i+1, i] = disc * (0.5 * val_up + 0.5 * val_down + Payment)

            # Callable (Prepayment)
            # Prepay Option: Borrower pays Remaining Principal (PP[i]) to retire debt.
            # They do so if Value of Liability (VNoC) > Principal (PP[i]).
            # Market Value of Debt > Face Value.
            # If they prepay, Value to Investor = PP[i] + Interest?
            # Usually Prepayment happens, Investor gets Principal.
            # Value of Callable Mortgage VC = min(VNoC, PP[i] + Accrued?).
            # Guide: "Call[j,i] = max(VNoC - PP, 0)"?
            # "VC[j,i] = VNoC[j,i] - Call[j,i]"
            # "if Call == VNoC - PP: ExIdx = 1".
            # This implies Call Value is VNoC - PP (Intrinsic).
            # But Option Value is Expectation...
            # "Call[j,i] = Disc * (0.5*Call_up + 0.5*Call_down)".
            # "Check Exercise: Call = max(Call_Hold, VNoC - PP)".

            c_up = VC[0:i+1, i+1] # Wait, separate Option Value array?
            # Guide tracks Call Value separately.
            # Let's track VC directly.
            # VC = min(VNoC, PP[i]).
            # Wait, strictly refinancing means evaluating option to call.
            # Guide Logic:
            # Call[j,i] = Disc * (Avg Next Calls).
            # Payoff = VNoC[j,i] - PP[i].
            # Call[j,i] = max(Call_Hold, Payoff).
            # VC[j,i] = VNoC - Call.

            # Let's follow Guide.
            # Initialize Call at terminal = 0.

            call_up = Call[0:i+1, i+1]
            call_down = Call[1:i+2, i+1]

            call_hold = disc * (0.5 * call_up + 0.5 * call_down)
            intrinsic = VNoC[0:i+1, i] - PP[i] # Current Value of Liability - Current Principal Payoff Cost
            # Note: Should we add interest to PP[i]? PP[i] is principal balance.
            # Usually prepay = Pay Balance.
            # VNoC includes current payment?
            # VNoC[j,i] is value at t=i (including payment at t=i+1? No.
            # VNoC[j,i] computed as Disc * (Next + Payment).
            # This is Ex-Payment at i?
            # Or Cum-Payment at i+1?
            # Standard: V_t = E[D * (V_{t+1} + CF_{t+1})]. This is Ex-CF_t value.
            # So VNoC is value of remaining flows.
            # If prepay, we pay PP[i].
            # So comparison VNoC vs PP[i] is correct.

            Call[0:i+1, i] = maximum(call_hold, intrinsic)
            VC[0:i+1, i] = VNoC[0:i+1, i] - Call[0:i+1, i]

        print(f"Value No Call: {VNoC[0,0]:.4f}")
        print(f"Value Callable: {VC[0,0]:.4f}")

        return {
            "VNoC": VNoC[0,0],
            "VC": VC[0,0],
            "fig1": fig1,
            "fig2": fig2,
            "fig3": fig3
        }

    if __name__ == "__main__":
        solve_pset7()

    return

if __name__ == "__main__":
    app.run()
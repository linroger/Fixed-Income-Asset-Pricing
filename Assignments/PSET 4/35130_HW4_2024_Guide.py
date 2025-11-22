# %% Packages
import math
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from numpy import arange, exp, sqrt, log, mean, std, nonzero, isnan, array, zeros, ones, amax, amin, hstack, vstack
from numpy import concatenate, arange, interp, diff, cov, var, diag, eye, cumsum, tile, transpose
from scipy.optimize import fmin
from NLLS_Func import NLLS, NLLS_Min
from YTM_Func import YTMsolver
from scipy.interpolate import interp1d, PchipInterpolator

plt.close('all')   # Close all charts
print("\n" * 100)  # Clear Console




#%% Controls
part1 = 1  # put = 1 to do part 1
part2 = 0  # put = 1 to do part 2. It only works if part1=0.



#%% PART I: TIPS
if part1==1:
    DataR     = array(pd.read_excel('DataTIPS.xlsx',sheet_name='TIPS_01152013',skiprows=4,usecols=arange(2,11)))
    CashFlowR = array(pd.read_excel('DataTIPS.xlsx',sheet_name='TIPS_01152013',skiprows=4,usecols=arange(12,44)))  # This is the cash flow matrix (N x T). Cash flow of each bond corresponding to each maturity.
    MatR      = array(pd.read_excel('DataTIPS.xlsx',sheet_name='TIPS_01152013',skiprows=4,usecols=arange(45,77)))  # Maturity Matrix (N x T) where N=number of bonds, T=max number of maturities.

    BidR   = DataR[:,1]
    AskR   = DataR[:,2]
    PriceR = (BidR+AskR)/2

    # Use Nelson Siegel Model.
    vecR0    = array((1.3774,-2.1906,-6.7484,244.0966))/100       # use solution as starting value.
    vecR     = fmin(func=NLLS_Min,x0=vecR0,args=(PriceR,MatR,CashFlowR))
    J,PPhatR = NLLS(vecR,PriceR,MatR,CashFlowR)

    # translate solution back in "formulas"
    th0R = vecR[0]; th1R = vecR[1]; th2R = vecR[2]; laR = vecR[3]

    plt.figure(1)
    plt.plot(DataR[:,4],PPhatR,'o',DataR[:,4],PriceR,'*')
    plt.legend(('Fitted','Data'),loc='best')
    plt.xlabel('Maturity')
    plt.ylabel('Price')
    plt.title('TIPS')

    hs   = 1/2         # time step
    T    = arange(hs,30+hs,hs)  # times
    YccR = ?? # real yield [insert formula from teaching notes]

    # Real Discount
    ZZccR = ??   # Real discount function Z_{real}(T)

    plt.figure(2)
    plt.plot(T,ZZccR,linewidth=2);
    plt.xlabel('Maturity')
    plt.ylabel('Z')
    plt.title('Real Discount')

    plt.figure(3)
    plt.plot(T,YccR,linewidth=2)
    plt.xlabel('Maturity')
    plt.ylabel('Yield')
    plt.title('Real Yield')
    #plt.show()





#%% PART I: Nominal Rates
if part1==1:
    DataN     = array(pd.read_excel('DataTIPS.xlsx',sheet_name='Treasuries_01152013',skiprows=5,usecols=arange(1,9)))
    CashFlowN = array(pd.read_excel('DataTIPS.xlsx',sheet_name='Treasuries_01152013',skiprows=5,usecols=arange(10,70)))  # This is the cash flow matrix (N x T). Cash flow of each bond corresponding to each maturity.
    MatN      = array(pd.read_excel('DataTIPS.xlsx',sheet_name='Treasuries_01152013',skiprows=5,usecols=arange(71,131)))  # Maturity Matrix (N x T) where N=number of bonds, T=max number of maturities.

    Bid        = DataN[:,3]
    Ask        = DataN[:,4]
    CleanPrice = (Bid+Ask)/2
    AccrInt    = DataN[:,7]/2

    # Compute Dirty Prices
    PriceN = ??

    # Use Nelson Siegel Model.
    vec0    = array((3.9533,-2.6185,-7.3917,200.9313))/100       # use solution as starting value.
    vec     = fmin(func=NLLS_Min,x0=vec0,args=(PriceN,MatN,CashFlowN))
    J,PPhat = NLLS(vec,PriceN,MatN,CashFlowN)

    # translate solution back in "formulas"
    th0 = vec[0]; th1 = vec[1]; th2 = vec[2]; la = vec[3]

    plt.figure(4)
    plt.plot(DataN[:,6],PPhat,'o',DataN[:,6],PriceN,'*')
    plt.legend(('Fitted','Data'),loc='best')
    plt.xlabel('Maturity')
    plt.ylabel('Price')
    plt.title('Treasuries')


    hs   = 1/2         # time step
    T    = arange(hs,30+hs,hs)  # times
    Ycc = ?? # nominal rate [insert formula. Use hs and T defined above for real rate]

    # Nominal Discount
    ZZcc = ??

    # Fwd rates
    FWD = ??
    FWD = hstack((Ycc[0],FWD))

    plt.figure(5)
    plt.plot(T,ZZcc,linewidth=2)
    plt.xlabel('Maturity')
    plt.ylabel('Z')
    plt.title('Nominal Discount')

    plt.figure(6)
    plt.plot(T,Ycc,T,FWD,'--',linewidth=2)
    plt.xlabel('Maturity')
    plt.ylabel('Yield')
    plt.legend(('yield', 'forward'))
    plt.title('Nominal Yield and Forward')


    # Break Even Rates(CC)
    pi = ??

    plt.figure(7)
    plt.plot(T,pi,linewidth=2)
    plt.xlabel('Maturity')
    plt.ylabel('Rate')
    plt.title('Break Even Rate')
    plt.show()



# %% PART II
if part2==1:
    print('==================================================================')
    print('PART 1 - SWAP SPREAD TRADES')
    print('==================================================================')

    #%% Examine Historical Data

    #Load and Prep Data
    H15_SWAPS = array(pd.read_csv('H15_SWAPS.txt', sep='\s+', header=None, skiprows=15))

    II        = (np.isnan(H15_SWAPS[:,2])==0)  # elimiinate NaN's
    H15_SWAPS = H15_SWAPS[II,:]                # redefine data without NaN's

    Dates = H15_SWAPS[:,0]                # dates
    LIBOR = H15_SWAPS[:,1]                # LIBOR rates
    Repo  = H15_SWAPS[:,2]                # Repo rates
    Swaps = H15_SWAPS[:,3:11]             # Swap rates across maturity
    CMT   = H15_SWAPS[:,11:]              # US Treasury constant maturity rates


    # Creat a plot
    plt.figure(10)
    plt.plot(Swaps[:,4]-CMT[:,3],linewidth=1)       # plot Swap Spread
    plt.plot(LIBOR-Repo) # plot LIBOR - Repo spread
    plt.title('5 year Swap Spread and Funding Spread over time')
    plt.legend(('Swap Spread','Funding Spread'))
    #plt.show()


    #%% Load and Prep Swap and Treasury Data for trading exercise

    linethick = 2 # for use with plots
    freq = 2

    # load prices for 5 year T-note

    data = array(pd.read_excel('HW4_Data.xls', sheet_name='Daily_Bond_Swaps', skiprows=12, usecols=arange(1,29)))
    QuoteDates = data[:,0]
    MatDates   = data[:,1]
    Coupon     = data[:,2]
    bid        = data[:,3]
    ask        = data[:,4]
    maturity   = data[:,5]
    AccInt     = data[:,6]
    LIBOR      = data[:,7]
    Repo       = data[:,8]
    SWAPS      = data[:,arange(9,17)]
    CMT        = data[:,17:]
    PriceTnote = (bid + ask)/2



    #%% Setup Trade
    # Examine the usual side of the trade. At the end take negative to get
    # profit on other side of the trade.

    # (A) reverse Repo the 5 year note, and short in to the market.
    #      Pay coupon and receive Repo rate
    # (B) enter swap, receive fixed swap rate, pay floating, 3 M LIBOR

    Swap30     = SWAPS[:,-1]              # time seirs of 30-year swap rates
    SwapRate   = Swap30[0]                # 30-year swap rate at trade date
    CouponRate = Coupon[0]                # Bond coupon rate
    SS         = SwapRate-CouponRate;     # Swap Spread
    LRS        = LIBOR[0]-Repo[0]         # LIBOR-Repo Spread

    print('Swap Spread minus Carry')
    print(100*(SS - LRS))

    # Yield to maturity of Note is
    YTM = YTMsolver(PriceTnote[0],CouponRate,freq,5)

    # LIBOR Swap curve at time 0 (this only required for later plot)
    DataMat   = hstack((0.25,arange(1,6),7,10,30))   # maturity points given by data
    IntMat    = arange(0.25,30+0.25,0.25)            # desired points on curve to do interpolation
    DataRates = hstack((LIBOR[0],SWAPS[0,:]))

    f       = PchipInterpolator(DataMat,DataRates)
    IntSwap = f(IntMat)

    # Compute Discounts of Swap
    ZL = zeros(IntSwap.shape[0])
    ZL[0]=??
    for jj in range(1,IntSwap.shape[0]): # bootstrap method: For every other maturity, use formula in TN3 and IntSwap as the swap rates
        ZL[jj] = ??
    LIBOR_Curve_0 = ?? # continuous compounded curve;

    # POSITION OF TRADE
    position = 1e8 # size of trade = 100 million
    Tnotes   = position/(PriceTnote[0]+AccInt[0])  # numbers of T-notes bought


    #%% Trade Value after one Quarter

    ValueTNotes = Tnotes*(PriceTnote[0]+AccInt[0])        # Value of notes on the first date
    ValueRepo   = position-Tnotes*(PriceTnote[0]+AccInt[0]) # Value of notes on the second date
    SwapSpread  = SS

    # One quarter later
    Today = 20090518

    # find this date in dataset
    idxToday = nonzero(Today==QuoteDates)

    # compute cash flows
    CF_SS_Today  = ??   # Cash flow from Swap Spread (assume you get accrued interest every quarter for simplicity)
    CF_LRS_Today = ??   # Cash flow from LIBOR-REPO Spread

    # compute value of position
    ValueTNotes_Today = ??  # Total value of T-Bonds?;
    ValueRepo_Today   = ??  # Value Repo (assume no haircut, and this is ex-payment of Repo rate)

    # to value the old swap, we must first extract the LIBOR curve
    # Compute LIBOR Curve Z^{LIBOR}(t;T) by INTERPOLATION
    DataMat   = hstack((0.25,arange(1,6),7,10,30))   # maturity points given by data
    IntMat    = arange(0.25,30+0.25,0.25)            # desired points on curve to do interpolation
    DataRates = hstack((LIBOR[idxToday][0],SWAPS[idxToday,:][0,0,:]))

    f       = PchipInterpolator(DataMat,DataRates)
    IntSwap = f(IntMat)

    # Compute Discounts of Swap
    ZL = zeros(IntSwap.shape[0])
    ZL[0]=??  # What is the first discount?
    for jj in range(1,IntSwap.shape[0]):
        ZL[jj] = ?? # Use formula from TN to bootstrap the LIBOR curve.
    LIBOR_Curve_Today = ?? # Trasform Z(T) into continuously compounded rate.

    # Given swap curve, value swap as long fixed coupon rate and
    # short floating. (Recall to adjust for the maturity)
    npay            = 30*4-1  # number of payments left AFTER today
    ValueSwap_Today = ??  # value of the swap

    print('Total Values on May 8, 2009')
    print('Repo,  Swap , Swap + Repo')
    print([ValueRepo_Today[0],ValueSwap_Today,ValueSwap_Today+ValueRepo_Today[0]])

    plt.figure(20)
    plt.plot(IntMat,LIBOR_Curve_0,IntMat,LIBOR_Curve_Today,'--',linewidth=2)
    plt.title('LIBOR curve on Feb 2 and May 18, 2009')
    plt.xlabel('Maturity')
    plt.ylabel('yield')
    plt.legend(('Feb 2, 2009', 'May 8,2009'))
    plt.show()

a=1


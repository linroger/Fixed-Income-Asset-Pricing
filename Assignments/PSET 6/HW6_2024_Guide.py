# %% Packages
import math
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from numpy.linalg import inv
from numpy import arange, exp, sqrt, log, mean, std, nonzero, isnan, array, zeros, ones, amax, amin, maximum, minimum
from numpy import concatenate, arange, interp, diff, cov, var, diag, eye, cumsum, tile, transpose, nan, hstack, vstack
from numpy import setdiff1d
from scipy.interpolate import interp1d, PchipInterpolator
from scipy.optimize import fmin
from scipy.stats import norm
from numpy.random import normal
from NLLS_Func import NLLS, NLLS_Min
from HoLee_SimpleBDT_Tree import fmin_HoLee_SimpleBDT_Tree,HoLee_SimpleBDT_Tree




plt.close('all')   # Close all charts
print("\n" * 100)  # Clear Console



print('==================================================================')
print('INTEREST RATE TREES  & VALUATION')
print('==================================================================')




#%% load data

Data     = array(pd.read_excel('/Users/rogerlin/Downloads/HW6_Data_Bonds.xls',sheet_name='Sheet1',skiprows=4,usecols=arange(9)))
Mat      = array(pd.read_excel('/Users/rogerlin/Downloads/HW6_Data_Bonds.xls',sheet_name='Sheet1',skiprows=4,usecols=arange(10,70)))    # Maturity Matrix (N x T) where N=number of bonds, T=max number of maturities.
CashFlow = array(pd.read_excel('/Users/rogerlin/Downloads/HW6_Data_Bonds.xls',sheet_name='Sheet1',skiprows=4,usecols=arange(72,132)))   # % This is the cash flow matrix (N x T). Cash flow of each bond corresponding to each maturity.


# assign names to some of the data
Bid        = Data[:,5]
Ask        = Data[:,6]
CleanPrice = (Bid+Ask)/2
Price      = CleanPrice + Data[:,8]     # add accrued interest to the clean prices.



#%% Use Nelson Siegel Model.

vec0 = array([5.3664,-0.1329,-1.2687,132.0669])/100    # use solution as starting value.

vec  = fmin(func=NLLS_Min,x0=vec0,args=(Price,Mat,CashFlow))  
# minimization algorithm. The 'NLLS' calls the matlab file with the minimization.
# vec0 is starting value; Price, Mat, CashFlow are passed to ENLLS.m to compute the minimizing function.
# translate solution back in "formulas"
th0 = vec[0]; th1 = vec[1]; th2 = vec[2]; la = vec[3]

J, PPhat = NLLS(vec, Price, Mat, CashFlow)

plt.figure(1)
plt.plot(Data[:,7],PPhat,Data[:,7],Price,'*')
plt.legend(('Fitted','Data'))
plt.xlabel('Maturity')
plt.ylabel('Price')

hs  = 1/2
T   = arange(hs,30+hs,hs)
Ycc = th0+(th1+th2)*(1-exp(-T/la))/(T/la)-th2*exp(-T/la)

# Discount
ZZcc=exp(-Ycc*T)

# Fwd rates
FWD = -log(ZZcc[1:]/ZZcc[0:-1])/hs
FWD = hstack((Ycc[0],FWD))

plt.figure(2)
plt.plot(T,ZZcc)
plt.xlabel('Maturity')
plt.ylabel('Z')

plt.figure(3)
plt.plot(T,Ycc,T,FWD,'--')
plt.xlabel('Maturity')
plt.ylabel('Yield')
plt.legend(('yield','forward','location','northwest'))




#%% BUILDING A RECOMBINING BINOMIAL TREE

# load time series of six month rates
data = array(pd.read_csv('HW6_FRB_H15.csv',skiprows=5,usecols=[1]))[:,0]
n = data.shape[0]
DataY6 = data[-12*10-1:]/100  # back in decimals
P6     = (1-180/360*DataY6)
rr6    =  -1/0.5*log(P6)

r0 = Ycc[0]

# Build tree, semi-annual increments
BDT_Flag=0 # = 0: HoLee; =1: SimpleBDT
if BDT_Flag==1:
    sigma = ???    # note: units (month) must be annualized
else:
    sigma = ???    # note: units (month) must be annualized

ImTree = zeros((int(30/hs),int(30/hs)))
ZZTree = zeros((int(30/hs+1),int(30/hs+1),int(30/hs)))

ImTree[0,0]     = r0              # root of the tree
ZZTree[0,0,0]   = exp(-r0*hs)     # first zero coupon bond (maturity i=2)
ZZTree[0:1,1,0] = 1

ImTree_1 = ImTree

FFF   = zeros(int(30/hs)-1)
theta = zeros(int(30/hs)-1)
for i in arange(1,int(30/hs)):

    theta_i    = fmin(func=fmin_HoLee_SimpleBDT_Tree,x0=0,args=(ZZcc[i],ImTree_1,i,sigma,hs,BDT_Flag))

    theta[i-1] = theta_i

    FF,ImTree,ZZTreei = HoLee_SimpleBDT_Tree(theta_i,ZZcc[i],ImTree_1,i,sigma,hs,BDT_Flag)
    FFF[i-1] = FF
    ImTree_1 = ImTree

    ZZTree[0:i+2,0:i+2,i] = ZZTreei


yyTree = -log(ZZTree[0,0,:])/T

plt.figure(4)
plt.plot(T,yyTree,T,Ycc,'--',linewidth=2)
plt.title('Yields: Data versus Tree')
plt.xlabel('Maturity')
plt.ylabel('yield')
plt.legend(('Tree','Data'),loc='best')

if BDT_Flag==0: # this only works if we are in the original HoLee model
    plt.figure(5)
    plt.plot(T[:-1],theta,T[:-1],(FWD[1:]-FWD[:-1])/hs+sigma**2*T[:-1],'--',linewidth=2)
    plt.title('Yields: Data versus Tree')
    plt.xlabel('Maturity')
    plt.ylabel('yield')
    plt.legend(('theta','d fwd_dt + sigma^2 T'),loc='best')




#%% Price of the Freddie Mac callable bond on the tree

coupon = ???
TBond  = ???
FCT    = ???                  # first call time
iT     = int(TBond/hs+1)      # step at maturity: always 1 more step than T/hs for maturity
iFCT   = int(FCT/hs+1)        # step at first call time

pi     = 0.5             # Risk Neutral Probability of up movement

PPTree_NC = zeros((iT,iT))    # initialize the matrix for the non-callable coupon bond with maturity i.
Call      = zeros((iT,iT))    # initialize the matrix for American call coupon bond with maturity i.

# final payoff of non-callable bond
PPTree_NC[0:iT,iT-1] = 100      # final price is equal to 100


# backward algorithm
for j in range(iT-1,0,-1):
    PPTree_NC[0:j,j-1] = ???
    if j>=iFCT:
        Call[0:j,j-1] = ???
    else:
        Call[0:j,j-1] = ???

PPTree_C = ???

print('Price Freddie Mac Callable Bond')
print('P_NC, Call, P_Call')
print(np.round(array([PPTree_NC[0,0],Call[0,0],PPTree_NC[0,0]-Call[0,0]]),4))

plt.figure(6,figsize=(8,6))
for j in range(1,5):
    plt.subplot(2,2,j)
    plt.plot(ImTree[0:iFCT-(j-1),iFCT-(j-1)],PPTree_NC[0:iFCT-(j-1),iFCT-(j-1)],'--',ImTree[0:iFCT-(j-1),iFCT-(j-1)],PPTree_C[0:iFCT-(j-1),iFCT-(j-1)],linewidth=2)
    #plt.xlabel('Interest Rate')
    plt.ylabel('Bond Prices')
    plt.legend(('Non-Callable','Callable'),loc='best')
    if j==1:
        titlestr = 'FCD'
    else:
        titlestr = str(j-1) + ' periods before FCD'
    plt.title(titlestr)





#%% Duration and Convexity

# Duration - Non-Callable
DNC = ???

# Duration - Callable
DCallable = ???

# Convexity - Non-Callable
Delta_NC_1u = (PPTree_NC[0,2]-PPTree_NC[1,2])/(ImTree[0,2]-ImTree[1,2])
Delta_NC_1d = (PPTree_NC[1,2]-PPTree_NC[2,2])/(ImTree[1,2]-ImTree[2,2])
C_NC        = ???

# Convexity - Callable
Delta_Call_1u = (PPTree_C[0,2]-PPTree_C[1,2])/(ImTree[0,2]-ImTree[1,2])
Delta_Call_1d = (PPTree_C[1,2]-PPTree_C[2,2])/(ImTree[1,2]-ImTree[2,2])
C_Callable    = ???



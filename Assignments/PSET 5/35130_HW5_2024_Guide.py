# %% Packages
import math
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from numpy import arange, exp, sqrt, log, mean, std, nonzero, isnan, array, zeros, ones, amax, amin, hstack, vstack
from numpy import concatenate, arange, interp, diff, cov, var, diag, eye, cumsum, tile, transpose, nan
from scipy.interpolate import interp1d, PchipInterpolator
from BlackScholes import BSfwd

plt.close('all')   # Close all charts
print("\n" * 100)  # Clear Console



#%% Input data
dataMat   = hstack((1/12,arange(1,7)))                                 # maturity points given by data
intMat    = arange(0.25,6+0.25,0.25)                                         # desired points on curve to do interpolation
dataRates = array((???  ???  ???  ???  ???  ???  ???  ???))/100 # rates points given by data

# Interpolated swap curve at quarterly frequency
f        = PchipInterpolator(dataMat,dataRates)
intRates = f(intMat)



#%% Bootstrap

# Discount Factor calculations via Bootstrap
dt    = 0.25
ZZ = zeros(intRates.shape[0])
ZZ[0] = 1/(1+dt*intRates[0])
for i in range(1,intRates.shape[0]):
    ZZ[i] = (1-intRates[i]*dt*np.sum(ZZ[0:i]))/(1+intRates[i]*dt)

fwd_Z          = hstack((nan,ZZ[1:]/ZZ[0:-1]))           # Forward Discount
fwd_Rate       = (1/fwd_Z-1)/dt                          # Forward Rate
fwd_Z_Swaption = hstack((nan,nan,nan,nan,ZZ[4:]/ZZ[3]))  # Forward Discount for Swaption calculations



#%% 1-year Cap

T     = 1                     # Time-to-maturity (years)
X     = ???                   # Strike Rate (from data)
sigma = ???                    # Volatility  (from data)
F     = fwd_Rate[1:int(T/dt)] # Forward Rates
Z     = ZZ[1:int(T/dt)]       # Discount Factors
CallF = 1                     # 1 for calls and 2 for puts
T_Cap = arange(dt,T,dt)       # caplet maturities

Caplets_1Year = 100*dt*BSfwd(F,X,Z,sigma,T_Cap,CallF)   # Caplet prices (use BSfwd.py) 
Cap_1Year     = np.sum(Caplets_1Year)                  # Cap price

TABLE_1Year = vstack((arange(dt,T+dt,dt),hstack((nan,F)),hstack((nan,Z)),hstack((nan,Caplets_1Year)))).T
print(('1-year Cap Price = ' + round(Cap_1Year,5).astype('str')))




#%% 2-year Cap

T     = 2                     # Time-to-maturity (years)
X     = ???                   # Strike Rate (from data)
sigma = ???                   # Volatility (from data)
F     = fwd_Rate[1:int(T/dt)] # Forward Rates
Z     = ZZ[1:int(T/dt)]       # Discount Factors
CallF = 1                     # 1 for calls and 2 for puts
T_Cap = arange(dt,T,dt)       # caplet maturities

Caplets_2Year = 100*dt*BSfwd(F,X,Z,sigma,T_Cap,CallF)  # Caplet prices (use BlackScholes.py)
Cap_2Year     = np.sum(Caplets_2Year)                  # Cap price

TABLE_2Year = vstack((arange(dt,T+dt,dt),hstack((nan,F)),hstack((nan,Z)),hstack((nan,Caplets_2Year)))).T
print(('1-year Cap Price = ' + round(Cap_2Year,5).astype('str')))




#%% Swaption
T     = 1        # Swaption Maturity
X     = ???      # Strike Rate (5-year swap rate) (from data)
sigma = ???     # Swaption Volatility (from data)
Z     = 1        # Discount Factors
CallF = 1        # 1 for calls and 2 for puts

# Fwd 5-year Swap Rate
SwapRate = 1/dt*(1-fwd_Z_Swaption[-1])/np.sum(fwd_Z_Swaption[4:])

# A-factor
A = A = dt*np.sum(ZZ[4:])

# Swaption price
Swaption = A*BSfwd(SwapRate,X,Z,sigma,T,CallF) # use BlackScholes.py.  Compare formula with TNs to understand what inputs to use.

print(('Swaption Price = ' + round(Swaption,5).astype('str')))




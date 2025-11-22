#%% Bus 35130 HW1

# Guide to the solution
# This file is not going to work as is. Your task is to complete the file
# by filling in the ?? scattered through the file. 


# %% Packages
import math
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from numpy import exp, sqrt, log, mean, std, nonzero, isnan, array, zeros, ones, amax, amin
from numpy import concatenate, arange, interp, diff, cov, var, diag, eye, cumsum, tile, transpose

plt.close('all')       # Close all charts
print("\n" * 100)      # Clear Console


#%% =================== Question 1 ===========================
# Determining time series of BEY and provide its plot

# Read time series of 3-month Treasury Bills from 'DTB3_2023.xls' Excel file, sheet 'Dtb3'
Data_TBill = pd.read_excel('DTB3_2024.xls',sheet_name='DTB3',skiprows=10,names=list(['DATE,'DTB3']))
rates = array(Data_TBill.DTB3)

II = ~np.isnan(rates)              # find business dates and ignore days when markets are closed
rates = rates[II]/100                 # get rid of percentages by dividing by 100
h
# input in BEY formula
N1 = 365
N2 = 360
N3 = 91
BEY = N1*rates/(N2-rates*N3)          # BEY calc according to the formula

# Plot the results
fig = plt.figure(1,figsize=(8,8))
plt.plot(rates)                         # plot of the T-bill rates
plt.plot(BEY)                           # actual plot  
plt.ylabel('3-month T-bill rates')      # label on y axis
plt.legend(['Quoted discounts', 'BEY'])   # plot legend






#%% =================== Question 2 ===========================
# Estimating of the AR(1) process for interest rates

m = ??     # the number of regression observations
Y = BEY[??]            # the dependent variable
X = BEY[??]          # the independent variable

# OLS regression
ybar = mean(Y) # Unconditional mean of Y variable defined above
xbar = mean(X) # Unconditional mean ofXY variable defined above

# beta_hat = cov(X,Y)/var(X), but the Python function 'cov' gives the
# 2 by 2 covariance matrix, not just the covariance, so we use just one number s(1,2)
# Complete the following

s         = ??                # Covariance calculation
beta_hat  = ??           # Regression slope, beta
alpha_hat = ??                # Regression intercept, alpha

# Calculating error terms
eps = Y-alpha_hat-beta_hat*X    # Regression residuals    
sig = np.sqrt(var(eps)*m/(m-1)) # Sample standard error of residuals

# Displaying the results in matlab's Command Window
print(' Standard approach ')
print('===========================================================================')
print('Regression coefficients for rates:')
print('beta_hat = ',str(beta_hat))
print('alpha_hat = ',str(alpha_hat))
print('Standard error of residuals:')
print('sig = ',str(sig))
print('===========================================================================')
print(' ')




#%% =================== Question 3 ===========================
# Interest rate forecasts over the period of 5 years

n = ??                                 # number of years to forecast

rate_forecast = zeros(n*252+1)     # vector to store the forecasts
rate_forecast[0] = BEY[-1]            # r_today - the most recent interest rate

# Loop to carry out AR(1) forecast
# r_forecast(i+1) = alpha + beta_hat * r_forecast(i). Fill in the blanks
for i in range(n*252):
    rate_forecast[??] = ??+??*rate_forecast[??]

# Plot the results
plt.figure(2,figsize=(8,8))
x = arange(0,n+1/252,1/252)
LR_mean = alpha_hat/(1-beta_hat)   # Long-run mean of the interest rate

plt.plot(x,rate_forecast)
plt.plot(x,LR_mean*ones((len(x),1)),'--',linewidth=2)
plt.ylim((0,0.07))
plt.xlabel('forecasting horizon (years)')
plt.ylabel('3-month T-bill rates')
plt.legend(('Time Series Forecast','Long-term interest rate'))




#%% =================== Question 4 ===========================
# Computing both the current yield curve and forward rates for all 
# maturities and comparing the forecasts of future interest rates 
# that are implicit in the forward rates to those obtained in question 3

strips_data = pd.read_excel('DTB3_2024.xls', sheet_name='Strip Prices', names=list(['Mat','Price']))

Mat  = array(strips_data.Mat)                    # Store maturities into a vector
Imat = (nonzero(Mat<n+0.25))                     # Find index with maturity n
Mat  = Mat[Imat]                                 # Take all the maturities up to n
Zfun = array(strips_data.Price); 
Zfun = Zfun[Imat]


yield1 = ??                      # Compute yields from strips (quoted term structure)
fwds = ??            # Compute forward rates


# Plot the results: Z-function only
plt.figure(3)
plt.plot(Mat,Zfun,linewidth=2)
plt.xlabel('forecasting horizon (years)')
plt.ylabel('Z function')
plt.title('Discount function')

# Plot the results: yields vs forward rates
plt.figure(4)
plt.plot(Mat,yield1,Mat,fwds,'-.',linewidth=2)
plt.xlabel('Maturity')
plt.ylabel('spot rate')
plt.title('Yields and Forwards')
plt.legend(('Yield','Forward'),loc='best')

# Plot the results: forward rates vs AR(1) forecast of interest rates
plt.figure(5)
plt.plot(??,??,'-.',??,??,linewidth=2)
plt.xlabel('forecasting horizon (years)')
plt.ylabel('spot rate')
plt.title('Two forecasts of future interest rates')
plt.legend(('Forward','Time Series Forecast'),loc='best')



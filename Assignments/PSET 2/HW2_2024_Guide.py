#%% Bus 35130 HW2

# Guide to the solution
# This file is not going to work as is. Your task is to complete the file
# by filling in the ?? scattered through the fil

# %% Packages
import math
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from numpy import exp, sqrt, log, mean, std, nonzero, isnan, array, zeros, ones, amax, amin
from numpy import concatenate, arange, interp, diff, cov, var, diag, eye, cumsum, tile, transpose
from numpy.linalg import inv
from scipy.stats import norm

plt.close('all')       # Close all charts
print("\n" * 100)      # Clear Console

#%% Question 1. Bootstraping the term structure

linethick = 2 # Thinkness of the line for use with plots

# displaying the results in the command window
print('==================================================================')
print('LEVERAGED INVERSE FLOATER')
print('==================================================================')

for i in range(2):
    if i==0:
        bond_data = pd.read_excel('HW2_Data.xls', sheetname='Quotes_Semi',usecols=range(12,16))
    elif i==1:
        bond_data = pd.read_excel('HW2_Data.xls', sheetname='Quotes_Semi',usecols=range(2,6))
    
    bond_data = array(bond_data)
    coupon    = bond_data[:,0] # create coupon data
    bid       = bond_data[:,1] # create bid price data
    ask       = bond_data[:,2] # create ask price data
    maturity  = bond_data[:,3] # create maturity data
    
    Nmat     = len(maturity)             # time to maturity
    freq     = 2                         # frequency of coupons, compounding
    price    = (bid+ask)/2               # take price as avg of bid,ask
    maturity = np.round(maturity,1)      # round maturity (from .4944 to .5)
    
    # Actual Bootstrap of the Zero Curve - See appendix of Chapter 2 for matrix formulation
    CF=zeros((Nmat,Nmat))
    for ii in range(Nmat):
        CF[ii,0:ii+1] = coupon[ii]/freq
    CF = CF+100*eye(Nmat)      # add principal
    Z = ??
    
    # convert to yield (both continuously and semi-annually compounded for comparison)
    yield1     = ??
    yield_semi = ??
    
    if i==0:
        ZZ     = Z                              # save the zero coupon
        yyield = yield1                         # save the bootstrapped yield data
    elif i==1:
        ZZ = np.vstack((ZZ,Z)).T                # save the zero coupon
        yyield = np.vstack((yyield,yield1)).T    # save the bootstrapped yield data

# plot discount and yield curves
plt.figure(1)
plt.plot(maturity,ZZ[:,0],'-.',maturity,ZZ[:,1],linewidth=linethick)
plt.plot(array((10,10)),array((0.62,1)),'k:',linewidth=linethick)
plt.title('Bootstapped Discount')
plt.xlabel('maturity')
plt.legend(('Use Old Bonds','Use New Bonds'),loc='best')

# plot discount and yield curves
plt.figure(2)
plt.plot(maturity,yyield[:,0],'-.',maturity,yyield[:,1],linewidth=linethick)
plt.plot(array((10,10)),array((0,0.04)),'k:',linewidth=linethick)
plt.title('Boostrapped Term Structure')
plt.xlabel('maturity')
plt.legend(('Use Old Bonds','Use New Bonds'),loc='best')

#Note that the directions said to bootstrap as far as we could---12.5 years.  
#However, the rest of Part 1 is usually concerned with 5-year maturities.
#Rather than truncate the yield and discount vectors at 5, I continually 
#make use of the number freq*T.  Remember that this is simply the number 10 
#here because freq=2 (semi-annual), and T is 5.




#%% Question 2. Pricing Leveraged Inverse Floater

coup_fix = 10 # number of coupon payments
T        = 5  # years to maturit

CF_fixed      = (coup_fix/freq)*ones((T*freq))      # fixed coupon
CF_fixed[-1]  = CF_fixed[-1]+100                    # principal amount
P_Fixed       = Z[0:T*freq]@CF_fixed                # price of fixed part
P_Float       = 100                                 # price of floating part
P_zero        = 100*Z[T*freq-1]                     # part of zero part
P_LIF         = ??                                  # LIF price

# displaying the results in the command window
print('       ')
print('Price of Inverse Floater:')
print(str(np.round(P_LIF,4)))

# Comparing price and coupon of LevInvFlt (holding short rate fixed) to fixed 5-year bond
coup_IFLconst    = coup_fix-2*100*yield1[0]
CF_IFLconst      = (coup_IFLconst/freq)*ones(T*freq)
CF_IFLconst[-1]  = CF_IFLconst[-1]+100;
P_IFLconst       = Z[0:T*freq]@CF_IFLconst

# displaying the results in the command window
print('IF SHORT-TERM RATE HELD CONSTANT (at ' + str(round(100*yield1[0],2)) + '%)...')
print('Leveraged Inverse Floater enhances coupon from ' + str(round(coupon[freq*T-1],2)) + '% to ' + str(round(coup_IFLconst,2)) + '%')
print('Price of fixed coupon bond with this rate would be ',str(round(P_IFLconst,2)));




#%% Question 3: Duration and Convexity analysis

# Duration calculations
stripweights     = (Z[0:freq*T]/P_Fixed)*(coup_fix/2)        # coupon weights
stripweights[-1] = stripweights[-1]+(Z[freq*T-1]*100)/P_Fixed # principal weight
D_Fixed          = stripweights@maturity[0:freq*T]          # fixed duration
D_Float          = 1/freq                                    # floating duration
D_Zero           = maturity[freq*T-1]                          # zero duration

# LIF duration
D_LIF = ??

# compare to duration of fixed 5-year coupon bond
D_fixed5 = ??

# displaying the results in the command window
print(' ')
print('Duration of LEV.INV.FLT.: ' + str(round(D_LIF,2)))
print('Duration of FIXED COUPON NOTE')
print('(with same maturity as LIF): ' + str(round(D_fixed5,2)))

# Graphically display duration effects 
# Plot sensitivity of portfolio to parallel shift in term structure

yshift = arange(-0.005,0.05+0.0005,0.0005)   #sizes of term structure shifts
Nyshift = len(yshift)

#initialize vectors computed by loop
Plif_shift    = zeros(Nyshift)
Pfixed5_shift = zeros(Nyshift)

# compute price of both LevInvFlt and fixed 5-yr coupon for each shift
# Pfloat is still the same, par at reset dates
for ii in range(Nyshift):
    Zshift            = exp(-(yield1+yshift[ii])*maturity)
    Pfixed_shift      = np.sum(Zshift[0:freq*T])*coup_fix/2+Zshift[freq*T-1]*100
    Plif_shift[ii]    = Pfixed_shift-2*P_Float + 2*Zshift[freq*T-1]*100
    Pfixed5_shift[ii] = np.sum(Zshift[0:freq*T])*coupon[freq*T-1]/2+Zshift[freq*T-1]*100

# plot of the results
plt.figure(3)
plt.plot(yshift,Plif_shift,linewidth=linethick)
plt.plot(yshift,Pfixed5_shift,'-.',linewidth=linethick)
plt.xlabel('size of parallel shift')
plt.ylabel('price')
plt.legend(('Leveraged Inverse Floater','Fixed Rate 5-yr Bond'))

# Convexity calculations
# Convexity of portfolio = weighted average of portfolio of convexities.
C_Fixed = stripweights@(maturity[0:freq*T]**2)    # fixed convexity
C_Float =(1/freq)**2                              # floating convexity
C_Zero  = maturity[freq*T-1]**2                   # zero convexity

# LIF convexity
C_LIF = ??

# note convexity
C_fixed5 = ??

# displaying the results in the command window
print('   ')
print('Convexity of LIF: ' + str(round(C_LIF,2)))
print('Convexity of 5-yr fixed coupon bond: ' + str(round(C_fixed5,2)))




#%% Question 4. Value at Risk calculation

# read data from Excel and remove NaN
d6  = pd.read_excel('HW2_Data.xls', sheetname='DTB6', skiprows=5, usecols=range(1,2), names=list(['d6']))
id1 = (d6.d6!='ND')
a   = d6.d6[id1]
d6  = array(a.astype('float'))

P6 = ??      # compute P6 using formula. n=182

# continuous compounding
r6  = -log(P6)*2
dr6 = diff(r6)
# compute distribution of d r6
mu6  = mean(dr6)
sig6 = std(dr6,ddof=1)

# For the LIF
mu_LIF  = -D_LIF*P_LIF*mu6
sig_LIF = sqrt((-D_LIF*P_LIF*sig6)**2)

VaR95 = ??
VaR99 = ??

# For the Fixed
mu_Fixed  = -D_Fixed*P_Fixed*mu6
sig_Fixed = sqrt((-D_Fixed*P_Fixed*sig6)**2)

VaR95_Fixed = ??
VaR99_Fixed = ??

# historical distribution for the LIF
dP_LIF = -D_LIF*P_LIF*dr6

VaR95_H = ??
VaR99_H = ??

# plot the results
plt.figure(41)
histogram_data = plt.hist(dP_LIF,100)
plt.close(41)
NN = histogram_data[0]
XX = histogram_data[1]
plt.figure(4)
plt.bar(XX[0:-1],NN/sum(NN)/mean(diff(XX)),width=0.25,color='cyan', edgecolor='blue')
plt.xlim((-7,7))
plt.plot(XX,mu_LIF+sig_LIF*norm.pdf(XX),'-g')
plt.plot(array((-VaR99_H,-VaR99_H)),array((0,0.8)),':k',array((-VaR99,-VaR99)),array((0,0.8)),'-.k',linewidth=2)
plt.title('VaR and Historical Distribution for LIF')
plt.legend(('Historical Distribution','Normal Distribution','99 % VaR: Historical','99 % VaR: Normal'),loc='best')

print('95 percent Normal VaR of LIF: ' + str(round(VaR95,2)))
print('95 percent Historical VaR of LIF: ' + str(round(VaR95_H,2)))
print('99 percent Normal VaR of LIF: ' + str(round(VaR99,2)))
print('99 percent Historical VaR of LIF: ' + str(round(VaR99_H,2)))

# historical distribution for the Fixed
dP_Fixed = -D_Fixed*P_Fixed*dr6

VaR95_H_Fixed = ??
VaR99_H_Fixed = ??

# plot the results
plt.figure(51)
histogram_data = plt.hist(dP_Fixed,100)
plt.close(51)
NN = histogram_data[0]
XX = histogram_data[1]
plt.figure(5)
plt.bar(XX[0:-1],NN/sum(NN)/mean(diff(XX)),width=0.25,color='cyan', edgecolor='blue')
plt.xlim((-7,7))
plt.plot(XX,mu_Fixed+sig_Fixed*norm.pdf(XX),'-g')
plt.plot(array((-VaR99_H_Fixed,-VaR99_H_Fixed)),array((0,2)),':k',array((-VaR99_Fixed,-VaR99_Fixed)), array((0,2)), '-.k',linewidth=2)
plt.title('VaR and Historical Distribution for Fixed Rate Bond')
plt.legend(('Historical Distribution','Normal Distribution','99 % VaR: Historical','99 % VaR: Normal'))

# displaying the results in the command window
print('95 percent Normal VaR of Fixed Rate Bond: ' + str(round(VaR95_Fixed,2)))
print('95 percent Historical VaR of Fixed Rate Bond: ' + str(round(VaR95_H_Fixed,2)))
print('99 percent Normal VaR of Fixed Rate Bond: ' + str(round(VaR99_Fixed,2)))
print('99 percent Historical VaR of Fixed Rate Bond: ' + str(round(VaR99_H_Fixed,2)))




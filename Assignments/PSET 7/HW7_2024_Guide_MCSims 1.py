# %% Packages
import math
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from numpy.linalg import inv,cholesky
from numpy import arange, exp, sqrt, log, mean, std, nonzero, isnan, array, zeros, ones, maximum, minimum
from numpy import concatenate, arange, interp, diff, cov, var, diag, eye, cumsum, tile, transpose, nan, hstack, vstack
from numpy import setdiff1d,polyfit,polyval,gradient,argmin,argmax,amax,amin,flipud,corrcoef
from scipy.interpolate import interp1d, PchipInterpolator
from scipy.optimize import fmin,fsolve
from scipy.stats import norm
from scipy.special import ndtri # inverse of norm.cdf
from numpy.random import normal,uniform
from mpl_toolkits.mplot3d import Axes3D
from BSfwd import BSfwd
from minfunBDTNew import minfunBDTNew,minfunBDTNew_fsolve



plt.close('all')   # Close all charts
print("\n" * 100)  # Clear Console



#%% BDT Model

# DATA
SwRates = array([0.152,0.2326,0.3247,0.346,0.7825,1.2435,1.599,1.853,2.052,2.2085,2.3371,2.4451,2.539,2.843,2.9863,3.0895])    # includes 1-, 3-, 6- month LIBOR as first 3 entries
CapsVol = array([68.53,63.63,54.06,48.43,44.87,42.03,43.35,38.03,36.54,38.45,33.13,29.97,26.91,24.95,23.65])

SwRates = SwRates/100  # transform in decimals
CapsVol = CapsVol/100  # transform in decimals

MaturitySwaps = array([1/12,3/12,6/12,1,2,3,4,5,6,7,8,9,10,15,20,30])    # maturities of Swaps
MaturityCaps  = array([1,2,3,4,5,6,7,8,9,10,12,15,20,25,30])             # maturities of Caps

dt = 0.25     # time step: Quarterly (as caps are quarterly).

# Augment Volatility data vector assuming *constant* volatility for short term caps
CapsVol = hstack((CapsVol[0],CapsVol))

# Augment maturity using 3 month
MaturityCaps = hstack((dt,MaturityCaps))

# Interpolation using dt steps
IntMat = arange(dt,30+dt,dt)
f      = interp1d(MaturitySwaps,SwRates,kind='cubic'); IntSwRate = f(IntMat)
f      = interp1d(MaturityCaps,CapsVol,kind='cubic');  IntVol    = f(IntMat)




#%% BOOSTRAP DISCOUNTS FROM INTERPOLATED SWAP RATES (TN 1)
ZSw      = zeros(IntSwRate.shape[0])
ZSw[0] = 1/(1+IntSwRate[0]*dt)
NN       = ZSw.shape[0]
for i in range(1,ZSw.shape[0]):
    ZSw[i] = (1-IntSwRate[i]*dt*np.sum(ZSw[0:i]))/(1+IntSwRate[i]*dt)


# double check: Does ZSw imply the original swap curve?
c2 = zeros(ZSw.shape[0])
for i in range(ZSw.shape[0]):
    c2[i] = (1-ZSw[i])/(np.sum(ZSw[0:i+1])*dt)

# Compute continuously compounded yield
Zyieldcc = -log(ZSw)/IntMat
# Transform them into quarterly compounding
Zyield = 1/dt*(exp(Zyieldcc*dt)-1)

# short term interest rate
r0cc = -log(ZSw[0])/IntMat[0]
# in quarterly compounding
r0 = 1/dt*(exp(r0cc*dt)-1)

# compute continuously compounded forwards
fwdcc = -log(ZSw[1:NN]/ZSw[0:NN-1])/dt
# compute quarterly compounded fwds
Fwd = 1/dt*(exp(fwdcc*dt)-1)


plt.figure(1)
plt.plot(IntMat[1:NN],Zyield[0:NN-1],'-.r',IntMat[1:NN],Fwd[0:NN-1],'-k')
plt.legend(('LIBOR Spot Curve','Forward Rates','location','southeast'))
plt.xlabel('Maturity')
plt.ylabel('yield')
plt.title('LIBOR Curve and Forward Curve')




#%% Extract Forward Volatities


# Initialize cap vector
Caps = zeros(NN-1)

# FIRST, compute the dollar values of the caps for all maturities.
#       It uses BSfwd = Black's formula m-file
for i in range(NN-1):
    Caps[i] = np.sum(dt*BSfwd(Fwd[0:i+1],IntSwRate[i+1],ZSw[1:i+2],IntVol[i+1],IntMat[0:i+1],1))
    # Note:
    # IntMat:   maturity "T" into Black's model (to compute d1, d2)
    # Fwd:      Fwd Rates to be used in Blacks model
    # IntSwRate: Swap Rate. This is "shifted" one quarter ahead. The reason
    #           is that this is the strike rate of the (T+1 quarter) cap, which we
    #           are computing.
    # ZSw:      Discount Function. Also this is shifter one quarter ahead,
    #           as recall that something determined at T is paid at T+1
    #           quarter
    # IntVol:   Flat Volatility interpolated from the data.






#%% SECOND, compute caplets which use forward vols instead of Flat vols.

ImplVol      = zeros(NN-1)
Caplet       = zeros(NN-1)
CapletMatrix = zeros((NN-1,NN-1))

VVV = arange(0.00001,1.4+0.00001,0.00001)    # possible vectors of volatilities. It will be used below to compute the implied Fwd volatilities

# Start computing FWD Vols.
ImplVol[0]      = IntVol[1]         # the first Fwd vol = Flat Vol
Caplet[0]       = Caps[0]         # the first caplet = cap with the first maturity
CapletMatrix[0,0] = Caplet[0]       # initialization of a caplet matrix

# Loop for all the other maturities
for i in range(1,NN-1):

    # compute caplets corresponding to maturities 1 to i-1 using the previously computed Fwd Volatilities (Vector ImplVol)
    CapletMatrix[0:i,i] = dt*BSfwd(Fwd[0:i],IntSwRate[i+1],ZSw[1:i+1],ImplVol[0:i],IntMat[0:i],1)

    # Compute the value of the sum of caplets up to maturity i-1
    SumCaplets = np.sum(dt*BSfwd(Fwd[0:i],IntSwRate[i+1],ZSw[1:i+1],ImplVol[0:i],IntMat[0:i],1))

    # The value of caplet with maturity i must equal the value of the cap with that maturity minus the previously compute sum of caplets
    Caplet[i] = Caps[i] - SumCaplets

    # Fill in the Caplet matrix for the missing i caplet
    CapletMatrix[i,i] = Caplet[i]

    # Obtained implied volatility. Idea is: compute the value of the caplet for each of the values in VVV vector,
    # and find the minimum of the distance between the Black's formula (on the grid VVV) and the Caplet value computed above.
    iV = argmin((dt*BSfwd(Fwd[i],IntSwRate[i+1],ZSw[i+1],VVV,IntMat[i],1)-Caplet[i])**2)

    # The implied volatility is just the value of VVV that minimizes the distance in the previous step, that is, element iV in the vector VVV;
    ImplVol[i] = VVV[iV]


plt.figure(2)
plt.plot(IntMat[1:NN],ImplVol[0:NN-1],'-.r',IntMat[1:NN],IntVol[1:NN],'-k',MaturityCaps,CapsVol,'*')
plt.legend(('Forward Volatilities','Flat Volatilities','Data'))
plt.xlabel('Maturity')
plt.ylabel('Volatility')
plt.title('Forward and Flat Volatility')






#%%% BUILD THE BDT TREE for MBS

dtstep = 1/4  # monthly steps in BDT model (Note: by choosing dtstep = 1/4 we can check whether the BDT model prices zeros and caps correctly. See below)

# add shorter maturity to all of the vectors
if dtstep==1/4:
    IntMat2  = IntMat
    ImplVol2 = ImplVol
    ZSw2     = ZSw
elif dtstep<1/4:
    ImplVol2 = hstack((ImplVol[0],ImplVol))
    ZSw2     = hstack((1/(1+SwRates[0]*1/12),ZSw))

Maturity = arange(dtstep,amax(IntMat[0:-1])+dtstep,dtstep)
f        = interp1d(IntMat2,ZSw2); ZSwInt = f(Maturity)
ZYield   = -log(ZSwInt[0:-1])/Maturity[0:-1]
f        = interp1d(IntMat2[0:-1],ImplVol2); FwdVol = f(Maturity)

NN = ZYield.shape[0]  # size of zero-coupon yield vector

# parallell shift for duration computation
dy     = 0
ZYield = ZYield + dy

# Intialize the Implied Tree
ImTree = zeros((NN,NN))

# First note is the c.c. yield
ImTree[0,0] = ZYield[0]

xx = ZYield[0] # starting value in the search engine

#Begin the loop across all of the other values
for j in range(NN-1):
    xx = xx*0.75 # starting value of search

    # use minfunBDTNew.m file to solve the for the interest rate such
    # that the zero coupon bond out of the tree equals the zero coupon bond from the data
    xx = fsolve(func=minfunBDTNew_fsolve,x0=xx,args=(ImTree,ZYield[j+1],FwdVol[j],Maturity[j+1]-Maturity[j],j+2))

    # plug back the solution in the mfile minfunBDTNew.m and obtain the vector of interest rate values
    F,vec = minfunBDTNew(xx,ImTree,ZYield[j+1],FwdVol[j],Maturity[j+1]-Maturity[j],j+2)

    # update the tree
    ImTree[0:j+2,j+1] = vec




#%%% Check if tree prices zeros and caps correctly
ZTree = zeros((NN,NN,NN))
for i in arange(1,NN): # add one step to maturity to ensure the last bond has 30/dt to maturity (i=2 has 6 months to maturity)
    ZTree[0:i+1,i,i] = 1 # initialize tree with maturity i
    # backward algorithm
    for j in arange(i,0,-1):
        ZTree[0:j,j-1,i] = exp(-ImTree[0:j,j-1]*dtstep)*(0.5*(ZTree[0:j,j,i]+ZTree[1:j+1,j,i]))

plt.figure(3)
plt.plot(Maturity[0:NN-1],ZTree[0,0,1:NN],'--*',IntMat2[0:-1],ZSw2[0:-1],'.')
plt.title('Tree implied discount function versus LIBOR discount')
plt.xlabel('Maturity')
plt.legend(('Tree-implied discount','LIBOR discount'))




#%%% Check if tree prices caps correctly
if dtstep==0.25:
    # Use the binomial tree to check if we price zeros and caps correctly.
    # only works if dtstep==0.25

    # compute caps and zeros on the tree
    CapTree = zeros((NN,NN,NN))
    for i in range(1,NN): # add one step to maturity to ensure the last bond has 30/dt to maturity (i=2 has 6 months to maturity)
        CapTree[0:i+1,i,i] = dt*exp(-ImTree[0:i+1,i]*dt)*maximum((exp(ImTree[0:i+1,i]*dt)-1)/dt-IntSwRate[i+1],0)
        # backward algorithm
        for j in range(i,0,-1):
            if j>1:
                CapTree[0:j,j-1,i] = exp(-ImTree[0:j,j-1]*dtstep)*(0.5*(CapTree[0:j,j,i]+CapTree[1:j+1,j,i]) + dtstep*maximum((exp(ImTree[0:j,j-1]*dtstep)-1)/dt-IntSwRate[i+1],0))
            else:
                CapTree[0:j,j-1,i] = exp(-ImTree[0:j,j-1]*dtstep)*(0.5*(CapTree[0:j,j,i]+CapTree[1:j+1,j,i]))

        a1 = CapTree[:,:, 0]
        a2 = CapTree[:,:, 1]
        if i==2:
            a3 = CapTree[:,:, 2]

    plt.figure(4)
    plt.plot(IntMat[1:],Caps,'*',IntMat[1:-1],CapTree[0,0,:],'o')
    plt.legend(('Data','Binomial Tree'),loc='best')
    plt.xlabel('Maturity')
    plt.ylabel('Dollars')






#%%% Mortgages

# mortgage characteristics
WAC = 4.492/100                  # weighted averge mortgage rate
WAM = int(round(311/12)/dtstep)  # number of periods, given step size in BDT model
PP0 = 100                        # reset current principal to 100
aa  = 1/(1+WAC*dtstep)

# pass through coupon rate
rbar = 4/100


NN      = WAM # Redefine maturity using actual number of months left.
MCoupon = PP0*(1-aa)/(aa-aa**(NN+1)) # Monthly dollar coupon at time 0





#%%% Monte Carlo Simulations to price securities using additional frictions.

# Number of simulations
NSim = 5000

# Pre-Payment Parameters
c_1 = ???        # Pre-Payment Speed
c_2 = ???        # Pre-payment Lower Bound
rm  = ???  # Pre-payment threshold
sp  = ???    # Pre-payment threshold's additional margin/spread

rK = ???      # Cap Rate
N  = ???      # Cap Notional

# Set a seed for spot rate computations. By uncommenting the line below,
# the random draws are always the same (from here on) and so we can use
# different draws to compute spot rate duration etc

#rand('seed',10); randn('seed',10);

PPsim2 = zeros(NSim) # initializate value of pass through security

# intialize Principal Only and Interest Only price vectors
PPPOsim = zeros(NSim); PPIOsim = zeros(NSim); PPCAPsim = zeros(NSim); PPACAPsim = zeros(NSim)

# same calculation but for a shift up of all simulated rates
PPsim2_Up = zeros(NSim); PPPOsim_Up = zeros(NSim); PPIOsim_Up = zeros(NSim); PPCAPsim_Up = zeros(NSim); PPACAPsim_Up = zeros(NSim)
# same calculation but for a shift down of all simulated rates
PPsim2_Do = zeros(NSim); PPPOsim_Do = zeros(NSim); PPIOsim_Do = zeros(NSim); PPCAPsim_Do = zeros(NSim); PPACAPsim_Do = zeros(NSim)

CCFACAP = zeros(NSim); CCFACAP_Up = zeros(NSim); CCFACAP_Do = zeros(NSim); CCFCAP = zeros(NSim); CCFCAP_Up = zeros(NSim)
CCFCAP_Do = zeros(NSim); CCFIO = zeros(NSim); CCFIO_Up = zeros(NSim); CCFIO_Do = zeros(NSim); CCFPT2 = zeros(NSim); CCFPT2_Up = zeros(NSim); CCFPT2_Do = zeros(NSim)
CCFPO = zeros(NSim); CCFPO_Up = zeros(NSim); CCFPO_Do = zeros(NSim)

dr = 0.5/100


for sim in range(NSim):      # simulation loop
    if sim%100==0: print('SIM = ' + str(sim))

    Pri_t    = PP0 # initial principal
    Pri_t_Up = PP0 # initial principal (up rate scenario)
    Pri_t_Do = PP0 # initial principal (down rate senario)

    Ct    = MCoupon  # Initial total coupon
    Ct_Up = MCoupon  # Initial total coupon (up rate scenario)
    Ct_Do = MCoupon  # Initial total coupon (down rate scenario)

    i=1
    j=1
    rrsim    = zeros(NN)
    rrsim_Up = zeros(NN)
    rrsim_Do = zeros(NN)

    rrsim[0]    = ImTree[0,0]    # initialize interest rate simulation vector
    rrsim_Up[0] = rrsim[0] + dr  # (up rate scenario)
    rrsim_Do[0] = rrsim[0] - dr  # (down rate scenario)

    CCt    = Ct            # initialize coupon vector
    CCt_Up = Ct            # initialize coupon vector (up rate scenario)
    CCt_Do = Ct            # initialize coupon vector (down rate scenario)

    CFPT2  = zeros(NN)         # initialize vector for CF from GSNF4
    CFIO   = zeros(NN)         # initialize vector for CF for IO and PO strips
    CFPO   = zeros(NN)         # initialize vector for CF for IO and PO strips
    CFCAP  = zeros(NN)         # initialize vector for CF for Caps
    CFACAP = zeros(NN)         # initialize vector for CF for Asian Caps

    CFPT2_Up  = zeros(NN)      # initialize vector for CF from GSNF4
    CFIO_Up   = zeros(NN)      # initialize vector for CF for IO and PO strips
    CFPO_Up   = zeros(NN)      # initialize vector for CF for IO and PO strips
    CFCAP_Up  = zeros(NN)      # initialize vector for CF for Caps
    CFACAP_Up = zeros(NN)      # initialize vector for CF for Asian Caps

    CFPT2_Do  = zeros(NN)      # initialize vector for CF from GSNF4
    CFIO_Do   = zeros(NN)      # initialize vector for CF for IO and PO strips
    CFPO_Do   = zeros(NN)      # initialize vector for CF for IO and PO strips
    CFCAP_Do  = zeros(NN)      # initialize vector for CF for Caps
    CFACAP_Do = zeros(NN)      # initialize vector for CF for Asian Caps


    eep = 0
    for i in range(1,NN-1):        # time loop
        #print('     i = ' + str(i))
        ep = uniform(size=1)       # random uniform
        #eep = [eep ep]
        if ep<1/2:                 # move up in the tree. j remains the same
            j=j                    #update current node
            rsim = ImTree[j,i]
        else:                      # move down the tree. j = j+1
            j=j+1                  # update current node
            rsim = ImTree[j,i]

        rsim_Up = rsim + dr
        rsim_Do = rsim - dr

        rrsim[i-1]    = rsim        # save simulated interest rate for later discounting

        # Interest paid (applied on remaining principal only)
        IntRatePay    = Pri_t*WAC*dtstep

        PriSchedule    = Ct-IntRatePay

        # Prepayments due to exogenous reasons
        PrepayFunc    = minimum(c_1*maximum(rm-sp-rsim,c_2),1)
        SMM        = PrepayFunc     # SMM = fraction of agents who exercise
        Prepay     = SMM*(Pri_t-PriSchedule)

        # interest rate cash flows
        IntRatePayPT    = Pri_t*rbar*dtstepp

        # Total Cash Flow PT
        CashFlowPT    = IntRatePayPT + PriSchedule + Prepay
        CFPT2[i-1]         = CashFlowPT


        # Cash Flow IO
        CashFlowIO    = IntRatePayPT
        CFIO[i-1]          = CashFlowIO


        # Total Cash Flow PO
        CashFlowPO    = PriSchedule + Prepay
        CFPO[i-1]          = CashFlowPO


        # Total repayment of Principal
        PriPay    = PriSchedule + Prepay


        # Adjust principal for next period.
        Pri_t    = maximum(Pri_t-PriPay,0)


        # The coupon is recalculated on this principal
        Ct    = Ct*(1-SMM)

        # CAPS AND ASIAN CAPS
        # -------------------

        # Cap Cash Flows
        CashFlowCAP    = ???
        CFCAP[i-1]          = CashFlowCAP


        # Asian Cap Cash Flows
        CashFlowACAP    = ???
        CFACAP[i-1]          = CashFlowACAP


    # Pass Through
    CCFPT2[sim]    = rbar*dtstep*Pri_t+Pri_t
    Psim2    = exp(-cumsum(rrsim)*dtstep)@CFPT2
    PPsim2[sim]    = Psim2


    # IO
    CCFIO[sim]    = rbar*dtstep*Pri_t
    PIOsim    = exp(-cumsum(rrsim)*dtstep)@CFIO
    PPIOsim[sim]    = PIOsim


    # PO
    CCFPO[sim]    = Pri_t
    PPOsim    = exp(-cumsum(rrsim)*dtstep)@CFPO
    PPPOsim[sim]    = PPOsim


    # CAP
    CCFCAP[sim]    = 0
    PCAPsim    = exp(-cumsum(rrsim)*dtstep)@CFCAP
    PPCAPsim[sim]    = PCAPsim

    # Asian CAP
    CCFACAP[sim]    = 0
    PACAPsim    = exp(-cumsum(rrsim)*dtstep)@CFACAP
    PPACAPsim[sim]    = PACAPsim


PPPT2    = mean(PPsim2)
SEPT2    = std(PPsim2,ddof=1)/sqrt(NSim)


PPIO    = mean(PPIOsim)
SEIO    = std(PPIOsim,ddof=1)/sqrt(NSim)

PPPO    = mean(PPPOsim)
SEPO    = std(PPPOsim,ddof=1)/sqrt(NSim)
PPPO_Up = mean(PPPOsim_Up)
PPPO_Do = mean(PPPOsim_Do)

PPCAP    = mean(PPCAPsim)
SECAP    = std(PPCAPsim,ddof=1)/sqrt(NSim)

PPACAP    = mean(PPACAPsim)
SEACAP    = std(PPACAPsim,ddof=1)/sqrt(NSim)
2)


print(' Pricing by Monte Carlo Simulations with "Irrational" Refinancing')
print(' ----------------------------------------------------------------')
print('Pass Through, IO Tranche, PO Tranche, CAP, Asian Cap')
print(np.round(array([PPPT2/PP0*100, PPIO/PP0*100, PPPO/PP0*100, PPCAP, PPACAP]),4))
print('stdandard errors')
print(np.round(array([SEPT2/PP0*100, SEIO/PP0*100, SEPO/PP0*100, SECAP, SEACAP]),4))

plt.show()

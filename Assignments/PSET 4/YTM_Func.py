import math
import numpy as np
from numpy import ones, arange
from scipy.optimize import fsolve


#%% YTMsolver Function
def YTMsolver(Ptrue,coup,freq,T):
    #YTM computes the yield to maturity of a fixed coupon bond
    #Ptrue is the observed price of the bond
    #enter coup as a decimal, not a percentage
    #enter freq as number per year (semiannual would be 2, not .5)

    # YTM_value Function
    def YTM_value(y):  # define the function to solve Perr (=Pricing error)
        if y > 0:
            disc  = 1/((1+y/freq)**(freq*cash_dates))
            Pcalc = disc@cash_flow
            Perr  = Ptrue-Pcalc
        else:
            Perr = 1e4

        return Perr

    #Build Cash Flow
    cash_flow    = (coup/freq)*100*ones(T*freq)
    cash_flow[-1]= cash_flow[-1]+100        #add principal
    cash_dates   = arange(1/freq,T+1/freq,1/freq)

    #Use fsolve to plug in values until Pcalc-Ptrue=0
    y0  = 0.04                         # y0 is an arbitrary initial guess
    ytm = fsolve(YTM_value,y0)        # fsolve is a Python function to solve non-linear equations

    return ytm




    a=1






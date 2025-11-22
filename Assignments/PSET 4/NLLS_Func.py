import math
import numpy as np
from numpy import exp



#%% NLLS_Min Function - For fmin only
def NLLS_Min(vec,Price,Maturity,CashFlow):

    J, PPhat = NLLS(vec, Price, Maturity, CashFlow)

    return J




#%% NLLS Function
def NLLS(vec,Price,Maturity,CashFlow):

    # Assign     variables
    th0 = vec[0]
    th1 = vec[1]
    th2 = vec[2]
    la  = vec[3]

    T  = np.maximum(Maturity,1e-10) # there are some zeros that do not make the computation below possible. Such cases are automatically eliminated when we multiply for zero cash flows
    RR = th0+(th1+th2)*(1-exp(-T/la))/(T/la)-th2*exp(-T/la)

    # Discount
    ZZhat = exp(-RR*T)

    # Prices
    PPhat = np.sum(CashFlow*ZZhat,axis=1)

    # Compute the squared distance between actual prices and theoretical prices
    J = np.sum((Price - PPhat)**2); #<-- this is the function we want to minimize!

    return J, PPhat



# %% Packages
import numpy as np
from numpy import sqrt,exp,arange,zeros


#%% For fmin only
def fmin_HoLee_SimpleBDT_Tree(theta_i,ZZi,ImTree,i,sigma,hs,BDT_Flag):

    FF,ImTree,ZZTree = HoLee_SimpleBDT_Tree(theta_i,ZZi,ImTree,i,sigma,hs,BDT_Flag)

    return FF



#%% Actual function
def HoLee_SimpleBDT_Tree(theta_i,ZZi,ImTree,i,sigma,hs,BDT_Flag):

    if BDT_Flag==0:
        # given theta, compute the next step of the tree
        ImTree[0,i] = ImTree[0,i-1] + theta_i*hs + sigma*sqrt(hs)
        for j in arange(1,i+1):
            ImTree[j,i] = ImTree[j-1,i-1] + theta_i*hs - sigma*sqrt(hs)
    else:
        # given theta, compute the next step of the tree
        ImTree[0,i] = ImTree[0,i-1]*exp(theta_i*hs+sigma*sqrt(hs))
        for j in arange(1,i+1):
            ImTree[j,i] = ImTree[j-1,i-1]*exp(theta_i*hs-sigma*sqrt(hs))

    # Use the tree to compute the value of a zero coupon bond

    # note: The zero coupon ZZ(i) in data expires at i+1 in Tree. For
    # instance, the first data point ZZ(1) is a 2-period bond, so expires
    # at i+1.
    ZZTree = zeros((i+2,i+2))      # initialize the matrix for the zero coupon bond with maturity i+1.
    ZZTree[0:i+2,i+1] = 1          # final price is equal to 1
    pi=0.5
    # backward algorithm
    for j in arange(i+1,0,-1):
        ZZTree[0:j,j-1] = exp(-ImTree[0:j,j-1]*hs)*(pi*ZZTree[0:j,j]+(1-pi)*ZZTree[1:j+1,j])

    FF=(ZZTree[0,0]-ZZi)**2

    return FF,ImTree,ZZTree

from numpy import exp,sqrt,arange,zeros,ones

def minfunBDTNew_fsolve(rmin,ImTree,yield1,vol,h,N):

    F,vec = minfunBDTNew(rmin,ImTree,yield1,vol,h,N)

    return F






def minfunBDTNew(rmin,ImTree,yield1,vol,h,N):

    mult = arange(N-1,-1,-1)
    vec           = rmin*exp(2*vol*sqrt(h)*mult)
    ImTree[0:N,N-1] = vec


    RateMatrix = ImTree[0:N,0:N]
    T          = N
    BB         = zeros((T+1,T+1))
    BB[:,T]    = ones(T+1)

    for t in arange(T,0,-1):
        BB[0:t,t-1] = exp(-RateMatrix[0:t,t-1]*h)*(0.5*BB[0:t,t] + 0.5*BB[1:t+1,t])

    PZero = BB[0,0]

    F = exp(-yield1*vec.shape[0]*h) - PZero

    return F,vec
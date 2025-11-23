
from scipy.stats import norm
from numpy import exp, sqrt, log

# %% Call Option
def Bsc(S,X,rd,q,sigma,T):

    d1 = (log(S/X)+(rd-q+sigma**2/2)*T)/(sigma*sqrt(T));
    d2 = d1-sigma*sqrt(T);

    nd1  = norm.cdf(d1)
    nd2  = norm.cdf(d2)
    npd1 = norm.pdf(d1)

    C     = S*exp(-q*T)*nd1-X*exp(-rd*T)*nd2
    Delta = exp(-q*T)*nd1;
    Gamma = exp(-q*T)/(sigma*S*sqrt(T))*npd1;
    Theta = -sigma*S*exp(-q*T)*npd1/(2*sqrt(T))+q*S*nd1*exp(-q*T)-rd*X*exp(-rd*T)*nd2;
    Vega  = S*sqrt(T)*exp(-q*T)*npd1;
    Rho   = X*T*exp(-rd*T)*nd2;

    return C, Delta, Gamma, Theta, Vega, Rho

# %% Put Option
def Bsp(S,X,rd,q,sigma,T):

    d1 = (log(S/X)+(rd-q+sigma**2/2)*T)/(sigma*sqrt(T));
    d2 = d1-sigma*sqrt(T);

    nd1  = norm.cdf(-d1)
    nd2  = norm.cdf(-d2)
    npd1 = norm.pdf(d1)

    P     = -S*exp(-q*T)*nd1+X*exp(-rd*T)*nd2;
    Delta = -exp(-q*T)*nd1;
    Gamma = exp(-q*T)/(sigma*S*sqrt(T))*npd1;
    Theta = -sigma*S*exp(-q*T)*npd1/(2*sqrt(T))-q*S*nd1*exp(-q*T)+rd*X*exp(-rd*T)*nd2;
    Vega  = S*sqrt(T)*exp(-q*T)*npd1;
    Rho   = -X*T*exp(-rd*T)*nd2;

    return P, Delta,Gamma, Theta, Vega, Rho

# %% Call-Put Option - Black Forward
def BSfwd(F,X,Z,sigma,T,CallF):

    # BSfwd(F,X,Z,sigma,T,CallF)
    # Compute Black's option premium
    # F = forward price
    # X = strike
    # Z = discount
    # sigma = volatility of forward
    # T = time-to-maturity
    # CallF = flag for call. = 1 for calls and =2 for puts

    if CallF==1:
        d1    = (log(F/X)+(sigma**2/2)*T)/(sigma*sqrt(T))
        d2    = d1-sigma*sqrt(T)
        Price = Z*(F*norm.cdf(d1)-X*norm.cdf(d2))
    elif CallF==2:
        d1    = (log(F/X)+(sigma**2/2)*T)/(sigma*sqrt(T))
        d2    = d1-sigma*sqrt(T)
        Price = Z*(-F*norm.cdf(-d1)+X*norm.cdf(-d2))

    return Price

# %% Put Option - Implied Vol calc
def ImpVolP(P, S, X, rd, q, T):
    sigma = 0.3
    error = 0.000001
    dv = error + 1

    while abs(dv) > error:
        Calc = Bsp(S, X, rd, q, sigma, T)
        P_Calc = Calc[0]
        Vega_Calc = Calc[4]
        PriceError = P_Calc - P
        dv = PriceError / Vega_Calc
        sigma = sigma - dv

    return sigma

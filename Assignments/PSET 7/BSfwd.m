function [Price]=BSfwd(F,X,Z,sigma,T,CallF)

% BSfwd(F,X,Z,sigma,T,CallF)
% Compute Black's option premium
% F = forward price
% X = strike
% Z = discount
% sigma = volatility of forward
% T = time-to-maturity
% CallF = flag for call. = 1 for calls and =2 for puts

if CallF==1
d1=(log(F./X)+(sigma.^2/2).*T)./(sigma.*sqrt(T))
d2=d1-sigma.*sqrt(T)
Price=Z.*(F.*normcdf(d1)-X.*normcdf(d2));
elseif CallF==2
d1=(log(F./X)+(sigma.^2/2).*T)./(sigma.*sqrt(T));
d2=d1-sigma.*sqrt(T);
Price=Z.*(-F.*normcdf(-d1)+X.*normcdf(-d2));
end
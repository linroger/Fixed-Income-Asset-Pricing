function [J,PPhat]=NLLS(vec,Price,Maturity,CashFlow)

% Assign variables
th0=vec(1); th1=vec(2); th2=vec(3); la=vec(4);    

T=max(Maturity,1e-10); % there are some zeros that do not make the computation below possible. Such cases are automatically eliminated when we multiply for zero cash flows
RR=th0+(th1+th2)*(1-exp(-T/la))./(T/la)-th2*exp(-T/la);


% Discount
ZZhat=exp(-RR.*T);

% Prices
PPhat=sum((CashFlow.*ZZhat)')';

% Compute the squared distance between actual prices and theoretical prices
J=sum((Price - PPhat).^2); %<-- this is the function we want to minimize!


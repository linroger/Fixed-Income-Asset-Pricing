function ytm=YTMsolver(Ptrue,coup,freq,T)
%YTM computes the yield to maturity of a fixed coupon bond
%Ptrue is the observed price of the bond
%enter coup as a decimal, not a percentage
%enter freq as number per year (semiannual would be 2, not .5)

%Build Cash Flow
cash_flow=(coup/freq)*100*ones(T*freq,1);
cash_flow(end)=cash_flow(end)+100;        %add principal
cash_dates=(1/freq:1/freq:T)';

%Use fsolve to plug in values until Pcalc-Ptrue=0
y0=.04;                         %y0 is an arbitrary initial guess
ytm=fsolve(@YTM_value,y0);      % fsolve.m is a matlab function to solve non-linear equations


function Perr=YTM_value(y)      % define the function to solve Perr (=Pricing error)
    if y>0
        disc=1./((1+y/freq).^(freq*cash_dates));
        Pcalc=disc'*cash_flow;
        Perr=Ptrue-Pcalc;
    else
        Perr=1e4;
    end
end


end
%% Bus 35130 HW2

% Guide to the solution
% This file is not going to work as is. Your task is to complete the file
% by filling in the ?? scattered through the file. 


%% Use F5 button to run the entire program


%% To execute code line-by-line: first create breakpoint at any line 
%  (line #12 for example), then press F5 to execute until the breakpoint,
%  then press F10 to execute one line at a time


%% Just to clear Matlab memory

clear all; % clear all of the data
close all; % close all of the figures
clc        % clears the writing in the command window



%% Question 1. Bootstraping the term structure

linethick=2; % Thinkness of the line for use with plots

% displaying the results in the command window
disp('==================================================================')
disp('LEVERAGED INVERSE FLOATER')
disp('==================================================================')

% load and arrange data 
bond_data = xlsread('HW2_Data','Quotes_Semi','C2:F27'); % read data from the Excel file
coupon    = bond_data(:,1); % create coupon data
bid       = bond_data(:,2); % create bid price data
ask       = bond_data(:,3); % create ask price data
maturity  = bond_data(:,4); % create maturity data

Nmat     = length(maturity);      % time to maturity
freq     = 2;                     % frequency of coupons, compounding
price    = (bid+ask)/2;           % take price as avg of bid,ask
maturity = round(maturity*10)/10; % round maturity (from .4944 to .5)


% Actual Bootstrap of the Zero Curve - See appendix of Chapter 2 for matrix formulation
CF=zeros(Nmat);
for ii=1:Nmat
    CF(ii,1:ii) = coupon(ii)/freq;
end
CF = CF+100*eye(ii);      % add principal
Z = ??;

%convert to yield (both continuously and semi-annually compounded for comparison)
yield      = ??;      
yield_semi = ??;

% plot discount and yield curves
figure
hold all
plot(maturity,Z,'linewidth',linethick)
plot([10 10],[.62 1],'k:','linewidth',linethick)
title('Bootstapped Discount')
xlabel('maturity')
saveas(gcf,'hw2_1_discount','epsc')

% plot discount and yield curves
figure
hold all
plot(maturity,yield,'linewidth',linethick)
plot([10 10],[0 .04],'k:','linewidth',linethick)
title('Boostrapped Term Structure')
xlabel('maturity')
saveas(gcf,'hw2_1_yield','epsc')

%{
Note that the directions said to bootstrap as far as we could---12.5 years.  
However, the rest of Part 1 is usually concerned with 5-year maturities.
Rather than truncate the yield and discount vectors at 5, I continually 
make use of the number freq*T.  Remember that this is simply the number 10 
here because freq=2 (semi-annual), and T is 5.
%}



%% Question 2. Pricing Leveraged Inverse Floater

coup_fix = 10; % number of coupon payments
T        = 5; % years to maturity

CF_fixed      = (coup_fix/freq)*ones(T*freq,1); % fixed coupon
CF_fixed(end) = CF_fixed(end)+100;              % principal amount
P_Fixed       = Z(1:T*freq)'*CF_fixed;          % price of fixed part
P_Float       = 100;                            % price of floating part
P_zero        = 100*Z(T*freq);                  % part of zero part
P_LIF         = ??;                             % LIF price

% displaying the results in the command window
disp(' ')
disp('Price of Inverse Floater:')
disp(P_LIF)

% Comparing price and coupon of LevInvFlt (holding short rate fixed) to fixed 5-year bond
coup_IFLconst    = coup_fix-2*100*yield(1);
CF_IFLconst      = (coup_IFLconst/freq)*ones(T*freq,1);
CF_IFLconst(end) = CF_IFLconst(end)+100;
P_IFLconst       = Z(1:T*freq)'*CF_IFLconst;

% displaying the results in the command window
disp(sprintf('IF SHORT-TERM RATE HELD CONSTANT (at %2.2f%%)...',100*yield(1)))
disp(sprintf('Leveraged Inverse Floater enhances coupon from %2.2f%% to %2.2f%%',coupon(freq*T),coup_IFLconst));
disp(sprintf('Price of fixed coupon bond with this rate would be %2.2f',P_IFLconst));




%% Quastion 3: Duration and Convexity analysis

% Duration calculations

stripweights      = (Z(1:freq*T)/P_Fixed)*(coup_fix/2);        % coupon weights
stripweights(end) = stripweights(end)+(Z(freq*T)*100)/P_Fixed; % principal weight
D_Fixed           = stripweights'*maturity(1:freq*T);          % fixed duration
D_Float           = 1/freq;                                    % floating duration
D_Zero            = maturity(freq*T);                          % zero duration

% LIF duration
D_LIF = ??; 

% compare to duration of fixed 5-year coupon bond
D_fixed5 = ??;

% displaying the results in the command window
disp(' ')
disp(sprintf('Duration of LEV.INV.FLT.: %2.2f',D_LIF))
disp(sprintf('Duration of FIXED COUPON NOTE'))
disp(sprintf('(with same maturity as LIF): %2.2f',D_fixed5))


% Graphically display duration effects 
% Plot sensitivity of portfolio to parallel shift in term structure

yshift = (-.005:.0005:.05)';  %sizes of term structure shifts
Nyshift = length(yshift);

%initialize vectors computed by loop
Plif_shift    = zeros(Nyshift,1);
Pfixed5_shift = zeros(Nyshift,1);

% compute price of both LevInvFlt and fixed 5-yr coupon for each shift
% Pfloat is still the same, par at reset dates
for ii=1:length(yshift)
    Zshift            = exp(-(yield+yshift(ii)).*maturity);
    Pfixed_shift      = sum(Zshift(1:freq*T))*(coup_fix/2)+Zshift(freq*T)*100;    
    Plif_shift(ii)    = Pfixed_shift-2*P_Float + 2*Zshift(freq*T)*100;
    Pfixed5_shift(ii) = sum(Zshift(1:freq*T))*(coupon(freq*T)/2)+Zshift(freq*T)*100;
end

% plot of the results
figure
hold all
plot(yshift,Plif_shift,'linewidth',linethick)
plot(yshift,Pfixed5_shift,'-.','linewidth',linethick)
xlabel('size of parallel shift')
ylabel('price')
legend('Leveraged Inverse Floater','Fixed Rate 5-yr Bond')
saveas(gcf,'hw2_1_duration','epsc')


% Convexity calculations

% Convexity of portfolio = weighted average of portfolio of convexities.
C_Fixed = stripweights'*(maturity(1:freq*T).^2); % fixed convexity
C_Float =(1/freq)^2;                             % floating convexity
C_Zero  = maturity(freq*T).^2;                   % zero convexity

% LIF convexity
C_LIF = ??;

% note convexity
C_fixed5 = ??;

% displaying the results in the command window
disp(' ')
disp(sprintf('Convexity of LIF: %2.2f',C_LIF))
disp(sprintf('Convexity of 5-yr fixed coupon bond: %2.2f',C_fixed5))




%% Question 4. Value at Risk calculation

d6   = xlsread('HW2_Data.xls','DTB6','B7:B13102'); % read data from Excel
INaN = find(isnan(d6)==0); % remove NaN
d6   = d6(INaN);           % remove NaN

P6 = ??; % compute P6 using formula. n=182

% continuous compounding
r6 = -log(P6)*2;
dr6 = diff(r6);
% compute distribution of d r6
mu6 = mean(dr6);
sig6 = std(dr6);

% For the LIF
mu_LIF  = -D_LIF*P_LIF*mu6;
sig_LIF = sqrt((-D_LIF*P_LIF*sig6).^2);

VaR95 = ??;
VaR99 = ??;

% For the Fixed
mu_Fixed  = -D_Fixed*P_Fixed*mu6;
sig_Fixed = sqrt((-D_Fixed*P_Fixed*sig6).^2);

VaR95_Fixed = ??;
VaR99_Fixed = ??;



% historical distribution for the LIF
dP_LIF = -D_LIF*P_LIF*dr6;

VaR95_H = ??;
VaR99_H = ??;

% plot the results
figure
[NN,XX] = hist(dP_LIF,100);
bar(XX,NN/sum(NN)/mean(diff(XX)));

AAA=axis;
AAA(1)=-7; min(dP_LIF); 
AAA(2)=7; max(dP_LIF);
axis(AAA);
hold on
plot(XX,normpdf(XX,mu_LIF,sig_LIF),'-g',[-VaR99_H,-VaR99_H],[AAA(3),AAA(4)],':k',[-VaR99,-VaR99],[AAA(3),AAA(4)],'-.k','linewidth',2)
axis(AAA)
hold off
legend('Historical Distribution','Normal Distribution','99 % VaR: Historical','99 % VaR: Normal')
title('VaR and Historical Distribution for LIF')
disp(sprintf('95 percent Normal VaR of LIF: %2.2f',VaR95))
disp(sprintf('95 percent Historical VaR of LIF: %2.2f',VaR95_H))
disp(sprintf('99 percent Normal VaR of LIF: %2.2f',VaR99))
disp(sprintf('99 percent Historical VaR of LIF: %2.2f',VaR99_H))
saveas(gcf,'LIF_Var','epsc')



% historical distribution for the Fixed
dP_Fixed = -D_Fixed*P_Fixed*dr6;

VaR95_H_Fixed = ??;
VaR99_H_Fixed = ??;

% plot the results
figure
[NN,XX] = hist(dP_Fixed,100);
bar(XX,NN/sum(NN)/mean(diff(XX)));

% plot the results
AAA=axis; AAA(1)=min(dP_Fixed); AAA(2)=max(dP_Fixed); axis(AAA);
hold on
plot(XX,normpdf(XX,mu_Fixed,sig_Fixed),'-g',[-VaR99_H_Fixed,-VaR99_H_Fixed],[AAA(3),AAA(4)],':k',[-VaR99_Fixed,-VaR99_Fixed],[AAA(3),AAA(4)],'-.k','linewidth',2)
axis(AAA)
hold off
legend('Historical Distribution','Normal Distribution','99 % VaR: Historical','99 % VaR: Normal')
title('VaR and Historical Distribution for Fixed Rate Bond')

% displaying the results in the command window
disp(sprintf('95 percent Normal VaR of Fixed Rate Bond: %2.2f',VaR95_Fixed))
disp(sprintf('95 percent Historical VaR of of Fixed Rate Bond: %2.2f',VaR95_H_Fixed))
disp(sprintf('99 percent Normal VaR of Fixed Rate Bond: %2.2f',VaR99_Fixed))
disp(sprintf('99 percent Historical VaR of Fixed Rate Bond: %2.2f',VaR99_H_Fixed))
saveas(gcf,'Fixed_Var','epsc')




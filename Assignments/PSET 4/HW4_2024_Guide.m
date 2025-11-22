%% Commands
clear all; close all; clc

part1=1; % put = 1 to do part 1
part2=0; % put = 1 to do part 2. It only works if part1=0.


%%
if part1==1

disp('==================================================================')
disp('PART 1 - TIPS')
disp('==================================================================')


% load data into Matlab
DataR=xlsread('DataTIPS.xlsx','TIPS_01152013','C6:K29');        % Data for TIPS ("R" stands for "Real")      
MatR=xlsread('DataTIPS.xlsx','TIPS_01152013','AT6:BY29');       % Maturity Matrix (N x T) where N=number of bonds, T=max number of maturities. This will be entered in the function to compute "Model TIPS prices"
CashFlowR=xlsread('DataTIPS.xlsx','TIPS_01152013','M6:AR29');   % This is the cash flow matrix (N x T). Cash flow of each bond corresponding to each maturity. This will be intered in the function to compute "Model TIPS prices"

BidR=DataR(:,2);   % Bids
AskR=DataR(:,3);   % Asks
PriceR=(BidR+AskR)/2; % Mid

% Use Nelson Siegel Model.
% ------------------------
options=optimset('Display','iter','MaxFunEvals',50000,'MaxIter',50000,'TolFun',10^-10,'TolX',10^-10); % these are options that define the default in the numerical procedure
vecR0 = rand(1,4);   % initial guess use a random vector.

[vecR,JJ]=fminsearch('NLLS',vecR0,options,PriceR, MatR, CashFlowR);    % minimization algorithm fminunc (like "solver" in Excel)  
                                                                    % The 'NLLS' calls the matlab file with the minimization. 
                                                                    % Inputs are the traded prices PriceR, the Maturity Matrix MatR and the Cash Flow Matrix R
                                                                    % Output is the vector of parameters VecR = [theta0, theta1, theta2, lambda]
% translate solution back in "formulas"
th0R=vecR(1); th1R=vecR(2); th2R=vecR(3); laR=vecR(4);

% compute "fitted" TIPS prices. We can use the NLLS function directly to do so
[J,PPhatR]=NLLS(vecR,PriceR, MatR, CashFlowR);

figure
plot(DataR(:,5),PPhatR,'O',DataR(:,5),PriceR,'*')
legend('Fitted','Data','location','northwest')
xlabel('Maturity')
ylabel('Price')
title('TIPS') 
saveas(gcf,'figures/hw4_TIPS','epsc')

hs=1/2;         % time step 
T=[hs:hs:30]';  % times
YccR = ??       % real yield [insert formula from teaching notes]

% Real Discount
ZZccR = ??   ; % Real discount function Z_{real}(T)

figure
plot(T,ZZccR,'linewidth',2);
xlabel('Maturity')
ylabel('Z')
title('Real Discount')

figure
plot(T,YccR,'linewidth',2);
xlabel('Maturity')
ylabel('Yield')
title('Real Yield')


% Do Nominal Rates
% -----------------
DataN=xlsread('DataTIPS','Treasuries_01152013','B4:I303');            % load data
MatN=xlsread('DataTIPS.xlsx','Treasuries_01152013','BT4:EA303');      % Maturity Matrix (N x T) where N=number of bonds, T=max number of maturities. 
CashFlowN=xlsread('DataTIPS.xlsx','Treasuries_01152013','K4:BR303');  % This is the cash flow matrix (N x T). Cash flow of each bond corresponding to each maturity.

Bid=DataN(:,4);             % Bid
Ask=DataN(:,5);             % Ask
CleanPrice=(Bid+Ask)/2;     % Clean price = Mid price
AccrInt=DataN(:,8)/2;       % Accrued interest
PriceN = ??  ;              % Invoice price [add] 

% Use Nelson Siegel Model.

options=optimset('Display','iter','MaxFunEvals',50000,'MaxIter',50000,'TolFun',10^-10,'TolX',10^-10); % these are options that define the default in the numerical procedure

vec0 = rand(1,4); % initialize search from random number

[vec,JJ]=fminsearch('NLLS',vec0,options,PriceN, MatN, CashFlowN);    % minimization algorithm (like "Solver" in Excel). 
                                                                    % The 'NLLS' calls the matlab file with the minimization. 
                                                                    % Inputs are the traded prices PriceR, the Maturity Matrix MatR and the Cash Flow Matrix R
                                                                    % Output is the vector of parameters VecR = [theta0, theta1, theta2, lambda]


% translate solution back in "formulas"
th0=vec(1); th1=vec(2); th2=vec(3); la=vec(4);
% compute fitted prices. Use NLLS function again
[J,PPhat]=NLLS(vec,PriceN, MatN, CashFlowN);

figure
plot(DataN(:,7),PPhat,'O',DataN(:,7),PriceN,'*')
legend('Fitted','Data')
xlabel('Maturity')
ylabel('Price')
title('Treasuries') 

Ycc = ?? ; % nominal rate [insert formula. Use hs and T defined above for real rate]

% Nominal Discount function
ZZcc = ??;

% Fwd rates (at hs frequency)
FWD = ?? ; 

FWD=[Ycc(1);FWD];

figure
plot(T,ZZcc,'linewidth',2);
xlabel('Maturity')
ylabel('Z')
title('Nominal Discount')

figure
plot(T,Ycc,T,FWD,'--','linewidth',2);
xlabel('Maturity')
ylabel('Yield')
legend('yield','forward')
title('Nominal Yield and Forward')


% Break Even Rates (CC)
pi = ??  ; % insert formula for break-even rates

figure
plot(T,pi,'linewidth',2);
xlabel('Maturity')
ylabel('Rate')
title('Break Even Rate')

end


%%
if part2==1

disp('==================================================================')
disp('PART 2 - SWAP SPREAD TRADES')
disp('==================================================================')



%% Examine Historical Data

%%% Load and Prep Data %%%
load H15_SWAPS.txt    

II=find(isnan(H15_SWAPS(:,3))==0); % elimiinate NaN's
H15_SWAPS=H15_SWAPS(II,:);         % redefine data without NaN's

Dates=H15_SWAPS(:,1);              % dates
LIBOR=H15_SWAPS(:,2);              % LIBOR rates
Repo=H15_SWAPS(:,3);               % Repo rates
Swaps=H15_SWAPS(:,4:11);           % Swap rates across maturity
CMT = H15_SWAPS(:,12:end);         % US Treasury constant maturity rates


%%% Plot History %%%
DatesHistory=linspace(2000.5,2008.5,length(Dates));  %used for plot axis
linethick=1;

% Creat a plot
figure
hold all    % "hold all" makes sure the next plots are plotted "on top of each other"
plot(DatesHistory,Swaps(:,5)-CMT(:,4),'linewidth',linethick)    % plot Swap Spread
plot(DatesHistory,LIBOR(:,1)-Repo(:,1),'-.','linewidth',linethick) % plot LIBOR - Repo spread
title('5 year Swap Spread and Funding Spread over time')
legend('Swap Spread','Funding Spread')



%% Load and Prep Swap and Treasury Data for trading exercise

clear all           % clear all of the data before, to avoid overlap in definitions

linethick=2;        %for use with plots
freq=2;

% load prices 
    data=xlsread('HW3_Data','Daily_Bond_Swaps','B14:AC358');
    QuoteDates=data(:,1);           % quote date
    MatDates=data(:,2);             % maturity date
    Coupon=data(:,3);               % coupon rate
    bid=data(:,4);                  % bid
    ask=data(:,5);                  % ask
    maturity=data(:,6);             % time to maturity
    AccInt=data(:,7);               % accrued interest
    LIBOR=data(:,8);                % 3-month LIBOR rate
    Repo=data(:,9);                 % 3-month Repo rate
    SWAPS=data(:,10:17);            % Swap rates
    CMT = data(:,18:end);           % Constant Maturity Rates
    

PriceTnote=(bid+ask)/2;             % Compute Bond Price




        
%% Setup Trade
% Examine the usual side of the trade. At the end take negative to get
% profit on other side of the trade.

% (A) reverse Repo the 5 year note, and short in to the market.
%      Pay coupon and receive Repo rate
% (B) enter swap, receive fixed swap rate, pay floating, 3 M LIBOR


Swap30=SWAPS(:,end);        % time seirs of 30-year swap rates
SwapRate=Swap30(1);         % 30-year swap rate at trade date
CouponRate=Coupon(1);       % Bond coupon rate
SS=SwapRate-CouponRate;     % Swap Spread
LRS=LIBOR(1)-Repo(1);       % LIBOR-Repo Spread

disp('Swap Spread minus Carry')
disp(100*(SS-LRS));

% Yield to maturity of Note is
YTM=YTMsolver(PriceTnote(1),CouponRate,freq,5); % This uses the m-function YTMsolve.m, also available. 

%   LIBOR Swap curve at time 0 (this only required for later plot)  
    DataMat=[.25 1:5 7 10 30];   % maturity points given by data
    IntMat=[0.25:0.25:30];       % desired points on curve to do interpolation
    IntSwap=interp1(DataMat,[LIBOR(1);SWAPS(1,:)'],IntMat,'makima'); % interpolated swap curve at quarterly frequency 

    %Compute Discounts of Swap
    ZL(1)= ?? % First discount 
    for jj=2:length(IntSwap)    % bootstrap method: For every other maturity, use formula in TN1 and IntSwap as the swap rates
        ZL(jj)= ?? ;
    end

    LIBOR_Curve_0=?? % continuous compounded curve;

% POSITION OF TRADE    
position=1e8; % size of trade = 100 million
Tnotes=position/(PriceTnote(1)+AccInt(1));  % numbers of T-notes bought 



%% Valuation One quarter later
Today = 20090518; % select date

% find this date in dataset 
idxToday=find(Today==QuoteDates);   % idxToday refers to the numerical index in the dataset

% compute cash flows
CF_SS_Today= ?? ; % Cash flow from Swap Spread (assume you get accrued interest every quarter for simplicity)
CF_LRS_Today= ??; % Cash flow from LIBOR-REPO Spread 

% compute value of position
ValueTNotes_Today= ??; % Total value of T-Bonds?;
ValueRepo_Today=??; % Value Repo (assume no haircut, and this is ex-payment of Repo rate) 

% To value the old swap, we must first extract the LIBOR curve
 %Compute LIBOR Curve Z^{LIBOR}(t;T) by INTERPOLATION
    DataMat=[.25 1:5 7 10 30];   % maturity points given by data
    IntMat=[0.25:0.25:30];       % desired points on curve to do interpolation
    IntSwap=interp1(DataMat,[LIBOR(idxToday);SWAPS(idxToday,:)'],IntMat,'makima'); % interpolated swap curve at quarterly frequency 

    %Compute Discounts of Swap
    ZL(1)=??;           % What is the first discount? 
    for jj=2:length(IntSwap)
        ZL(jj)=?? ;     % Use formula from TN to bootstrap the LIBOR curve. 
    end
    LIBOR_Curve_Today=?? ; % Trasform Z(T) into continuously compounded rate.
    
    % Given swap curve, value swap 
    
    npay=30*4-1; % number of payments left AFTER today
    ValueSwap_Today=??; % use the formula for the value of the swap
    
    disp('Total Values on May 8, 2009')
    disp('Repo,  Swap , Swap + Repo')
    disp([ValueRepo_Today,ValueSwap_Today,ValueSwap_Today+ValueRepo_Today])
    
    
figure
plot(IntMat,LIBOR_Curve_0,IntMat,LIBOR_Curve_Today,'--','linewidth',2)
title('LIBOR curve on Feb 2 and May 18, 2009')
xlabel('Maturity')
ylabel('yield')
legend('Feb 2, 2009','May 8,2009')

end

clear all; close all; clc

%%
disp('==================================================================')
disp('INTEREST RATE TREES  & VALUATION')
disp('==================================================================')

clear all; close all; clc


% load data

Data=xlsread('HW6_Data_Bonds.xls','Sheet1','A6:I54');      
Mat=xlsread('HW6_Data_Bonds.xls','Sheet1','K6:BR54');       % Maturity Matrix (N x T) where N=number of bonds, T=max number of maturities.
CashFlow=xlsread('HW6_Data_Bonds.xls','Sheet1','BU6:EB54'); % This is the cash flow matrix (N x T). Cash flow of each bond corresponding to each maturity.

% assign names to some of the data
Bid=Data(:,6); Ask=Data(:,7); 
CleanPrice=(Bid+Ask)/2;

Price=CleanPrice+Data(:,9);     % add accrued interest to the clean prices.

% Use Nelson Siegel Model.

options=optimset('Display','iter','MaxFunEvals',50000,'MaxIter',50000,'TolFun',10^-10,'TolX',10^-10); % these are options that define the default in the numerical procedure

vec0=[5.3664   -0.1329   -1.2687  132.0669]/100; % use solution of HW3 as starting value.

[vec]=fminunc('NLLS_v2',vec0,options,Price, Mat, CashFlow);    % minimization algorithm. The 'NLLS' calls the matlab file with the minimization. 
                                                            % vec0 is starting value; Price, Mat, CashFlow are passed to ENLLS.m to compute the minimizing function.

                                                                
% translate solution back in "formulas"
th0=vec(1); th1=vec(2); th2=vec(3); la=vec(4);

[J,PPhat]=NLLS_v2(vec,Price, Mat, CashFlow);

figure
plot(Data(:,8),PPhat,Data(:,8),Price,'*')
legend('Fitted','Data')
xlabel('Maturity')
ylabel('Price')
 
hs=1/2; 
T=[hs:hs:30]';
Ycc=th0+(th1+th2)*(1-exp(-T/la))./(T/la)-th2*exp(-T/la);


% Discount
ZZcc=exp(-Ycc.*T);

% Fwd rates
FWD=-log(ZZcc(2:end)./ZZcc(1:end-1))/hs;
FWD=[Ycc(1);FWD];

figure
plot(T,ZZcc);
xlabel('Maturity')
ylabel('Z')

figure
plot(T,Ycc,T,FWD,'--');
xlabel('Maturity')
ylabel('Yield')
legend('yield','forward','location','northwest')



   %% ============
   % BUILDING A RECOMBINING BINOMIAL TREE
   % ============
   
   % load time series of six month rates
   DataY6 = xlsread('HW6_FRB_H15.csv','B7:B588');
   DataY6 =DataY6/100; % back in decimals

   r0=Ycc(1);
   
   % Build tree, semi-annual increments
   BDT_Flag=0; % = 0: HoLee; =1: SimpleBDT
   if BDT_Flag==1
       sigma= ?? ;% Need estimate of sigma using the Simple BDT
   else
       sigma= ??; % Need estimate of sigma using the Ho-Lee model
   end
   
   % initialize Implied Tree (ImTree)
   ImTree=zeros(30/hs,30/hs);
   % initialize zero coupon bond trees
   ZZTree=zeros(30/hs+1,30/hs+1,30/hs);
   
   % root of the tree
   ImTree(1,1)=r0;
   % first zero coupon bond (maturity i=2)
   ZZTree(1,1,1)=exp(-r0*hs);
   ZZTree(1:2,2,1)=1;
   
   % define a "minus 1" tree, to be used in the iterative method below
   ImTree_1=ImTree;
   
for i=2:30/hs

    % search for the "theta_i" that minimize the differerence between data
    % (in ZZcc(i)) and tree. Note the inputs require the "tree built so
    % far" (ImTree_1), the step number "i", the volatility sigma, the step
    % size hs, and whether we are using HoLee or Simple BDT (the flag)
    
[theta_i]=fminsearch('HoLee_SimpleBDT_Tree',0,[],ZZcc(i),ImTree_1,i,sigma,hs,BDT_Flag); % fminsearch is a minimizer algorithm

% store the resulting "theta"
theta(i-1)=theta_i;

% use the same function to record the current tree so far (step i) and the
% implied zero coupon bond tree from the minimization.
[FF,ImTree,ZZTreei]=HoLee_SimpleBDT_Tree(theta_i,ZZcc(i),ImTree_1,i,sigma,hs,BDT_Flag);

% record the minimized function (FF should be close to zero)
FFF(i-1)=FF;
% record the tree so far (it is used as input in the next step)
ImTree_1=ImTree;
% record the binomial tree of the zero coupon bond so far
ZZTree(1:i+1,1:i+1,i)=ZZTreei;


end

% compute the yield curve out of the tree
yyTree=squeeze(-log(ZZTree(1,1,1:end)))./T;
   
   figure
   plot(T,yyTree,T,Ycc,'--','linewidth',2)
   title('Yields: Data versus Tree')
   xlabel('Maturity')
   ylabel('yield')
   legend('Tree','Data')

   
   
%% Price of the Freddie Mac callable bond on the tree

% data
coupon = ??; % coupon rate from term sheet
TBond  = ??;  % maturity from term sheet
FCT    = ??;    % first call time from the term sheet

% transate inputs into "steps" on the tree
iT   = TBond/hs+1; % step at maturity: always 1 more step than T/hs for maturity
iFCT = FCT/hs+1; % step at first call time

pi = 0.5; % Risk Neutral Probability of up movement (assumption of both HoLee and simple BDT)
   
   PPTree_NC = zeros(iT,iT); % initialize the matrix for the non-callable coupon bond with maturity i. 
   Call      = zeros(iT,iT); % initialize the matrix for American call coupon bond with maturity i. 
   
   % final payoff of non-callable bond
   PPTree_NC(1:iT,iT) = 100; % final price is equal to 100
   
   
     % backward algorithm
 for j=iT-1:-1:1
                  
        PPTree_NC(1:j,j) = ?? ; % what is the price of the Non-Callable tree at time j?
        
        if j>=iFCT           ; % the backward calculation of call option depend on whether one can exercise it or not. j>iFCT is after first call date
        Call(1:j,j) = ??;      % add formula if it is possible to exercise the call option
        else
        Call(1:j,j) = ??        % add formula if it is not possible to exercise the call option
        end
 end
 
 PPTree_C=??            % what is the value of the CALLABLE bond on the tree?
 
 disp('Price Freddie Mac Callable Bond')
 disp('P_NC, Call, P_Call')
 disp([PPTree_NC(1,1),Call(1,1),PPTree_NC(1,1)-Call(1,1)]);
 

 % set the terms of the axis for the next plot.
 AAA=[0,.1,50,150];     % AAA = [minX, maxX, minY, maxY] as we desire for the plots
 figure
 for j=1:4
 subplot(2,2,j)
 plot(ImTree(1:iFCT-(j-1),iFCT-(j-1)),PPTree_NC(1:iFCT-(j-1),iFCT-(j-1)),'--',ImTree(1:iFCT-(j-1),iFCT-(j-1)),PPTree_C(1:iFCT-(j-1),iFCT-(j-1)),'linewidth',2);
 xlabel('Interest Rate')
 ylabel('Bond Prices')
 legend('Non-Callable','Callable','location','southwest')
 if j==1
 titlestr=['FCD'];      % use strings to define the title. For the first plot, title will be "FCD"
 else
 titlestr=[num2str(j-1) ' periods before FCD'];         % for the other plots will be the number of periods before FCD
 end
 title(titlestr)
 axis(AAA)
 end 
   
 
 
%% Duration and Convexity 
 % Duration - Non-Callable
 DNC=???
 % Duration - Callable
 DCallable=???

 % Convexity - Non-Callable
 Delta_NC_1u = (PPTree_NC(1,3)-PPTree_NC(2,3))/(ImTree(1,3)-ImTree(2,3));
 Delta_NC_1d = (PPTree_NC(2,3)-PPTree_NC(3,3))/(ImTree(2,3)-ImTree(3,3));
 C_NC        = ???
 
 % Convexity - Callable
 Delta_Call_1u = (PPTree_C(1,3)-PPTree_C(2,3))/(ImTree(1,3)-ImTree(2,3));
 Delta_Call_1d = (PPTree_C(2,3)-PPTree_C(3,3))/(ImTree(2,3)-ImTree(3,3));
 C_Callable    = ???
 
 
 
 

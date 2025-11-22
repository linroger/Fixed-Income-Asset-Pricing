% BDT Model

clear all; close all; clc;


%%
% DATA
SwRates=[0.152	0.2326	0.3247	0.346	0.7825	1.2435	1.599	1.853	2.052	2.2085	2.3371	2.4451	2.539	2.843	2.9863	3.0895]; % includes 1-, 3-, 6- month LIBOR as first 3 entries
CapsVol=[68.53	63.63	54.06	48.43	44.87	42.03	43.35	38.03	36.54	38.45	33.13	29.97	26.91	24.95	23.65];

SwRates=SwRates/100;  % transform in decimals
CapsVol=CapsVol/100;  % transform in decimals

MaturitySwaps=[1/12 3/12 6/12 1:10,15,20,30];   % maturities of Swaps 
MaturityCaps=[1:10,12,15,20,25,30];             % maturities of Caps

dt=.25;     % time step: Quarterly (as caps are quarterly).


% Augment Volatility data vector assuming *constant* volatility for short term caps
CapsVol=[CapsVol(1) CapsVol];

% Augment maturity using 3 month
MaturityCaps=[dt MaturityCaps];

% Interpolation using dt steps
IntMat=[dt:dt:30];
IntSwRate=interp1(MaturitySwaps,SwRates,IntMat,'spline');
IntVol=interp1(MaturityCaps,CapsVol,IntMat,'spline');




%% BOOSTRAP DISCOUNTS FROM INTERPOLATED SWAP RATES (TN 1)
ZSw=zeros(size(IntSwRate));
ZSw(1,1)=1/(1+IntSwRate(1,1)*dt);
NN=length(ZSw);
for i=2:length(ZSw)
    ZSw(i)=(1-IntSwRate(i)*dt*sum(ZSw(1,1:i-1)))/(1+IntSwRate(i)*dt);
end

% double check: Does ZSw imply the original swap curve?
for i=1:length(ZSw)
c2(i,1)=(1-ZSw(i))./(sum(ZSw(1:i))*dt);
end

% Compute continuously compounded yield
Zyieldcc=-log(ZSw)./IntMat;
% Transform them into quarterly compounding
Zyield=1/dt*(exp(Zyieldcc*dt)-1);

% short term interest rate
r0cc=-log(ZSw(1))/IntMat(1);
% in quarterly compounding
r0=1/dt*(exp(r0cc*dt)-1);

% compute continuously compounded forwards
fwdcc=-log(ZSw(2:NN)./ZSw(1:NN-1))/dt;
% compute quarterly compounded fwds
Fwd=1/dt*(exp(fwdcc*dt)-1);


figure
plot(IntMat(2:NN),Zyield(1:NN-1),'-.r',IntMat(2:NN),Fwd(1:NN-1),'-k')
legend('LIBOR Spot Curve','Forward Rates','location','southeast')
xlabel('Maturity')
ylabel('yield')
title('LIBOR Curve and Forward Curve')
saveas(gcf,'hw7_LIBOR_Curve','epsc')
%%
% Extract Forward Volatities


% Initialize cap vector
Caps=zeros(NN-1,1);

% FIRST, compute the dollar values of the caps for all maturities.
%       It uses BSfwd = Black's formula m-file
for i=1:NN-1
    if i==NN-1
        dddd = 1
    end
    dddd = dt*BSfwd(Fwd(1:i),IntSwRate(i+1),ZSw(2:i+1),IntVol(i+1),IntMat(1:i),1);
    Caps(i,1)=sum(dt*BSfwd(Fwd(1:i),IntSwRate(i+1),ZSw(2:i+1),IntVol(i+1),IntMat(1:i),1));
% Note:
    % IntMat:   maturity "T" into Black's model (to compute d1, d2)
    % Fwd:      Fwd Rates to be used in Blacks model
    % IntSwRate: Swap Rate. This is "shifted" one quarter ahead. The reason
    %           is that this is the strike rate of the (T+1 quarter) cap, which we
    %           are computing.
    % ZSw:      Discount Function. Also this is shifter one quarter ahead,
    %           as recall that something determined at T is paid at T+1
    %           quarter
    % IntVol:   Flat Volatility interpolated from the data.
    
end

aaa = [[Fwd 0];ZSw]';

%% SECOND, compute caplets which use forward vols instead of Flat vols.

Caplet=zeros(NN-1,1);

VVV=[0.00001:0.00001:1.4];   % possible vectors of volatilities. It will be used below to compute the implied 
                        % Fwd volatilities

% Start computing FWD Vols.
ImplVol(1,1)=IntVol(2); % the first Fwd vol = Flat Vol
Caplet(1,1)=Caps(1,1);  % the first caplet = cap with the first maturity
CapletMatrix(1,1)=Caplet(1,1); % initialization of a caplet matrix

% Loop for all the other maturities
for i=2:NN-1

    % compute caplets corresponding to maturities 1 to i-1 using the previously
    %           computed Fwd Volatilities (Vector ImplVol)
    CapletMatrix(1:i-1,i)=dt*BSfwd(Fwd(1:i-1),IntSwRate(i+1),ZSw(2:i),ImplVol(1:i-1),IntMat(1:i-1),1)';

    % Compute the value of the sum of caplets up to maturity i-1
    SumCaplets=sum(dt*BSfwd(Fwd(1:i-1),IntSwRate(i+1),ZSw(2:i),ImplVol(1:i-1),IntMat(1:i-1),1));
    fff = dt*BSfwd(Fwd(1:i-1),IntSwRate(i+1),ZSw(2:i),ImplVol(1:i-1),IntMat(1:i-1),1);

    % The value of caplet with maturity i must equal the value of the cap with
    %           that maturity minus the previously compute sum of caplets
    Caplet(i,1)=Caps(i,1) - SumCaplets;

    % Fill in the Caplet matrix for the missing i caplet
    CapletMatrix(i,i)=Caplet(i,1);
    
    % Obtained implied volatility. Idea is: compute the value of the caplet for
    % each of the values in VVV vector, and find the minimum of the distance between the
    % Black's formula (on the grid VVV) and the Caplet value computed above.
    [mV,iV]=min((dt*BSfwd(Fwd(i),IntSwRate(i+1),ZSw(i+1),VVV,IntMat(i),1)-Caplet(i,1)).^2);
    
    % The implied volatility is just the value of VVV that minimizes the
    % distance in the previous step, that is, element iV in the vector VVV;
    ImplVol(1,i)=VVV(iV);  
end

figure
plot(IntMat(2:NN),ImplVol(1:NN-1),'-.r',IntMat(2:NN),IntVol(2:NN),'-k',MaturityCaps,CapsVol,'*')
legend('Forward Volatilities','Flat Volatilities','Data')
xlabel('Maturity')
ylabel('Volatility')
title('Forward and Flat Volatility')
saveas(gcf,'hw7_fwdvol','epsc')



%%  Data Table

Table=[IntMat(2:NN);IntSwRate(2:NN);IntVol(2:NN);ImplVol(1:NN-1)]';

%%% Save Results to Table %%%
save Data_HW7.txt Table -ascii





%% 
% BUILD THE BDT TREE for MBS

%    dtstep=1/12; % monthly steps in BDT model (Note: by choosing dtstep = 1/4 we can check whether the BDT model prices zeros and caps correctly. See below) 
    dtstep=1/4; % monthly steps in BDT model (Note: by choosing dtstep = 1/4 we can check whether the BDT model prices zeros and caps correctly. See below) 

    % add shorter maturity to all of the vectors
    if dtstep==1/4
        IntMat2=IntMat;
        ImplVol2=ImplVol;
        ZSw2=ZSw;
    elseif dtstep<1/4
        IntMat2=[dtstep,IntMat];
        ImplVol2=[ImplVol(1),ImplVol];
        ZSw2=[1/(1+SwRates(1,1)*1/12),ZSw];
    end
    
    Maturity=[dtstep:dtstep:max(IntMat(1:end-1))]';
    ZSwInt=interp1(IntMat2,ZSw2,Maturity);
    ZYield=-log(ZSwInt(1:end-1)')./Maturity(1:end-1)';
    FwdVol=interp1(IntMat2(1:end-1),ImplVol2,Maturity); 
       
    options=optimset('Display','final','MaxFunEvals',10000,'MaxIter',10000,'TolFun',10^-10,'TolX',10^-10);
    
    NN=length(ZYield);  % size of zero-coupon yield vector
    
    % parallell shift for duration computation
    dy=0; .1/100;
    ZYield=ZYield+dy;
    
    % Intialize the Implied Tree
    ImTree=zeros(NN,NN);
    
    % First note is the c.c. yield
    ImTree(1,1)=ZYield(1);
    
    xx=ZYield(1); % starting value in the search engine
    
    %Begin the loop across all of the other values
    for j=1:NN-1
        xx=[xx*(.75)]; % starting value of search
        
        % use minfunBDTNew.m file to solve the for the interest rate such
        % that the zero coupon bond out of the tree equals the zero coupon
        % bond from the data
        %xx=fsolve('minfunBDTNew',xx,options,ImTree,ZYield(j+1),FwdVol(j+1),Maturity(j+1)-Maturity(j),j+1);
        xx=fsolve('minfunBDTNew',xx,options,ImTree,ZYield(j+1),FwdVol(j),Maturity(j+1)-Maturity(j),j+1);
        
        % plug back the solution in the mfile minfunBDTNew.m and obtain the
        % vector of interest rate values
        %[F,vec]=minfunBDTNew(xx,ImTree,ZYield(j+1),FwdVol(j+1),Maturity(j+1)-Maturity(j),j+1);
        [F,vec]=minfunBDTNew(xx,ImTree,ZYield(j+1),FwdVol(j),Maturity(j+1)-Maturity(j),j+1);
        
        % update the tree
        ImTree(1:j+1,j+1)=vec';
    end
    pause(1)
%    clc


%%
% Check if tree prices zeros and caps correctly

for i=2:NN % add one step to maturity to ensure the last bond has 30/dt to maturity (i=2 has 6 months to maturity)
   ZTree(1:i,i,i)=1; % initialize tree with maturity i
     % backward algorithm
        for j=i-1:-1:1
            ZTree(1:j,j,i) = exp(-ImTree(1:j,j)*dtstep).*(0.5*(ZTree(1:j,j+1,i)+ZTree(2:j+1,j+1,i)));
        end
end

figure
plot(Maturity(1:NN-1),squeeze(ZTree(1,1,2:NN)),'--*',IntMat2(1:end-1),ZSw2(1:end-1),':O')
title('Tree implied discount function versus LIBOR discount')
xlabel('Maturity')
legend('Tree-implied discount','LIBOR discount')
saveas(gcf,'hw7_Zero_Tree','epsc')

%%
% Check if tree prices caps correctly
if dtstep==0.25;
% Use the binomial tree to check if we price zeros and caps correctly. 
% only works if dtstep==0.25

  % compute caps and zeros on the tree
   
   for i=2:NN ; % add one step to maturity to ensure the last bond has 30/dt to maturity (i=2 has 6 months to maturity)
   CapTree(1:i,i,i)=dt*exp(-ImTree(1:i,i)*dt).*max((exp(ImTree(1:i,i)*dt)-1)/dt-IntSwRate(i+1),0); %
     % backward algorithm
        for j=i-1:-1:1
            if j>1
            CapTree(1:j,j,i) = exp(-ImTree(1:j,j)*dtstep).*(0.5*(CapTree(1:j,j+1,i)+CapTree(2:j+1,j+1,i))+dtstep*max((exp(ImTree(1:j,j)*dtstep)-1)/dt-IntSwRate(i+1),0));
            else
            CapTree(1:j,j,i) = exp(-ImTree(1:j,j)*dtstep).*(0.5*(CapTree(1:j,j+1,i)+CapTree(2:j+1,j+1,i)));
            end
        end
        
      
   end


figure
plot(IntMat(2:end),Caps,'*',IntMat(2:end-1),squeeze(CapTree(1,1,:)),'O')
legend('Data','Binomial Tree','location','northeast')
xlabel('Maturity')
ylabel('Dollars')
saveas(gcf,'hw7_Caps_Tree','epsc')

end




    
    %% 
    % Pricing of GNSF 4
    % -------------------

   
   % mortgage characteristics
   WAC=4.492/100;            % weighted averge mortgage rate
   WAM=round(311/12)/dtstep; % number of periods, given step size in BDT model   
   PP0=100;                  % reset current principal to 100
   aa=1/(1+WAC*dtstep);
   
   % pass through coupon rate
   rbar=4/100;
   
       
   NN=WAM; % Redefine maturity using actual number of months left.
   MCoupon = PP0*(1-aa)/(aa-aa^(NN+1)); % Monthly dollar coupon at time 0

   PP=zeros(NN+1,1);        % initialize vector of principal balance
   PriPaid=zeros(NN,1);     % initialize vector of principal paid
   IntPaid=zeros(NN,1);     % initialize vector interest paid
   
   PP(1)=100;  % we redefine the current principal = 100, even if it has been repaid. Everything scales up linearly. Without prepayments, the current amount of principal left should be given by MCoupon*(aa-aa^(WAM+1))/(1-aa); 
   
   % scheduled interest and principal payments
   for i=1:NN
   IntPaid(i) = ???;     % interest paid
   PriPaid(i) = ???;     % principal paid
   PP(i+1)    = ???;     % principal remaining
   end
   aa = 1000*[[0;IntPaid] [0;PriPaid] PP]';
 
   figure
   subplot(2,1,1)
   plot([1:NN],IntPaid,[1:NN],PriPaid,'--','linewidth',2)
   xlabel('Months')
   ylabel('dollars')
   title('Scheduled Interest and Principal Payments')
   legend('Scheduled Interest','Scheduled Principal')
   AAA=axis; AAA(1)=1; AAA(2)=NN; AAA(3)=0; AAA(4)=MCoupon; axis(AAA);
   
   subplot(2,1,2)
   plot([1:NN],PP(1:end-1),'linewidth',2)
   xlabel('Months')
   ylabel('Principal')
   title('Principal Balance')
   AAA=axis; AAA(1)=1; AAA(2)=NN; AAA(3)=0; AAA(4)=100; axis(AAA);
   saveas(gcf,'hw7_Scheduled_Principal','epsc')

   
   

   % initialize variables
   VNoC   = zeros(NN,NN);       % value of future scheduled coupon without prepayment option
   Call   = zeros(NN,NN);       % value of the call option
   VC     = zeros(NN,NN);         % value of mortgages with call
   VPT    = zeros(NN,NN);        % value of pass through security
   VPTNoC = zeros(NN,NN);     % value of pass through security without call option
   ExIdx  = zeros(NN,NN);      % index where exercise takes place
   
   % final value of call option at maturity
    for j=1:NN
        Call(j,NN)=0; 
    end
    
    % Backward loop on the binomial tree starts here
    
    for i=NN-1:-1:1   % maturity loop
        for j=1:i     % node loop
            
            CFNoPreP = ???;  % CF if there is no prepayment = Monthly Coupon
            
            % Value of mortgage without prepayment
            VNoC(j,i) = ???;
            
            if i>1  % assume exercise can only be after time 1
            
                % call option to 
                Call(j,i) = ???;
            else    % at time 1 canot exercise
                
                Call(j,i) = ???;
            end
            
            % value of callable mortgage
            VC(j,i)=VNoC(j,i)-Call(j,i);
            
            if Call(j,i)==VNoC(j,i)-PP(i) % if exercise in node (j,i)
                
                VPT(j,i) = ????; % value of pass through security equal to principal
            else % if not exercise in node (j,i)
                CFPassThrough(i+1,1) = rbar*dtstep*PP(i)+PriPaid(i);  % if no exercise, cash flow of PT equal to interest plus scheduled principal paid (all next period)
                VPT(j,i) = ???;  % value of PT security given by backward computation 
            end
            
            % value of pass through without call feature.
            CFPassThrough2(i+1,1) = rbar*dtstep*PP(i)+PriPaid(i);  % if no exercise, cash flow of PT equal to interest plus scheduled principal paid (all next period)
            VPTNoC(j,i)           = exp(-ImTree(j,i)*dtstep)*(CFPassThrough2(i+1,1)+1/2*(VPTNoC(j,i+1)+VPTNoC(j+1,i+1)));
            
        end
    end
   
 
    disp('Rational Pricing on Binomial Trees')
    disp('----------------------------------')

    disp('Value of Mortgage Pool without Option to Exercise')
    disp(VNoC(1,1)/PP(1)*100)

    disp('Value of Mortgage Pool with Option to Exercise')
    disp(VC(1,1)/PP(1)*100)

    disp('Value of PT without Option to Exercise')
    disp(VPTNoC(1,1)/PP(1)*100)
    
    disp('Value of PT with Optimal Exercise')
    disp(VPT(1,1)/PP(1)*100)
    

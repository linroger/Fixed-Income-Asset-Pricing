%% Commands
clear all; close all; clc

part1=1; % put = 1 to do part 1
part2=1; % put = 1 to do part 2. It requires part 1 to do one of the exercises



%%
if part1==1
    disp('==================================================================')
    disp('PART 1 - PRINCIPAL COMPONENT ANALYSIS')
    disp('==================================================================')
    
    %% data preparation, yields, zeros
    
    % read data
    data = xlsread('FByields_2024.xlsx','FBYields','A6:I864');
    data = data(:,[1 3:end]);
    
    % find dates
    date1 = 20090331; I_1 = find(data(:,1)==date1);
    date2 = 20090430; I_2 = find(data(:,1)==date2);
    
    % term structure on the two dates
    Mat     =[1/12,3/12,1:5]';
    MatInt  = [.5:.5:5]';
    yields1 = data(I_1,2:end)'/100; % put in decimals
    yields2 = data(I_2,2:end)'/100; % put in decimals
    
    % compute relevant yields at semi-annual frequencies
    Y_Int_1 = interp1(Mat,yields1,MatInt,'pchip');
    Y_Int_2 = interp1(Mat,yields2,MatInt,'pchip');
    
    % compute zeros
    Z1=??
    Z2=??
    
    % plot results
    figure;
    plot(MatInt,Y_Int_1,MatInt,Y_Int_2,':','linewidth',2)
    title('Change in the term structure between March 31st and April 30th');
    xlabel('Maturity');
    ylabel('Yield')
    legend('March 31, 2009','April 30,2009');
    %saveas(gcf,'hw3_yieldc','epsc')
    
    
    %% Compute value of LIF on both cases
    
    freq=2;
    coup_fix=10;
    TT=5;
    
    % compute C(T), the cash flows at various maturities, of the "fixed" rate
    % bond underlying the LIF
    CF_fixed=(coup_fix/freq)*ones(TT*freq,1);
    CF_fixed(end)=CF_fixed(end)+100;
    
    P_Fixed1=Z1(1:TT*freq)'*CF_fixed;
    P_Fixed2=Z2(1:TT*freq)'*CF_fixed;
    P_Float=100;
    P_zero1=100*Z1(TT*freq);
    P_zero2=100*Z2(TT*freq);
    % as from HW2, the value of the LIF is given by?
    P_LIF1=??
    P_LIF2=??
    
    disp(' ')
    disp('P_LIF March, P_LIF April')
    disp([P_LIF1, P_LIF2,])
    
    % compute average change in the yield curve between date 1 and date 2
    dr=mean(Y_Int_2-Y_Int_1);
    D_LIF_HW2=11.29;    % the duration of LIF from HW2
    C_LIF_HW2=58.45;    % the convexity of LIF from HW2
    
    dPP_D = ??         % the change in price due to duration (add formula)
    dPP_D_C = ??       % the change in price due to convexity (add formula)
    
    disp(' ')
    disp('Dur-based  Dur/Conv-based   Actual ')
    disp('Return (%)   Return (%)   Return (%)')
    disp([dPP_D*100,dPP_D_C*100,  (P_LIF2-P_LIF1)./P_LIF1*100])
    
    
    
    
    %% Calculating Principal Components
    
    YY=data(1:I_1-1,2:end);
    T=length(YY);
    
    % We need to build a dataset. Moreover, we need the "betas" from PCA at
    % given horizons. Thus, for each month, we build the dataset and interpolate the yields
    % on the relevant  horizons, to compute proper betas for the duration
    yields=zeros(T,length(MatInt));
    for ii=1:T
        yields(ii,:)=interp1(Mat',YY(ii,:),MatInt','pchip');
    end
    
    % compute changes in yields
    dy=diff(yields);
    % compute the covariance matrix of the changes in yields
    SIGMA=cov(dy);
    % compute the eigenvalues and eigenvectors of SIGMA
    [V E]=eig(SIGMA);
    % compute the vectors of eigenvalues from the diagonal matrix E
    E=diag(E);
    % compute number of eigenvalues. "Eig" command orders eigenvalues and
    % eigenvectors from the smallest to the largest. The methodology requires
    % we start from the largest eigenvalue.
    mm=length(E);
    
    % compute the betas "explicitly"
    % component 1
    
    % compute the data
    z1=dy*V(:,mm);          % transform data using eigenvector corresponding to largest eigenvalue (mm)
    Level=cumsum(z1);       % Level factor
    z11=[ones(T-1,1),z1];   % make a vector for the regressions (include vector of ones)
    b1=inv(z11'*z11)*z11'*dy;   % Regression beta = first component (note: first component is just the second row. The first is just the intercept)
    e1=dy-z11*b1;               % Residuals to be used in the next step
    
    RSS1=diag(e1'*e1);  %Residual sum of squares
    sisq1=RSS1/(T-3);   % variance of residuals
    t1=b1./(sqrt(diag(inv(z11'*z11))*sisq1'));  % t-statitstics
    
    % component 2
    
    z2=e1*V(:,mm-1);            % transform the residuals using the eigenvector correposponding to the second largest eigenvalue (mm-1)
    Slope=cumsum(z2);           % Slope factor
    z22=[ones(T-1,1),z1,z2];    % prepare vector for regression
    b2=inv(z22'*z22)*z22'*dy;   % second component = third row of regression beta
    e2=dy-z22*b2;               % residuals from regression to compute next component
    
    RSS2=diag(e2'*e2);          % to compute t-stats
    sisq2=RSS2/(T-3);
    t2=b2./(sqrt(diag(inv(z22'*z22))*sisq2'));
    
    % assign the two principal components a new name for simplicity
    betas=b2(2:3,:);
    disp('betas: 6 months - 5 years')
    disp(betas')
    
    figure
    plot(MatInt,betas(1,:),MatInt,betas(2,:),':','linewidth',2)
    title('Factors');
    xlabel('Maturity');
    ylabel('Factor Beta')
    legend('Level','Slope');
    %saveas(gcf,'hw3_factor_beta','epsc')
    
    % make a vector of dates for the plot
    BegDate=1952.5;         % beginning of data
    EndDate=2009+2/12;      % end of data
    NDates=length(Level)-1; % number of data points
    DatePlot=BegDate+(EndDate-BegDate)*[0:NDates]/NDates; % Vector of dates for the plot
    
    figure
    subplot(2,1,1)
    plot(DatePlot,Level,'linewidth',2)
    title('Level Factor');
    subplot(2,1,2)
    plot(DatePlot,Slope,'linewidth',2)
    title('Slope Factor');
    %saveas(gcf,'hw3_factors','epsc')
    
    % change in levels and slope
    Diff_Level=dy*betas(1,:)';
    Diff_Slope=dy*betas(2,:)';
    
    
    %% Computing factor durations of LIF
    
    maturity=MatInt';      % redefine some variables, to use same codes as HW2
    stripweights=(Z1(1:freq*TT)/P_Fixed1)*(coup_fix/2);  % we use the "1" price above, as it refers to March
    stripweights(end)=stripweights(end)+(Z1(freq*TT)*100)/P_Fixed1;    %principal weight
    
    % Duration against the level factor
    D_Fixed_L=??
    D_Float_L=??
    D_Zero_L=??
    %take weighted average of level-factor durations
    D_LIF_L=??
    
    % Duration against the slope factor
    D_Fixed_S=??
    D_Float_S=??
    D_Zero_S=??
    %take weighted average of slope-factor durations
    D_LIF_S=??
    
    Diff_Level_March_April=?? % What is the change in value of LIF due to Level factor?
    Diff_Slope_March_April=?? % What is the change in value of LIF due to Slope factor?
    
    dPP_Factors=Diff_Level_March_April+Diff_Slope_March_April;     % What is the change in value of LIF due to Level and Slope factors?
    
    disp(' ')
    disp('Dur-based Dur/Conv-based Factor-based  Actual ')
    disp('Return (%)  Return (%)    Return (%)  Return (%)')
    disp([dPP_D*100,dPP_D_C*100, dPP_Factors*100, (P_LIF2-P_LIF1)./P_LIF1*100])
    
end


%%
if part2==1
    disp('==================================================================')
    disp('PART 2 - PREDICTABILITY OF EXCESS RETURNS                         ')
    disp('==================================================================')
    
    % Load all the data at annual frequency
    DataB=xlsread('FBYields_2024_v2.xlsx','Annual','A6:AE76');
    
    DateB=round(DataB(:,1)/100); % Date vector
    yields=DataB(:,2:6);    % Yields: Available in Spreadsheet
    fwd=DataB(:,18:21);     % Forwards: Available in Spreadsheet
    RetB=DataB(:,26:29);    % Holding Period Return
    AveRetB=DataB(:,30);
    CP=DataB(:,31);         % Cochrane-Piazzesi Factor
    
    %BegDateB=round(DateB(1,1));   % Initial date in sample
    %EndDateB=round(DateB(end,1)); % End date in sample
    %NDatesB=size(DateB,1);        % number of data points
    %DatePlotB=BegDateB+(EndDateB-BegDateB)*[0:NDatesB]/NDatesB; % make a vector of dates for the plot
    DatePlotB = DateB;
    
    % Predictive Regressions Regressions
    
    TABLE_FB=[];  % initiate table to save Fama - Bliss Results
    TABLE_CP=[];  % initiate table to save Cochrane - Piazzesi Results
    for jpred=1:4  % jpred = inded of maturity of the bond jpred = 1 means maturity 2, jpred = 2 means maturity 3 etc...
        
        YY=RetB(2:end,jpred);  % bond to predict with maturity jpred + 1
        
        % Fama Bliss.
        XX=[ones(size(YY,1),1),??];              % DEFINE PREDICTIVE VARIABLE (the "X")
        [BB,tBB,R2]=regression_35130(YY,XX);         % run the regression (codes in "regression_35130.m")
        TABLE_FB=[TABLE_FB;jpred+1,BB',tBB',R2];     % save result on a table
        
        % Cochrane Piazzesi
        XX=[ones(size(YY,1),1),??];               % DEFINE PREDICTIVE VARIABLES (the "X")
        [BB,tBB,R2]=regression_35130(YY,XX);         % run the regression (codes in "regression_35130.m"
        TABLE_CP=[TABLE_CP;jpred+1,BB',tBB',R2];     % save result on a table
        
        
    end
    
    disp('Fama-Bliss');
    disp(TABLE_FB);
    disp(' ');
    disp('Cochrane-Piazzesi');
    disp(TABLE_CP);
    
    % FIGURES FAMA BLISS
    ibond=4; % use bond 4 ==> 5 year to maturity
    
    figure
    YY=RetB(2:end,ibond);
    XX=[ones(size(YY,1),1),??];  % see above: Need to select regressor!
    [BB,tBB,R2]=regression_35130(YY,XX);
    plot(XX(:,2),YY,'*',XX(:,2),XX*BB,'-k','linewidth',2);
    ylabel('realized bond return'); xlabel('5-year forward spread')
    legend('data','regression fit','location','southeast')
    title('Realized 5-year Bond Return vs 5-year Forward Spread')
    
    figure
    yyaxis left;
    plot(DatePlotB(1:end-1),YY,':','linewidth',2);
    yyaxis right;
    plot(DatePlotB(1:end-1),XX(:,2),'-','linewidth',2);
    legend('lagged realized bond return','forward spread','location','northwest')
    title('5-year Bond Return and 5-year Forward Spread')
    
    figure
    plot(DatePlotB(1:end-1),YY,':',DatePlotB(1:end-1),XX*BB,'linewidth',2);
    legend('lagged realized bond return','predicted return','location','northwest')
    title('5-year Bond Return and Predicted Return from 5-year Forward Spread')
    
    % FIGURES COCHRANE PIAZZESI
    
    figure
    YY=RetB(2:end,ibond);
    XX=[ones(size(YY,1),1),??];  % see above: Need to select regressor!
    [BB,tBB,R2]=regression_35130(YY,XX);
    plot(XX(:,2),YY,'*',XX(:,2),XX*BB,'-k','linewidth',2);
    ylabel('realized bond return'); xlabel('5-year forward spread')
    legend('data','regression fit','location','southeast')
    title('Realized 5-year Bond Return vs CP factor')
    
    figure
    yyaxis left;
    plot(DatePlotB(2:end),YY,':','linewidth',2);
    yyaxis right;
    plot(DatePlotB(1:end-1),XX(:,2),'-','linewidth',2);
    legend('lagged realized bond return','CP factor','location','northwest')
    title('5-year Bond Return and CP factor')
    
    figure
    plot(DatePlotB(1:end-1),YY,':',DatePlotB(1:end-1),XX*BB,'linewidth',2);
    legend('lagged realized bond return','predicted return','location','northwest')
    title('5-year Bond Return and Predicted Return from CP factor')
    
end
%% Bus 35130 HW1

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




%% =================== Question 1 ===========================
% Determining time series of BEY and provide its plot

% Read time series of 3-month Treasury Bills from 'DTB3.xls' Excel file, sheet 'Dtb3'
rates=readtable('DTB3_2024.xls','Sheet','DTB3','Range','B2:B18315');
rates = table2array(rates);

II=find(rates>0);                % find business dates and ignore days when markets are closed
rates = rates(II)/100;           % get rid of percentages by dividing by 100

%input in BEY formula
N1=??;
N2=??;
N3=??;
BEY = N1*rates./(N2-rates*N3); % BEY calc according to the formula

% Plot the results
figure(1)
plot(rates,'-.');                  % plot of the T-bill rates
hold on;                           % this command allows multiple plots in figure               
plot(BEY,'r','linewidth',1);       % actual plot  
ylabel('3-month T-bill rates');    % label on y axis
legend('Quoted discounts', 'BEY'); % plot legend





%% =================== Question 2 ===========================
% Estimating of the AR(1) process for interest rates

m = ?? ;              % the number of regression observations
Y = BEY(??:??);       % the dependent variable
X = BEY(??:??);       % the independent variable

% OLS regression
ybar = mean(Y); % Unconditional mean of Y variable defined above
xbar = mean(X); % Unconditional mean ofXY variable defined above

% beta_hat = cov(X,Y)/var(X), but the Matlab function 'cov' gives the
% 2 by 2 covariance matrix, not just the covariance, so we use just one number s(1,2)
% Complete the following

s = ??;              % Covariance calculation
beta_hat = ??;       % Regression slope, beta
alpha_hat = ??;      % Regression intercept, alpha

% Calculating error terms
eps = Y-alpha_hat-beta_hat*X; % Regression residuals    
sig = sqrt(var(eps)*m/(m-1)); % Sample standard error of residuals

% Displaying the results in matlab's Command Window
disp(' Standard approach ')
disp('===========================================================================')
disp('Regression coefficients for rates:')
disp(sprintf('beta_hat = %g',beta_hat))
disp(sprintf('alpha_hat = %g',alpha_hat))
disp('Standard error of residuals:')
disp(sprintf('sig = %g',sig))
disp('===========================================================================')
disp(' ')



%ALTERNATIVELY:
% If you like to do regression in matrix form, the code below is
% another way to estimate the parameters of the AR(1) process.
% The results is the same as the approach above
% You can use the one above of this one, whichever you prefer

m = ??;              % m is the number of observations
X = [ones(m-1,1) BEY(??:??)]; % X variable, as matrix, can include many explanatory variables
Y = BEY(??:??);                 % Y variable, as column vector

% OLS regression: b(1) is alpha hat, b(2) is beta hat
b = inv(X'*X)*X'*Y;
alpha_hat = b(1); beta_hat = b(2);

% error terms
eps = Y - X*b;
sig = sqrt(var(eps)*m/(m-1));    % won't make a difference

% Displaying the results in matlab's Command Window
disp(' Matrix based approach ')
disp('===========================================================================')
disp('Regression coefficients for rates:')
disp(sprintf('beta_hat = %g',b(2)))
disp(sprintf('alpha_hat = %g',b(1)))
disp('Standard error of residuals:')
disp(sprintf('sig = %g',sig))
disp('===========================================================================')
disp(' ')

% Some descriptive statistic - t-test for alpha hat and beta hat
cov_b = sig^2*inv(X'*X);   
std_b = sqrt(diag(cov_b));
T_b   = b./std_b





%% =================== Question 3 ===========================
% Interest rate forecasts over the period of 5 years

n = ??;                            % number of years to forecast

rate_forecast = zeros(n*252+1,1); % vector to store the forecasts
rate_forecast(1) = BEY(end);      % r_today - the most recent interest rate

% Loop to carry out AR(1) forecast
% r_forecast(i+1) = alpha + beta_hat * r_forecast(i). Fill in the blanks
for i = 1:n*252
    rate_forecast(??) = ??+??*rate_forecast(??);
end


% Plot the results
linethick = 2;    %set line thickness to be used in plot commands below
figure(2);
x = [0:1/252:n]';   
LR_mean = alpha_hat/(1-beta_hat);   % Long-run mean of the interest rate

plot(x,rate_forecast, x,LR_mean*ones(length(x),1),'--','linewidth',linethick)
AAA=axis; AAA(4)=.07; axis(AAA);
xlabel('forecasting horizon (years)');
ylabel('3-month T-bill rates');
legend('Time Series Forecast','Long-term interest rate')




%% =================== Question 4 ===========================
% Computing both the current yield curve and forward rates for all maturities and comparing the 
% forecasts of future interest rates that are implicit in the forward rates to those obtained in question 3



strips_data = readtable('DTB3_2024.xls','Sheet','Strip Prices','Range','A2:B61');  % Load the Excel data on strips
strips_data = table2array(strips_data);

Mat  = strips_data(:,1);      % Sotore maturities into a vector
Imat = max(find(Mat<n+.25));  % Find index with maturity n
Mat  = Mat(1:Imat);           % Take all the maturities up to n
Zfun = strips_data(:,2);      % Strip prices
Zfun = Zfun(1:Imat);

yield = ??;                % Compute yields from strips (quoted term structure)
fwds = ??                  % Compute forward rates

% Plot the results: Z-function only
figure(3);
plot(Mat,Zfun,'linewidth',linethick)
xlabel('forecasting horizon (years)');
ylabel('Z function')
title('Discount function')

% Plot the results: yields vs forward rates
figure(4)
plot(Mat,yield,Mat(1:end-1),fwds,'-.','linewidth',linethick)
legend('Yield','Forward','location','best')
xlabel('Maturity');ylabel('spot rate')
title('Yields and Forwards')

% Plot the results: forward rates vs AR(1) forecast of interest rates
figure(5)
plot(??,??,'-.',??,??,'linewidth',linethick)  
legend('Forward','Time Series Forecast','location','best')
xlabel('forecasting horizon (years)');ylabel('spot rate')
title('Two forecasts of future interest rates')
saveas(gcf,'hw1fig3','epsc')





%% Saving the results

saveas(figure(1),'HW1_fig1.fig') % Saving Figure 1
saveas(figure(1),'HW1_fig1.jpg') % Saving Figure 1

saveas(figure(2),'HW1_fig2.fig') % Saving Figure 2
saveas(figure(2),'HW1_fig2.jpg') % Saving Figure 2

saveas(figure(3),'HW1_fig3.fig') % Saving Figure 3
saveas(figure(3),'HW1_fig3.jpg') % Saving Figure 3

saveas(figure(4),'HW1_fig4.fig') % Saving Figure 4
saveas(figure(4),'HW1_fig4.jpg') % Saving Figure 4

saveas(figure(5),'HW1_fig5.fig') % Saving Figure 5
saveas(figure(5),'HW1_fig5.jpg') % Saving Figure 5




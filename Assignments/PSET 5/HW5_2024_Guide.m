%% Close all windows and clear all variables
clear all; close all; clc



%% Input data
dataMat   = [1/12  1:7]';          % maturity points given by data
intMat    = [0.25:0.25:6]';             % desired points on curve to do interpolation

% Enter Libor/Swap rates from Bloomberg screen to the following variable 
% (8 numbers in percentage format)
dataRates = [???  ???  ???  ???  ???  ???  ???  ???]'/100;   % rates points given by data

% Interpolated swap curve at quarterly frequency
intRates=interp1(dataMat,dataRates,intMat,'pchip'); 






%% Bootstrap

% Discount Factor calculations via Bootstrap
dt = 0.25;
ZZ(1) = 1/(1+dt*intRates(1));
for i=2:length(intRates)
    ZZ(i,1) = (1-intRates(i)*dt*sum(ZZ(1:i-1)))/(1+intRates(i)*dt);    
end

fwd_Z          = [NaN;ZZ(2:end)./ZZ(1:end-1)]; % Forward Discount
fwd_Rate       = (1./fwd_Z-1)/dt;              % Forward Rate
fwd_Z_Swaption = [NaN(4,1);ZZ(5:end)./ZZ(4)];  % Forward Discount for Swaption calculations




%% 1-year Cap
T     = 1;                % Time-to-maturity (years)
X     = ??;           % Strike Rate (from data)
sigma = ??;           % Volatility  (from data)
F     = fwd_Rate(2:T/dt); % Forward Rates
Z     = ZZ(2:T/dt);       % Discount Factors
CallF = 1;                % 1 for calls and 2 for puts
T_Cap = [dt:dt:T-dt]';    % caplet maturities

Caplets_1Year = 100*dt*BSfwd(F,X,Z,sigma,T_Cap,CallF);  % Caplet prices (use the BSfwd.m function. Note that if inputs are vectors, so will its output)
Cap_1Year     = sum(Caplets_1Year);                    % Cap price

disp(['1-year Cap Price = ' num2str(Cap_1Year)])



%% 2-year Cap
T     = 2;                % Time-to-maturity (years)
X     = ??;           % Strike Rate (from data)
sigma = ??;           % Volatility  (from Data)
F     = fwd_Rate(2:T/dt); % Forward Rates
Z     = ZZ(2:T/dt);       % Discount Factors
CallF = 1;                % 1 for calls and 2 for puts
T_Cap = [dt:dt:T-dt]';    % caplet maturities

Caplets_2Year = 100*dt*BSfwd(F,X,Z,sigma,T_Cap,CallF); % Caplet prices (use BSfwd.m function)
Cap_2Year     = sum(Caplets_2Year);                    % Cap price

disp(['2-year Cap Price = ' num2str(Cap_2Year)])



%% Swaption
T     = 1;       % Swaption Maturity
X     = ??; % Strike Rate (5-year swap rate) (from data)
sigma = ??;  % Swaption Volatility  (from data)
Z     = 1;       % Discount Factors
CallF = 1;       % 1 for calls and 2 for puts

% Fwd 5-year Swap Rate
SwapRate = 1/dt*(1-fwd_Z_Swaption(end))/ sum(fwd_Z_Swaption(5:end));

% A-factor
A = dt*sum(ZZ(5:end));

% Swaption price
Swaption = A*BSfwd(SwapRate,X,Z,sigma,T,CallF);  % use BSfwd.m function. Compare formula with TNs to understand what inputs to use.

disp(['Swaption Price = ' num2str(Swaption)])


# Packages
import math
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from numpy import exp, sqrt, log, mean, std, nonzero, isnan, array, zeros, ones, amax, amin
from numpy import concatenate, arange, interp, diff, cov, var, diag, eye, cumsum, tile, transpose
from numpy.linalg import eigh,inv
from scipy.stats import norm
from scipy.interpolate import interp1d, PchipInterpolator

print('==================================================================')
print('PART 1 - PRINCIPAL COMPONENT ANALYSIS')
print('==================================================================')

## data preparation, yields, zeros

# read data
data = pd.read_excel('FByields_2024_v2.xlsx',sheet_name='FBYields',header=None,skiprows=5,usecols='A,C:I')
data = array(data)

# find dates 
date1 = 20090331
I_1 = data[:,0]==date1
date2 = 20090430 
I_2 = data[:,0]==date2

# term structure on the two dates
Mat     = array([np.hstack([1/12,3/12,arange(1,6)])]).T
MatInt  = array([arange(.5,5+.5,.5)]).T
yields1 = (data[I_1,1:]).T/100 # put in decimals
yields2 = (data[I_2,1:]).T/100 # put in decimals

# compute relevant yields at semi-annual frequencies
f = PchipInterpolator(Mat.squeeze(),yields1.squeeze())
Y_Int_1 = f(MatInt)
f = PchipInterpolator(Mat.squeeze(),yields2.squeeze())
Y_Int_2 = f(MatInt)

# compute zeros
Z1=??
Z2=??

# plot results
plt.figure(0)
plt.plot(MatInt,Y_Int_1,MatInt,Y_Int_2,linewidth=2)
plt.title('Change in the term structure between March 31st and April 30th')
plt.xlabel('Maturity')
plt.ylabel('Yield')
plt.legend(['March 31, 2009','April 30,2009'])

## Compute value of LIF on both cases

freq=2    
coup_fix=10
TT=5

# compute C(T), the cash flows at various maturities, of the "fixed" rate
# bond underlying the LIF
CF_fixed=(coup_fix/freq)*ones((TT*freq))
CF_fixed[-1]=CF_fixed[-1]+100

P_Fixed1=(Z1[0:TT*freq]).T@CF_fixed.T
P_Fixed2=(Z2[0:TT*freq]).T@CF_fixed.T

P_Float=100
P_zero1=100*Z1[TT*freq-1]
P_zero2=100*Z2[TT*freq-1]
# as from HW2, the value of the LIF is given by?
P_LIF1=??
P_LIF2=??

print(' ')
print('P_LIF March, P_LIF April')
print([P_LIF1, P_LIF2,])

# compute average change in the yield curve between date 1 and date 2
dr=mean(Y_Int_2-Y_Int_1)
D_LIF_HW2=11.29 # the duration of LIF from HW2
C_LIF_HW2=58.45 # the convexity of LIF from HW2

dPP_D = ?? # the change in price due to duration (add formula)
dPP_D_C = ?? # the change in price due to convexity (add formula)

print(' ')
print('Dur-based  Dur/Conv-based   Actual ')
print('Return (%)   Return (%)   Return (%)')
print([dPP_D*100,dPP_D_C*100,  (P_LIF2-P_LIF1)/P_LIF1*100])

## Calculating Principal Components

loc=data[:,0]<=date1
YY=data[loc,1:]
T=YY.shape[0]

# We need to build a dataset. Moreover, we need the "betas" from PCA at
# given horizons. Thus, for each month, we build the dataset and interpolate the yields
# on the relevant  horizons, to compute proper betas for the duration
yields=zeros((T,MatInt.shape[0]))
for ii in range(T):
    f=PchipInterpolator(Mat.squeeze(),YY[ii,:])
    yields[ii,:]=f(MatInt).T
    
# compute changes in yields
dy=diff(yields,axis=0)
# compute the covariance matrix of the changes in yields
SIGMA=cov(dy,rowvar=False,ddof=1)
# compute the eigenvalues and eigenvectors of SIGMA
E,V=eigh(SIGMA)
# compute the vectors of eigenvalues from the diagonal matrix E
E=diag(E)
# compute number of eigenvalues. "Eig" command orders eigenvalues and
# eigenvectors from the smallest to the largest. The methodology requires
# we start from the largest eigenvalue.
mm=E.shape[0]

# compute the betas "explicitly"
# component 1

z1=dy@V[:,-1]                 # transform data using eigenvector corresponding to largest eigenvalue (mm)
Level=array([cumsum(z1)]).T   # Level factor
z11=np.vstack((ones(T-1),z1)).T # make a vector for the regressions (include vector of ones)
b1=inv(z11.T@z11)@z11.T@dy    # Regression beta = first component (note: first component is just the second row. The first is just the intercept)
e1=dy-z11@b1                  # Residuals to be used in the next step

RSS1=diag(e1.T@e1)                                      # Residual sum of squares
sisq1=(RSS1/(T-3)).reshape((dy.shape[1],1))             # variance of residuals
t1=b1/(sqrt(diag(inv(z11.T@z11)).reshape(2,1)@sisq1.T)) # t-statitstics

# component 2

z2=e1@V[:,-2]                      # transform the residuals using the eigenvector correposponding to the second largest eigenvalue (mm-1)
Slope=array([cumsum(z2)]).T        # Slope factor
z22=np.vstack((ones(T-1),z1,z2)).T # prepare vector for regression
b2=inv(z22.T@z22)@z22.T@dy         # second component = third row of regression beta 
e2=dy-z22@b2                       # residuals from regression to compute next component

RSS1=diag(e1.T@e1)                                      # to compute t-stats
sisq2=(RSS1/(T-3)).reshape((dy.shape[1],1))             
t2=b2/(sqrt(diag(inv(z22.T@z22)).reshape(3,1)@sisq2.T)) 

# assign the two principal components a new name for simplicity
betas=array([b2[1,:],b2[2,:]])
print('betas: 6 months - 5 years')
print(betas.T)

plt.figure(1)
plt.plot(MatInt,betas[0,:],MatInt,betas[1,:],linewidth=2)
plt.title('Factors')
plt.xlabel('Maturity')
plt.ylabel('Factor Beta')
plt.legend(['Level','Slope'])

# make a vector of dates for the plot
BegDate=1952.5        # beginning of data
EndDate=2009+2/12     # end of data
NDates=Level.shape[0] # number of data points
DatePlot=BegDate+(EndDate-BegDate)*array([arange(0,NDates)]).T/NDates # Vector of dates for the plot

plt.figure(2)
plt.plot(DatePlot,Level,linewidth=2)
plt.title('Level Factor')

plt.figure(3)
plt.plot(DatePlot,Slope,linewidth=2)
plt.title('Slope Factor')

# change in levels and slope
Diff_Level=dy@betas[0,:].T
Diff_Slope=dy@betas[1,:].T

## Computing factor durations of LIF
   
maturity=MatInt # redefine some variables, to use same codes as HW2
stripweights=(Z1[0:freq*TT]/P_Fixed1)*(coup_fix/2) # we use the "1" price above, as it refers to March
stripweights[-1]=stripweights[-1]+(Z1[freq*TT-1]*100)/P_Fixed1 # principal weight

# Duration against the level factor
D_Fixed_L=??
D_Float_L=??
D_Zero_L=?? 
#take weighted average of level-factor durations 
D_LIF_L=??

print(' ')
print('Level Duration: Fixed, Floating, Zero-Coupon, LIF')
print([D_Fixed_L, D_Float_L, D_Zero_L, D_LIF_L])

# Duration against the slope factor
D_Fixed_S=??
D_Float_S=??
D_Zero_S=?? 
#take weighted average of slope-factor durations
D_LIF_S=??

print(' ')
print('Slope Duration: Fixed, Floating, Zero-Coupon, LIF')
print([D_Fixed_S, D_Float_S, D_Zero_S, D_LIF_S])

Diff_Level_March_April=?? # What is the change in value of LIF due to Level factor? 
Diff_Slope_March_April=?? # What is the change in value of LIF due to Slope factor? 

dPP_Factors=Diff_Level_March_April+Diff_Slope_March_April # What is the change in value of LIF due to Level and Slope factors? 

print(' ')
print('Dur-based Dur/Conv-based Factor-based  Actual ')
print('Return (#)  Return (#)    Return (#)  Return (#)')
print([dPP_D*100,dPP_D_C*100, dPP_Factors*100, (P_LIF2-P_LIF1)/P_LIF1*100])

def regression_35130(Y,X):
    T=Y.shape[0]
    B=inv(X.T@X)@X.T@Y
    eps=Y-X@B # residuals
 
    RSS=eps.T@eps # residual sum of squares
    RSSToT=(Y-mean(Y)).T@(Y-mean(Y))
    sigsq=RSS/(T-3)
    tstat=B/(sqrt(diag(inv(X.T@X))*sigsq))
    R2=1-RSS/RSSToT   
    
    return B,tstat,R2

print('==================================================================')
print('PART 2 - PREDICTABILITY OF EXCESS RETURNS                         ')
print('==================================================================')

# Load all the data at annual frequency
DataB = pd.read_excel('FBYields_2024_v2.xlsx',sheet_name='Annual',header=None,skiprows=5)
DataB = array(DataB)
DataB = DataB[:,0:31]

DateB=np.round(DataB[:,0]/100)       # Date vector
yields=DataB[:,arange(1,6)]          # Yields: Available in Spreadsheet
fwd=DataB[:,arange(17,21)]           # Forwards: Available in Spreadsheet
RetB=DataB[:,arange(25,29)]          # Holding Period Return 
AveRetB=DataB[:,29]
CP=DataB[:,30]                       # Cochrane-Piazzesi Factor

#BegDateB=round(DateB(1,1));   % Initial date in sample
#EndDateB=round(DateB(end,1)); % End date in sample
#NDatesB=size(DateB,1);        % number of data points
#DatePlotB=BegDateB+(EndDateB-BegDateB)*[0:NDatesB]/NDatesB; % make a vector of dates for the plot
DatePlotB = DateB

# Predictive Regressions Regressions
TABLE_FB = zeros((4,6)) # initiate table to save Fama - Bliss Results
TABLE_CP = zeros((4,6)) # initiate table to save Cochrane - Piazzesi Results
for jpred in range(4):
    YY=RetB[1:,jpred]

    # Fama Bliss
    XX0=ones(YY.shape[0])
    XX1=?? # DEFINE PREDICTIVE VARIABLE (the "X")
    XX=np.vstack([XX0,XX1]).T
    
    BB,tBB,R2=regression_35130(YY,XX)
    TABLE_FB[jpred,:]=array([jpred+2,BB[0],BB[1],tBB[0],tBB[1],R2])
    
    # Cochrane Piazzesi
    XX1=?? # DEFINE PREDICTIVE VARIABLE (the "X")
    XX=np.vstack([XX0,XX1]).T
    BB,tBB,R2=regression_35130(YY,XX) 
    TABLE_CP[jpred,:]=array([jpred+2,BB[0],BB[1],tBB[0],tBB[1],R2])

print('Fams-Bliss')
print(TABLE_FB)
print(' ')
print('Cochrane-Piazzesi') 
print(TABLE_CP)

# FIGURES FAMA BLISS
ibond=3 # use bond 4 ==> 5 year to maturity

plt.figure(4)
YY=RetB[1:,ibond]
XX0=ones(YY.shape[0])
XX1=?? # see above: Need to select regressor!
XX=np.vstack([XX0,XX1]).T
BB,tBB,R2 = regression_35130(YY,XX)
plt.plot(XX[:,1],YY,'bo',XX[:,1],XX@BB,'k-')
plt.title('Realized 5-year Bond Return vs 5-year Forward Spread')
plt.xlabel('5-year forward spread')
plt.ylabel('realized bond return')
plt.legend(['data','regression fit'])

plt.figure(5)
plt.plot(DatePlotB[0:-1],YY,DatePlotB[0:-1],XX@BB,linewidth=2)
plt.title('5-year Bond Return and Predicted Return from 5-year Forward Spread')
plt.legend(['lagged realized bond return','predicted return'])

# FIGURES COCHRANE PIAZZESI

plt.figure(6)
YY=RetB[1:,ibond]
XX0=ones(YY.shape[0])
XX1=?? # see above: Need to select regressor!
XX=np.vstack([XX0,XX1]).T
BB,tBB,R2 = regression_35130(YY,XX)
plt.plot(XX[:,1],YY,'bo',XX[:,1],XX@BB,'k-')
plt.title('Realized 5-year Bond Return vs CP factor')
plt.xlabel('5-year forward spread')
plt.ylabel('realized bond return')
plt.legend(['data','regression fit'])

plt.figure(7)
plt.plot(DatePlotB[0:-1],YY,DatePlotB[0:-1],XX@BB,linewidth=2)
plt.title('5-year Bond Return and Predicted Return from CP factor')
plt.legend(['lagged realized bond return','predicted return'])

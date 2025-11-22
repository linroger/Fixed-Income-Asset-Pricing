function [B,tstat,R2]=regression_35130(Y,X)

        T=length(Y);
        B=inv(X'*X)*X'*Y;  
        eps=Y-X*B; % residuals

        RSS=(eps'*eps);  %Residual sum of squares
        RSSToT=(Y-mean(Y))'*(Y-mean(Y));  
        sigsq=RSS/(T-3);
        tstat=B./(sqrt(diag(inv(X'*X))*sigsq'));
        R2=1-RSS/RSSToT;




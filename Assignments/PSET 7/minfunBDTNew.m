function [F,vec]=minfunBDT(rmin,ImTree,yield,vol,h,N)

vec=rmin*exp(2*vol*sqrt(h)*[N-1:-1:0]);
ImTree(1:N,N)=vec';


RateMatrix=ImTree(1:N,1:N);
T=N;
   BB=zeros(T+1,T+1);
   BB(:,T+1)=ones(T+1,1);
   
   for t=T:-1:1;
   BB(1:t,t)=exp(-RateMatrix(1:t,t)*h).*(.5*BB(1:t,t+1)+.5*BB(2:t+1,t+1));   
   end
   PZero=BB(1,1);      

F=exp(-yield*length(vec)*h)-PZero;


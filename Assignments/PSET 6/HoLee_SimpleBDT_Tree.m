function [FF,ImTree,ZZTree]=HoLee_SimpleBDT_Tree(theta_i,ZZi,ImTree,i,sigma,hs,BDT_Flag)

if BDT_Flag==0
      % given theta, compute the next step of the tree
         ImTree(1,i)=ImTree(1,i-1)+theta_i*hs+sigma*sqrt(hs);
      for j=2:i
         ImTree(j,i)=ImTree(j-1,i-1)+theta_i*hs-sigma*sqrt(hs);
      end
else
       % given theta, compute the next step of the tree
       
         ImTree(1,i)=ImTree(1,i-1)*exp(theta_i*hs+sigma*sqrt(hs));
      for j=2:i
         ImTree(j,i)=ImTree(j-1,i-1)*exp(theta_i*hs-sigma*sqrt(hs));
      end
     
end
      
      % Use the tree to compute the value of a zero coupon bond
      
      % note: The zero coupon ZZ(i) in data expires at i+1 in Tree. For
      % instance, the first data point ZZ(1) is a 2-period bond, so expires
      % at i+1. 
      ZZTree(:,:)=zeros(i+1,i+1); % initialize the matrix for the zero coupon bond with maturity i+1. 
      ZZTree(1:i+1,i+1)=1; % final price is equal to 1
      pi=0.5;
     % backward algorithm
        for j=i:-1:1
        ZZTree(1:j,j) = exp(-ImTree(1:j,j)*hs).*(pi*ZZTree(1:j,j+1)+(1-pi)*ZZTree(2:j+1,j+1));
        end
        
        FF=(ZZTree(1,1)-ZZi)^2;
        
       
    
      
      

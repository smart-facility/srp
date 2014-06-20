function [outcome]=mmv(D,X,num)
    outcome = ompmmv(D,X,num);

function [AFA] = ompmmv(D,X,L)
residual = X;
c = zeros(size(D,2),1);
A = zeros(size(D,2),size(X,2));
indx1=zeros(L,1);
for j=1:1:L
    proj1=D'*residual;    
    for i=1:1:size(proj1,1);
        c(i) = norm(proj1(i,:));
    end
    [maxVal1,pos1]=max(abs(c));
    indx1(j)=pos1;
    a=pinv(D(:,indx1(1:j)))*X;    
    residual=X-D(:,indx1(1:j))*a;    
    if sum(residual.^2) < 1e-6
        break;
    end
end
for i=1:1:j
    A(indx1(i),:)=a(i,:);
end
AFA = A;
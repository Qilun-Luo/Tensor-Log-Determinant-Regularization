function [Z, objV] = prox_logdet(Y, tau)

[n1,n2,n3] = size(Y);
X = zeros(n1,n2,n3);
Y = fft(Y,[],3);
objV = 0;
trank = 0;
        
% first frontal slice
[U,S,V] = svd(Y(:,:,1),'econ');
S = diag(S);
S = truncate_logdet(S, tau);
X(:,:,1) = U*diag(S)*V';
objV = objV+sum(log(1+S.^2)/n3);
trank = max(trank,sum(S>0));

% i=2,...,halfn3
halfn3 = round(n3/2);
for i = 2 : halfn3
    [U,S,V] = svd(Y(:,:,i),'econ');
    S = diag(S);
    S = truncate_logdet(S, tau);
    X(:,:,i) = U*diag(S)*V';
    objV = objV+sum(log(1+S.^2)/n3);
    trank = max(trank,sum(S>0));
    X(:,:,n3+2-i) = conj(X(:,:,i));
end

% if n3 is even
if mod(n3,2) == 0
    i = halfn3+1;
    [U,S,V] = svd(Y(:,:,i),'econ');
    S = diag(S);
    S = truncate_logdet(S, tau);
    X(:,:,i) = U*diag(S)*V';
    objV = objV+sum(log(1+S.^2)/n3);
    trank = max(trank,sum(S>0));
end
Z = ifft(X,[],3);   
    
end


function [Z, objV] = prox_epsilon_logdet(Y, tau)

[n1,n2,n3] = size(Y);
X = zeros(n1,n2,n3);
Y = fft(Y,[],3);
objV = 0;
trank = 0;
        
% first frontal slice
[U,S,V] = svd(Y(:,:,1),'econ');
S = diag(S);
delta = (-S+sqrt(S.^2+4*exp(2)))/2;
rho = S-tau./(S+4*delta);
ind = (rho>0);
if length(ind)>=1
    S = rho(ind);
    X(:,:,1) = U(:,ind)*diag(S)*V(:,ind)';
    objV = objV+sum(log(4*delta(ind)+S.^2)/n3);
    trank = max(trank,length(ind));
end
% i=2,...,halfn3
halfn3 = round(n3/2);
for i = 2 : halfn3
    [U,S,V] = svd(Y(:,:,i),'econ');
    S = diag(S);
    delta = (-S+sqrt(S.^2+4*exp(2)))/2;
    rho = S-tau./(S+4*delta);
    ind = (rho>0);
    if length(ind)>=1
        S = rho(ind);
        X(:,:,i) = U(:,ind)*diag(S)*V(:,ind)';
        objV = objV+sum(log(4*delta(ind)+S.^2)/n3);
        trank = max(trank,length(ind));
    end
    X(:,:,n3+2-i) = conj(X(:,:,i));
end

% if n3 is even
if mod(n3,2) == 0
    i = halfn3+1;
    [U,S,V] = svd(Y(:,:,i),'econ');
    S = diag(S);
    delta = (-S+sqrt(S.^2+4*exp(2)))/2;
    rho = S-tau./(S+4*delta);
    ind = (rho>0);
    if length(ind)>=1
        S = rho(ind);
        X(:,:,i) = U(:,ind)*diag(S)*V(:,ind)';
        objV = objV+sum(log(4*delta(ind)+S.^2)/n3);
        trank = max(trank,length(ind));
    end
end
Z = ifft(X,[],3);

end
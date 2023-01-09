function x = truncate_logdet(sigma, tau)


p = (3*(2*tau+1) - sigma.^2)/3;
q = (-2*sigma.^3 + 18*(tau-1).*sigma)/27;

if tau < 4
    x = my_cubic_solver(p, q, sigma, 1);
else
    delta = sqrt(tau*(tau-4)^3);
    sigma1 = (tau^2+10*tau-2 - delta)/2;
    sigma2 = (tau^2+10*tau-2 + delta)/2;
    x = zeros(size(sigma));
    ind1 = (sigma.^2 > sigma1) & (sigma.^2 < sigma2);
    ind2 = ~ind1;
    if sum(ind1)>0
        p1 = p(ind1);
        q1 = q(ind1);
        omega1 = sigma(ind1);
        x1 = my_cubic_solver(p1, q1, omega1, 0);
        y1 = func_logdet(x1, omega1, tau);
        [~,ind] = min(y1,[],2);
        x(ind1) = x1(sub2ind(size(x1),1:size(x1,1),ind'))';
    end
    if sum(ind2)>0
        x(ind2) = my_cubic_solver(p(ind2), q(ind2), sigma(ind2), 1);
    end
end


end

function y = my_cubic_solver(p, q, omega, type)
    % solve for only one real root
    if type == 1 
        t = sqrt(q.^2/4+p.^3/27);
        y = nthroot(-q/2+t, 3)+nthroot(-q/2-t,3)+ omega/3;
    % there are 3 real roots
    else 
        t = acos((3*q)./(2*p).*sqrt(-3./p))/3;
        w = 2*sqrt(-p/3);
        y1 = w.*cos(t)+ omega/3;
        y2 = w.*cos(t-2*pi/3)+ omega/3;
        y3 = w.*cos(t-4*pi/3)+ omega/3;
        y = [y1 y2 y3];
    end
end

function y = func_logdet(omega, sigma, lambda)
    sigma = repmat(sigma, 1, 3);
    y = (sigma-omega).^2/2 + lambda*log(1+omega.^2);
end

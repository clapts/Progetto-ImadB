function [SSR] = calcSSR2var(phi, thetaLS, Y)

    eps = (Y - phi * thetaLS);
    SSR = eps'*eps;
        
end
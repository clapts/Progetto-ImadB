function [thetaLS, SSR, ste] = autolscov(gradoMaxPolinomio, X, Y)

    phi = zeros(length(X), gradoMaxPolinomio+1);

    for i = 0:gradoMaxPolinomio

        phi(:, i + 1) = X.^i;

    end

    [thetaLS, ste] = lscov(phi, Y);
    eps = (Y - phi * thetaLS);
    SSR = eps'*eps;
        
end
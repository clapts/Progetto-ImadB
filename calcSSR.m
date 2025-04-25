function [SSR] = calcSSR(X, Y, thetaLS)
    % se thetaLS ha 1 param -> grado 0 (costante)
    % se thetaLS ha 2 param -> grado 1 (retta)
    gradoMaxPolinomio = length(thetaLS)-1;

    phi = zeros(length(X), gradoMaxPolinomio+1);

    for i = 0:gradoMaxPolinomio

        phi(:, i + 1) = X.^i;

    end

    eps = (Y - phi * thetaLS);
    SSR = eps'*eps;
        
end
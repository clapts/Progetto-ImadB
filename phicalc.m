function [phi] = phicalc(gradoMaxPolinomio, X1, X2, Y)
    n = length(X1);
    phi = ones(n, 1); % Partiamo con il termine di grado 0
    
    for grado = 1:gradoMaxPolinomio
        for j = 0:grado
            k = grado - j;
            % aggiungiamo X1^j * X2^k come nuova colonna
            nuovaColonna = (X1.^j) .* (X2.^k);
            phi = [phi, nuovaColonna];
        end
    end
end

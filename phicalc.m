function [phi, phigrid, X, Y] = phicalc(gradoMaxPolinomio, X1, X2, samples)
    
    if(nargout>1)
        xgrid=linspace(min(X1),max(X1),samples);
        ygrid=linspace(min(X2),max(X2),samples);
        
        
        [X,Y]=meshgrid(xgrid, ygrid);
        phigrid = ones(length(X(:)), 1);

    end

    n = length(X1);
    phi = ones(n, 1); % Partiamo con il termine di grado 0
    
    
    for grado = 1:gradoMaxPolinomio
        for j = 0:grado
            k = grado - j;
            % aggiungiamo X1^j * X2^k come nuova colonna
            nuovaColonna = (X1.^j) .* (X2.^k);
            phi = [phi, nuovaColonna];
            
            if(nargout>1)
                colonnaGriglia=(X(:).^j) .* (Y(:).^k);
                phigrid=[phigrid, colonnaGriglia];
            end
        end
    end
end

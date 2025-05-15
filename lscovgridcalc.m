function [res_grid] = lscovgridcalc(thetaLS, xgrid, ygrid)
    % thetaLS parametri di un polinomio, da grado 0 a grado N (dimensione
    % del polinomio

    % xgrid è il vettore di valori dei quali calcolare le coordinate y
    % corrispondenti

    % la matrice phigrid ha dim(xgrid) righe e dim(thetaLS) colonne
    numeroParametri = length(thetaLS);

    phigrid = zeros(length(xgrid), numeroParametri);

    for i = 1:numeroParametri
        
        % si posiziona su una liga
        phigrid(:, i) = xgrid.^(i-1);

    end
    res_grid = phigrid*thetaLS;
        
end
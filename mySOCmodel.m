function soc_pred = mySOCmodel(voltage_vec, temp_vec)
    load model.mat
    
    n = length(voltage_vec);
    phi = ones(n, 1); % Partiamo con il termine di grado 0
    
    for grado = 1:gradoPolinomio
        for j = 0:grado
            k = grado - j;
            % aggiungiamo X1^j * X2^k come nuova colonna
            nuovaColonna = (voltage_vec.^j) .* (temp_vec.^k);
            phi = [phi, nuovaColonna];
            
        end
    end

    soc_pred = phi * thetaModel;

end
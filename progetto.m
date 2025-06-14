%{

La sfida ingegneristica consiste nel descrivere la relazione fra:
• SOC (in [0, 1])    (stato di carica)
• Tensione misurata
• Temperatura operativa

%}


clc,
clear,
close all;

load train_data.mat
load val_data.mat


Vtrain=data_train.Voltage;
Ttrain=data_train.Temperature;
SOCtrain=data_train.SOC;
Vval=data_val.Voltage;
Tval=data_val.Temperature;
SOCval=data_val.SOC;

% plotting dei dati di iddentificazione
figure();
scatter3(Vtrain, Ttrain, SOCtrain, 'DisplayName', 'training', 'MarkerEdgeColor', 'b');
hold on;
grid on;

% plotting dei dati di validazione
scatter3(Vval, Tval, SOCval, 'DisplayName', 'validazione', 'MarkerEdgeColor', 'r');

xlabel("Voltage");
ylabel("Temperature");
zlabel("SOC");
title("Dati nello spazio originale");
legend();


%% 2.2

% filtrare dati identificazione e validazione
filtro=(SOCtrain>(1e-4)) & SOCtrain<(1-(1e-4));

V=Vtrain(filtro);
T=Ttrain(filtro);
SOC=SOCtrain(filtro);

filtro=(SOCval>(1e-4)) & SOCval<(1-(1e-4));

Vval=Vval(filtro);
Tval=Tval(filtro);
SOCval=SOCval(filtro);


% trasformazione applico la logit sia ai dati di identificazioni che valid
SOClogit = logit(SOC);
SOCval = logit(SOCval);


figure();
scatter3(V, T, SOClogit);
grid on;
xlabel("Voltage");
ylabel("Temperature");
zlabel("SOC");
title("Dati nello spazio logit");

%% modelli
nV=length(SOClogit);

[theta0, SSR(1)]=autolscov(0, V, SOClogit);
[theta1, SSR(2)]=autolscov(1, V, SOClogit);
[theta2, SSR(3)]=autolscov(2, V, SOClogit);
[theta3, SSR(4)]=autolscov(3, V, SOClogit);
[theta4, SSR(5)]=autolscov(4, V, SOClogit);
[theta5, SSR(6)]=autolscov(5, V, SOClogit);
[theta6, SSR(7)]=autolscov(6, V, SOClogit);
[theta7, SSR(8)]=autolscov(7, V, SOClogit);
[theta8, SSR(9)]=autolscov(8, V, SOClogit);
[theta9, SSR(10)]=autolscov(9, V, SOClogit);
[theta10, SSR(11)]=autolscov(10, V, SOClogit);
[theta11, SSR(12)]=autolscov(11, V, SOClogit);

maxParametri = length(SSR);


% Criteri
AIC = zeros(1, maxParametri); % Preallocazione per i valori di AIC
k_values = 1:1:maxParametri;


for i = 1:length(k_values)
    % q = grado modello
    q = k_values(i);
    % nV è il numero di osservazioni
    AIC(i) = (2*q/nV) + log(SSR(i));
end


FPE = zeros(1, maxParametri);
for i = 1:length(k_values)
    % q = grado modello
    q = k_values(i);
    % nV è il numero di osservazioni
    FPE(i) = (nV+q)/(nV-q) * SSR(i);
end


MDL = zeros(1, maxParametri);
for i = 1:length(k_values)
    % q = grado modello
    q = k_values(i);
    % nV è il numero di osservazioni
    MDL(i) = (log(nV)*q)/(nV) + log(SSR(i));
end


% TEST F
TestF = zeros(1, maxParametri-1);
% ottengo da tabella
Fa = zeros(1, maxParametri-1);
for i = 2:length(k_values)
    q = k_values(i);
    TestF(i-1)=(nV-q)*(SSR(i-1)-SSR(i))/(SSR(i));

    % valori di f di fisher
    % (1, N-q) Gradi di libertà
    Fa(i-1) = finv(1 - 0.05, 1, nV-q);
end


passedTestF=TestF<Fa;
% il vettore è sempre vero perché N=4000 e quindi N-q è sempre 3.87 per
% valori di q piccoli, quindi il test f è sempre superato

%% crossvalidazione

% calcolo SSRv
% SSRv = epsv'*epsv
% epsv = Y - Yv
% Yv = PhiVal * thetaLS_ident

SSRv(1) = calcSSR(Vval, SOCval, theta0);
SSRv(2) = calcSSR(Vval, SOCval, theta1);
SSRv(3) = calcSSR(Vval, SOCval, theta2);
SSRv(4) = calcSSR(Vval, SOCval, theta3);
SSRv(5) = calcSSR(Vval, SOCval, theta4);
SSRv(6) = calcSSR(Vval, SOCval, theta5);
SSRv(7) = calcSSR(Vval, SOCval, theta6);
SSRv(8) = calcSSR(Vval, SOCval, theta7);
SSRv(9) = calcSSR(Vval, SOCval, theta8);
SSRv(10) = calcSSR(Vval, SOCval, theta9);
SSRv(11) = calcSSR(Vval, SOCval, theta10);
SSRv(12) = calcSSR(Vval, SOCval, theta11);

figure();
sgtitle("Modello con solo Tensione come regressore");
grid on;
legend(); % attivo la legenda

subplot(2,2,1);
hold on;
plot(0:(length(SSRv)-1), SSRv, 'DisplayName', 'validazione', 'Color', 'r');
plot(0:(length(SSR)-1), SSR, 'DisplayName', 'identificazione', 'Color', 'b');
hold off;
title("Andamento SSR");
ylabel("SSR");
xlabel("Grado Polinomio");
legend();


subplot(2,2,2);
plot(0:(length(FPE)-1), FPE, 'DisplayName', 'FPE');
title("FPE");
xlabel("Grado Polinomio");
legend();

subplot(2,2,3);
plot(0:(length(AIC)-1), AIC, 'DisplayName', 'AIC');
title("AIC");
xlabel("Grado Polinomio");
legend();

subplot(2,2,4);
plot(0:(length(MDL)-1), MDL, 'DisplayName', 'MDL');
title("MDL");
xlabel("Grado Polinomio");
legend();

% guardando AIC noto che il grande miglioramento lo ottengo dal 4° al 5°
% modello, dopo il miglioramento diminuisce tantissimo, e vedo il gomito
% della curva. tengo modello quinto grado

% guardando la crossvalidazione noto che 


%% visualizzazione modello scelto (5° grado, 6 parametri)
figure();
sgtitle("Modello di grado 5 (in 2D)");
hold on;

xgrid=linspace(min(V), max(V), 1000);

y5 = lscovgridcalc(theta5, xgrid);

subplot(1,2,1);
hold on;
scatter(V, expit(SOClogit), 'x');
plot(xgrid, expit(y5), 'LineWidth', 3, 'Color', 'r');
title("Modello polinomiale spazio originario");

subplot(1,2,2);
hold on;
scatter(V, SOClogit, 'x');
plot(xgrid, y5, 'LineWidth', 3, 'Color', 'r');
title("Modello polinomiale spazio logit");

% scelto modello 6
SSRgrado5regrV=SSR(6);
SSRgrado5regrVval=SSRv(6);


%% aggiungo temperatura
% avevo scelto quinto grado (modello 6) per la tensione

% Nuovi modelli
[theta02v, SSR2(1)] = autolscov2var(0, V, T, SOClogit);
[theta12v, SSR2(2)] = autolscov2var(1, V, T, SOClogit);
[theta22v, SSR2(3)] = autolscov2var(2, V, T, SOClogit);
[theta32v, SSR2(4)] = autolscov2var(3, V, T, SOClogit);
[theta42v, SSR2(5), phi42var, STE] = autolscov2var(4, V, T, SOClogit);
[theta52v, SSR2(6)] = autolscov2var(5, V, T, SOClogit);
[theta62v, SSR2(7)] = autolscov2var(6, V, T, SOClogit);
[theta72v, SSR2(8)] = autolscov2var(7, V, T, SOClogit);



maxParametri = length(SSR2);


% Criteri
AIC2 = zeros(1, maxParametri); % Preallocazione per i valori di AIC
k_values = 1:1:maxParametri;  % gradi del polinomio

for i = 1:length(k_values)
    % q = grado modello
    q = k_values(i);
    % nV è il numero di osservazioni
    AIC2(i) = (2*q/nV) + log(SSR2(i));
end


FPE2 = zeros(1, maxParametri);
for i = 1:length(k_values)
    % q = grado modello
    q = k_values(i);
    % nV è il numero di osservazioni
    FPE2(i) = (nV+q)/(nV-q) * SSR2(i);
end


MDL2 = zeros(1, maxParametri);
for i = 1:length(k_values)
    % q = grado modello
    q = k_values(i);
    % nV è il numero di osservazioni
    MDL2(i) = (log(nV)*q)/(nV) + log(SSR2(i));
end

% come prima il test f è sempre superato

%% crossvalidazione

% calcolo SSR2v
% SOCval è già in logit

phi02 = phicalc(0, Vval, Tval);
phi12 = phicalc(1, Vval, Tval);
phi22 = phicalc(2, Vval, Tval);
phi32 = phicalc(3, Vval, Tval);
phi42 = phicalc(4, Vval, Tval);
phi52 = phicalc(5, Vval, Tval);
phi62 = phicalc(6, Vval, Tval);
phi72 = phicalc(7, Vval, Tval);


SSR2v(1) = calcSSR2var(phi02, theta02v, SOCval);
SSR2v(2) = calcSSR2var(phi12, theta12v, SOCval);
SSR2v(3) = calcSSR2var(phi22, theta22v, SOCval);
SSR2v(4) = calcSSR2var(phi32, theta32v, SOCval);
SSR2v(5) = calcSSR2var(phi42, theta42v, SOCval);
SSR2v(6) = calcSSR2var(phi52, theta52v, SOCval);
SSR2v(7) = calcSSR2var(phi62, theta62v, SOCval);
SSR2v(8) = calcSSR2var(phi72, theta72v, SOCval);


% plotting di AIC, MDL, FPE, Crossval

figure();
sgtitle("Modello con Temperatura e Tensione come regressori");
legend(); % attivo la legenda

subplot(2,2,1);
hold on;
plot(0:(length(SSR2v)-1), SSR2v, 'DisplayName', 'validazione', 'Color', 'r');
plot(0:(length(SSR2)-1), SSR2, 'DisplayName', 'identificazione', 'Color', 'b');
hold off;
title("Andamento SSR 2 Variabili");
ylabel("SSR");
xlabel("Grado Polinomio");
legend();


subplot(2,2,2);
plot(0:(length(FPE2)-1), FPE2, 'DisplayName', 'FPE');
title("FPE");
xlabel("Grado Polinomio");
legend();

subplot(2,2,3);
plot(0:(length(AIC2)-1), AIC2, 'DisplayName', 'AIC');
title("AIC");
xlabel("Grado Polinomio");
legend();

subplot(2,2,4);
plot(0:(length(MDL2)-1), MDL2, 'DisplayName', 'MDL');
title("MDL");
xlabel("Grado Polinomio");
legend();


%% plot del modello con 2 variabili come regressori


% è superficie logit polinomiale
[phi5asd, Phi5_grid, X, Y] = phicalc(4, V, T, 100);

z_grid = Phi5_grid*theta42v;
SOCgrid=reshape(z_grid, size(X));

figure();
sgtitle("Modello polinomiale di grado 4 in 3D");
scatter3(V, T, SOClogit);
%è in logit
hold on;

mesh(X, Y, SOCgrid);

scatter3(Vval, Tval, SOCval);

% scelto modello grado 4
SSRgrado4regrVT=SSR2(5);
SSRgrado4regrVTval=SSR2v(5);

%% RMSE

for i=1:8
    RMSE(i)=sqrt(SSR2(i)/nV);
end

%% confronto funzione mySOCmodel

thetaModel = theta42v;
gradoPolinomio = 4;
save('model.mat', 'thetaModel', 'gradoPolinomio');

% test
predizione1 = mySOCmodel(V, T);
predizioneModello = phi42var*theta42v;

SSRgrado5regrV=SSR(6);

%% intervallo di confidenza, verifico STE polinomio grado 4

IC_inf = theta42v - 2*STE;
IC_sup = theta42v + 2*STE;

% Verifica se l'intervallo NON contiene lo zero
significativo = (IC_inf > 0 & IC_sup > 0) | (IC_inf < 0 & IC_sup < 0);

disp('Parametro   StdErr     IC_inf     IC_sup   Significativo');
for i = 1:length(theta42v)
    fprintf('%9.4f %9.4f %11.4f %11.4f    %d\n', theta42v(i), STE(i), IC_inf(i), IC_sup(i), significativo(i));
end

%% IDEE FALLIMENTARI

%% modello strisce centrali

load("modelCentralTemp.mat");

% ora che ho caricato il modello con le strisce centrali faccio il plotting
figure();
sgtitle("Modello fallimentare di grado 5 (in 2D)");
hold on;

xgrid=linspace(min(V), max(V), 1000);

y5 = lscovgridcalc(thetaModelCentral5, xgrid);

subplot(1,2,1);
hold on;
scatter(V, expit(SOClogit), 'x');
plot(xgrid, expit(y5), 'LineWidth', 3, 'Color', 'r');
title("Modello polinomiale spazio originario");

subplot(1,2,2);
hold on;
scatter(V, SOClogit, 'x');
plot(xgrid, y5, 'LineWidth', 3, 'Color', 'r');
title("Modello polinomiale spazio logit");


% con 2 regressori

% è superficie logit polinomiale
[phi5asd, Phi5_grid, X, Y] = phicalc(4, V, T, 100);

z_grid = Phi5_grid*thetaModelCentral4_multivar;
SOCgrid=reshape(z_grid, size(X));

figure();
sgtitle("Modello fallimentare di grado 4 in 3D");
scatter3(V, T, SOClogit);
%è in logit
hold on;

mesh(X, Y, SOCgrid);

scatter3(Vval, Tval, SOCval);

SSRfallimentare1val = calcSSR(Vval, SOCval, thetaModelCentral5);
SSRfallimentare2val = calcSSR2var(phi42, thetaModelCentral4_multivar, SOCval);
SSRfallimentare1id = calcSSR(V, SOClogit, thetaModelCentral5);
SSRfallimentare2id = calcSSR2var(phi42var, thetaModelCentral4_multivar, SOClogit);


%% stepwise regression

% tabella con i dati
tbl = table(V, T, SOClogit, 'VariableNames', {'V', 'T', 'SOClogit'});


mdl5 = stepwiselm(tbl, 'poly55', 'ResponseVar', 'SOClogit', 'Verbose', 1);

% poly44 è il polinomio di grado 4 in 2 variabili, Verbose per stampare i dettagli
mdl = stepwiselm(tbl, 'poly44', 'ResponseVar', 'SOClogit', 'Verbose', 1);


% griglia
V_grid = linspace(min(V), max(V), 50);
T_grid = linspace(min(T), max(T), 50);
[VV, TT] = meshgrid(V_grid, T_grid);

% rifaccio tabella
tbl_grid = table(VV(:), TT(:), 'VariableNames', {'V', 'T'});

% predizione valori
SOClogit_pred = predict(mdl, tbl_grid);
% reshape per usare mesh
SOClogit_pred_grid = reshape(SOClogit_pred, size(VV));

% plot modello scelto
figure;
mesh(VV, TT, SOClogit_pred_grid);
hold on;
scatter3(V, T, SOClogit, 'filled', 'MarkerEdgeColor', 'b');
scatter3(Vval, Tval, SOCval, 'filled', 'MarkerEdgeColor', 'r');
xlabel('V');
ylabel('T');
zlabel('SOClogit');
title('Superficie modello stepwise');
hold off;


% ORA CALCOLO SSR validazione
tbl_val = table(Vval, Tval, SOCval, 'VariableNames', {'V', 'T', 'SOClogit'});
% predizione valori
SOClogit_pred_val = predict(mdl, tbl_val);
% calcolo
residui_val = SOCval - SOClogit_pred_val;
SSR_valSTEP(1) = sum(residui_val.^2);

% stessa cosa per quinto grado
SOClogit_pred_val = predict(mdl5, tbl_val);
% calcolo
residui_val = SOCval - SOClogit_pred_val;
SSR_valSTEP(2) = sum(residui_val.^2);


% ORA CALCOLO SSR identificazione
tbl_id = table(V, T, SOClogit, 'VariableNames', {'V', 'T', 'SOClogit'});
% predizione valori
SOClogit_pred_id = predict(mdl, tbl_id);
% calcolo
residui_id = SOClogit - SOClogit_pred_id;
SSR_idSTEP(1) = sum(residui_id.^2);

% predizione valori
SOClogit_pred_id = predict(mdl5, tbl_id);
% calcolo
residui_id = SOClogit - SOClogit_pred_id;
SSR_idSTEP(2) = sum(residui_id.^2);


% confronto Stepwise con stima LS
figure();
title("Confronto Stepwise e stima LS");
hold on;
plot(4:5, SSR2v(5:6), 'DisplayName', 'LS validazione', 'Color', 'r');
plot(4:5, SSR2(5:6), 'DisplayName', 'LS identificazione', 'Color', 'b');

plot(4:5, SSR_idSTEP, '--ko', 'DisplayName', 'SW identificazione', 'LineWidth', 2);
plot(4:5, SSR_valSTEP, '--mo', 'DisplayName', 'SW validazione', 'LineWidth', 2);

xlim([3.95 5.05]);
ylim([50 2800]);

ylabel("SSR");
xlabel("Grado Polinomio");
legend();


SSR_idSTEPgr4=SSR_idSTEP(1);
SSR_valSTEPgr4=SSR_valSTEP(1);


%% confronto modelli fallimentari

Models = ["Modello 1 regr. (5°)";
            "Modello 1 regr. temp. cent. (5°)";
            "Modello 2 regr. (4°)";
            "Modello 2 regr. temp. cent. (4°)";
            "Modello 2 regr. SW (4°)"];

SSRid = [SSRgrado5regrV; SSRfallimentare1id; SSRgrado4regrVT;SSRfallimentare2id;SSR_idSTEPgr4];
SSRvAll = [SSRgrado5regrVval; SSRfallimentare1val; SSRgrado4regrVTval;SSRfallimentare2val;SSR_valSTEPgr4];

modelli=table(Models, SSRid, SSRvAll)



%% stepwise regression

% tabella con i dati
tbl = table(V, T, SOClogit, 'VariableNames', {'V', 'T', 'SOClogit'});

mdl54 = stepwiselm(tbl, 'poly54', 'ResponseVar', 'SOClogit', 'Verbose', 1);


% griglia
V_grid = linspace(min(V), max(V), 50);
T_grid = linspace(min(T), max(T), 50);
[VV, TT] = meshgrid(V_grid, T_grid);

% rifaccio tabella
tbl_grid = table(VV(:), TT(:), 'VariableNames', {'V', 'T'});

% predizione valori
SOClogit_pred = predict(mdl54, tbl_grid);
% reshape per usare mesh
SOClogit_pred_grid = reshape(SOClogit_pred, size(VV));

% plot modello scelto
figure;
mesh(VV, TT, SOClogit_pred_grid);
hold on;
scatter3(V, T, SOClogit, 'filled', 'MarkerEdgeColor', 'b');
scatter3(Vval, Tval, SOCval, 'filled', 'MarkerEdgeColor', 'r');
xlabel('V');
ylabel('T');
zlabel('SOClogit');
title('Superficie modello stepwise 54');
hold off;


% calcolo ssrv
tbl_val = table(Vval, Tval, SOCval, 'VariableNames', {'V', 'T', 'SOClogit'});
SOClogit_pred_val = predict(mdl54, tbl_val);
residui_val = SOCval - SOClogit_pred_val;
SSR_valSTEP54 = sum(residui_val.^2);


% ORA CALCOLO SSR identificazione
tbl_id = table(V, T, SOClogit, 'VariableNames', {'V', 'T', 'SOClogit'});
% predizione valori
SOClogit_pred_id = predict(mdl54, tbl_id);
% calcolo
residui_id = SOClogit - SOClogit_pred_id;
SSR_idSTEP54 = sum(residui_id.^2);

RMSEstep54 = sqrt(SSR_idSTEP54/nV)

%% LS 54

Phi54 = [V.^0 , V , T ,  ...
    V.^2 , T.^2 , V.*T , ...
    V.^3 , T.^3 , V.^2.*T , V.*T.^2 ,  ...
    V.^4 , T.^4 , V.^3.*T , V.^2.*T.^2 , V.*T.^3 ...
    V.^5 , V.^4.*T , V.^3.*T.^2 , V.^2.*T.^3 , V.*T.^4];

theta54=lscov(Phi54, SOClogit);

V_grid = linspace(min(V), max(V), 50);
T_grid = linspace(min(T), max(T), 50);
[VV, TT] = meshgrid(V_grid, T_grid);

phi54grid = [ ...
    ones(numel(VV),1), ...
    VV(:), TT(:), ...
    VV(:).^2, TT(:).^2, VV(:).*TT(:), ...
    VV(:).^3, TT(:).^3, VV(:).^2.*TT(:), VV(:).*TT(:).^2, ...
    VV(:).^4, TT(:).^4, VV(:).^3.*TT(:), VV(:).^2.*TT(:).^2, VV(:).*TT(:).^3, ...
    VV(:).^5, VV(:).^4.*TT(:), VV(:).^3.*TT(:).^2, VV(:).^2.*TT(:).^3, VV(:).*TT(:).^4 ...
];

zMesh = phi54grid * theta54;
zMesh = reshape(zMesh, size(VV));

figure();
mesh(VV, TT, zMesh);
hold on;
scatter3(V, T, SOClogit, 'filled', 'MarkerEdgeColor', 'b');
scatter3(Vval, Tval, SOCval, 'filled', 'MarkerEdgeColor', 'r');
xlabel('V');
ylabel('T');
zlabel('SOClogit');
title('Superficie modello LS 54');
hold off;


RMSEls54 = sqrt(SSR_idSTEP54/nV)
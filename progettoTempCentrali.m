%{

La sfida ingegneristica consiste nel descrivere la relazione fra:
• SOC (in [0, 1])    (stato di carica)
• Tensione misurata
• Temperatura operativa

%}

clc,
close all,
clear;

load train_data.mat
load val_data.mat


Vtrain=data_train.Voltage;
Ttrain=data_train.Temperature;
SOCtrain=data_train.SOC;
Vval=data_val.Voltage;
Tval=data_val.Temperature;
SOCval=data_val.SOC;



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

%% filtro i dati di identificazione e metto gli altri in validazione

% voglio solo i dati di identificazione compresi tra -15 e 15 gradi, la variabile è T
filtro= (T>-15) & (T<15);

V=V(filtro);
T=T(filtro);
SOC=SOC(filtro);

% voglio mettere insieme i dati di identificazione e validazione
V = [V; Vval];
T = [T; Tval];
SOC = [SOC; SOCval];




%ora creo di nuovi dati di validazione, sovrascrivendo Vval, Tval e SOCval facendo un
% campionamento uniforme di T, V e SOC sui dati di identificazione prendendo solo dati
% appartenenti ai vettori usando la funzione datasample()

% Campionamento uniforme per creare nuovi dati di validazione
numSamples = 1000; % Numero di campioni da estrarre

% Estrai indici in modo coerente per tutti i vettori
idx = datasample(1:length(V), numSamples, 'Replace', false);

% Usa gli indici per estrarre i campioni
Vval = V(idx);
Tval = T(idx);
SOCval = SOC(idx);

% Rimuovi i campioni estratti dai vettori originali
V(idx) = [];
T(idx) = [];
SOC(idx) = [];

figure();
scatter3(Vval, Tval, SOCval);
hold on;
grid on;
scatter3(V, T, SOC);
xlabel("Voltage");
ylabel("Temperature");
zlabel("SOC");
title("dati modello");
legend();

%% 2.2


% trasformazione applico la logit sia ai dati di identificazioni che valid
SOClogit = logit(SOC);
SOCval = logit(SOCval);


figure();
scatter3(V, T, SOClogit);
grid on;
xlabel("Voltage");
ylabel("Temperature");
zlabel("SOC");

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
sgtitle("Modello temp. centrali con solo Tensione come regressore");
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
sgtitle("Modello temp. centrali con Temperatura e Tensione come regressori");
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

%% RMSE

for i=1:8
    RMSE(i)=SSR2(i)/i;
end

%% intervallo di confidenza, verifico STE polinomio grado 4

IC2_inf = theta42v - 2*STE;
IC2_sup = theta42v + 2*STE;

% Verifica se l'intervallo NON contiene lo zero
significativo = (IC2_inf > 0 & IC2_sup > 0) | (IC2_inf < 0 & IC2_sup < 0);

disp('Parametro   StdErr     IC2_inf     IC2_sup   Significativo');
for i = 1:length(theta42v)
    fprintf('%9.4f %9.4f %11.4f %11.4f    %d\n', theta42v(i), STE(i), IC2_inf(i), IC2_sup(i), significativo(i));
end

%% salvo modello temperature centrali

thetaModelCentral5 = theta5;
gradoModelCentral5 = 5;
thetaModelCentral4_multivar = theta42v;
gradoModelCentral4_multivar = 4;
save('modelCentralTemp.mat', ...
    'thetaModelCentral5', ...
    'thetaModelCentral4_multivar', ...
    'gradoModelCentral5', ...
    'gradoModelCentral4_multivar');


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

figure();
Vtrain=data_train.Voltage;
Ttrain=data_train.Temperature;
SOCtrain=data_train.SOC;
Vval=data_val.Voltage;
Tval=data_val.Temperature;
SOCval=data_val.SOC;

hold on;
grid on;

scatter3(Vtrain, Ttrain, SOCtrain);

scatter3(Vval, Tval, SOCval);
xlabel("Voltage");
ylabel("Temperature");
zlabel("SOC");


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
SOClogit = log(SOC./(1-SOC));
SOCval = log(SOCval./(1-SOCval));


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
    MDL(i) = (log(nV)*q)/(nV) * log(SSR(i));
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
hold on;

xgrid=linspace(min(V), max(V), 1000);

y5 = lscovgridcalc(theta5, xgrid);

subplot(1,2,1);
hold on;
scatter(V, expit(SOClogit), 'x');
plot(xgrid, expit(y5), 'LineWidth', 2);
title("Modello polinomiale spazio originario");

subplot(1,2,2);
hold on;
scatter(V, SOClogit, 'x');
plot(xgrid, y5, 'LineWidth', 2);
title("Modello polinomiale spazio logit");


%% aggiungo temperatura
% avevo scelto quinto grado (modello 6) per la tensione

% Nuovi modelli
[theta02v, SSR2(1)] = autolscov2var(0, V, T, SOClogit);
[theta12v, SSR2(2)] = autolscov2var(1, V, T, SOClogit);
[theta22v, SSR2(3)] = autolscov2var(2, V, T, SOClogit);
[theta32v, SSR2(4)] = autolscov2var(3, V, T, SOClogit);
[theta42v, SSR2(5)] = autolscov2var(4, V, T, SOClogit);
[theta52v, SSR2(6)] = autolscov2var(5, V, T, SOClogit);
[theta62v, SSR2(7)] = autolscov2var(6, V, T, SOClogit);
[theta72v, SSR2(8)] = autolscov2var(7, V, T, SOClogit);
[theta82v, SSR2(9)] = autolscov2var(8, V, T, SOClogit);



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
    MDL2(i) = (log(nV)*q)/(nV) * log(SSR2(i));
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
phi82 = phicalc(8, Vval, Tval);


SSR2v(1) = calcSSR2var(phi02, theta02v, SOCval);
SSR2v(2) = calcSSR2var(phi12, theta12v, SOCval);
SSR2v(3) = calcSSR2var(phi22, theta22v, SOCval);
SSR2v(4) = calcSSR2var(phi32, theta32v, SOCval);
SSR2v(5) = calcSSR2var(phi42, theta42v, SOCval);
SSR2v(6) = calcSSR2var(phi52, theta52v, SOCval);
SSR2v(7) = calcSSR2var(phi62, theta62v, SOCval);
SSR2v(8) = calcSSR2var(phi72, theta72v, SOCval);
SSR2v(9) = calcSSR2var(phi82, theta82v, SOCval);


% plotting di AIC, MDL, FPE, Crossval

figure();
grid on;
legend(); % attivo la legenda

subplot(2,2,1);
hold on;
%plot(0:(length(SSR2v)-1), SSR2v, 'DisplayName', 'validazione', 'Color', 'r');
plot(0:(length(SSR2)-1), SSR2, 'DisplayName', 'identificazione', 'Color', 'b');
hold off;
title("Andamento SSR 2 Variabili");
ylabel("SSR 2 Variabili");
xlabel("Grado Polinomio");
legend();


subplot(2,2,2);
plot(0:(length(FPE2)-1), FPE2, 'DisplayName', 'FPE');
title("FPE 2 var");
xlabel("Grado Polinomio");
legend();

subplot(2,2,3);
plot(0:(length(AIC2)-1), AIC2, 'DisplayName', 'AIC');
title("AIC 2 var");
xlabel("Grado Polinomio");
legend();

subplot(2,2,4);
plot(0:(length(MDL2)-1), MDL2, 'DisplayName', 'MDL');
title("MDL 2 var");
xlabel("Grado Polinomio");
legend();


lscov



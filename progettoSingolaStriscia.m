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

figure(1)
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

% copio i dati filtrati per metterli in validazione

Vval = [V; Vval];
Tval = [T; Tval];
SOCval = [SOC; SOCval];

% ora devo filtrarli nel range esterno a quello di temperatura scelto
filtro=(Tval>-5)|(Tval<-15);

Vval=Vval(filtro);
Tval=Tval(filtro);
SOCval=SOCval(filtro);


% filtro range temperatura bello
filtro=(T<-5)&(T>-15);

V=V(filtro);
T=T(filtro);
SOC=SOC(filtro);



% trasformazione applico la logit sia ai dati di identificazioni che valid
SOClogit = log(SOC./(1-SOC));
SOCval = log(SOCval./(1-SOCval));


figure(2)
scatter3(V, T, SOClogit);
grid on;
xlabel("Voltage");
ylabel("Temperature");
zlabel("SOC");

%% modelli

maxParametri=7;
nV=length(SOClogit);

%primo grado
Phi0 = ones(nV, 1);
Phi1=[Phi0 V];
Phi2=[Phi1 V.^2];
Phi3=[Phi2 V.^3];
Phi4=[Phi3 V.^4];
Phi5=[Phi4 V.^5];
Phi6=[Phi5 V.^6];

[theta0, ste0] = lscov(Phi0, SOClogit);
[theta1, ste1] = lscov(Phi1, SOClogit);
[theta2, ste2] = lscov(Phi2, SOClogit);
[theta3, ste3] = lscov(Phi3, SOClogit);
[theta4, ste4] = lscov(Phi4, SOClogit);
[theta5, ste5] = lscov(Phi5, SOClogit);
[theta6, ste6] = lscov(Phi6, SOClogit);


% SSR = eps'*eps
% eps = Y - Yls
% in questo caso la Y è la SOC

eps0 = (SOClogit - Phi0*theta0);
SSR0 = eps0'*eps0;

eps1 = (SOClogit - Phi1*theta1);
SSR1 = eps1'*eps1;

eps2 = (SOClogit - Phi2*theta2);
SSR2 = eps2'*eps2;

eps3 = (SOClogit - Phi3*theta3);
SSR3 = eps3'*eps3;

eps4 = (SOClogit - Phi4*theta4);
SSR4 = eps4'*eps4;

eps5 = (SOClogit - Phi5*theta5);
SSR5 = eps5'*eps5;

eps6 = (SOClogit - Phi6*theta6);
SSR6 = eps6'*eps6;


% Criteri
AIC = zeros(1, maxParametri); % Preallocazione per i valori di AIC
k_values = 1:1:maxParametri;

SSR = [SSR0, SSR1, SSR2, SSR3, SSR4, SSR5, SSR6]; % Somme dei residui al quadrato per ogni modello

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
nV=length(SOCval);

Phi0v = ones(nV, 1);
Phi1v=[Phi0v Vval];
Phi2v=[Phi1v Vval.^2];
Phi3v=[Phi2v Vval.^3];
Phi4v=[Phi3v Vval.^4];
Phi5v=[Phi4v Vval.^5];
Phi6v=[Phi5v Vval.^6];

% SSR = eps'*eps
% eps = Y - Yv
% Y = rendimentoValidazione
% Yv = Phiv*theta = rendimento stimato
eps0v = (SOCval - Phi0v*theta0);
SSRv(1) = eps0v'*eps0v;

eps1v = (SOCval - Phi1v*theta1);
SSRv(2) = eps1v'*eps1v;

eps2v = (SOCval - Phi2v*theta2);
SSRv(3) = eps2v'*eps2v;

eps3v = (SOCval - Phi3v*theta3);
SSRv(4) = eps3v'*eps3v;

eps4v = (SOCval - Phi4v*theta4);
SSRv(5) = eps4v'*eps4v;

eps5v = (SOCval - Phi5v*theta5);
SSRv(6) = eps5v'*eps5v;

eps6v = (SOCval - Phi6v*theta6);
SSRv(7) = eps6v'*eps6v;


figure(44);
hold on;
grid on;
legend(); % attivo la legenda

plot(0:(length(SSRv)-1), SSRv, 'DisplayName', 'validazione', 'Color', 'r');
title("Andamento SSR");
ylabel("SSR");
xlabel("Ordine modello");

plot(0:(length(SSR)-1), SSR, 'DisplayName', 'identificazione', 'Color', 'b');


%% visualizzazione modelli 
figure(29);
hold on;

xgrid=linspace(min(V), max(V), 1000);

phi0grid=[ones(1000,1)];
phi1grid=[phi0grid, xgrid'];
phi2grid=[phi1grid, (xgrid.^2)'];
phi3grid=[phi2grid, (xgrid.^3)'];
phi4grid=[phi3grid, (xgrid.^4)'];
phi5grid=[phi4grid, (xgrid.^5)'];
phi6grid=[phi5grid, (xgrid.^6)'];

y0 = phi0grid*theta0;
y1 = phi1grid*theta1;
y2 = phi2grid*theta2;
y3 = phi3grid*theta3;
y4 = phi4grid*theta4;
y5 = phi5grid*theta5;
y6 = phi6grid*theta6;
%clearvars phi0grid phi2grid phi3grid phi4grid phi5grid phi6grid xgrid; 

scatter(V, expit(SOClogit), 'x');

scatter(Vval, expit(SOCval), 'x');

plot(xgrid, expit(y0), 'LineWidth', 2);
plot(xgrid, expit(y1), 'LineWidth', 2);
plot(xgrid, expit(y2), 'LineWidth', 2);
plot(xgrid, expit(y3), 'LineWidth', 2);
plot(xgrid, expit(y4), 'LineWidth', 2);
plot(xgrid, expit(y5), 'LineWidth', 2);
plot(xgrid, expit(y6), 'LineWidth', 2);








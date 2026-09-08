%% Coupled BSDE analysis for the stochastically forced Lorenz system
% Forward stochastic simulation, coupled BSDE solution, parameter scans,
% robustness tests, and predictability diagnostics.

clear; close all; clc;
rng(1);

%% ======================== Global style ======================================
FS = 20;          % adjustable font size
FS11_1 = 18;     
LW = 2.2;         % line width

set(groot,'defaultTextInterpreter','latex');
set(groot,'defaultAxesTickLabelInterpreter','latex');
set(groot,'defaultLegendInterpreter','latex');

%% ======================== Main experiment parameters =========================
T  = 5;
N  = 500;                 % stored BSDE / output time intervals
dt = T/N;
tgrid = linspace(0,T,N+1);
maxForwardDt = 5e-3;    % effective EM micro-step for the forward Lorenz SDE

M  = 50000;        % Monte Carlo paths

epsNoise = 1;

% Lorenz parameters
sigmaL = 10;
rhoL   = 28;
betaL  = 8/3;

% Terminals
alpha  = 0.35;
delta0 = 1e-4*[1;0;0];
etaReg = 1e-12;

% Stability caps (meaning-preserving monotone saturation)
g2cap = 25;    % cap for sensitivity proxy
ycap  = 30;    % cap for Y during scan Picard updates (prevents rare blow-ups)

% BSDE drivers
kappa1 = 0.7; eta1 = 0.2; gamma = 0.7; xi = 0.3; omega = 8.0;
kappa2 = 1;   eta2 = 0.4;
zmax   = 10;  
Kpic   = 6;              % six points are retained in the Picard diagnostic
theta  = 1.0;

% Regression
useRidge = true;
ridgeLam = 1e-6;
useQuadraticBasis = true;

% Plot controls
Mplot1_3D = 24;  
Mplot2    = 55;    

%% ======================== Forward simulation (X, Xtilde) ====================
x0 = [-8; 7; 27];

% Forward integration with Euler-Maruyama micro-substeps.
% Each BSDE interval is internally subdivided so that the micro-step does not
% exceed maxForwardDt.
% The Brownian micro-increments are summed to obtain the macro increment used
% by the BSDE recursion.
[X, Xtilde, dW, nForwardSub] = simulate_lorenz_pair_substepped( ...
    M, T, N, epsNoise, sigmaL, rhoL, betaL, x0, delta0, maxForwardDt);

fprintf('Forward EM micro-substeps per BSDE interval = %d; effective dt = %.6g\n', ...
    nForwardSub, dt/nForwardSub);

% Separation and sensitivity proxy
Delta = Xtilde - X;                                % Mx3x(N+1)
Sep   = squeeze(sqrt(sum(Delta.^2,2)));            % Mx(N+1)
sep0  = norm(delta0);

g2  = (1/T) * log( (Sep(:,end) + etaReg) / sep0 );
g2  = max(min(g2, g2cap), -g2cap);
g2b = tanh(g2);

XT  = X(:,:,end);
XTmean = mean(XT,1);                 % ensemble-mean terminal state E[X_T]
XTanom = XT - XTmean;                % terminal anomaly X_T - E[X_T]
E_T = sum(XTanom.^2,2);              % centered/anomaly energy
g1  = tanh(alpha * log(1 + E_T));

%% ======================== Solve coupled BSDE (main run) ======================
[Y1, Y2, Z1, Z2, diagMain] = solve_bsde_main( ...
    X, dW, tgrid, dt, g1, g2, ...
    kappa1, eta1, gamma, xi, omega, ...
    kappa2, eta2, zmax, Kpic, theta, ...
    useQuadraticBasis, useRidge, ridgeLam);

fprintf('E[Y_0^1] = %.6f,  SE = %.6f\n', mean(Y1(:,1)), std(Y1(:,1))/sqrt(M));
fprintf('E[Y_0^2] = %.6f,  SE = %.6f\n', mean(Y2(:,1)), std(Y2(:,1))/sqrt(M));

%% ======================== Figure 1: Forward diffusion overview ===============
idx1 = randperm(M, 500);
k0 = idx1(1);

figure('Name','Figure 1: Forward diffusion overview','Color','w');
tl = tiledlayout(2,2,'TileSpacing','compact','Padding','compact');

nexttile;
hold on;
trail = squeeze(X(k0,:,:)).';  % (N+1)x3
scatter3(trail(:,1), trail(:,2), trail(:,3), 8, 'filled', ...
    'MarkerFaceAlpha', 0.10, 'MarkerEdgeAlpha', 0.10);
nShow3D = min(Mplot1_3D, numel(idx1));
for ii = 1:nShow3D
    k = idx1(ii);
    plot3(squeeze(X(k,1,:)), squeeze(X(k,2,:)), squeeze(X(k,3,:)), 'LineWidth', 1.6);
end
hold off;
grid on;
xlabel('$X_1$'); ylabel('$X_2$'); zlabel('$X_3$');
% title('3D state-space view with a trajectory bundle');
set_axes_font(FS); view(42,18);

  xlim([-20 20]);
 xticks(-20:10:20);
  ylim([-30 30]);
 yticks(-30:30:30);
  zlim([0 60]);
 zticks(0:20:60);

nexttile;
hold on;
tidx = [round(0.2*N)+1, round(0.5*N)+1, N+1];
msel = randperm(M, 2600);

Xi = X(msel,1,tidx(1)); Xj = X(msel,2,tidx(1));
scatter(Xi(:), Xj(:), 14, 'o', 'filled', ...
    'MarkerFaceAlpha', 0.65, 'MarkerEdgeAlpha', 0.65, 'DisplayName','Early');

Xi = X(msel,1,tidx(2)); Xj = X(msel,2,tidx(2));
scatter(Xi(:), Xj(:), 14, 's', 'filled', ...
    'MarkerFaceAlpha', 0.65, 'MarkerEdgeAlpha', 0.65, 'DisplayName','Middle');

Xi = X(msel,1,tidx(3)); Xj = X(msel,2,tidx(3));
scatter(Xi(:), Xj(:), 14, '^', 'filled', ...
    'MarkerFaceAlpha', 0.65, 'MarkerEdgeAlpha', 0.65, 'DisplayName','Terminal');

hold off; grid on;
xlabel('$X_1$'); ylabel('$X_2$');
% title('Ensemble footprint on $(X_1,X_2)$');
lg = legend('Location','best'); lg.Box = 'on';
set_axes_font(FS);

  xlim([-20 20]);
 xticks(-20:10:20);
  ylim([-30 30]);
 yticks(-30:15:30);

nexttile;
medSep = median(Sep,1);
q25Sep = quantile(Sep,0.25,1);
q75Sep = quantile(Sep,0.75,1);
plot(tgrid, medSep, 'LineWidth', LW); hold on;
plot(tgrid, q25Sep, 'LineWidth', 1.6);
plot(tgrid, q75Sep, 'LineWidth', 1.6);
hold off; grid on;
xlabel('$t$'); ylabel('$\|\Delta(t)\|$');
% title('Separation statistics under common noise');
legend({'Median','25\%','75\%'},'Location','best');
set_axes_font(FS);

  xlim([0 5]);
 xticks(0:1:5);
  ylim([0 0.04]);
 yticks(0:0.01:0.04);

% Instantaneous logarithmic separation-growth rate on the right axis.

nexttile;
E_one   = squeeze(sum(X(k0,:,:).^2,2));
Sep_one = Sep(k0,:);

yyaxis left;
plot(tgrid, E_one, 'LineWidth', LW);
ylabel('$\|X(t)\|^2$');

  ylim([0 3000]);
 yticks(0:1000:3000);

yyaxis right;
% Instantaneous logarithmic separation-growth rate.
dlogSep = [0, diff(log(Sep_one + etaReg))] / dt;   % forward difference
plot(tgrid, dlogSep, '--', 'LineWidth', LW);
ylabel('$\frac{d}{dt}\log(\|\Delta(t)\|+\eta)$');

  ylim([-10 10]);
 yticks(-10:5:10);

  xlim([0 5]);
 xticks(0:1:5);

grid on; xlabel('$t$');
% title('Energy magnitude and instantaneous separation growth rate (one path)');
set_axes_font(FS);



drawnow;

%% ======================== Figure 2: Separation and sensitivity proxy =========
idx2 = randperm(M, 600);

figure('Name','Figure 2: Paired separation and sensitivity proxy','Color','w');
tl = tiledlayout(2,2,'TileSpacing','compact','Padding','compact');

nexttile;
hold on;
nShow = min(Mplot2, numel(idx2));
for ii = 1:nShow
    k = idx2(ii);
    plot(tgrid, Sep(k,:), 'LineWidth', 1.6);
end
hold off; grid on;
xlabel('$t$'); ylabel('$\|\Delta(t)\|$');
% title('Separation magnitude under common noise');
set_axes_font(FS);


  xlim([0 5]);
 xticks(0:1:5);
  ylim([0 0.3]);
 yticks(0:0.1:0.3);



nexttile;
hold on;
nShow = min(Mplot2, numel(idx2));
for ii = 1:nShow
    k = idx2(ii);
    plot(tgrid, log((Sep(k,:)+etaReg)/sep0), 'LineWidth', 1.6);
end
hold off; grid on;
xlabel('$t$'); ylabel('$\log((\|\Delta(t)\|+\eta)/\|\delta_0\|)$');
% title('Log separation ratio');
set_axes_font(FS);

  xlim([0 5]);
 xticks(0:1:5);
  ylim([-2 8]);
 yticks(-2:2:8);

nexttile;
Lambda_t = (1./max(tgrid,1e-6)) .* log((Sep + etaReg)/sep0);
Lambda_t(:,1) = 0;
medL = median(Lambda_t,1);
q10L = quantile(Lambda_t,0.10,1);
q90L = quantile(Lambda_t,0.90,1);
plot(tgrid, medL, 'LineWidth', LW); hold on;
plot(tgrid, q10L, 'LineWidth', 1.6);
plot(tgrid, q90L, 'LineWidth', 1.6);
hold off; grid on;
xlabel('$t$'); ylabel('$\Lambda(t)$');
% title('Finite-time sensitivity proxy over time');
legend({'Median','10\%','90\%'},'Location','best');
set_axes_font(FS);

  xlim([0 5]);
 xticks(0:1:5);
  ylim([-10 5]);
 yticks(-10:5:5);

nexttile;
SepT = Sep(:,end);
sel = randperm(M, min(3500,M));
scatter(E_T(sel), log((SepT(sel)+etaReg)/sep0), 10, 'filled');
grid on;
xlabel('$\|X_T-\overline{X}_T\|^2$'); ylabel('$\log((\|\Delta_T\|+\eta)/\|\delta_0\|)$');
% title('Terminal anomaly energy vs. terminal separation (log-ratio)');
set_axes_font(FS);



  xlim([0 1000]);
 xticks(0:250:1000);
  ylim([0 12]);
 yticks(0:3:12);

drawnow;

%% ======================== Figure 3: Terminal maps and relationships ==========
figure('Name','Figure 3: Terminal maps and relationships','Color','w');
tl = tiledlayout(2,3,'TileSpacing','compact','Padding','compact');

nexttile;
histogram(g1, 50, 'Normalization','pdf');
grid on;
xlabel('$g_1(X_T)$'); ylabel('Density');
% title('Terminal saturated anomaly-energy map $g_1$');
set_axes_font(FS);

  xlim([0.6 1]);
 xticks(0.6:0.1:1);
  ylim([0 10]);
 yticks(0:2:10);

nexttile;
histogram(g2, 50, 'Normalization','pdf');
grid on;
xlabel('$g_2(X_T,\tilde X_T)$'); ylabel('Density');
% title('Terminal sensitivity proxy $g_2$');
set_axes_font(FS);

  xlim([0 2.5]);
 xticks(0:0.5:2.5);
  ylim([0 2]);
 yticks(0:0.5:2);


nexttile;
sel = randperm(M, min(3200,M));
scatter(E_T(sel), g2(sel), 10, 'filled');
grid on;
xlabel('$\|X_T-\overline{X}_T\|^2$'); ylabel('$g_2$');
% title('Anomaly energy versus sensitivity proxy at $T$');
set_axes_font(FS);

  xlim([0 1000]);
 xticks(0:250:1000);
  ylim([0 2.5]);
 yticks(0:0.5:2.5);

nexttile;
sel = randperm(M, min(3500,M));
scatter(g1(sel), g2(sel), 10, 'filled');
xlabel('$g_1(X_T)$');
ylabel('$g_2(X_T,\tilde X_T)$');
grid on;

% title('Relation between terminal observables $g_1$ and $g_2$');
set_axes_font(FS);

  xlim([0.6 1]);
 xticks(0.6:0.1:1);
  ylim([0 2.5]);
 yticks(0:0.5:2.5);

nexttile;
sel = randperm(M, min(3500,M));
scatter3(XT(sel,1), XT(sel,2), XT(sel,3), 10, g1(sel), 'filled');
grid on;
xlabel('$X_{T,1}$'); ylabel('$X_{T,2}$'); zlabel('$X_{T,3}$');
% title('Terminal state cloud colored by $g_1$');
cb = colorbar;
  clim([0.6 1]);
 cb.Ticks = 0.6:0.1:1;
cb.Label.String = '$g_1$'; cb.Label.Interpreter = 'latex';
set_axes_font(FS); view(40,18);

  xlim([-20 20]);
 xticks(-20:20:20);
  ylim([-30 30]);
 yticks(-30:30:30);
  zlim([0 60]);
 zticks(0:20:60);

nexttile;
scatter3(XT(sel,1), XT(sel,2), XT(sel,3), 10, g2(sel), 'filled');
grid on;
xlabel('$X_{T,1}$'); ylabel('$X_{T,2}$'); zlabel('$X_{T,3}$');
% title('Terminal state cloud colored by $g_2$');
cb = colorbar;
  clim([0 2.5]);
 cb.Ticks = 0:0.5:2.5;
cb.Label.String = '$g_2$'; cb.Label.Interpreter = 'latex';
set_axes_font(FS); view(40,18);

  xlim([-20 20]);
 xticks(-20:20:20);
  ylim([-30 30]);
 yticks(-30:30:30);
  zlim([0 60]);
 zticks(0:20:60);

drawnow;

%% ======================== Figure 4: Backward dynamics and control ============
figure('Name','Figure 4: Backward dynamics and control behavior','Color','w');
tl = tiledlayout(2,2,'TileSpacing','compact','Padding','compact');

nexttile;
plot(tgrid, mean(Y1,1), 'LineWidth', LW); hold on;
plot(tgrid, mean(Y2,1), 'LineWidth', LW);
hold off; grid on;
xlabel('$t$'); ylabel('Mean');
% title('Means of $Y^1(t)$ and $Y^2(t)$');
legend({'$E[Y^1]$','$E[Y^2]$'},'Location','best');
set_axes_font(FS);

 xlim([0 5]);
 xticks(0:1:5);
 ylim([0 30]);
 yticks(0:10:30);

nexttile;
q10 = quantile(Y2,0.10,1);
q50 = quantile(Y2,0.50,1);
q90 = quantile(Y2,0.90,1);
plot(tgrid, q50, 'LineWidth', LW); hold on;
plot(tgrid, q10, 'LineWidth', 1.6);
plot(tgrid, q90, 'LineWidth', 1.6);
hold off; grid on;
xlabel('$t$'); ylabel('$Y^2(t)$');
% title('Quantile trajectories of $Y^2(t)$');
legend({'Median','10\%','90\%'},'Location','best');
set_axes_font(FS);

 xlim([0 5]);
 xticks(0:1:5);
 ylim([0 30]);
 yticks(0:10:30);

nexttile;
Z1norm = squeeze(sqrt(sum(Z1.^2,2)));
Z2norm = squeeze(sqrt(sum(Z2.^2,2)));
plot(tgrid(1:end-1), mean(Z1norm,1), 'LineWidth', LW); hold on;
plot(tgrid(1:end-1), mean(Z2norm,1), 'LineWidth', LW);
hold off; grid on;
xlabel('$t$'); ylabel('Mean $\|Z\|$');
% title('Mean control magnitudes');
legend({'$E[\|Z^1\|]$','$E[\|Z^2\|]$'},'Location','best');
set_axes_font(FS);

 xlim([0 5]);
 xticks(0:1:5);
 ylim([0 10]);
 yticks(0:2:10);

nexttile;
idxS = randperm(M, 10);
hold on;
for ii = 1:numel(idxS)
    plot(tgrid(1:end-1), Z1norm(idxS(ii),:), 'LineWidth', 1.6);
end
hold off; grid on;
xlabel('$t$'); ylabel('$\|Z^1(t)\|$');
% title('Several sample paths of $\|Z^1(t)\|$');
set_axes_font(FS);

 xlim([0 5]);
 xticks(0:1:5);
 ylim([0 10]);
 yticks(0:2:10);

drawnow;

%% ======================== Figure 5: Driver decomposition and balances ========
tN = tgrid(1:end-1);
X1n = squeeze(X(:,1,1:end-1));
Y1n = Y1(:,1:end-1);
Y2n = Y2(:,1:end-1);
Z1sq = squeeze(sum(Z1.^2,2));
q1 = min(Z1sq, zmax^2);

S1 = kappa1 * tanh(Y1n);
S2 = eta1   * q1;
S3 = gamma  * tanh(Y2n);
S4 = xi * (sin(omega * tN)) .* tanh(X1n);

m1 = mean(abs(S1),1);
m2 = mean(abs(S2),1);
m3 = mean(abs(S3),1);
m4 = mean(abs(S4),1);

A1 = mean(S1,2);
A2 = mean(S2,2);
A3 = mean(S3,2);

Dabs = mean(abs(S1+S2+S3+S4),2);

figure('Name','Figure 5: Driver decomposition and pathwise balances','Color','w');
tl = tiledlayout(2,2,'TileSpacing','compact','Padding','compact');

nexttile;
plot(tN, m1, 'LineWidth', LW); hold on;
plot(tN, m2, 'LineWidth', LW);
plot(tN, m3, 'LineWidth', LW);
plot(tN, m4, 'LineWidth', LW);
hold off; grid on;
xlabel('$t$'); ylabel('Mean absolute magnitude');
% title('Mean absolute profiles of the four driver components in $f_1$');
legend({'$\kappa_1\tanh(Y^1)$','$\eta_1\,q_1$','$\gamma\tanh(Y^2)$','$\xi\sin(\omega t)\tanh(X_1)$'}, ...
    'Location','best');
set_axes_font(FS);

 xlim([0 5]);
 xticks(0:1:5);
 ylim([-1 4]);
 yticks(-1:1:4);


nexttile;
c1 = cumtrapz(tN, mean(S1,1));
c2 = cumtrapz(tN, mean(S2,1));
c3 = cumtrapz(tN, mean(S3,1));
c4 = cumtrapz(tN, mean(S4,1));
plot(tN, c1, 'LineWidth', LW); hold on;
plot(tN, c2, 'LineWidth', LW);
plot(tN, c3, 'LineWidth', LW);
plot(tN, c4, 'LineWidth', LW);
hold off; grid on;
xlabel('$t$'); ylabel('Cumulative mean contribution');
% title('Cumulative mean contributions $\int_0^t E[\cdot]\,ds$');
legend({'$\kappa_1\tanh(Y^1)$','$\eta_1\,q_1$','$\gamma\tanh(Y^2)$','$\xi\sin(\omega t)\tanh(X_1)$'}, ...
    'Location','best');
set_axes_font(FS);

 xlim([0 5]);
 xticks(0:1:5);
 ylim([-1 6]);
 yticks(-1:1:6);


% Terminal sensitivity and time-averaged coupling contribution.
nexttile;
sel = randperm(M, min(4500,M));

% Relationship between terminal sensitivity proxy and time-averaged coupling contribution
scatter(g2(sel), A3(sel), 10, 'filled');
grid on;
xlabel('Terminal sensitivity proxy $g_2$');
ylabel('Time-avg coupling term');
% title('Terminal sensitivity versus time-avg coupling contribution');
set_axes_font(FS);

 xlim([0 2.5]);
 xticks(0:0.5:2.5);
 ylim([0.64 0.7]);
 yticks(0.64:0.02:0.7);

nexttile;
scatter3(A1(sel), A3(sel), Dabs(sel), 10, g2(sel), 'filled');
grid on;
xlabel('Time-avg $\kappa_1\tanh(Y^1)$');
ylabel('Time-avg $\gamma\tanh(Y^2)$');
zlabel('Time-avg $|f_1|$');
% title('Pathwise summaries colored by $g_2$');
cb = colorbar;
  clim([0 2.5]);
  cb.Ticks = 0:0.5:2.5;
cb.Label.String = '$g_2$'; cb.Label.Interpreter = 'latex';
set_axes_font(FS); view(38,18);

  xlim([0.67 0.69]);
 xticks(0.67:0.01:0.69);
  ylim([0.66 0.70]);
 yticks(0.66:0.02:0.70);
  zlim([0 6]);
 zticks(0:2:6);

drawnow;

%% ======================== Figure 6: Solver diagnostics =======================
figure('Name','Figure 6: Regression and Picard diagnostics','Color','w');
tl = tiledlayout(2,2,'TileSpacing','compact','Padding','compact');

nexttile;
plot(tgrid(1:end-1), diagMain.RY(:,1), 'LineWidth', LW); hold on;
plot(tgrid(1:end-1), diagMain.RY(:,2), 'LineWidth', LW);
hold off; grid on;
xlabel('$t$'); ylabel('$R_Y$');
% title('Normalized regression residual for $Y$ updates');
legend({'$R_Y^1$','$R_Y^2$'},'Location','best');
set_axes_font(FS);

 xlim([0 5]);
 xticks(0:1:5);
 ylim([0 1]);
 yticks(0:0.2:1);

nexttile;
RZ1 = mean(diagMain.RZ(:,1,:),3);
RZ2 = mean(diagMain.RZ(:,2,:),3);
plot(tgrid(1:end-1), RZ1, 'LineWidth', LW); hold on;
plot(tgrid(1:end-1), RZ2, 'LineWidth', LW);
hold off; grid on;
xlabel('$t$'); ylabel('$R_Z$');
% title('Average normalized residual for $Z$ updates');
legend({'$R_Z^1$','$R_Z^2$'},'Location','best');
set_axes_font(FS);

 xlim([0 5]);
 xticks(0:1:5);
 ylim([0.985 1]);
 yticks(0.985:0.005:1);

nexttile;
pic1 = squeeze(mean(diagMain.picardErr(:,:,1),1,'omitnan'));
pic2 = squeeze(mean(diagMain.picardErr(:,:,2),1,'omitnan'));
plot(1:Kpic, pic1, 'o-','LineWidth', LW); hold on;
plot(1:Kpic, pic2, 'o-','LineWidth', LW);
hold off; grid on;
xlabel('Picard iteration $k$'); ylabel('$E[|Y^{k+1}-Y^{k}|]$');
% title('Picard update magnitude');
legend({'Component 1','Component 2'},'Location','best');
set_axes_font(FS);

 xlim([1 6]);
 xticks(1:1:6);
 ylim([0 0.15]);
 yticks(0:0.05:0.15);

nexttile;
Z1abs = squeeze(abs(Z1));                 % M x 3 x N
Z2abs = squeeze(abs(Z2));                 % M x 3 x N

mZ1c = squeeze(mean(Z1abs,1));            % 3 x N
mZ2c = squeeze(mean(Z2abs,1));            % 3 x N

p1 = mZ1c ./ max(sum(mZ1c,1), 1e-12);     % 3 x N
p2 = mZ2c ./ max(sum(mZ2c,1), 1e-12);     % 3 x N

H1 = -sum(p1 .* log(max(p1,1e-12)), 1) / log(3);
H2 = -sum(p2 .* log(max(p2,1e-12)), 1) / log(3);

plot(tgrid(1:end-1), H1, 'LineWidth', LW); hold on;
plot(tgrid(1:end-1), H2, 'LineWidth', LW);
hold off; grid on;

xlabel('$t$'); ylabel('Normalized entropy');
% title('Directional anisotropy of control: entropy of component weights');
legend({'BSDE 1','BSDE 2'}, 'Location','best');
set_axes_font(FS);

 xlim([0 5]);
 xticks(0:1:5);
 ylim([0.8 1]);
 yticks(0.8:0.05:1);

drawnow;

%% ======================== Parameter scan surfaces for Fig 7 & 8 ==============
epsList   = linspace(0.2,1.2,10);
gammaList = linspace(0.0,1.6,10);
[EE,GG] = meshgrid(epsList, gammaList);  % columns -> epsilon, rows -> gamma

% meshgrid convention: epsilon varies across columns and gamma across rows.
assert(max(abs(EE(1,:) - epsList)) < 1e-12, 'epsilon grid orientation mismatch.');
assert(max(abs(GG(:,1) - gammaList(:))) < 1e-12, 'gamma grid orientation mismatch.');

scanOpts.Mscan = 1200;
scanOpts.Nscan = 40;
scanOpts.Kpic  = 2;
scanOpts.theta = 0.9;              % mild damping (scan only) to enhance stability
scanOpts.useQuadraticBasis = false;
scanOpts.useRidge = true;
scanOpts.ridgeLam = 1e-6;
scanOpts.maxForwardDt = 1.0e-2;    % smaller forward micro-step even on coarse scan grids
scanOpts.commonNoiseSeed = 314159; % common random numbers across (epsilon,gamma)

[Y0surf1, Y0surf2] = param_scan_surfaces( ...
    epsList, gammaList, T, ...
    sigmaL, rhoL, betaL, x0, delta0, etaReg, alpha, ...
    kappa1, eta1, xi, omega, kappa2, eta2, zmax, ...
    scanOpts, g2cap, ycap);

[Y0surf1, ok1] = force_finite_grid(Y0surf1, EE, GG);
[Y0surf2, ok2] = force_finite_grid(Y0surf2, EE, GG);
if ~(ok1 && ok2)
    warning('Scan grids had too many non-finite values; used conservative fallback filling.');
end

% The gamma=0 baseline is the first row of the meshgrid output.
Y0gamma0_1 = Y0surf1(1,:);
Y0gamma0_2 = Y0surf2(1,:);
DeltaMat1 = Y0surf1 - Y0gamma0_1;
DeltaMat2 = Y0surf2 - Y0gamma0_2;
fprintf('gamma=0 baseline check: max residual Y1 = %.3e, Y2 = %.3e\n', ...
    max(abs(DeltaMat1(1,:))), max(abs(DeltaMat2(1,:))));

epsFine = linspace(min(epsList), max(epsList), 110);
gamFine = linspace(min(gammaList), max(gammaList), 110);
[EEf,GGf] = meshgrid(epsFine, gamFine);

Y1fine = interp2(EE, GG, Y0surf1, EEf, GGf, 'makima');
Y2fine = interp2(EE, GG, Y0surf2, EEf, GGf, 'makima');

dE = epsFine(2)-epsFine(1);
dG = gamFine(2)-gamFine(1);

% First derivatives: epsilon is dimension 2; gamma is dimension 1.
dY1dE = first_derivative_dim2(Y1fine, dE);
dY2dE = first_derivative_dim2(Y2fine, dE);
dY1dG = first_derivative_dim1(Y1fine, dG);
dY2dG = first_derivative_dim1(Y2fine, dG);

% Epsilon curvature surfaces.
d2e1 = second_derivative_dim2(Y1fine, dE);
d2e2 = second_derivative_dim2(Y2fine, dE);

%% ======================== Figure 7: 3D parameter-scan responses ==============
figure('Name','Figure 7: 3D scan for E[Y0^1] (all 3D)','Color','w');
tl = tiledlayout(2,2,'TileSpacing','compact','Padding','compact');

nexttile;
surf(EEf, GGf, Y1fine, 'EdgeColor','none');
grid on; view(45,30);
xlabel('$\varepsilon$'); ylabel('$\gamma$'); zlabel('$E[Y_0^1]$');
% title('Smoothed surface of $E[Y_0^1]$');
set_axes_font(FS);
% Direct coupling sensitivity.

 xlim([0 1.2]);
 xticks(0:0.4:1.2);
 ylim([0 2]);
 yticks(0:0.5:2);
 zlim([5 20]);
 zticks(5:5:20);

nexttile;
surf(EEf, GGf, dY1dG, 'EdgeColor','none');
grid on; view(45,30);
xlabel('$\varepsilon$'); ylabel('$\gamma$'); zlabel('$\partial E[Y_0^1]/\partial \gamma$');
% title('Coupling sensitivity $\partial E[Y_0^1]/\partial \gamma$');
set_axes_font(FS);

 xlim([0 1.2]);
 xticks(0:0.4:1.2);
 ylim([0 2]);
 yticks(0:0.5:2);
 zlim([6 10]);
 zticks(6:2:10);

nexttile;
surf(EEf, GGf, dY1dE, 'EdgeColor','none');
grid on; view(45,30);
xlabel('$\varepsilon$'); ylabel('$\gamma$'); zlabel('$\partial E[Y_0^1]/\partial \varepsilon$');
% title('Noise sensitivity $\partial E[Y_0^1]/\partial \varepsilon$');
set_axes_font(FS);

 xlim([0 1.2]);
 xticks(0:0.4:1.2);
 ylim([0 2]);
 yticks(0:0.5:2);
 zlim([-10 5]);
 zticks(-10:5:10);


nexttile;
surf(EEf, GGf, d2e1, 'EdgeColor','none');
grid on; view(45,30);
xlabel('$\varepsilon$'); ylabel('$\gamma$'); zlabel('$\partial^2 E[Y_0^1]/\partial \varepsilon^2$');
% title('Curvature in $\varepsilon$ for $E[Y_0^1]$');

 xlim([0 1.2]);
 xticks(0:0.4:1.2);
 ylim([0 2]);
 yticks(0:0.5:2);
 zlim([-200 200]);
 zticks(-200:100:200);


set_axes_font(FS);
drawnow;

%% ======================== Figure 8: Y0^2 response surfaces ===================
figure('Name','Figure 8: 3D scan surfaces for Y0^2','Color','w');
tl = tiledlayout(2,2,'TileSpacing','compact','Padding','compact');

nexttile;
surf(EEf, GGf, Y2fine, 'EdgeColor','none');
grid on; view(45,30);
xlabel('$\varepsilon$'); ylabel('$\gamma$'); zlabel('$E[Y_0^2]$');
% title('Smoothed surface of $E[Y_0^2]$');
set_axes_font(FS);
% Coupling sensitivity of the second backward component.

 xlim([0 1.2]);
 xticks(0:0.4:1.2);
 ylim([0 2]);
 yticks(0:0.5:2);
 zlim([8 14]);
 zticks(8:2:14);



nexttile;
surf(EEf, GGf, dY2dG, 'EdgeColor','none');
grid on; view(45,30);
xlabel('$\varepsilon$'); ylabel('$\gamma$'); zlabel('$\partial E[Y_0^2]/\partial \gamma$');
% title('Coupling sensitivity $\partial E[Y_0^1]/\partial \gamma$');
set_axes_font(FS);

 xlim([0 1.2]);
 xticks(0:0.4:1.2);
 ylim([0 2]);
 yticks(0:0.5:2);
%zlim([2.8 3.6]);
%zticks(2.8:0.2:3.6);


nexttile;
surf(EEf, GGf, dY2dE, 'EdgeColor','none');
grid on; view(45,30);
xlabel('$\varepsilon$'); ylabel('$\gamma$'); zlabel('$\partial E[Y_0^2]/\partial \varepsilon$');
% title('Noise sensitivity $\partial E[Y_0^2]/\partial \varepsilon$');
set_axes_font(FS);

 xlim([0 1.2]);
 xticks(0:0.4:1.2);
 ylim([0 2]);
 yticks(0:0.5:2);
 zlim([-30 20]);
 zticks(-30:10:20);


nexttile;
surf(EEf, GGf, d2e2, 'EdgeColor','none');
grid on; view(45,30);
xlabel('$\varepsilon$'); ylabel('$\gamma$'); zlabel('$\partial^2 E[Y_0^2]/\partial \varepsilon^2$');
%title('Curvature in $\varepsilon$ for $E[Y_0^2]$');
set_axes_font(FS);

 xlim([0 1.2]);
 xticks(0:0.4:1.2);
 ylim([0 2]);
 yticks(0:0.5:2);
 zlim([-400 400]);
 zticks(-400:200:400);


drawnow;

%% ======================== Figure 9: Robustness-oriented parameter scans (Y0^1)
figure('Name','Figure 9: Robustness-oriented parameter scans','Color','w');
tl = tiledlayout(2,2,'TileSpacing','compact','Padding','compact');

Mrob = 900;
Trob = T;
robMaxForwardDt = 1.0e-2;

% Picard-depth scan with K=1,...,6.
Ngrid = [20 30 50 60 70];
Kgrid = 1:6;
[NN,KK] = meshgrid(Ngrid, Kgrid);
Y0_NK  = zeros(size(NN));
Y2_NK  = zeros(size(NN));
for i = 1:numel(NN)
    % Same seed for equal N values reduces Monte Carlo contamination when K changes.
    rng(8000 + NN(i));
    [Y0_NK(i), Y2_NK(i)] = quick_Y0_both(Mrob, Trob, NN(i), epsNoise, ...
        sigmaL, rhoL, betaL, x0, delta0, etaReg, alpha, g2cap, ycap, ...
        kappa1, eta1, gamma, xi, omega, kappa2, eta2, zmax, ...
        KK(i), 0.9, false, true, 1e-6, robMaxForwardDt);
end
[Y0_NK, ~] = force_finite_grid(Y0_NK, NN, KK);
[Y2_NK, ~] = force_finite_grid(Y2_NK, NN, KK);

% Refined reference for relative-error normalization.
MrobRef = 1800;
Nref    = 160;
Kref    = 6;
rng(18000);
[Yref1, Yref2] = quick_Y0_both(MrobRef, Trob, Nref, epsNoise, ...
    sigmaL, rhoL, betaL, x0, delta0, etaReg, alpha, g2cap, ycap, ...
    kappa1, eta1, gamma, xi, omega, kappa2, eta2, zmax, ...
    Kref, 0.9, false, true, 1e-6, 5.0e-3);

signalFloor = 1e-10;
RelErr_NK1 = 100 * abs(Y0_NK - Yref1) / max(abs(Yref1), signalFloor);

fprintf('Robustness reference signals: E[Y0^1]=%.6f, E[Y0^2]=%.6f\n', Yref1, Yref2);

Nfine = linspace(min(Ngrid), max(Ngrid), 80);
Kfine = linspace(min(Kgrid), max(Kgrid), 80);
[Nf,Kf] = meshgrid(Nfine, Kfine);
RelErr_NK1_f = interp2(NN, KK, RelErr_NK1, Nf, Kf, 'makima');
Y2_NK_f = interp2(NN, KK, Y2_NK, Nf, Kf, 'makima');

nexttile;
surf(Nf, Kf, RelErr_NK1_f, 'EdgeColor','none'); grid on; view(45,30);
xlabel('$N$'); ylabel('$K_{\mathrm{pic}}$'); zlabel('Relative error (\%)');
% title('Signal-normalized discretization/Picard error');
set_axes_font(FS);

 xlim([20 80]);
 xticks(20:20:80);
 ylim([1 6]);
 yticks(1:1:6);
 zlim([0 80]);
 zticks(0:20:80);

lamGrid = [1e-12 1e-10 1e-8 1e-6 1e-4];
thGrid  = [1.0 0.8 0.60 0.40];
[LL,TT] = meshgrid(lamGrid, thGrid);
Y0_LT = zeros(size(LL));
Y2_LT = zeros(size(LL));
for i = 1:numel(LL)
    rng(19000 + i);
    [Y0_LT(i), Y2_LT(i)] = quick_Y0_both(Mrob, Trob, 40, epsNoise, ...
        sigmaL, rhoL, betaL, x0, delta0, etaReg, alpha, g2cap, ycap, ...
        kappa1, eta1, gamma, xi, omega, kappa2, eta2, zmax, ...
        2, TT(i), false, true, LL(i), robMaxForwardDt);
end
[Y0_LT, ~] = force_finite_grid(Y0_LT, log10(LL), TT);
[Y2_LT, ~] = force_finite_grid(Y2_LT, log10(LL), TT);

lamFine = logspace(-12, -4, 90);
thFine  = linspace(min(thGrid), max(thGrid), 90);
[LF,TF] = meshgrid(lamFine, thFine);
LLlog = log10(max(LL,1e-12));
LFlog = log10(LF);
Y0_LT_f = interp2(LLlog, TT, Y0_LT, LFlog, TF, 'makima');
Y2_LT_f = interp2(LLlog, TT, Y2_LT, LFlog, TF, 'makima');

nexttile;
surf(LF, TF, Y0_LT_f, 'EdgeColor','none'); grid on; view(45,30);
set(gca,'XScale','log');
xlabel('$\lambda$'); ylabel('$\theta$'); zlabel('$E[Y_0^1]$');
% title('Robustness to ridge and Picard damping');
set_axes_font(FS);
zGrid   = [6 8 10 12 14];
eta1Grid = [0.1 0.2 0.3 0.4 0.60];
[ZZ,EE1] = meshgrid(zGrid, eta1Grid);
Y0_ZE = zeros(size(ZZ));
Y2_ZE = zeros(size(ZZ));
for i = 1:numel(ZZ)
    rng(20000 + i);
    [Y0_ZE(i), Y2_ZE(i)] = quick_Y0_both(Mrob, Trob, 40, epsNoise, ...
        sigmaL, rhoL, betaL, x0, delta0, etaReg, alpha, g2cap, ycap, ...
        kappa1, EE1(i), gamma, xi, omega, kappa2, eta2, ZZ(i), ...
        2, 0.9, false, true, 1e-6, robMaxForwardDt);
end
[Y0_ZE, ~] = force_finite_grid(Y0_ZE, ZZ, EE1);
[Y2_ZE, ~] = force_finite_grid(Y2_ZE, ZZ, EE1);

zFine = linspace(min(zGrid), max(zGrid), 90);
eFine = linspace(min(eta1Grid), max(eta1Grid), 90);
[Zf,Ef] = meshgrid(zFine, eFine);
Y0_ZE_f = interp2(ZZ, EE1, Y0_ZE, Zf, Ef, 'makima');
Y2_ZE_f = interp2(ZZ, EE1, Y2_ZE, Zf, Ef, 'makima');

 xlim([1e-12 1e-4]);
 xticks([1e-12,1e-8,1e-4]);
 ylim([0.4 1]);
 yticks(0.4:0.2:1);
 zlim([5 20]);
 zticks(5:5:20);


nexttile;
surf(Zf, Ef, Y0_ZE_f, 'EdgeColor','none'); grid on; view(45,30);
xlabel('$z_{\max}$'); ylabel('$\eta_1$'); zlabel('$E[Y_0^1]$');
% title('Robustness to truncation and quadratic weight');
set_axes_font(FS);
epsGrid = [0.2 0.5 0.8 1.1 1.4 1.6];
alpGrid = [0.10 0.20 0.30 0.5 0.80];
[EP,AL] = meshgrid(epsGrid, alpGrid);
Y0_EA = zeros(size(EP));
Y2_EA = zeros(size(EP));
for i = 1:numel(EP)
    rng(21000 + i);
    [Y0_EA(i), Y2_EA(i)] = quick_Y0_both(Mrob, Trob, 40, EP(i), ...
        sigmaL, rhoL, betaL, x0, delta0, etaReg, AL(i), g2cap, ycap, ...
        kappa1, eta1, gamma, xi, omega, kappa2, eta2, zmax, ...
        2, 0.9, false, true, 1e-6, robMaxForwardDt);
end
[Y0_EA, ~] = force_finite_grid(Y0_EA, EP, AL);
[Y2_EA, ~] = force_finite_grid(Y2_EA, EP, AL);

epFine = linspace(min(epsGrid), max(epsGrid), 90);
alFine = linspace(min(alpGrid), max(alpGrid), 90);
[EPf,ALf] = meshgrid(epFine, alFine);
Y0_EA_f = interp2(EP, AL, Y0_EA, EPf, ALf, 'makima');
Y2_EA_f = interp2(EP, AL, Y2_EA, EPf, ALf, 'makima');


 xlim([4 16]);
 xticks(4:4:16);
 ylim([0. 0.6]);
 yticks(0.:0.2:0.6);
 zlim([0 40]);
 zticks(0:10:40);

nexttile;
surf(EPf, ALf, Y0_EA_f, 'EdgeColor','none'); grid on; view(45,30);
xlabel('$\varepsilon$'); ylabel('$\alpha$'); zlabel('$E[Y_0^1]$');
% title('Robustness to noise intensity and terminal saturation');
set_axes_font(FS);


 xlim([0 2]);
 xticks(0:0.5:2);
 ylim([0. 0.8]);
 yticks(0.:0.2:0.8);
 zlim([8 14]);
 zticks(8:2:14);

drawnow;
%% ======================== Figure 10: Robustness scans (Y0^2) =================
figure('Name','Figure 10: Robustness-oriented parameter scans for Y0^2','Color','w');
tl = tiledlayout(2,2,'TileSpacing','compact','Padding','compact');

% Robustness surfaces for the second backward component.
nexttile;
surf(Nf, Kf, Y2_NK_f, 'EdgeColor','none'); grid on; view(45,30);
xlabel('$N$'); ylabel('$K_{\mathrm{pic}}$'); zlabel('$E[Y_0^2]$');
% title('Robustness to discretization and Picard depth');
set_axes_font(FS);

 xlim([20 80]);
 xticks(20:20:80);
 ylim([1 6]);
 yticks(1:1:6);
 zlim([0 30]);
 zticks(0:10:30);

nexttile;
surf(LF, TF, Y2_LT_f, 'EdgeColor','none'); grid on; view(45,30);
set(gca,'XScale','log');
xlabel('$\lambda$'); ylabel('$\theta$'); zlabel('$E[Y_0^2]$');
% title('Robustness to ridge and Picard damping');
set_axes_font(FS);

 xlim([1e-12 1e-4]);
 xticks([1e-12,1e-8,1e-4]);
 ylim([0.4 1]);
 yticks(0.4:0.2:1);
 zlim([0 30]);
 zticks(0:10:30);


nexttile;
surf(Zf, Ef, Y2_ZE_f, 'EdgeColor','none'); grid on; view(45,30);
xlabel('$z_{\max}$'); ylabel('$\eta_1$'); zlabel('$E[Y_0^2]$');
% title('Robustness to truncation and quadratic weight');
set_axes_font(FS);

 xlim([4 16]);
 xticks(4:4:16);
 ylim([0. 0.6]);
 yticks(0.:0.2:0.6);
 zlim([10 40]);
 zticks(10:10:40);


nexttile;
surf(EPf, ALf, Y2_EA_f, 'EdgeColor','none'); grid on; view(45,30);
xlabel('$\varepsilon$'); ylabel('$\alpha$'); zlabel('$E[Y_0^2]$');
% title('Robustness to noise intensity and terminal saturation');
set_axes_font(FS);

 xlim([0 2]);
 xticks(0:0.5:2);
 ylim([0. 0.8]);
 yticks(0.:0.2:0.8);
 zlim([10 25]);
 zticks(10:5:25);

drawnow;


%% ======================== Figure 11: Predictability and BSDE diagnostics =====
figure('Name','Figure 11: Quantitative relevance of terminals and BSDE closure','Color','w');
tl = tiledlayout(2,2,'TileSpacing','compact','Padding','compact');

% Diagnostics include terminal-threshold success probabilities, driver-omission
% closure errors, martingale-increment consistency, and linkage between the
% backward states and their terminal observables.

%% --- Figure 11, Tile 1: initial uncertainty versus terminal target ----------
nexttile;

% Probability of satisfying a terminal separation threshold over a range of
% initial perturbation magnitudes and terminal tolerance levels.
SepEnd = Sep(:,end);
SepEnd = SepEnd(isfinite(SepEnd) & SepEnd > 0);

deltaInit = logspace(-5, -3, 64);             % initial uncertainty: 1e-5 to 1e-3
targetSep = logspace(-4, -1, 64);             % terminal threshold: 1e-4 to 1e-1

Psuccess = zeros(numel(targetSep), numel(deltaInit));
for id = 1:numel(deltaInit)
    scaledSep = SepEnd * (deltaInit(id) / sep0);
    for js = 1:numel(targetSep)
        Psuccess(js,id) = mean(scaledSep <= targetSep(js));
    end
end

[DD,SS] = meshgrid(deltaInit, targetSep);
contourf(DD, SS, Psuccess, 0:0.1:1, 'LineColor','none');
hold on;
[C,hc] = contour(DD, SS, Psuccess, [0.50 0.80 0.95], 'k', 'LineWidth', 1.2);
clabel(C,hc,'Color','k','FontSize',FS11_1,'FontName','Times New Roman');
hold off;

ax11_1 = gca;
set(ax11_1, ...
    'XScale','log', ...
    'YScale','log', ...
    'FontName','Times New Roman', ...
    'FontSize',FS);
grid on;

    % 'XLim',[1e-5 1e-3], ...
    % 'YLim',[1e-4 1e-1], ...

cb = colorbar;
cb.Label.String = 'Success probability';
cb.Label.Interpreter = 'none';
cb.Label.FontName = 'Times New Roman';
cb.Label.FontSize = FS11_1;
cb.FontName = 'Times New Roman';
cb.FontSize = FS11_1;

 clim([0. 1]);
 cb.Ticks = 0.:0.2:1;
 xlim([1e-5 1e-3]);
 xticks([1e-5,1e-4,1e-3]);
 ylim([1e-4 1e-1]);
 yticks([1e-4,1e-3,1e-2,1e-1]);

% Axis text uses the same font size as the main manuscript figures.
xlabel('Initial uncertainty ||\delta_0||', ...
    'Interpreter','tex','FontName','Times New Roman','FontSize',FS);
ylabel('Terminal threshold S_T^*', ...
    'Interpreter','tex','FontName','Times New Roman','FontSize',FS);
% title('Probability of meeting a terminal uncertainty target');

%% --- Figure 11, Tile 2: time-resolved driver-necessity lines -----------------
nexttile;

% Change in one-step BSDE closure error after omitting each driver term while
% holding the computed Y and Z trajectories fixed.
tN11b = tgrid(1:end-1);
X1n11b = squeeze(X(:,1,1:end-1));
Y1n11b = Y1(:,1:end-1);
Y2n11b = Y2(:,1:end-1);

Z1sq11b = squeeze(sum(Z1.^2,2));
Z2sq11b = squeeze(sum(Z2.^2,2));
q1_11b = min(Z1sq11b, zmax^2);
q2_11b = min(Z2sq11b, zmax^2);

D11_k1 = kappa1 * tanh(Y1n11b);
D11_e1 = eta1   * q1_11b;
D11_g  = gamma  * tanh(Y2n11b);
D11_xi = xi * (sin(omega*tN11b)) .* tanh(X1n11b);

D11_k2 = kappa2 * tanh(Y2n11b);
D11_e2 = eta2   * q2_11b;

f1_11b = D11_k1 + D11_e1 + D11_g + D11_xi;
f2_11b = D11_k2 + D11_e2;

mart1_11b = squeeze(sum(Z1 .* dW,2));
mart2_11b = squeeze(sum(Z2 .* dW,2));

resFull1_11b = Y1(:,2:end) + f1_11b*dt - Y1(:,1:end-1) - mart1_11b;
resFull2_11b = Y2(:,2:end) + f2_11b*dt - Y2(:,1:end-1) - mart2_11b;

resNo_k1 = resFull1_11b - D11_k1*dt;
resNo_e1 = resFull1_11b - D11_e1*dt;
resNo_g  = resFull1_11b - D11_g *dt;
resNo_xi = resFull1_11b - D11_xi*dt;

resNo_k2 = resFull2_11b - D11_k2*dt;
resNo_e2 = resFull2_11b - D11_e2*dt;

rmseFull1_t = sqrt(mean(resFull1_11b.^2,1,'omitnan'));
rmseFull2_t = sqrt(mean(resFull2_11b.^2,1,'omitnan'));

deg_k1 = 100 * (sqrt(mean(resNo_k1.^2,1,'omitnan')) ./ max(rmseFull1_t,1e-12) - 1);
deg_e1 = 100 * (sqrt(mean(resNo_e1.^2,1,'omitnan')) ./ max(rmseFull1_t,1e-12) - 1);
deg_g  = 100 * (sqrt(mean(resNo_g.^2 ,1,'omitnan')) ./ max(rmseFull1_t,1e-12) - 1);
deg_xi = 100 * (sqrt(mean(resNo_xi.^2,1,'omitnan')) ./ max(rmseFull1_t,1e-12) - 1);

deg_k2 = 100 * (sqrt(mean(resNo_k2.^2,1,'omitnan')) ./ max(rmseFull2_t,1e-12) - 1);
deg_e2 = 100 * (sqrt(mean(resNo_e2.^2,1,'omitnan')) ./ max(rmseFull2_t,1e-12) - 1);

plot(tN11b, deg_k1, 'LineWidth', LW); hold on;
plot(tN11b, deg_e1, 'LineWidth', LW);
plot(tN11b, deg_g , 'LineWidth', LW);
plot(tN11b, deg_xi, 'LineWidth', LW);
plot(tN11b, deg_k2, '--', 'LineWidth', LW);
plot(tN11b, deg_e2, '--', 'LineWidth', LW);
hold off;
grid on;


 xlim([0 5]);
 xticks(0:1:5);
 ylim([-5 25]);
 yticks(-5:5:25);

xlabel('$t$');
ylabel('Closure-error increase (\%)');
legend({ ...
    '$\kappa_1\tanh(Y^1)$ omitted', ...
    '$\eta_1 q_1$ omitted', ...
    '$\gamma\tanh(Y^2)$ omitted', ...
    '$\xi\sin(\omega t)\tanh(X_1)$ omitted', ...
    '$\kappa_2\tanh(Y^2)$ omitted', ...
    '$\eta_2 q_2$ omitted'}, ...
    'Location','best');
% title('Time-resolved necessity of the deterministic driver terms');
set_axes_font(FS);

%% --- Figure 11, Tile 3: role of Z in stochastic innovation ------------------
nexttile;

tN11 = tgrid(1:end-1);
X1n11 = squeeze(X(:,1,1:end-1));
Y1n11 = Y1(:,1:end-1);
Y2n11 = Y2(:,1:end-1);

Z1sq11 = squeeze(sum(Z1.^2,2));
Z2sq11 = squeeze(sum(Z2.^2,2));
q1_11 = min(Z1sq11, zmax^2);
q2_11 = min(Z2sq11, zmax^2);

f2_11 = kappa2 * tanh(Y2n11) + eta2 * q2_11;
f1_11 = kappa1 * tanh(Y1n11) + eta1 * q1_11 + gamma * tanh(Y2n11) ...
       + xi * (sin(omega*tN11)) .* tanh(X1n11);

% From dY = -f dt + Z dW, the one-step stochastic innovation is
% Delta Y + f dt and should be represented by Z * Delta W.
innov1 = Y1(:,2:end) - Y1(:,1:end-1) + f1_11*dt;
innov2 = Y2(:,2:end) - Y2(:,1:end-1) + f2_11*dt;

mart1 = squeeze(sum(Z1 .* dW,2));
mart2 = squeeze(sum(Z2 .* dW,2));

innov1c = innov1 - mean(innov1,1);
innov2c = innov2 - mean(innov2,1);
mart1c  = mart1  - mean(mart1,1);
mart2c  = mart2  - mean(mart2,1);

num1 = sum(innov1c .* mart1c,1).^2;
num2 = sum(innov2c .* mart2c,1).^2;
den1 = sum(innov1c.^2,1) .* sum(mart1c.^2,1);
den2 = sum(innov2c.^2,1) .* sum(mart2c.^2,1);

R2mart1 = num1 ./ max(den1,1e-20);
R2mart2 = num2 ./ max(den2,1e-20);
R2mart1(~isfinite(R2mart1)) = NaN;
R2mart2(~isfinite(R2mart2)) = NaN;

plot(tN11, R2mart1, 'LineWidth', LW); hold on;
plot(tN11, R2mart2, 'LineWidth', LW);
hold off; grid on;
xlabel('$t$');
ylabel('$R^2(\Delta Y+f\Delta t,\; Z\cdot\Delta W)$');
legend({'Component 1','Component 2'},'Location','best');
% title('Brownian innovation captured by the martingale term');

 xlim([0 5]);
 xticks(0:1:5);
 ylim([0 0.8]);
 yticks(0:0.2:0.8);

set_axes_font(FS);

%% --- Figure 11, Tile 4: terminal-observable linkage --------------------------
nexttile;

% Statistical linkage between the backward variables and their terminal
% observables as the remaining forecast horizon decreases.
g1c = g1 - mean(g1);
g2c = g2 - mean(g2);

Y1c11 = Y1 - mean(Y1,1);
Y2c11 = Y2 - mean(Y2,1);

rhoY1g1 = sum(Y1c11 .* g1c,1) ./ ...
    sqrt(max(sum(Y1c11.^2,1) * sum(g1c.^2),1e-20));
rhoY2g2 = sum(Y2c11 .* g2c,1) ./ ...
    sqrt(max(sum(Y2c11.^2,1) * sum(g2c.^2),1e-20));

rhoY1g1(~isfinite(rhoY1g1)) = NaN;
rhoY2g2(~isfinite(rhoY2g2)) = NaN;

R2terminal1 = rhoY1g1.^2;
R2terminal2 = rhoY2g2.^2;

plot(tgrid, R2terminal1, 'LineWidth', LW); hold on;
plot(tgrid, R2terminal2, 'LineWidth', LW);
hold off; grid on;
xlabel('$t$');
ylabel('Terminal-observable linkage $R^2$');
legend({'$Y^1(t)$ vs. $g_1$','$Y^2(t)$ vs. $g_2$'},'Location','best');
% title('Backward variables as terminal-risk-to-go summaries');
set_axes_font(FS);

 xlim([0 5]);
 xticks(0:1:5);
 ylim([0 1]);
 yticks(0:0.2:1);

drawnow;


%% ======================== Local helper functions ============================

function set_axes_font(FS)
set(gca,'FontName','Times New Roman','FontSize',FS);
end

function [Vout, ok] = force_finite_grid(V, Xg, Yg)
Vout = V;
Vout(~isfinite(Vout)) = NaN;

mask = isfinite(Vout);
nfinite = nnz(mask);
ok = true;

if nfinite == 0
    ok = false;
    Vout(:) = 0;
    return;
end

if nfinite < 6
    ok = false;
    v0 = median(Vout(mask),'omitnan');
    Vout(~mask) = v0;
    return;
end

x = Xg(mask); y = Yg(mask); v = Vout(mask);
F = scatteredInterpolant(x(:), y(:), v(:), 'natural', 'nearest');
Vout(~mask) = F(Xg(~mask), Yg(~mask));

if any(~isfinite(Vout(:)))
    ok = false;
    v0 = median(v,'omitnan');
    Vout(~isfinite(Vout)) = v0;
end
end

function b = lorenz_drift(x, sigma, rho, beta)
x1 = x(:,1); x2 = x(:,2); x3 = x(:,3);
b = zeros(size(x));
b(:,1) = sigma*(x2 - x1);
b(:,2) = x1.*(rho - x3) - x2;
b(:,3) = x1.*x2 - beta*x3;
end

function [X, Xtilde, dWmacro, nSub] = simulate_lorenz_pair_substepped( ...
    M, T, N, epsNoise, sigmaL, rhoL, betaL, x0, delta0, maxForwardDt)
% Integrate each stored interval using smaller Euler-Maruyama microsteps.
% The summed Brownian increment is returned for the BSDE recursion.

dtMacro = T/N;
nSub = max(1, ceil(dtMacro / maxForwardDt));
dtSub = dtMacro / nSub;

X      = zeros(M,3,N+1);
Xtilde = zeros(M,3,N+1);
dWmacro = zeros(M,3,N);

X(:,:,1)      = repmat(x0.', M, 1);
Xtilde(:,:,1) = repmat((x0 + delta0).', M, 1);

for n = 1:N
    xn  = X(:,:,n);
    xtn = Xtilde(:,:,n);
    dWn = zeros(M,3);

    for isub = 1:nSub
        dWs = sqrt(dtSub) * randn(M,3);
        bn  = lorenz_drift(xn,  sigmaL, rhoL, betaL);
        btn = lorenz_drift(xtn, sigmaL, rhoL, betaL);

        xn  = xn  + bn  * dtSub + epsNoise * dWs;
        xtn = xtn + btn * dtSub + epsNoise * dWs;
        dWn = dWn + dWs;
    end

    X(:,:,n+1)      = xn;
    Xtilde(:,:,n+1) = xtn;
    dWmacro(:,:,n)  = dWn;
end
end

function Phi = basis_features(X, useQuadratic)
x1 = X(:,1); x2 = X(:,2); x3 = X(:,3);
if useQuadratic
    Phi = [ ...
        ones(size(x1)), ...
        x1, x2, x3, ...
        x1.^2, x2.^2, x3.^2, ...
        x1.*x2, x1.*x3, x2.*x3, ...
        tanh(x1), tanh(x2), tanh(x3) ...
    ];
else
    Phi = [ones(size(x1)), x1, x2, x3, tanh(x1), tanh(x2), tanh(x3)];
end
Phi(~isfinite(Phi)) = 0;
end

function reg = prepare_regression(Phi, useRidge, lam)
% SVD-based ridge regression avoids forming normal equations and improves
% conditioning for the polynomial regression basis.

A = Phi;
A(~isfinite(A)) = 0;

[U,S,V] = svd(A,'econ');
s = diag(S);

if useRidge
    % Exact ridge filter for min ||A*b-y||^2 + lam*||b||^2.
    denom = s.^2 + max(lam,0);
    tiny = eps(max(1,max(s.^2)));
    filt = zeros(size(s));
    keep = denom > tiny;
    filt(keep) = s(keep) ./ denom(keep);
else
    tol = max(size(A)) * eps(max(1,max(s)));
    filt = zeros(size(s));
    keep = s > tol;
    filt(keep) = 1 ./ s(keep);
end

reg.A = A;
reg.U = U;
reg.V = V;
reg.filt = filt;
end

function Yhat = stable_regress(reg, Y)
Y(~isfinite(Y)) = 0;
coef = reg.V * (reg.filt .* (reg.U.' * Y));
Yhat = reg.A * coef;
Yhat(~isfinite(Yhat)) = 0;
end

function Zp = project_Z(Z, zcap)
zn = sqrt(sum(Z.^2,2));
scale = ones(size(zn));
idx = zn > zcap;
scale(idx) = zcap ./ (zn(idx) + 1e-12);
Zp = Z .* scale;
Zp(~isfinite(Zp)) = 0;
end

function [Y1, Y2, Z1, Z2, diagOut] = solve_bsde_main( ...
    X, dW, tgrid, dt, g1, g2, ...
    kappa1, eta1, gamma, xi, omega, ...
    kappa2, eta2, zmax, Kpic, theta, ...
    useQuadraticBasis, useRidge, ridgeLam)

[M,~,N1] = size(X);
N = N1-1;

Y1 = zeros(M,N+1); Y2 = zeros(M,N+1);
Z1 = zeros(M,3,N); Z2 = zeros(M,3,N);

Y1(:,N+1) = g1;
Y2(:,N+1) = g2;

RY = nan(N,2);
RZ = nan(N,2,3);
picardErr = nan(N,Kpic,2);

for n = N:-1:1
    Xn  = X(:,:,n);
    dWn = dW(:,:,n);
    tn  = tgrid(n);

    Phi = basis_features(Xn, useQuadraticBasis);
    reg = prepare_regression(Phi, useRidge, ridgeLam);

    Y1n = Y1(:,n+1);
    Y2n = Y2(:,n+1);
    Z1n = zeros(M,3);
    Z2n = zeros(M,3);

    Tfit = zeros(M,2);
    U1fit = zeros(M,3);
    U2fit = zeros(M,3);

    for k = 1:Kpic
        q1 = min(sum(Z1n.^2,2), zmax^2);
        q2 = min(sum(Z2n.^2,2), zmax^2);

        f2 = kappa2 * tanh(Y2n) + eta2 * q2;
        f1 = kappa1 * tanh(Y1n) + eta1 * q1 + gamma * tanh(Y2n) ...
             + xi * sin(omega*tn) .* tanh(Xn(:,1));

        T1 = Y1(:,n+1) + f1 * dt;
        T2 = Y2(:,n+1) + f2 * dt;

        U1 = Y1(:,n+1) .* dWn;
        U2 = Y2(:,n+1) .* dWn;

        Tfit = stable_regress(reg, [T1,T2]);
        Y1new = Tfit(:,1);
        Y2new = Tfit(:,2);

        U1fit = stable_regress(reg, U1);
        U2fit = stable_regress(reg, U2);
        Z1new = U1fit / dt;
        Z2new = U2fit / dt;

        Z1new = project_Z(Z1new, zmax);
        Z2new = project_Z(Z2new, zmax);

        picardErr(n,k,1) = mean(abs(Y1new - Y1n),'omitnan');
        picardErr(n,k,2) = mean(abs(Y2new - Y2n),'omitnan');

        Y1n = (1-theta)*Y1n + theta*Y1new;
        Y2n = (1-theta)*Y2n + theta*Y2new;
        Z1n = (1-theta)*Z1n + theta*Z1new;
        Z2n = (1-theta)*Z2n + theta*Z2new;

        Y1n(~isfinite(Y1n)) = 0;
        Y2n(~isfinite(Y2n)) = 0;
        Z1n(~isfinite(Z1n)) = 0;
        Z2n(~isfinite(Z2n)) = 0;
    end

    Y1(:,n) = Y1n;  Y2(:,n) = Y2n;
    Z1(:,:,n) = Z1n; Z2(:,:,n) = Z2n;

    RY(n,1) = normalized_residual(T1, Tfit(:,1));
    RY(n,2) = normalized_residual(T2, Tfit(:,2));

    for j = 1:3
        RZ(n,1,j) = normalized_residual(U1(:,j), U1fit(:,j));
        RZ(n,2,j) = normalized_residual(U2(:,j), U2fit(:,j));
    end
end

diagOut.RY = RY;
diagOut.RZ = RZ;
diagOut.picardErr = picardErr;
end

function r = normalized_residual(y, yhat)
den = norm(y - mean(y))^2;
num = norm(y - yhat)^2;
r = num / max(den,1e-12);
if ~isfinite(r), r = 0; end
end

function [Y1, Y2] = solve_bsde_fast_stable( ...
    X, dW, tgrid, dt, g1, g2, ...
    kappa1, eta1, gamma, xi, omega, kappa2, eta2, zmax, ...
    Kpic, theta, useQuadraticBasis, useRidge, ridgeLam, ycap)

[M,~,N1] = size(X);
N = N1-1;

Y1 = zeros(M,N+1); Y2 = zeros(M,N+1);
Y1(:,N+1) = g1;
Y2(:,N+1) = g2;

for n = N:-1:1
    Xn  = X(:,:,n);
    dWn = dW(:,:,n);
    tn  = tgrid(n);

    Phi = basis_features(Xn, useQuadraticBasis);
    reg = prepare_regression(Phi, useRidge, ridgeLam);

    Y1n = Y1(:,n+1);
    Y2n = Y2(:,n+1);
    Z1n = zeros(M,3);
    Z2n = zeros(M,3);

    for k = 1:Kpic
        q1 = min(sum(Z1n.^2,2), zmax^2);
        q2 = min(sum(Z2n.^2,2), zmax^2);

        f2 = kappa2 * tanh(Y2n) + eta2 * q2;
        f1 = kappa1 * tanh(Y1n) + eta1 * q1 + gamma * tanh(Y2n) ...
             + xi * sin(omega*tn) .* tanh(Xn(:,1));

        T1 = Y1(:,n+1) + f1 * dt;
        T2 = Y2(:,n+1) + f2 * dt;

        U1 = Y1(:,n+1) .* dWn;
        U2 = Y2(:,n+1) .* dWn;

        Tfit = stable_regress(reg, [T1,T2]);
        Y1new = Tfit(:,1);
        Y2new = Tfit(:,2);

        Z1new = stable_regress(reg, U1) / dt;
        Z2new = stable_regress(reg, U2) / dt;

        Z1new = project_Z(Z1new, zmax);
        Z2new = project_Z(Z2new, zmax);

        Y1n = (1-theta)*Y1n + theta*Y1new;
        Y2n = (1-theta)*Y2n + theta*Y2new;
        Z1n = (1-theta)*Z1n + theta*Z1new;
        Z2n = (1-theta)*Z2n + theta*Z2new;

        Y1n = max(min(Y1n, ycap), -ycap);
        Y2n = max(min(Y2n, ycap), -ycap);

        Y1n(~isfinite(Y1n)) = 0;
        Y2n(~isfinite(Y2n)) = 0;
        Z1n(~isfinite(Z1n)) = 0;
        Z2n(~isfinite(Z2n)) = 0;
    end

    Y1(:,n) = Y1n;
    Y2(:,n) = Y2n;
end
end

function [Y0surf1, Y0surf2] = param_scan_surfaces( ...
    epsList, gammaList, T, ...
    sigmaL, rhoL, betaL, x0, delta0, etaReg, alpha, ...
    kappa1, eta1, xi, omega, kappa2, eta2, zmax, ...
    scanOpts, g2cap, ycap)

Ne = numel(epsList);
Ng = numel(gammaList);
Y0surf1 = nan(Ng,Ne);
Y0surf2 = nan(Ng,Ne);

Nscan = scanOpts.Nscan;
dt = T/Nscan;
tgrid = linspace(0,T,Nscan+1);
M = scanOpts.Mscan;

rngState = rng;
for ig = 1:Ng
    for ie = 1:Ne
        epsNoise = epsList(ie);
        gamma    = gammaList(ig);

        % Reset to the same random stream for every parameter pair.  This is
        % a common-random-number construction and makes surface differences
        % attributable to parameters rather than Monte Carlo resampling.
        rng(scanOpts.commonNoiseSeed);
        [X, Xtilde, dW, ~] = simulate_lorenz_pair_substepped( ...
            M, T, Nscan, epsNoise, sigmaL, rhoL, betaL, x0, delta0, ...
            scanOpts.maxForwardDt);

        Delta = Xtilde - X;
        Sep = squeeze(sqrt(sum(Delta.^2,2)));
        sep0 = norm(delta0);

        g2 = (1/T) * log((Sep(:,end) + etaReg) / sep0);
        g2 = max(min(g2, g2cap), -g2cap);

        XTloc = X(:,:,end);
        XTmeanLoc = mean(XTloc,1);
        XTlocAnom = XTloc - XTmeanLoc;
        E_Tloc = sum(XTlocAnom.^2,2);
        g1loc = tanh(alpha*log(1+E_Tloc));

        [Y1loc, Y2loc] = solve_bsde_fast_stable( ...
            X, dW, tgrid, dt, g1loc, g2, ...
            kappa1, eta1, gamma, xi, omega, kappa2, eta2, zmax, ...
            scanOpts.Kpic, scanOpts.theta, scanOpts.useQuadraticBasis, ...
            scanOpts.useRidge, scanOpts.ridgeLam, ycap);

        y01 = mean(Y1loc(:,1));
        y02 = mean(Y2loc(:,1));

        if isfinite(y01) && isfinite(y02)
            Y0surf1(ig,ie) = y01;
            Y0surf2(ig,ie) = y02;
        end
    end
end
rng(rngState);
end

function [Y0mean1, Y0mean2] = quick_Y0_both(M, T, N, epsNoise, ...
    sigmaL, rhoL, betaL, x0, delta0, etaReg, alpha, g2cap, ycap, ...
    kappa1, eta1, gamma, xi, omega, kappa2, eta2, zmax, ...
    Kpic, theta, useQuadraticBasis, useRidge, ridgeLam, maxForwardDt)

dt = T/N;
tgrid = linspace(0,T,N+1);

[X, Xtilde, dW, ~] = simulate_lorenz_pair_substepped( ...
    M, T, N, epsNoise, sigmaL, rhoL, betaL, x0, delta0, maxForwardDt);

Delta = Xtilde - X;
Sep = squeeze(sqrt(sum(Delta.^2,2)));
sep0 = norm(delta0);

g2 = (1/T) * log((Sep(:,end) + etaReg) / sep0);
g2 = max(min(g2, g2cap), -g2cap);

XTloc = X(:,:,end);
XTmeanLoc = mean(XTloc,1);
XTlocAnom = XTloc - XTmeanLoc;
E_Tloc = sum(XTlocAnom.^2,2);
g1loc = tanh(alpha*log(1+E_Tloc));

[Y1loc, Y2loc] = solve_bsde_fast_stable( ...
    X, dW, tgrid, dt, g1loc, g2, ...
    kappa1, eta1, gamma, xi, omega, kappa2, eta2, zmax, ...
    Kpic, theta, useQuadraticBasis, useRidge, ridgeLam, ycap);

Y0mean1 = mean(Y1loc(:,1));
Y0mean2 = mean(Y2loc(:,1));
if ~isfinite(Y0mean1), Y0mean1 = 0; end
if ~isfinite(Y0mean2), Y0mean2 = 0; end
end

function D = first_derivative_dim1(A, h)
D = zeros(size(A));
D(2:end-1,:) = (A(3:end,:) - A(1:end-2,:)) / (2*h);
D(1,:) = (A(2,:) - A(1,:)) / h;
D(end,:) = (A(end,:) - A(end-1,:)) / h;
end

function D = first_derivative_dim2(A, h)
D = zeros(size(A));
D(:,2:end-1) = (A(:,3:end) - A(:,1:end-2)) / (2*h);
D(:,1) = (A(:,2) - A(:,1)) / h;
D(:,end) = (A(:,end) - A(:,end-1)) / h;
end

function D2 = second_derivative_dim2(A, h)
D2 = zeros(size(A));
D2(:,2:end-1) = (A(:,3:end) - 2*A(:,2:end-1) + A(:,1:end-2)) / (h^2);
D2(:,1) = (A(:,3) - 2*A(:,2) + A(:,1)) / (h^2);
D2(:,end) = (A(:,end) - 2*A(:,end-1) + A(:,end-2)) / (h^2);
end

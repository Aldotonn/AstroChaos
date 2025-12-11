function Sigma_time = load_cov_json(filename)
% Carica una covarianza salvata da Python (matrix.tolist())
% Se è (K,6,6) → restituisce (K,6,6)
% Se è cell{K} di 6x6 → la converte in (K,6,6)

    raw = jsondecode(fileread(filename));

    if iscell(raw)
        % Caso: cell array 1xK, ognuno 6x6
        K = numel(raw);
        N = size(raw{1},1);
        Sigma_time = zeros(K, N, N);
        for k = 1:K
            Sigma_time(k,:,:) = raw{k};
        end
    else
        % Caso: array numerico Kx6x6 (o 6x6 per il tf)
        Sigma_time = raw;
    end
end
clear; clc;

%% ==== 1. CARICO I FILE ====

Sigma_MC_time  = load_cov_json("Sigma_MC_time.json");                % K x 6 x 6
Sigma_PCE_time = load_cov_json("Sigma_PCE_time_order5.json");        % K x 6 x 6

Sigma_MC_tf    = load_cov_json("Sigma_MC_tf.json");                  % 6 x 6
Sigma_PCE_tf   = load_cov_json("Sigma_PCE_tf_order5.json");          % 6 x 6

[K, N, ~] = size(Sigma_MC_time);   % K snapshot, N stati (N=6)

%% ==== 2. ANALISI DI BASE: VARIANZE, VOLUME, ERRORE FROBENIUS ====

var_MC  = zeros(N, K);
var_PCE = zeros(N, K);
det_MC  = zeros(1, K);
det_PCE = zeros(1, K);
fro_err = zeros(1, K);
fro_MC  = zeros(1, K);

for k = 1:K
    S_MC  = squeeze(Sigma_MC_time(k,:,:));
    S_PCE = squeeze(Sigma_PCE_time(k,:,:));

    var_MC(:,k)  = diag(S_MC);
    var_PCE(:,k) = diag(S_PCE);

    det_MC(k)  = sqrt(max(real(det(S_MC)), 0));
    det_PCE(k) = sqrt(max(real(det(S_PCE)), 0));

    fro_err(k) = norm(S_PCE - S_MC, 'fro');
    fro_MC(k)  = norm(S_MC, 'fro');
end

rel_err_time = fro_err ./ max(fro_MC, 1e-16);

%% ==== 3. PLOT VARIANZE ====

time = 1:K;  % se hai un vero vettore tempi puoi sostituirlo qui

figure;
for i = 1:N
    subplot(3,2,i);
    plot(time, var_MC(i,:), 'LineWidth', 1.5); hold on;
    plot(time, var_PCE(i,:), '--', 'LineWidth', 1.5);
    grid on;
    title(sprintf('Variance state %d', i));
    xlabel('snapshot');
    ylabel('\sigma^2');
    legend('MC','PCE');
end
sgtitle('Variances over time');

%% ==== 4. PLOT ERRORE FROBENIUS RELATIVO ====

figure;
plot(time, rel_err_time, 'LineWidth', 1.8);
grid on;
xlabel('snapshot');
ylabel('relative Frobenius error');
title('||\Sigma_{PCE} - \Sigma_{MC}||_F / ||\Sigma_{MC}||_F');

%% ==== 5. ANALISI FINALE A tf ====

Delta_tf = Sigma_PCE_tf - Sigma_MC_tf;
fro_mc_tf  = norm(Sigma_MC_tf, 'fro');
rel_err_tf = norm(Delta_tf, 'fro') / max(fro_mc_tf, 1e-16);

det_pce_tf = sqrt(max(real(det(Sigma_PCE_tf)), 0));
det_mc_tf  = sqrt(max(real(det(Sigma_MC_tf)), 0));
vol_ratio_tf = det_pce_tf / max(det_mc_tf, 1e-16);

fprintf('\n=== SUMMARY AT FINAL TIME tf ===\n');
fprintf('Relative Frobenius error  = %.3e\n', rel_err_tf);
fprintf('Volume ratio sqrt(det PCE)/sqrt(det MC) = %.3f\n', vol_ratio_tf);
fprintf('trace Sigma_MC_tf  = %.3e\n', trace(Sigma_MC_tf));
fprintf('trace Sigma_PCE_tf = %.3e\n', trace(Sigma_PCE_tf));

%% ==== 6. HEATMAP DELL\'ERRORE A tf ====

figure;
imagesc(Delta_tf);
colorbar;
axis equal tight;
title('\Sigma_{PCE}(tf) - \Sigma_{MC}(tf)');
xlabel('state index');
ylabel('state index');

%% ============================================
%  AUTOVALORI DELLA COVARIANZA (MC vs PCE)
% =============================================
K = size(Sigma_MC_time, 1);
N = size(Sigma_MC_time, 2);

eig_MC  = zeros(K, N);
eig_PCE = zeros(K, N);

for k = 1:K
    S_MC  = squeeze(Sigma_MC_time(k,:,:));
    S_PCE = squeeze(Sigma_PCE_time(k,:,:));

    % autovalori sortati in ordine decrescente
    eig_MC(k,:)  = sort(eig(S_MC), 'descend');
    eig_PCE(k,:) = sort(eig(S_PCE), 'descend');
end

%% ============================================
%  PLOT DEI PRIMI 3 AUTOVALORI (STRETCHING DIRECTIONS)
% =============================================

time = 1:K;

figure;
set(gcf, 'Position', [200 200 900 700])

for i = 1:3
    subplot(3,1,i)
    plot(time, eig_MC(:,i), 'LineWidth', 1.8); hold on;
    plot(time, eig_PCE(:,i), '--', 'LineWidth', 1.8);
    grid on;
    xlabel('snapshot index')
    ylabel(sprintf('\\lambda_%d(t)', i))
    title(sprintf('Principal eigenvalue %d of covariance', i))
    legend('MC', 'PCE')
end

sgtitle('Evolution of covariance eigenvalues: MC vs PCE')

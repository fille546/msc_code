%First set the random seed
rng(42);

% Then add the ranges
num_samples = 1000; 
T_range = [0, 5];
S0_range = [80, 120];
r_range = [0, 0.05];
sigma_range = [0.01, 0.5];
K_range = [50, 150];

% Generate random samples within the ranges
T_samples = rand(num_samples, 1) * diff(T_range) + T_range(1);
S0_samples = rand(num_samples, 1) * diff(S0_range) + S0_range(1);
r_samples = rand(num_samples, 1) * diff(r_range) + r_range(1);
K_samples = rand(num_samples, 1) * diff(K_range) + K_range(1);
sigma_samples = rand(num_samples, 1) * diff (sigma_range) + sigma_range(1);

NSim = 10000;
model = 'BS';
sym = 0;

% Create a matrix for the inputs
X = [T_samples, S0_samples, r_samples, K_samples, sigma_samples];

% Calculate the target values (with MC)
tic;
benchmark_prices = arrayfun(@(idx) MC_Option_Pricing(S0_samples(idx), K_samples(idx), T_samples(idx), r_samples(idx), sigma_samples(idx),NSim , model, 0, 0, 0, sym), 1:num_samples);
MC_target_values = toc;

% Calculate the AE-values
tic;
european_approx_prices = arrayfun(@(idx) european_BS(S0_samples(idx), K_samples(idx), T_samples(idx), sigma_samples(idx)), 1:num_samples);
AE_target_values = toc;

Y_method1 = benchmark_prices(:);
Y_method2 = (benchmark_prices - european_approx_prices);
Y_method2 = Y_method2(:);

% Split the data
train_ratio = 0.8;
val_ratio = 0.2; 
idx = randperm(num_samples);
train_idx = idx(1:round(train_ratio * num_samples));
val_idx = idx(round(train_ratio * num_samples) + 1:end);
X_train = X(train_idx, :);
X_val = X(val_idx, :);
Y_method1_train = Y_method1(train_idx);
Y_method1_val = Y_method1(val_idx);
Y_method2_train = Y_method2(train_idx);
Y_method2_val = Y_method2(val_idx);
european_approx_prices_val = european_approx_prices(val_idx);

% Standardize the features
X_mean = mean(X_train, 1); 
X_std = std(X_train, 0, 1); 
X_train = (X_train - X_mean) ./ X_std;
X_val = (X_val - X_mean) ./ X_std;

% Configure the ANN
layers = [
    featureInputLayer(5, 'Name', 'input')
    fullyConnectedLayer(16, 'Name', 'fc1') % For the complex change nodes to 64
    fullyConnectedLayer(16, 'Name', 'fc2') % For the complex change nodes to 64
    fullyConnectedLayer(16, 'Name', 'fc3') % For the complex change nodes to 64
    reluLayer('Name', 'relu1')
    fullyConnectedLayer(1, 'Name', 'output') 
    regressionLayer('Name', 'output_reg')
];

options = trainingOptions('adam', ...
    'MaxEpochs', 10, ...
    'MiniBatchSize', 2, ...
    'ValidationData', {X_val, Y_method1_val}, ...
    'ValidationFrequency', 15, ...
    'ValidationPatience', 200, ...
    'Plots', 'training-progress');

% Train M1
tic;
net_method1 = trainNetwork(X_train, Y_method1_train, layers, options);
M1_training = toc;

% Train M2
options.ValidationData = {X_val, Y_method2_val};
tic;
net_method2 = trainNetwork(X_train, Y_method2_train, layers, options);
M2_training = toc;

% Use the validation data for predictions (M2)
tic;
Y_pred_method2_val = predict(net_method2, X_val);
M2_predict = toc;
M2_predict_full = M2_predict + (0.2 * AE_target_values);
MC_val = 0.2 * MC_target_values;

if size(Y_pred_method2_val, 1) ~= size(european_approx_prices_val, 1)
    european_approx_prices_val = reshape(european_approx_prices_val, size(Y_pred_method2_val));
end

% Add back the AE-values for M2
Y_pred_method2_val = Y_pred_method2_val + european_approx_prices_val;
tic;

% Use the validation data for predictions (M1)
Y_pred_method1_val = predict(net_method1, X_val);
M1_predict = toc;

% Calculate the metrics for M1
mse_method1_val = mean((Y_method1_val - Y_pred_method1_val).^2);
mae_method1_val = mean(abs(Y_method1_val - Y_pred_method1_val));
ss_res_method1_val = sum((Y_method1_val - Y_pred_method1_val).^2);
ss_tot_method1_val = sum((Y_method1_val - mean(Y_method1_val)).^2);
r_squared_method1_val = 1 - ss_res_method1_val/ss_tot_method1_val;

% Calculate the metrics for M2
mse_method2_val = mean((Y_method1_val - Y_pred_method2_val).^2);
mae_method2_val = mean(abs(Y_method1_val - Y_pred_method2_val));
ss_res_method2_val = sum((Y_method1_val - Y_pred_method2_val).^2);
ss_tot_method2_val = sum((Y_method1_val - mean(Y_method1_val)).^2);
r_squared_method2_val = 1 - ss_res_method2_val/ss_tot_method2_val;
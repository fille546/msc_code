% First set the random seed, using the rng function (MathWorks, n.d.-b).
rng(42);

% Then add the ranges.
num_samples = 1000;
T_range = [0, 5];
S0_range = [80, 120];
r_range = [0, 0.05];
sigma_range = [0.01, 0.5];
K_range = [50, 150];

% Generate random samples within the ranges, using rand and diff 
% functions (MathWorks, n.d.-g, n.d.-c).
T_samples = rand(num_samples, 1) * diff(T_range) + T_range(1);
S0_samples = rand(num_samples, 1) * diff(S0_range) + S0_range(1);
r_samples = rand(num_samples, 1) * diff(r_range) + r_range(1);
K_samples = rand(num_samples, 1) * diff(K_range) + K_range(1);
sigma_samples = rand(num_samples, 1) * diff (sigma_range) + sigma_range(1);

NSim = 10000;
model = 'BS';

% Create a matrix for the inputs.
X = [T_samples, S0_samples, r_samples, K_samples, sigma_samples];

% Calculate the target values (with MC), using arrayfun to apply the
% function on each element (MathWorks, n.d.-a).
% Timing with the tic function (MathWorks, n.d.-f).
tic;
benchmark_prices = arrayfun(@(idx) MC(S0_samples(idx), ...
    K_samples(idx), T_samples(idx), r_samples(idx), sigma_samples(idx), ...
    NSim , model, 0, 0, 0), 1:num_samples);
MC_target_values = toc;

% Calculate the AE-values.
tic;
european_approx_prices = arrayfun(@(idx) european_BS(S0_samples(idx), ...
    K_samples(idx), T_samples(idx), sigma_samples(idx)), 1:num_samples);
AE_target_values = toc;

Y_method1 = benchmark_prices(:); 
Y_method2 = (benchmark_prices - european_approx_prices); 
Y_method2 = Y_method2(:);

% Split the data.
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

% Standardize the features.
X_mean = mean(X_train, 1); 
X_std = std(X_train, 0, 1); 
X_train = (X_train - X_mean) ./ X_std;
X_val = (X_val - X_mean) ./ X_std;

% Configure the DT (MathWorks, n.d.-d).
tic;
%tree_method1 = fitrtree(X_train, Y_method1_train, 'MaxNumSplits', 10, 
% 'MinLeafSize', 10);
tree_method1 = fitrtree(X_train, Y_method1_train, 'MaxNumSplits', 100, ...
    'MinLeafSize', 1);
M1_training = toc;
view(tree_method1, 'Mode', 'graph');
tic;
%tree_method2 = fitrtree(X_train, Y_method2_train, 'MaxNumSplits', 10, 
% 'MinLeafSize', 10);
tree_method2 = fitrtree(X_train, Y_method2_train, 'MaxNumSplits', 100, ...
    'MinLeafSize', 1); 
M2_training = toc;

% Use the validation data for predictions (M1).
tic;
Y_pred_method1_val = predict(tree_method1, X_val);
M1_predict = toc;

% Use the validation data for predictions (M2).
tic;
Y_pred_method2_val = predict(tree_method2, X_val);
M2_predict = toc;
M2_predict_full = M2_predict + (0.2 * AE_target_values);
MC_val = 0.2 * MC_target_values;

if size(Y_pred_method2_val, 1) ~= size(european_approx_prices_val, 1)
    european_approx_prices_val = reshape(european_approx_prices_val, ...
        size(Y_pred_method2_val));
end

% Add back the AE-values for M2.
Y_pred_method2_val = Y_pred_method2_val + european_approx_prices_val;

% Calculate the metrics for M1.
mse_method1_val = mean((Y_method1_val - Y_pred_method1_val).^2);
mae_method1_val = mean(abs(Y_method1_val - Y_pred_method1_val));
ss_res_method1_val = sum((Y_method1_val - Y_pred_method1_val).^2);
ss_tot_method1_val = sum((Y_method1_val - mean(Y_method1_val)).^2);
r_squared_method1_val = 1 - ss_res_method1_val/ss_tot_method1_val;

% Calculate the metrics for M2.
mse_method2_val = mean((Y_method1_val - Y_pred_method2_val).^2);
mae_method2_val = mean(abs(Y_method1_val - Y_pred_method2_val));
ss_res_method2_val = sum((Y_method1_val - Y_pred_method2_val).^2);
ss_tot_method2_val = sum((Y_method1_val - mean(Y_method1_val)).^2);
r_squared_method2_val = 1 - ss_res_method2_val/ss_tot_method2_val;

% References:
% MathWorks. (n.d.-a). Apply function to each element of array - MATLAB
% arrayfun. Retrieved December 10, 2023, from 
% https://se.mathworks.com/help/matlab/ref/arrayfun.html

% MathWorks. (n.d.-b). Control random number generator - MATLAB rng. 
% Retrieved December 10, 2023, from 
% https://se.mathworks.com/help/matlab/ref/rng.html

% MathWorks. (n.d.-c). Differences and approximate derivatives - 
% MATLAB diff. Retrieved December 10, 2023, from 
% https://se.mathworks.com/help/matlab/ref/diff.html

% MathWorks. (n.d.-d). Fit binary decision tree for regression -
% MATLAB fitrtree. Retrieved December 10, 2023, from 
% https://se.mathworks.com/help/stats/fitrtree.html% 

% MathWorks. (n.d.-e). Random permutation of integers - MATLAB randperm. 
% Retrieved December 10, 2023, from 
% https://se.mathworks.com/help/matlab/ref/randperm.html

% MathWorks. (n.d.-f). Start stopwatch timer - MATLAB tic. 
% Retrieved March 19, 2024, from 
% https://se.mathworks.com/help/matlab/ref/tic.html

% MathWorks. (n.d.-g). Uniformly distributed random numbers - 
% MATLAB rand. Retrieved December 10, 2023, from 
% https://se.mathworks.com/help/matlab/ref/rand.html


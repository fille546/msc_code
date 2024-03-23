function Price = MC(S_0, K, T, r, sigma, NSim, model, alpha, beta, mu)
    NSteps = round(T * 365);
    dt = T / NSteps;
    S = zeros(NSim, NSteps + 1);
    S(:,1) = S_0;

% Choose the correct model.
    for j = 1:NSteps
        Z = randn(NSim, 1);
        if strcmp(model, 'BS')
            S(:,j+1) = S(:,j) .* exp((r - 0.5 * sigma^2) * dt + sigma * ...
                sqrt(dt) * Z); % Defining the Black-Scholes-Merton 
            % volatility model (Black & Scholes, 1973)
        else
            if strcmp(model, 'CEV')
                sigma_f = sigma * S(:,j).^(beta - 1); % Defining the 
                % constant elasticity of variance volatility model 
                % according to Funahashi and Kijima (2016)

            elseif strcmp(model, 'LV')
                sigma_f = (alpha + beta * S(:,j)/S_0) .* ...
                    exp(-mu * S(:,j)/S_0); % Defining the non-linear 
                % volatility model according to Funahashi and Kijima (2016)

            end
            S(:,j+1) = S(:,j) + r * S(:,j) * dt + sigma_f .* ...
                S(:,j) .* Z * sqrt(dt);
        end
    end

% Calculate the payoff for every path.
    Payoff = max(S(:,end) - K, 0);
    
% Discount them back to present value.
    Payoff_discounted = exp(-r * T) * Payoff;
    
% Estimate the price from the average of all the discounted payoffs.
    Price = mean(Payoff_discounted);
end

% References:
% Black, F., & Scholes, M. (1973). The pricing of options and corporate 
% liabilities. Journal of Political Economy, 81(3), 637–654. 
% https://doi.org/10.1086/260062

% Funahashi, H., & Kijima, M. (2016). Analytical pricing of single 
% barrier options under local volatility models. Quantitative 
% Finance, 16(6), 867–886. https://doi.org/10.1080/14697688.2015.1101483
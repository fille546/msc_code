function Price = MC_Option_Pricing(S0, K, T, r, sigma, NSim, model, alpha, beta, mu)
    % This function will estimate prices of European Vanilla Call options
    % using Monte Carlo simulations.
    % 
    % This function is designed for the estimation of the European call option price using
    % Monte Carlo simulation approach. The option pricing models included are Black-Scholes ('BS'),
    % Constant Elasticity of Variance ('CEV'), and Local Volatility ('LV'). For 'LV', the parameters
    % alpha, beta, and mu must be specified, which influence the volatility function uniquely for this model.
    %
    % Input parameters:
    % S0 - Initial stock price
    % K - Strike price
    % T - Maturity time in years
    % r - Risk-free interest rate, annualized
    % sigma - Volatility (used in 'BS' and 'CEV' models)
    % NSim - Number of simulation paths
    % model - Indicates the model to use: 'BS', 'CEV', or 'LV'
    % alpha, beta, mu - Specific parameters for the 'LV' model
    %
    % Outputs:
    % Price - The estimated price of the option

    % Creating the MC simulation similarly as 

    % First defining the time step for simulations, assuming daily monitoring
    dt = 1/365;
    NSteps = round(T / dt);
    S = zeros(NSim, NSteps);
    S(:,1) = S0;
    
    % Iterating over each time step to simulate asset paths
    for t = 2:NSteps
        dW = sqrt(dt) * randn(NSim, 1);
        if strcmp(model, 'BS')
            S(:,t) = S(:,t-1) .* exp((r - 0.5 * sigma^2) * dt + sigma * dW);
        elseif strcmp(model, 'CEV')
            S(:,t) = S(:,t-1) .* exp((r - 0.5 * sigma^2 * (S(:,t-1)./S0).^beta) * dt + ...
                                     sigma * (S(:,t-1)./S0).^(beta/2) .* dW);
        elseif strcmp(model, 'LV')
            sigmaLV = (alpha + beta * S(:,t-1)./S0) .* exp(-mu * S(:,t-1)./S0);
            S(:,t) = S(:,t-1) .* exp((r - 0.5 * sigmaLV.^2) * dt + sigmaLV .* dW);
        end
    end
    
    % Calculating the payoff for each simulated path at maturity
    Payoff = max(S(:,end) - K, 0);
    
    % Discounting the payoffs to present value
    DiscountedPayoff = exp(-r * T) * Payoff;
    
    % Computing the average of the discounted payoffs to estimate the option price
    Price = mean(DiscountedPayoff);
end

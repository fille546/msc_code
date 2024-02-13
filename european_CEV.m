function [price] = european_CEV(S_0, K, T, sigma, beta)

% Defining the constant elasticity of variance volatility model according
% to Funahashi and Kijima (2016)
sigma_CEV=@(S) sigma*S^(beta-1);

% Pricing the option according to the equations by
% Funahashi and Kijima (2015)
K_tilde=1-K/S_0;
Omega_T=sigma_CEV(S_0)^2*T;
v_T=((sigma_CEV(S_0)^4)*T^2)/2;

price=S_0*normpdf(K_tilde,0,sqrt(Omega_T))/Omega_T*...
    (Omega_T^2-v_T*K_tilde)+S_0*K_tilde*...
    (1-normcdf(-K_tilde/sqrt(Omega_T)));

end

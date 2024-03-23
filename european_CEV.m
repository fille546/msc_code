function [price] = european_CEV(S_0, K, T, sigma, beta)

% Defining the constant elasticity of variance volatility model according
% to Funahashi and Kijima (2016)
sigma_CEV=@(S) sigma*S^(beta-1);

% Pricing the option according to the equations by
% Funahashi (2014)
K_tilde=1-K/S_0;
Omega_T=sigma_CEV(S_0)^2*T;
v_T=((sigma_CEV(S_0)^4)*T^2)/2;

price=S_0*normpdf(K_tilde,0,sqrt(Omega_T))/Omega_T*...
    (Omega_T^2-v_T*K_tilde)+S_0*K_tilde*...
    (1-normcdf(-K_tilde/sqrt(Omega_T)));

end

% References:
% Funahashi, H. (2014). A chaos expansion approach under hybrid 
% volatility models. Quantitative Finance, 14(11), 1923–1936. 
% https://doi.org/10.1080/14697688.2013.872283

% Funahashi, H., & Kijima, M. (2016). Analytical pricing of single 
% barrier options under local volatility models. Quantitative 
% Finance, 16(6), 867–886. https://doi.org/10.1080/14697688.2015.1101483
function [price] = european_BS(S_0, K, T, sigma)

% Pricing the option according to the equations by
% Funahashi and Kijima (2015)
K_tilde=1-K/S_0;
Omega_T=sigma^2*T;
v_T=((sigma^4)*T^2)/2;

price=S_0*normpdf(K_tilde,0,sqrt(Omega_T))/Omega_T*...
    (Omega_T^2-v_T*K_tilde)+S_0*K_tilde*...
    (1-normcdf(-K_tilde/sqrt(Omega_T)));
end
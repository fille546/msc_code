function [price] = barrier_BS(S_0, K, T, B, sigma)

%Following the approach by Funahashi and Kijima (2016) we begin
%initialising the mathematical expression for the barrier option.

x_0 = S_0 / B;

eta_tilde_0 = sigma;
eta_tilde_0_d1 = 0;
eta_tilde_0_d2 = 0;
eta_tilde_0_d3 = 0;

p1=eta_tilde_0+(x_0-1)*eta_tilde_0_d1+...
    ((x_0-1)^2)/2*eta_tilde_0_d2+((x_0-1)^3)...
    /6*eta_tilde_0_d3;

p2=eta_tilde_0+(x_0+1)*(eta_tilde_0_d1)+...
    x_0*(x_0-1)*eta_tilde_0_d2+(x_0-1)^2/2*(x_0+1)...
    *eta_tilde_0_d3;

p3n=@(n) 1/factorial(n)*((x_0-1)^n)*(eta_tilde_0+x_0*...
    eta_tilde_0_d1+(x_0-1)*(eta_tilde_0_d1+x_0*eta_tilde_0_d2)...
    +((x_0-1)^2)/2*(eta_tilde_0_d2+x_0*eta_tilde_0_d3));

Sigma_T=(p1^2)*T;

q_T=(p1^2)*p2*eta_tilde_0*(T^2)/2+(p1^2)*(T^2)*...
    (p3n(1)*eta_tilde_0_d1+p3n(2)*eta_tilde_0_d2+...
    p3n(3)*eta_tilde_0_d3)/2;

%For the barrier option under the symmetrized assumption we follow the
%equations by Funuhashi and Kijima (2016)

E_StildaG=(S_0-K)/2+(q_T*(K-S_0)*exp(-(K-S_0)^2/ ...
    (2*S_0^2*Sigma_T)))/(sqrt(2*pi)*Sigma_T^(3/2))+...
    (S_0*sqrt(Sigma_T)*exp(-(K-S_0)^2/(2*S_0^2*Sigma_T)))...
    /sqrt(2*pi)+1/2*abs(K-S_0)*(2*normcdf(abs(K-S_0)/(S_0*sqrt...
    (Sigma_T)))-1);

E_StildaGamma=(B-K*x_0)/2+(q_T*(B-K*x_0)*...
    exp(-(B-K*x_0)^2/(2*K^2*x_0^2*Sigma_T)))/(sqrt(2*pi)*...
    Sigma_T^(3/2))+(K*x_0*sqrt(Sigma_T)*exp(-(B-K*x_0)^2/(2*K^2 ...
    *x_0^2*Sigma_T)))/sqrt(2*pi)+1/2*abs(B-K*x_0)*(2*normcdf ...
    (abs(B-K*x_0)/(K*x_0*sqrt(Sigma_T)))-1);

% Pricing the european option according to the equations by
% Funahashi and Kijima (2015)

K_tilde = 1-K/S_0;
Omega_T = sigma^2*T;
v_T = ((sigma^4)*T^2)/2;

E_SG = S_0*normpdf(K_tilde,0,sqrt(Omega_T))/Omega_T*...
    (Omega_T^2-v_T*K_tilde)+S_0*K_tilde*...
    (1-normcdf(-K_tilde/sqrt(Omega_T)));

%Adding all into the barrier option

price= E_StildaGamma+(E_SG-E_StildaG);

end

function [price] = barrier_LV(S_0, K, T, B, beta, s, mu, alpha)


%Following the approach by Funahashi and Kijima (2016) we begin
%initialising the mathematical expression for the barrier option.

eta_tilde=@(s,x) (alpha+beta*B*exp(sqrt(s^2+(log(x))^2))/S_0)*...
    exp(-mu*B*exp(sqrt(s^2+(log(x))^2))/S_0);
eta_tilde_0=@(s) eta_tilde(s,1);
eta_tilde_0_d1=@(s) 0;
eta_tilde_0_d2=@(s) -B*exp(s-(B*mu*exp(s))/S_0)*(B*beta*mu*exp(s)+...
    S_0*alpha*mu-S_0*beta)/(s*S_0^2);
eta_tilde_0_d3=@(s) -3*eta_tilde_0_d2(s);
sigma_LV=@(S) (alpha+beta*S/S_0)*exp(-mu*S/S_0);

x_0 = S_0 / B;

p1=@(s) eta_tilde_0(s)+(x_0-1)*eta_tilde_0_d1(s)+...
    ((x_0-1)^2)/2*eta_tilde_0_d2(s)+((x_0-1)^3)...
    /6*eta_tilde_0_d3(s);

p2=@(s) eta_tilde_0(s)+(x_0+1)*(eta_tilde_0_d1(s))+...
    x_0*(x_0-1)*eta_tilde_0_d2(s)+(x_0-1)^2/2*(x_0+1)...
    *eta_tilde_0_d3(s);

p3n=@(n,s) 1/factorial(n)*((x_0-1)^n)*(eta_tilde_0(s)+x_0*...
    eta_tilde_0_d1(s)+(x_0-1)*(eta_tilde_0_d1(s)+x_0*eta_tilde_0_d2(s))...
    +((x_0-1)^2)/2*(eta_tilde_0_d2(s)+x_0*eta_tilde_0_d3(s)));

Sigma_T=@(s) (p1(s)^2)*T;

q_T=@(s) (p1(s)^2)*p2(s)*eta_tilde_0(s)*(T^2)/2+(p1(s)^2)*(T^2)*...
    (p3n(1,s)*eta_tilde_0_d1(s)+p3n(2,s)*eta_tilde_0_d2(s)+...
    p3n(3,s)*eta_tilde_0_d3(s))/2;

E_StildaG=@(s) (S_0-K)/2+(q_T(s)*(K-S_0)*exp(-(K-S_0)^2/ ...
    (2*S_0^2*Sigma_T(s))))/(sqrt(2*pi)*Sigma_T(s)^(3/2))+...
    (S_0*sqrt(Sigma_T(s))*exp(-(K-S_0)^2/(2*S_0^2*Sigma_T(s))))...
    /sqrt(2*pi)+1/2*abs(K-S_0)*(2*normcdf(abs(K-S_0)/(S_0*sqrt...
    (Sigma_T(s))))-1);

E_StildaGamma=@(s) (B-K*x_0)/2+(q_T(s)*(B-K*x_0)*...
    exp(-(B-K*x_0)^2/(2*K^2*x_0^2*Sigma_T(s))))/(sqrt(2*pi)*...
    Sigma_T(s)^(3/2))+(K*x_0*sqrt(Sigma_T(s))*exp(-(B-K*x_0)^2/(2*K^2 ...
    *x_0^2*Sigma_T(s))))/sqrt(2*pi)+1/2*abs(B-K*x_0)*(2*normcdf ...
    (abs(B-K*x_0)/(K*x_0*sqrt(Sigma_T(s))))-1);

K_tilde=1-K/S_0;
Omega_T=sigma_LV(S_0)^2*T;
v_T=((sigma_LV(S_0)^4)*T^2)/2;

E_SG=S_0*normpdf(K_tilde,0,sqrt(Omega_T))/Omega_T*...
    (Omega_T^2-v_T*K_tilde)+S_0*K_tilde*...
    (1-normcdf(-K_tilde/sqrt(Omega_T)));

DImSG=@(s) E_StildaGamma(s)+(E_SG-E_StildaG(s));

m=1/s;

price=DImSG(1/(m+2))-((DImSG(1/(m+2))-DImSG(1/(m+1)))^2)/(DImSG(1/(m+2))...
    -2*DImSG(1/(m+1))+DImSG(1/m));

end
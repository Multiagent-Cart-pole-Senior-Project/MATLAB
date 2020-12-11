%% Linear Controller for Cart-Pole System %%
% Author: Ryan Russell
% 
% Control Algorithm Developed from: 
% "ECE 221 Project: Inverted Pendulum"
%    By: Dr. Jing Wang



% State Variables:
% x1 = Theta
% x2 = Theta_dot
% x3 = x
% x4 = x_dot

close all
clear all
clc

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
% PARAMETERS
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
M = 1; % [kg] - Mass of the Cart
m = 0.1; % [kg] - Mass of the pendulum
b = 0; % [N/(m/s)] - Coefficient of friction of the cart
l = 0.5; % [m] - Length of pendulum center of mass
g = 9.81; % [m/s^2] - Gravitational Acceleration Constant

t0 = 0; % [s] - Start time
tf = 40; % [s] - End time
T = 0.01; % [s] - Sampling Time
t = t0:T:tf; % Time Vector
tt = t0:T:tf+1; % Time Vector

x(:,1) = [0; 0.01; 0.1; 0.01]; % Initial Conditions

% State Matrices
A = [0, 1, 0, 0; (g*(M+m))/(M*l), 0, 0, b/(M*l); 0, 0, 0, 1; -(m*g)/M, 0, 0, -b/M];
B = [0; -1/(M*l); 0; 1/M];
C = [1, 0, 0, 0; 0, 0, 1, 0];

% Desired Pole Locataions
zeta = 0.6;
w_n = 1;
pole_d = [-zeta*w_n+1i*w_n*sqrt(1-zeta^2); -zeta*w_n-1i*w_n*sqrt(1-zeta^2);...
                                                    -5*zeta*w_n; -8*zeta*w_n];

% Control Gain Vector
K = acker(A,B,pole_d);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
% Simulation (Discrete Time - Euler Integration)
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
k = 1; % Starting timestep
kf = tf/T; % Final timestep

while k <= kf+1
    u_temp = -K*x(:,k); % Control Input
    u(k) = sign(u_temp)*min(1 , abs(u_temp));
    
    x0k = [x(:,k); u(k)];
    [ttt, x0k1] = ode45('Cart_model', [tt(k) tt(k+1)], x0k);

    x(1,k+1) = x0k1(length(ttt),1);
    x(2,k+1) = x0k1(length(ttt),2);
    x(3,k+1) = x0k1(length(ttt),3);
    x(4,k+1) = x0k1(length(ttt),4);
    
    x_pos(k) = x(3,k);
    Theta(k) = x(1,k);
    
    k = k + 1; % Update Timestep
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
% PLOT RESULTS
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

figure
subplot(2,1,1)
plot(t,x_pos)
title('Cart Position')
ylabel('Position [m]')
xlabel('Time [s]')
grid on
subplot(2,1,2)
plot(t,Theta)
title('Pole Angle')
ylabel('Angle [rad]')
xlabel('Time [s]')
grid on
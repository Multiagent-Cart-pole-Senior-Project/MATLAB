% Control of Cart-pole systems

% Nonlinear Model: 
% (M+m) \ddot{x} + ml\cos\theta \ddot\theta - ml\sin\theta (\dot\theta)^2 +
% b\dot{x} = F

% (I + ml^2) \ddot\theta + ml\cos\theta \ddot{x} = mgl\sin\theta 

% M = 0.5kg; Mass of the cart
% m = 0.2kg; Mass of the pendulum
% b = 0.1 N/(m/s); coefficient of friction of cart
% l = 0.3m; length of pendulum center of mass
% I = 0.006 kg m^2; moment of inertia of the pendulum
% F: force applied to the cart
% theta: angular position of the pendulum
% x: horizontal position of the cart


clc;
clear;
close all;


%global control;

% Model parameters
M = 0.5;
m = 0.2;
b = 0.1;
l = 0.5;
I = 0.006;
g = 9.8; 

% Linear controller is designed based on the linearized model around
% x_1 =0, x_2 = 0
A = [0 1 0 0; (m*g*l*(M+m))/((M+m)*I+M*m*l^2) 0 0 b*m*l/((M+m)*I+M*m*l^2); ...
0 0 0 1; -m^2*l^2*g/((M+m)*I+M*m*l^2) 0 0 -b*(I+m*l^2)/((M+m)*I+M*m*l^2)];

B = [0; -m*l/((M+m)*I+M*m*l^2); 0; (I+m*l^2)/((M+m)*I+M*m*l^2)];

C = [1 0 0 0; 0 0 1 0];

% Desired poles:
zeta = 0.7;
omegan = 10;

s1d = -zeta*omegan + i*omegan*sqrt(1-zeta^2);
s2d = -zeta*omegan - i*omegan*sqrt(1-zeta^2);
%s3d = -zeta*omegan + i*omegan*sqrt(1-zeta^2);
%s4d = -zeta*omegan - i*omegan*sqrt(1-zeta^2);
s3d = -5*zeta*omegan;
s4d = -8*zeta*omegan;

Sd =[s1d; s2d; s3d; s4d];
K = acker(A,B, Sd);
k1 = K(1);
k2 = K(2);
k3 = K(3);
k4 = K(4);

% Simulation time

T = 5; 
Ts=0.01; 
tt=0:Ts:T; 

xx0 = [0.01; 0 ; 0; 0];
xd = [0; 0; 0; 0];

for j=1:length(tt)-1
   ttt(j)=tt(j);
end

for k = 1:length(tt)-1
    
   fprintf('\n k = %i', k);
    
   yyy(k,:) = xx0';
   con1 = -K*(xx0-xd)+2*sin(5*tt(k));
   u(k) = sign(con1)*min(1 , abs(con1)); % con1;  
   xx00 = [xx0; con1];
   [t, yy] = ode45('Cart_model', [tt(k) tt(k+1)], xx00);
   xx00=yy(length(t), :)';
   xx0=xx00(1:4);
  
   
end 


alw = 0.75;    % AxesLineWidth
fsz = 11;      % Fontsize
lw = 1.5;      % LineWidth 
% lw =1.5
msz = 8; 

figure
plot(tt(1:length(tt)-1), yyy(:,1), 'LineWidth',lw,'MarkerSize',msz);
xlabel('time');
ylabel('theta');

figure
plot(tt(1:length(tt)-1), yyy(:,2), 'LineWidth',lw,'MarkerSize',msz);
xlabel('time');
ylabel('theta dot');

figure
plot(tt(1:length(tt)-1), yyy(:,3), 'LineWidth',lw,'MarkerSize',msz);
xlabel('time');
ylabel('x');

figure
plot(tt(1:length(tt)-1), yyy(:,4), 'LineWidth',lw,'MarkerSize',msz);
xlabel('time');
ylabel('x dot');

figure
plot(tt(1:length(tt)-1), u, 'LineWidth',lw,'MarkerSize',msz);
xlabel('time');
ylabel('Control input');






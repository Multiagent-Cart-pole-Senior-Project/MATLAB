% Dr. Jing Wang, Nov. 2020

% Control of Cart-pole systems

% In this example, we consider the linear consensus control of multiple
% cart-pole systems



% Nonlinear Model: 
% (M+m) \ddot{x} + ml\cos\theta \ddot\theta - ml\sin\theta (\dot\theta)^2 +
% b\dot{x} = F

% (I + ml^2) \ddot\theta + ml\cos\theta \ddot{x} = mgl\sin\theta 

% M = 0.5kg; Mass of the cart
% m = 0.2kg; Mass of the pendulum
% b = 0.1 N/(m/s); coefficient of friction of cart
% l = 0.3 m; length of pendulum center of mass
% I = 0.006 kg m^2; moment of inertia of the pendulum
% F: force applied to the cart
% theta: angular position of the pendulum
% x: horizontal position of the cart


% Leader-follower strategy 
clc;
clear;
close all;


% Model parameters
M = 0.5;
m = 0.2;
b = 0.1;
l = 0.3;
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
s3d = -5*zeta*omegan;
s4d = -8*zeta*omegan;

%s3d = -10*zeta*omegan;
%s4d = -20*zeta*omegan;


Sd =[s1d; s2d; s3d; s4d];
K = acker(A,B, Sd);
k1 = K(1);
k2 = K(2);
k3 = K(3);
k4 = K(4);

% Simulation time

T = 2; 
Ts=0.001; 
tt=0:Ts:T; 

% Leader 
xx0 = [0.01; 0 ; 0; 0];

% Adjacency matrix

Ad = [0 1 0; 1 0 1; 0 1 0];

% leader matrix 
Bd = [1 0 0]';



% agent 1
xx0_1 = [0.02; 0 ; 0; 0];

% agent 2
xx0_2 = [-0.01; 0 ; 0; 0];

% agent 3
xx0_3 = [0.03; 0 ; 0; 0];



% Control gain
c=3;
K2 = c*K;
K1 = K2;

for j=1:length(tt)-1
   ttt(j)=tt(j);
end

for k = 1:length(tt)-1
    
   fprintf('\n k = %i', k);
   
   % Leader 
   yyy0(k,:) = xx0';
   
   con0 = -K*xx0+2*sin(10*tt(k)); %+0.1*sin(10*tt(k));
   u0(k) = con0;  
   xx00 = [xx0; con0];
   [t, yy] = ode45('Cart_model', [tt(k) tt(k+1)], xx00);
   xx00=yy(length(t), :)';
   xx0=xx00(1:4);
   
   
   
   % Agent 1 
   yyy1(k,:) = xx0_1';
   
   con1 = K2*(Ad(1,2)*(xx0_2-xx0_1) + Ad(1,3)*(xx0_3-xx0_1))+K1*Bd(1)*(xx0-xx0_1);
   
   u1(k) = con1;
   xx01 = [xx0_1; con1];
   [t, yy1] = ode45('Cart_model', [tt(k) tt(k+1)], xx01);
   xx01=yy1(length(t), :)';
   xx0_1 = xx01(1:4);
   
   % Agent 2 
   yyy2(k,:) = xx0_2';
   con2 = K2*(Ad(2,1)*(xx0_1-xx0_2) + Ad(2,3)*(xx0_3-xx0_2)) +K1*Bd(2)*(xx0-xx0_2);
   u2(k) = con2;  
   xx02 = [xx0_2; con2];
   [t, yy2] = ode45('Cart_model', [tt(k) tt(k+1)], xx02);
   xx02=yy2(length(t), :)';
   xx0_2 = xx02(1:4);
   
   
   % Agent 3 
   yyy3(k,:) = xx0_3';
   con3 = K2*(Ad(3,1)*(xx0_1-xx0_3) + Ad(3,2)*(xx0_2-xx0_3)) +K1*Bd(3)*(xx0-xx0_3);
   u3(k) = con3; 
   xx03 = [xx0_3; con3];
   [t, yy3] = ode45('Cart_model', [tt(k) tt(k+1)], xx03);
   xx03=yy3(length(t), :)';
   xx0_3 = xx03(1:4);
   
end 

alw = 0.75;    % AxesLineWidth
fsz = 11;      % Fontsize
lw = 1.5;      % LineWidth 
msz = 8; 

figure
plot(tt(1:length(tt)-1), yyy0(:,1), tt(1:length(tt)-1), yyy1(:,1), tt(1:length(tt)-1), yyy2(:,1),tt(1:length(tt)-1), yyy3(:,1),'LineWidth',lw,'MarkerSize',msz);
xlabel('time');
ylabel('theta');

figure
plot(tt(1:length(tt)-1), yyy0(:,2), tt(1:length(tt)-1), yyy1(:,2),tt(1:length(tt)-1), yyy2(:,2),tt(1:length(tt)-1), yyy3(:,2), 'LineWidth',lw,'MarkerSize',msz);
xlabel('time');
ylabel('theta dot');

figure
plot(tt(1:length(tt)-1), yyy0(:,3), tt(1:length(tt)-1), yyy1(:,3),tt(1:length(tt)-1), yyy2(:,3),tt(1:length(tt)-1), yyy3(:,3),'LineWidth',lw,'MarkerSize',msz);
xlabel('time');
ylabel('x');

figure
plot(tt(1:length(tt)-1), yyy0(:,4), tt(1:length(tt)-1), yyy1(:,4),tt(1:length(tt)-1), yyy2(:,4),tt(1:length(tt)-1), yyy3(:,4),'LineWidth',lw,'MarkerSize',msz);
xlabel('time');
ylabel('x dot');

figure
plot(tt(1:length(tt)-1), u0,tt(1:length(tt)-1), u1,tt(1:length(tt)-1), u2,tt(1:length(tt)-1), u3, 'LineWidth',lw,'MarkerSize',msz);
xlabel('time');
ylabel('Control input');






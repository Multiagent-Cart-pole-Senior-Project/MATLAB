%% Multiagent Consensus of Cart-Pole Systems Using Deep Q-Learning %%
%% Large Radial Basis Function %%
% Author: Ryan Russell
% 
% Linear Control Algorithm Developed from: 
% "ECE 221 Project: Inverted Pendulum"
%    By: Dr. Jing Wang
% 
% Nonlinear Model Developed by: Dr. Wang

% State Variables:
% x1 = Theta
% x2 = Theta_dot
% x3 = x
% x4 = x_dot

close all
clear all
clc

%% PARAMETERS
M = 0.5; % [kg] - Mass of the Cart
m = 0.2; % [kg] - Mass of the pendulum
b = 0.1; % [N/(m/s)] - Coefficient of friction of the cart
l = 0.3; % [m] - Length of pendulum center of mass
g = 9.81; % [m/s^2] - Gravitational Acceleration Constant

MAX_CONTROL = 100; % Maximum Absolute value of Control Input

t0 = 0; % [s] - Start time
tf = 10; % [s] - End time
T = 0.01; % [s] - Sampling Time
t = t0:T:tf; % Time Vector

N = 3; % Number of Agents (excluding leader)

% Initial Conditions
x_0(:,1) = [0.01; 0; -0.1; 0]; % Initial Conditions - Agent 0 (Leader)
x_1(:,1) = [0.02; 0; 0; 0]; % Initial Conditions - Agent 1 (Follower)
x_2(:,1) = [0.03; 0; 0; 0]; % Initial Conditions - Agent 2 (Follower)
x_3(:,1) = [-0.01; 0; 0; 0]; % Initial Conditions - Agent 3 (Follower)

% States
xl(:,1) = x_0(:,1);
x(:,1,1) = x_1(:,1);
x(:,2,1) = x_2(:,1);
x(:,3,1) = x_3(:,1);

%% Linear Controller

% State Matrices
A = [0, 1, 0, 0; (g*(M+m))/(M*l), 0, 0, b/(M*l); 0, 0, 0, 1; -(m*g)/M, 0, 0, -b/M];
B = [0; -1/(M*l); 0; 1/M];
C = [1, 0, 0, 0; 0, 0, 1, 0];

% Desired Pole Locataions
zeta = 0.6;
w_n = 1;
pole_d = [-zeta*w_n+1i*w_n*sqrt(1-zeta^2); -zeta*w_n-1i*w_n*sqrt(1-zeta^2);...
                                                    -5*zeta*w_n; -8*zeta*w_n];                                                
% Linear Control Gain Vector
K = acker(A,B,pole_d);


%% Communication Matrices

% Communication with Leader
Bd = [1 0 0];

% Adjacency Matrix
Ad = [0 1 0; 1 0 1; 0 1 0];

% Diagonal Matrix
for i = 1:N
    d(i) = 0;
    for j = 1:N
        d(i) = d(i) + Ad(i,j);
    end
end

% Degree Matrix
Dd = diag(d);

% Laplacian Matrix
Ld = Dd - Ad;

% Left Eigenvector
one = [1;1;1];
DD = Ad*one; %1; d2 =2; d3=1;
d1 = DD(1); d2 = DD(2); d3 = DD(3);
II = eye(3);
D = diag(DD);
L = D-Ad;
F = (II -inv(II+D)*L);
[V, DD] = eig(F');
w1 = V(1,3)/(V(1,3)+V(2,3)+V(3,3));
w2 = V(2,3)/(V(1,3)+V(2,3)+V(3,3));
w3 = V(3,3)/(V(1,3)+V(2,3)+V(3,3));

w = [w1; w2; w3]; % Left Eigenvector


%% Deep Q-Learning Parameters
LEARNING_RATE = 0.01;

Gamma = [1, 0, 0, 0; 0, 0, 0, 0; 0, 0, 1, 0; 0, 0, 0, 0];
Lambda = 0.1*eye(N);
Nu = [0.8; 1; 0.08];

theta_1(:,1) = 1e-6*rand(6,1); % Initial NN weights (6x1)
theta_2(:,1) = 1e-6*rand(11,1); % Initial NN weights (11x1)
theta_3(:,1) = 1e-6*rand(6,1); % Initial NN weights (11x1)

%% Simulation (Discrete Time - Euler Integration)
k = 1; % Starting timestep
kf = tf/T; % Final timestep

% Initial Values
x_pos_0(1) = x_0(3,1);
Theta_0(1) = x_0(1,1);
x_pos_1(1) = x_1(3,1);
Theta_1(1) = x_1(1,1);
x_pos_2(1) = x_2(3,1);
Theta_2(1) = x_2(1,1);
x_pos_3(1) = x_3(3,1);
Theta_3(1) = x_3(1,1);

% Episode Loop
while k <= kf
    clc
    fprintf('%.4f \n', T*k)
    
    %% Agent actions and next state

    %% Determine Actions to Take
    u(:,k) = action(x,N,Ad,Nu,theta_1,theta_2,theta_3,k);

    % Agent 0 (Leader)
    u0_temp = -K*x_0(:,k) + 0.05*sin(0.5*T*k); % Control Input
    u_0(k) = sign(u0_temp)*min(MAX_CONTROL, abs(u0_temp)); % Limit Control Input from -1 to 1

    x0k = [x_0(:,k); u_0(k)];
    [tt, x0k1] = ode45('Cart_model', [t(k) t(k+1)], x0k);

    x_0(1,k+1) = x0k1(length(tt),1);
    x_0(2,k+1) = x0k1(length(tt),2);
    x_0(3,k+1) = x0k1(length(tt),3);
    x_0(4,k+1) = x0k1(length(tt),4);
    
    u0_temp = -K*x_0(:,k+1); % Control Input
    u_0(k+1) = sign(u0_temp)*min(MAX_CONTROL, abs(u0_temp)); % Limit Control Input from -1 to 1

    x_pos_0(k+1) = x_0(3,k+1);
    Theta_0(k+1) = x_0(1,k+1);

    % Agent 1
    u1_temp = u(1,k);
    u_1(k) = sign(u1_temp)*min(MAX_CONTROL, abs(u1_temp)); % Limit Control Input from -1 to 1

    x1k = [x_1(:,k); u_1(k)];
    [tt, x1k1] = ode45('Cart_model', [t(k) t(k+1)], x1k);

    x_1(1,k+1) = x1k1(length(tt),1);
    x_1(2,k+1) = x1k1(length(tt),2);
    x_1(3,k+1) = x1k1(length(tt),3);
    x_1(4,k+1) = x1k1(length(tt),4);

    x_pos_1(k+1) = x_1(3,k+1);
    Theta_1(k+1) = x_1(1,k+1);

    % Agent 2  
    u2_temp = u(2,k);
    u_2(k) = sign(u2_temp)*min(MAX_CONTROL, abs(u2_temp)); % Limit Control Input from -1 to 1

    x2k = [x_2(:,k); u_2(k)];
    [tt, x2k1] = ode45('Cart_model', [t(k) t(k+1)], x2k);

    x_2(1,k+1) = x2k1(length(tt),1);
    x_2(2,k+1) = x2k1(length(tt),2);
    x_2(3,k+1) = x2k1(length(tt),3);
    x_2(4,k+1) = x2k1(length(tt),4);

    x_pos_2(k+1) = x_2(3,k+1);
    Theta_2(k+1) = x_2(1,k+1);  

    % Agent 3  
    u3_temp = u(3,k);
    u_3(k) = sign(u3_temp)*min(MAX_CONTROL, abs(u3_temp)); % Limit Control Input from -1 to 1

    x3k = [x_3(:,k); u_3(k)];
    [tt, x3k1] = ode45('Cart_model', [t(k) t(k+1)], x3k);

    x_3(1,k+1) = x3k1(length(tt),1);
    x_3(2,k+1) = x3k1(length(tt),2);
    x_3(3,k+1) = x3k1(length(tt),3);
    x_3(4,k+1) = x3k1(length(tt),4);

    x_pos_3(k+1) = x_3(3,k+1);
    Theta_3(k+1) = x_3(1,k+1);  

    %% Put States and Actions in Matrices
    % States
    xl(:,k) = x_0(:,k);
    x(:,1,k) = x_1(:,k);
    x(:,2,k) = x_2(:,k);
    x(:,3,k) = x_3(:,k);

    xl(:,k+1) = x_0(:,k+1);
    x(:,1,k+1) = x_1(:,k+1);
    x(:,2,k+1) = x_2(:,k+1);
    x(:,3,k+1) = x_3(:,k+1);

    %% Get Rewards
    rewards(:,k) = reward(x,xl,u,Gamma,Lambda,N,Ad,Bd,k);

    %% Next Action
    theta_1(:,k+1) = theta_1(:,k);
    theta_2(:,k+1) = theta_2(:,k);
    theta_3(:,k+1) = theta_3(:,k);
    u(:,k+1) = action(x,N,Ad,Nu,theta_1,theta_2,theta_3,k+1);

    %% Get Phis
    [phi_1(:,k), phi_2(:,k), phi_3(:,k)] = Phi(x,u,N,Ad,Nu,k);
    [phi_1(:,k+1), phi_2(:,k+1), phi_3(:,k+1)] = Phi(x,u,N,Ad,Nu,k+1);

    %% Get Estimations
    if k == 1
        % R vector
        R(:,k) = rewards(:,k);
        % G vector
        G(:,k) = [sum(phi_1(:,k+1)'.*theta_1(:,k)'), sum(phi_2(:,k+1)'.*theta_2(:,k)'), sum(phi_3(:,k+1)'.*theta_3(:,k)')] ... 
            + transpose(u(:,k).^2)*Lambda;
        % P vector
        P(:,k) = [sum(phi_1(:,k)'.*theta_1(:,k)'), sum(phi_2(:,k)'.*theta_2(:,k)'), sum(phi_3(:,k)'.*theta_3(:,k)')] ...
            + transpose(u(:,k).^2)*Lambda;
    else
        [R(:,k), G(:,k), P(:,k)] = estimates(R,G,P,N,Ad,d,w,theta_1,theta_2,theta_3,rewards,phi_1,phi_2,phi_3,k);
    end

    %% Update Theta
    theta_1(:,k+1) = theta_1(:,k) + LEARNING_RATE * (R(1,k) + G(1,k) - P(1,k)).* phi_1(:,k);
    theta_2(:,k+1) = theta_2(:,k) + LEARNING_RATE * (R(2,k) + G(2,k) - P(2,k)).* phi_2(:,k);
    theta_3(:,k+1) = theta_3(:,k) + LEARNING_RATE * (R(3,k) + G(3,k) - P(3,k)).* phi_3(:,k);

    k = k + 1; % Update Timestep
end


%% PLOT RESULTS
figure
subplot(2,1,1)
plot(t,x_pos_0(1,:))
hold on 
plot(t,x_pos_1(1,:))
hold on 
plot(t,x_pos_2(1,:))
hold on 
plot(t,x_pos_3(1,:)) 
title('Cart Position (Episode 1)')
ylabel('Position [m]')
xlabel('Time [s]')
legend('Agent 0 (Leader)', 'Agent 1', 'Agent 2', 'Agent 3')
grid on

subplot(2,1,2)
plot(t,Theta_0(1,:))
hold on
plot(t,Theta_1(1,:))
hold on
plot(t,Theta_2(1,:))
hold on
plot(t,Theta_3(1,:))
title('Pole Angle (Episode 1)')
ylabel('Angle [rad]')
xlabel('Time [s]')
legend('Agent 0 (Leader)', 'Agent 1', 'Agent 2', 'Agent 3')
grid on

figure
plot(1:kf, rewards(1,:))
hold on 
plot(1:kf, rewards(2,:))
hold on
plot(1:kf, rewards(3,:))
title('Agent Rewards each Time Step')
ylabel('Reward')
xlabel('Time Step')
grid on

fprintf('FINISHED RUNNING')

% Reward Function
function rew = reward(x,xl,u,Gamma,Lambda,N,Ad,Bd,k)
    rew = zeros(N,1);
    for i = 1:N
        for j = 1:N
            rew(i) = rew(i) + Ad(i,j)*...
                transpose((x(:,i,k)-x(:,j,k)))*Gamma*...
                (x(:,i,k)-x(:,j,k));
        end
        rew(i) = rew(i) + Bd(i) * transpose((x(:,i,k)-xl(:,k)))*Gamma*...
                (x(:,i,k)-xl(:,k));
    end
    rew = rew + transpose(u(:,k))*Lambda*...
        u(:,k);
end

% Phi Function
function [phi_1, phi_2, phi_3] = Phi(x,u,N,Ad,Nu,k)
    phi_1 = zeros(6,1);
    phi_2 = zeros(11,1);
    phi_3 = zeros(6,1);
    
    % Agent 1
    phi_1(1,1) = norm(x(:,2,k) - x(:,1,k))^2;
    phi_1(2,1) = (x(1,1,k)-x(1,2,k))*u(1,k);
    phi_1(3,1) = (x(2,1,k)-x(2,2,k))*u(1,k);
    phi_1(4,1) = (x(3,1,k)-x(3,2,k))*u(1,k);
    phi_1(5,1) = (x(4,1,k)-x(4,2,k))*u(1,k);
    phi_1(6,1) = u(1,k)^2;
    
    % Agent 2
    phi_2(1,1) = (norm(x(:,2,k) - x(:,1,k))^2);
    phi_2(2,1) = (norm(x(:,3,k) - x(:,1,k))^2);
    phi_2(3,1) = ((x(1,2,k)-x(1,1,k))*u(2,k));
    phi_2(4,1) = ((x(2,2,k)-x(2,1,k))*u(2,k));
    phi_2(5,1) = ((x(3,2,k)-x(3,1,k))*u(2,k)); 
    phi_2(6,1) = ((x(4,2,k)-x(4,1,k))*u(2,k));
    phi_2(7,1) = ((x(1,2,k)-x(1,3,k))*u(2,k));
    phi_2(8,1) = ((x(2,2,k)-x(2,3,k))*u(2,k));
    phi_2(9,1) = ((x(3,2,k)-x(3,3,k))*u(2,k));
    phi_2(10,1) = ((x(4,2,k)-x(4,3,k))*u(2,k));
    phi_2(11,1) = u(2,k)^2;    
    
    % Agent 3
    phi_3(1,1) = norm(x(:,2,k) - x(:,3,k))^2;
    phi_3(2,1) = (x(1,3,k)-x(1,2,k))*u(3,k);
    phi_3(3,1) = (x(2,3,k)-x(2,2,k))*u(3,k);
    phi_3(4,1) = (x(3,3,k)-x(3,2,k))*u(3,k);
    phi_3(5,1) = (x(4,3,k)-x(4,2,k))*u(3,k);
    phi_3(6,1) = u(3,k)^2;        
end

% Estimation Functions
function [Rout, Gout, Pout] = estimates(R,G,P,N,Ad,d,w,theta_1,theta_2,theta_3,rew,phi_1,phi_2,phi_3,k)
    for i = 1:N
        R_sum(i) = 0;
        G_sum(i) = 0;
        P_sum(i) = 0;
        % Summations
        for j = 1:N
            R_sum(i) = R_sum(i) + Ad(i,j)*(R(j,k-1)-R(i,k-1));
            G_sum(i) = G_sum(i) + Ad(i,j)*(G(j,k-1)-G(i,k-1));
            P_sum(i) = P_sum(i) + Ad(i,j)*(P(j,k-1)-P(i,k-1));
        end
    end
    
    % Estimations
    Rout(1) = R(1,k-1) + (1/(1+d(1)))*R_sum(1) + (1/w(1))*(rew(1,k)-rew(1,k-1));
    Gout(1) = G(1,k-1) + (1/(1+d(1)))*G_sum(1) + (1/w(1))*((transpose(phi_1(:,k+1))*theta_1(:,k))-(transpose(phi_1(:,k))*theta_1(:,k-1)));
    Pout(1) = P(1,k-1) + (1/(1+d(1)))*P_sum(1) + (1/w(1))*((transpose(phi_1(:,k))*theta_1(:,k))-(transpose(phi_1(:,k-1))*theta_1(:,k-1)));

    Rout(2) = R(2,k-1) + (1/(1+d(2)))*R_sum(2) + (1/w(2))*(rew(2,k)-rew(2,k-1));
    Gout(2) = G(2,k-1) + (1/(1+d(2)))*G_sum(2) + (1/w(2))*((transpose(phi_2(:,k+1))*theta_2(:,k))-(transpose(phi_2(:,k))*theta_2(:,k-1)));
    Pout(2) = P(2,k-1) + (1/(1+d(2)))*P_sum(2) + (1/w(2))*((transpose(phi_2(:,k))*theta_2(:,k))-(transpose(phi_2(:,k-1))*theta_2(:,k-1)));
    
    Rout(3) = R(3,k-1) + (1/(1+d(3)))*R_sum(3) + (1/w(3))*(rew(3,k)-rew(3,k-1));
    Gout(3) = G(3,k-1) + (1/(1+d(3)))*G_sum(3) + (1/w(3))*((transpose(phi_3(:,k+1))*theta_3(:,k))-(transpose(phi_3(:,k))*theta_3(:,k-1)));
    Pout(3) = P(3,k-1) + (1/(1+d(3)))*P_sum(3) + (1/w(3))*((transpose(phi_3(:,k))*theta_3(:,k))-(transpose(phi_3(:,k-1))*theta_3(:,k-1)));
end

% Action function
function u = action(x,N,Ad,Nu,theta_1,theta_2,theta_3,k)
    u = zeros(N,1);
    
    % u1
    u(1) = (((x(1,2,k) - x(1,1,k))*theta_1(1,k)) + ((x(2,2,k) - x(2,1,k))*theta_1(2,k))...
        + ((x(3,2,k) - x(3,1,k))*theta_1(3,k)) + ((x(4,2,k) - x(4,1,k))*theta_1(4,k)))/...
        (2*theta_1(6,k));
    
    % u2
    sum1 = 0;
    sum2 = 0;
    for j = 1:4 
        sum1 = sum1 + ((x(j,1,k)-x(j,2,k))*theta_2(j+2,k)); 
        sum2 = sum2 + ((x(j,3,k)-x(j,2,k))*theta_2(j+6,k));
    end
    u(2) = (sum1 + sum2)/(2*theta_2(11,k));
    
    % u3
    u(3) = (((x(1,2,k) - x(1,3,k))*theta_3(1,k)) + ((x(2,2,k) - x(2,3,k))*theta_3(2,k))...
        + ((x(3,2,k) - x(3,3,k))*theta_3(3,k)) + ((x(4,2,k) - x(4,3,k))*theta_3(4,k)))/...
        (2*theta_3(6,k));    
    
end

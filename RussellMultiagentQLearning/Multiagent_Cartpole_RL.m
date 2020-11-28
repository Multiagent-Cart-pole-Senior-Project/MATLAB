%% Multiagent Consensus of Cart-Pole Systems Using Deep Q-Learning %%
% Author: Ryan Russell
% 
% Linear Control Algorithm Developed from: 
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

%% PARAMETERS
M = .5; % [kg] - Mass of the Cart
m = 0.1; % [kg] - Mass of the pendulum
b = 0.1; % [N/(m/s)] - Coefficient of friction of the cart
l = 0.5; % [m] - Length of pendulum center of mass
g = 9.81; % [m/s^2] - Gravitational Acceleration Constant

t0 = 0; % [s] - Start time
tf = 10; % [s] - End time
T = 0.01; % [s] - Sampling Time
t = t0:T:tf; % Time Vector

N = 3; % Number of Agents (excluding leader)

% Initial Conditions
x_0(:,1) = [0.11; 0; 2; 0]; % Initial Conditions - Agent 0 (Leader)
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
Bd = [1 1 1 1];

% Adjacency Matrix
Ad = [0 1 1; 1 0 1; 1 1 0];

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
P = eye(N) - inv(Dd)*Ld;
[v,D,w] = eig(P);


%% Deep Q-Learning Parameters
EPISODES = 20000;
LEARNING_RATE = 0.1;

Gamma = [1, 0, 0, 0; 0, 0, 0, 0; 0, 0, 1, 0; 0, 0, 0, 0];
Lambda = 0.1*eye(N);
Nu = [0.8; 1; 0.08];

theta(:,:,1) = 1e-6*rand(N,3); % Initial NN weights


%% Simulation (Discrete Time - Euler Integration)
for i = 1:EPISODES
    clc
    fprintf('%.4f \n', i/EPISODES)
    % Reset Every Episode
    k = 1; % Starting timestep
    kf = tf/T; % Final timestep
    
    % Reward Summations for each agent
    rewards_1 = 0;
    rewards_2 = 0;
    rewards_3 = 0;
    
    % Episode Loop
    while k <= kf +1
        
        %% Agent actions and next state
        
        %% Determine Actions to Take
        u(:,k) = action(x,N,Ad,Nu,theta,k);
        
        % Agent 0 (Leader)
        u0_temp = -K*x_0(:,k); % Control Input
        u_0(k) = sign(u0_temp)*min(1 , abs(u0_temp)); % Limit Control Input from -1 to 1
        u_0(k+1) = u_0(k);
        
        x_0(1,k+1) = x_0(1,k) + T * x_0(2,k);
        x_0(2,k+1) = x_0(2,k) + T * ((g*(M+m))/(M*l)*x_0(1,k) + b/(M*l)*x_0(4,k) - (1/(M*l))*u_0(k));
        x_0(3,k+1) = x_0(3,k) + T * x_0(4,k);
        x_0(4,k+1) = x_0(4,k) + T * (-((m*g)/M)*x_0(1,k) - (b/M)* x_0(4,k) + (1/M)*u_0(k));

        x_pos_0(k) = x_0(3,k);
        Theta_0(k) = x_0(1,k);

        % Agent 1
        u1_temp = u(1,k);
        u_1(k) = sign(u1_temp)*min(1 , abs(u1_temp)); % Limit Control Input from -1 to 1
        
        x_1(1,k+1) = x_1(1,k) + T * x_1(2,k);
        x_1(2,k+1) = x_1(2,k) + T * ((g*(M+m))/(M*l)*x_1(1,k) + b/(M*l)*x_1(4,k) - (1/(M*l))*u_1(k));
        x_1(3,k+1) = x_1(3,k) + T * x_1(4,k);
        x_1(4,k+1) = x_1(4,k) + T * (-((m*g)/M)*x_1(1,k) - (b/M)* x_1(4,k) + (1/M)*u_1(k));

        x_pos_1(k) = x_1(3,k);
        Theta_1(k) = x_1(1,k);

        % Agent 2  
        u2_temp = u(2,k);
        u_2(k) = sign(u2_temp)*min(1 , abs(u2_temp)); % Limit Control Input from -1 to 1
        
        x_2(1,k+1) = x_2(1,k) + T * x_2(2,k);
        x_2(2,k+1) = x_2(2,k) + T * ((g*(M+m))/(M*l)*x_2(1,k) + b/(M*l)*x_2(4,k) - (1/(M*l))*u_2(k));
        x_2(3,k+1) = x_2(3,k) + T * x_2(4,k);
        x_2(4,k+1) = x_2(4,k) + T * (-((m*g)/M)*x_2(1,k) - (b/M)* x_2(4,k) + (1/M)*u_2(k));

        x_pos_2(k) = x_2(3,k);
        Theta_2(k) = x_2(1,k);  

        % Agent 3  
        u3_temp = u(3,k);
        u_3(k) = sign(u3_temp)*min(1 , abs(u3_temp)); % Limit Control Input from -1 to 1

        x_3(1,k+1) = x_3(1,k) + T * x_3(2,k);
        x_3(2,k+1) = x_3(2,k) + T * ((g*(M+m))/(M*l)*x_3(1,k) + b/(M*l)*x_3(4,k) - (1/(M*l))*u_3(k));
        x_3(3,k+1) = x_3(3,k) + T * x_3(4,k);
        x_3(4,k+1) = x_3(4,k) + T * (-((m*g)/M)*x_3(1,k) - (b/M)* x_3(4,k) + (1/M)*u_3(k));

        x_pos_3(k) = x_3(3,k);
        Theta_3(k) = x_3(1,k);  
        
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
        rewards_1 = rewards(1);
        rewards_2 = rewards(2);
        rewards_3 = rewards(3);
                
        %% Next Action
        theta(:,:,k+1) = theta(:,:,k);
        u(:,k+1) = action(x,N,Ad,Nu,theta,k+1);
        
        %% Get Phis
        phi(:,:,k) = Phi(x,u,N,Ad,Nu,k);
        phi(:,:,k+1) = Phi(x,u,N,Ad,Nu,k+1);
        
        %% Get Estimations
        if k == 1
            R(:,k) = rewards(:,k);
            G(:,k) = sum(phi(:,:,k+1)'.*theta(:,:,k)',1) + transpose(u(:,k).^2)*Lambda;
            P(:,k) = sum(phi(:,:,k)'.*theta(:,:,k)',1) + transpose(u(:,k).^2)*Lambda;
        else
            [R(:,k), G(:,k), P(:,k)] = estimates(R,G,P,N,Ad,d,w,theta,rewards,phi,k);
        end
        
        %% Update Theta
        for agent = 1:N
            theta(:,agent,k+1) = theta(:,agent,k) + LEARNING_RATE * (R(agent,k) + G(agent,k) - P(agent,k)).* phi(:,agent,k);
        end
        
        k = k + 1; % Update Timestep
    end
    
    % Record reward Summations for each agent after each episode
    reward_sum(:,i) = [rewards_1; rewards_2; rewards_3];
end


%% PLOT RESULTS
figure
subplot(2,1,1)
plot(t,x_pos_0)
hold on 
plot(t,x_pos_1)
hold on 
plot(t,x_pos_2)
hold on 
plot(t,x_pos_3) 
title('Cart Position')
ylabel('Position [m]')
xlabel('Time [s]')
grid on

subplot(2,1,2)
plot(t,Theta_0)
hold on
plot(t,Theta_1)
hold on
plot(t,Theta_2)
hold on
plot(t,Theta_3)
title('Pole Angle')
ylabel('Angle [rad]')
xlabel('Time [s]')
grid on

figure
plot(1:EPISODES, reward_sum(1,:))
hold on 
plot(1:EPISODES, reward_sum(2,:))
hold on
plot(1:EPISODES, reward_sum(3,:))
title('Agent Rewards each Episode')
ylabel('Reward')
xlabel('Episode')
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
function phi = Phi(x,u,N,Ad,Nu,k)
    phi = zeros(3,N);
    
    for i = 1:N
        sum1 = 0;
        sum2 = 0;
        for j = 1:N
            sum1 = sum1 + Ad(i,j) * (norm(x(:,i,k)-x(:,j,k))^2);
            for n = 1:4
                sum2 = sum2 + Ad(i,j) * ((x(n,i,k) - x(n,j,k))*u(i,k));
            end
        end
        phi(1,i) = exp(-sum1/(Nu(1)^2))*sum1;
        phi(2,i) = exp(-sum1/(Nu(2)^2))*sum2;
        phi(3,i) = u(i,k)^2;
    end
    
end

% Estimation Functions
function [Rout, Gout, Pout] = estimates(R,G,P,N,Ad,d,w,theta,rew,phi,k)
    for i = 1:N
        R_sum = 0;
        G_sum = 0;
        P_sum = 0;
        % Summations
        for j = 1:N
            R_sum = R_sum + Ad(i,j)*(R(j,k-1)-R(i,k-1));
            G_sum = G_sum + Ad(i,j)*(G(j,k-1)-G(i,k-1));
            P_sum = P_sum + Ad(i,j)*(P(j,k-1)-P(i,k-1));
        end
        % Estimations
        Rout(i) = R(i,k-1) + (1/(1+d(i)))*R_sum + (1/w(i,i))*(rew(i,k)-rew(i,k-1));
        Gout(i) = G(i,k-1) + (1/(1+d(i)))*G_sum + (1/w(i,i))*((transpose(phi(:,i,k+1))*theta(:,i,k))-(transpose(phi(:,i,k))*theta(:,i,k-1)));
        Pout(i) = P(i,k-1) + (1/(1+d(i)))*P_sum + (1/w(i,i))*((transpose(phi(:,i,k))*theta(:,i,k))-(transpose(phi(:,i,k-1))*theta(:,i,k-1)));
    end
        
end

% Action function
function u = action(x,N,Ad,Nu,theta,k)
    u = zeros(N,1);
    
    for i = 1:N
        sum1 = 0;
        sum2 = 0;
        for j = 1:N
            sum1 = sum1 + Ad(i,j) * (norm(x(:,i,k)-x(:,j,k))^2);
            for n = 1:4
                sum2 = sum2 + Ad(i,j) * ((x(n,i,k) - x(n,j,k)));
            end
        end
        u(i) = ((theta(2,i,k)*exp(-sum1/(Nu(2)^2)))/(2*theta(3,i,k)))*sum2;
    end
end

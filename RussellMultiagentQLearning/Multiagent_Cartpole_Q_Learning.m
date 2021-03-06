%% Multiagent Consensus of Cart-Pole Systems Using Q-Learning %%
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

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
% PARAMETERS
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
M = .5; % [kg] - Mass of the Cart
m = 0.1; % [kg] - Mass of the pendulum
b = 0.1; % [N/(m/s)] - Coefficient of friction of the cart
l = 0.5; % [m] - Length of pendulum center of mass
g = 9.81; % [m/s^2] - Gravitational Acceleration Constant

t0 = 0; % [s] - Start time
tf = 10; % [s] - End time
T = 0.01; % [s] - Sampling Time
t = t0:T:tf; % Time Vector

N = 4; % Number of Agents (including leader)
NUM_states = 10; % Number of Discrete States
NUM_actions = 11; % Number of possible actions to take
low_guess = [-0.2; -0.2; -2.5; -0.6]; % Low Guesses for States
window_size = (abs(low_guess)*2)/NUM_states;
u_q = -1:0.2:1; % Vector of Possible Control Inputs
EPISODES = 10000;

% Initial Conditions
x_0(:,1) = [0.11; 0; 2; 0]; % Initial Conditions - Agent 0 (Leader)
x_1(:,1) = [0.02; 0; 1; 0]; % Initial Conditions - Agent 1 (Follower)
x_2(:,1) = [0.03; 0; -1; 0]; % Initial Conditions - Agent 2 (Follower)
x_3(:,1) = [-0.01; 0; 0; 0]; % Initial Conditions - Agent 3 (Follower)

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

% Adjacency Matrix
Ad = [0 0 0 0; 1 0 1 0; 0 1 0 1; 0 0 1 0];

% Diagonal Matrix
for i = 1:N
    d(i) = 0;
    for j = 1:N
        d(i) = d(i) + Ad(i,j);
    end
end

Dd = diag(d);

% Initialize Q-tables for each follower (random values from 0 to 1)

% Q-table 1
if d(2) ~= 0
    for i = 1:d(2)
        for j = 1:4
            if exist('Q_size1') 
                Q_size1(length(Q_size1)+1) = NUM_states;
            else
                Q_size1(1) = NUM_states;
            end
        end
    end
    Q_size1(length(Q_size1)+1) = NUM_actions;
end
Q_1 = randn(Q_size1);

% Q-table 2
if d(3) ~= 0
    for i = 1:d(3)
        for j = 1:4
            if exist('Q_size2') 
                Q_size2(length(Q_size2)+1) = NUM_states;
            else
                Q_size2(1) = NUM_states;
            end
        end  
    end
    Q_size2(length(Q_size2)+1) = NUM_actions;
end
Q_2 = randn(Q_size2);

% Q-table 3
if d(4) ~= 0
    for i = 1:d(4)
        for j = 1:4
            if exist('Q_size3') 
                Q_size3(length(Q_size3)+1) = NUM_states;
            else
                Q_size3(1) = NUM_states;
            end
        end
    end
    Q_size3(length(Q_size3)+1) = NUM_actions;
end
Q_3 = randn(Q_size3);

DISCOUNT = 0.99;
Learning_Rate = 0.1;


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
% Simulation (Discrete Time - Euler Integration)
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
for i = 1:EPISODES
    k = 1; % Starting timestep
    kf = tf/T; % Final timestep
    
    % Reward Summations for each agent
    rewards_1 = 0;
    rewards_2 = 0;
    rewards_3 = 0;
    
    while k <= kf +1

        % Agent 0
        u_temp = -K*x_0(:,k); % Control Input
        u_0(k) = sign(u_temp)*min(1 , abs(u_temp)); % Limit Control Input from -1 to 1

        x_0(1,k+1) = x_0(1,k) + T * x_0(2,k);
        x_0(2,k+1) = x_0(2,k) + T * ((g*(M+m))/(M*l)*x_0(1,k) + b/(M*l)*x_0(4,k) - (1/(M*l))*u_0(k));
        x_0(3,k+1) = x_0(3,k) + T * x_0(4,k);
        x_0(4,k+1) = x_0(4,k) + T * (-((m*g)/M)*x_0(1,k) - (b/M)* x_0(4,k) + (1/M)*u_0(k));

        x_pos_0(k) = x_0(3,k);
        Theta_0(k) = x_0(1,k);

        % Agent 1
        dis_1 = getDiscreteState(x_1(:,k),low_guess,window_size);
        [argvalue, argmax] = max(Q_1(dis_1(1),dis_1(2),dis_1(3),dis_1(4),:));
        [x, y, action] = ind2sub(Q_size1, argmax);
        u_1(k) = u_q(action);

        x_1(1,k+1) = x_1(1,k) + T * x_1(2,k);
        x_1(2,k+1) = x_1(2,k) + T * ((g*(M+m))/(M*l)*x_1(1,k) + b/(M*l)*x_1(4,k) - (1/(M*l))*u_1(k));
        x_1(3,k+1) = x_1(3,k) + T * x_1(4,k);
        x_1(4,k+1) = x_1(4,k) + T * (-((m*g)/M)*x_1(1,k) - (b/M)* x_1(4,k) + (1/M)*u_1(k));

        x_pos_1(k) = x_1(3,k);
        Theta_1(k) = x_1(1,k);

        reward = 100 + A(2,1)*(-2/d(2)*abs(x_0(1,k) - x_1(1,k)) - 10/d(2)*abs(x_0(3,k) - x_1(3,k))) +  ...
            A(2,2)*(-2/d(2)*abs(x_1(1,k) - x_1(1,k)) - 10/d(2)*abs(x_1(3,k) - x_1(3,k))) + ...
            A(2,3)*(-2/d(2)*abs(x_2(1,k) - x_1(1,k)) - 10/d(2)*abs(x_2(3,k) - x_1(3,k))) + ...
            A(2,4)*(-2/d(2)*abs(x_3(1,k) - x_1(1,k)) - 10/d(2)*abs(x_3(3,k) - x_1(3,k)));
        
        rewards_1 = rewards_1 + reward;

        dis_11 = getDiscreteState(x_1(:,k+1),low_guess,window_size);
        [argvalue1, argmax1] = max(Q_1(dis_11(1),dis_11(2),dis_11(3),dis_11(4),:));

        current_q = argvalue;
        max_future_q = argvalue1;
        new_q = (1-Learning_Rate) * current_q + Learning_Rate * (reward + DISCOUNT * max_future_q);
        Q_1(dis_1(1),dis_1(2),dis_1(3),dis_1(4),action) = new_q;

        % Agent 2
        dis_1 = getDiscreteState(x_2(:,k),low_guess,window_size);
        [argvalue, argmax] = max(Q_2(dis_1(1),dis_1(2),dis_1(3),dis_1(4),:));
        [x, y, action] = ind2sub(Q_size2, argmax);  
        u_2(k) = u_q(action);

        x_2(1,k+1) = x_2(1,k) + T * x_2(2,k);
        x_2(2,k+1) = x_2(2,k) + T * ((g*(M+m))/(M*l)*x_2(1,k) + b/(M*l)*x_2(4,k) - (1/(M*l))*u_2(k));
        x_2(3,k+1) = x_2(3,k) + T * x_2(4,k);
        x_2(4,k+1) = x_2(4,k) + T * (-((m*g)/M)*x_2(1,k) - (b/M)* x_2(4,k) + (1/M)*u_2(k));

        x_pos_2(k) = x_2(3,k);
        Theta_2(k) = x_2(1,k);  

        reward = 100 + A(3,1)*(-2/d(3)*abs(x_0(1,k) - x_2(1,k)) - 10/d(3)*abs(x_0(3,k) - x_2(3,k))) +  ...
            A(3,2)*(-2/d(3)*abs(x_1(1,k) - x_2(1,k)) - 10/d(3)*abs(x_1(3,k) - x_2(3,k))) + ...
            A(3,3)*(-2/d(3)*abs(x_2(1,k) - x_2(1,k)) - 10/d(3)*abs(x_2(3,k) - x_2(3,k))) + ...
            A(3,4)*(-2/d(3)*abs(x_3(1,k) - x_2(1,k)) - 10/d(3)*abs(x_3(3,k) - x_2(3,k)));
        
        rewards_2 = rewards_2 + reward;

        dis_11 = getDiscreteState(x_2(:,k+1),low_guess,window_size);
        [argvalue1, argmax1] = max(Q_2(dis_11(1),dis_11(2),dis_11(3),dis_11(4),:));

        current_q = argvalue;
        max_future_q = argvalue1;
        new_q = (1-Learning_Rate) * current_q + Learning_Rate * (reward + DISCOUNT * max_future_q);
        Q_2(dis_1(1),dis_1(2),dis_1(3),dis_1(4),action) = new_q;

        % Agent 3
        dis_1 = getDiscreteState(x_3(:,k),low_guess,window_size);
        [argvalue, argmax] = max(Q_3(dis_1(1),dis_1(2),dis_1(3),dis_1(4),:));
        [x, y, action] = ind2sub(Q_size3, argmax);   
        u_3(k) = u_q(action);

        x_3(1,k+1) = x_3(1,k) + T * x_3(2,k);
        x_3(2,k+1) = x_3(2,k) + T * ((g*(M+m))/(M*l)*x_3(1,k) + b/(M*l)*x_3(4,k) - (1/(M*l))*u_3(k));
        x_3(3,k+1) = x_3(3,k) + T * x_3(4,k);
        x_3(4,k+1) = x_3(4,k) + T * (-((m*g)/M)*x_3(1,k) - (b/M)* x_3(4,k) + (1/M)*u_3(k));

        x_pos_3(k) = x_3(3,k);
        Theta_3(k) = x_3(1,k);  

        reward = 100 + A(4,1)*(-2/d(4)*abs(x_0(1,k) - x_3(1,k)) - 10/d(4)*abs(x_0(3,k) - x_3(3,k))) +  ...
            A(4,2)*(-2/d(4)*abs(x_1(1,k) - x_3(1,k)) - 10/d(4)*abs(x_1(3,k) - x_3(3,k))) + ...
            A(4,3)*(-2/d(4)*abs(x_2(1,k) - x_3(1,k)) - 10/d(4)*abs(x_2(3,k) - x_3(3,k))) + ...
            A(4,4)*(-2/d(4)*abs(x_3(1,k) - x_3(1,k)) - 10/d(4)*abs(x_3(3,k) - x_3(3,k)));
        
        rewards_3 = rewards_3 + reward;

        dis_11 = getDiscreteState(x_3(:,k+1),low_guess,window_size);
        [argvalue1, argmax1] = max(Q_3(dis_11(1),dis_11(2),dis_11(3),dis_11(4),:));

        current_q = argvalue;
        max_future_q = argvalue1;
        new_q = (1-Learning_Rate) * current_q + Learning_Rate * (reward + DISCOUNT * max_future_q);
        Q_3(dis_1(1),dis_1(2),dis_1(3),dis_1(4),action) = new_q;

        k = k + 1; % Update Timestep
    end
    
    % Record reward Summations for each agent after each episode
    reward_sum(:,i) = [rewards_1; rewards_2; rewards_3];
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
% PLOT RESULTS
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

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


% Function to get discrete state of cart-pole
function disc = getDiscreteState(x,low_guess,window_size)
    disc = round((x-low_guess)./window_size) + 1;
end

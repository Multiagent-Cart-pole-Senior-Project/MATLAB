clc;
clear;
close all;

global L m M g dt N d sigma sigmal mu mul Q Ql U b A action_space

%% Inputs

% Number of Agents
N = 8;

% Time
dt = 0.01;
t = 0:dt:100;

% Initial States
x0 = [0.1, 0, 0.01, 0].*randn(N,4); % Agents
xl0 = [0.1, 0, 0.01, 0].*randn(1,4); % Leader

% Optimization Parameters
alpha0 = 1e-3; % inital learning rate
kappa = 1e-1; % decay
B1 = 0.9; % Adam first moment update
B2 = 0.99; % Adam second moment update
epsilon = 1e-7; % Adam divide by zero gradient correction

% Cart Parameters
L = 1; % pole
m = 0.5; % plumbob 
M = 2; % cart
g = 9.81; % gravity
action_space = [-1, 1]; % action limits

% Communication Topology
b = [1, 1, 1, 1, 1, 1, 1, 1]; % to leader
A = [0, 1, 1, 0, 0, 0, 1, 1;
    1, 0, 1, 1, 0, 0, 0, 1;
    1, 1, 0, 1, 1, 0, 0, 0;
    0, 1, 1, 0, 1, 1, 0, 0;
    0, 0, 1, 1, 0, 1, 1, 0;
    0, 0, 0, 1, 1, 0, 1, 1;
    1, 0, 0, 0, 1, 1, 0, 1;
    1, 1, 0, 0, 0, 1, 1, 0]; % betweeen agent
A_ = ones(N) - eye(N); % full comm for comparison

% Cost Parameters
Q = [0, 0, 0, 0].*eye(4); % Consensus
Ql = [1, 0, 1, 0].*eye(4); % to Leader
U = 1e-1*eye(N); % Control

% Radial Basis Function (Phi(i) = phi'*theta + U(i,i)*u(i)^2)
theta0 = 1e-6*rand(N,16); % Initial weights
sigma = [1, 2, 1, 0.1].*ones(N,4); % consensus std dev
sigmal = [0.8, 1, 0.08, 1].*ones(N,4); % leader std dev
mu = zeros(N,4); % consensus mean
mul = zeros(N,4); % leader mean


%% Main Simulation

% Allocate memory and fill ICs
x = zeros(length(t),N,4);
x(1,:,:) = x0;
xl = zeros(length(t),4);
xl(1,:) = xl0;
v = zeros(length(t),N,N);
v(1,:,:) = eye(N);
r = zeros(length(t),N);
u = zeros(length(t),N);
theta = zeros(length(t),N,length(theta0(1,:)));
theta(1,:,:) = theta0;
R = zeros(length(t)-1,N);
G = zeros(length(t)-1,N);
P = zeros(length(t)-1,N);
G_compare = zeros(length(t)-1,N);
P_compare = zeros(length(t)-1,N);
alpha = zeros(1,length(t)-1);

% Find stable linear control
K = linear_control([-0.8, -0.6, -0.4, -0.7]);

% Find adjacencies
adj = find_adjacencies(A);
adj_ = find_adjacencies(A_);

% True Left eigenvector
d = sum(A,2);
D = diag(d);
La = D - A;
F = eye(N) - (eye(N) + D)\La;
[V,W] = eig(F');
[~,w_id] = min((diag(W)-1).^2);
v_true = V(:,w_id)';
v_true = abs(v_true);
v_true = v_true/sum(v_true);

% Adam Initialization
mt = 0;
vt = 0;

for i = 1:length(t)-1
    clc;
    disp(i/length(t));
    
    % Left eigenvector
    v(i+1,:,:) = left_eig(v(i,:,:),adj);
    
    % Learning Rate
    alpha(i) = alpha0*exp(-kappa*dt*(i-1));
    
    % Leader Dynamics
    ul = -xl(i,:)*K' + 0.2*sin(2*dt*i/pi);
    xl(i+1,:) = h(xl(i,:),ul);
    
    % Action
    u(i,:) = action(x(i,:,:),xl(i,:),adj,theta(i,:,:));
    
    % Cost
    r(i,:) = cost(x(i,:,:),u(i,:),xl(i,:),adj);
    
    % RBF Vector
    if i > 1
        phi_prev = phi; % store old one
    end
    phi = Phi(x(i,:,:),u(i,:),xl(i,:),adj);
    
    % Agent Dynamics
    for j = 1:N
        x_ = squeeze(x(i,j,:))';
        u_ = -x_*K' + u(i,j);
        x(i+1,j,:) = f(x_,u_);
    end
    
    % Next State Control, Cost, and RBF Vector
    u(i+1,:) = action(x(i+1,:,:),xl(i+1,:),adj,theta(i,:,:));
    r(i+1,:) = cost(x(i+1,:,:),u(i+1,:),xl(i+1,:),adj);
    phi_next = Phi(x(i+1,:,:),u(i+1,:),xl(i+1,:),adj);
    
    % Fill Initial Estimate Values and Estimate after i = 1
    if i == 1
        R(i,:) = r(i,:);
        G(i,:) = sum(phi_next'.*squeeze(theta(i,:,:))',1) + u(i,:).^2*U;
        P(i,:) = sum(phi'.*squeeze(theta(i,:,:))',1) + u(i,:).^2*U;
    else
        [R(i,:), G(i,:), P(i,:)] = estimate(R(i-1,:),G(i-1,:),P(i-1,:),...
            r(i-1:i,:),u(i-1:i+1,:),phi_prev,phi,phi_next,...
            theta(i-1:i,:,:),adj,v(i,:,:));
    end
    
    % Update RBF weights (Adam optimization)
    gt = -(R(i,:) + G(i,:) - P(i,:))'.*phi; % regular gradient
    mt = B1*mt + (1 - B1)*gt; % first moment update
    vt = B2*vt + (1 - B2)*gt.^2; % second moment update
    mth = mt/(1 - B1^i);
    vth = vt/(1 - B2^i);
    theta(i+1,:,:) = squeeze(theta(i,:,:)) + alpha(i)*mth./(sqrt(vth)+eps);
    % note: Adam derives its step with opposite step direction (+ instead
    % of -)
    
    % Evaluate Comparison values (fully connected communication)
    phi_total = Phi(x(i,:,:),u(i,:),xl(i,:),adj_);
    phi_next_total = Phi(x(i+1,:,:),u(i+1,:),xl(i+1,:),adj_);
    G_compare(i,:) = sum(phi_next_total'.*squeeze(theta(i,:,:))',1)...
                                                            + u(i,:).^2*U;
    P_compare(i,:) = sum(phi_total'.*squeeze(theta(i,:,:))',1) + ...
                                                              u(i,:).^2*U;
end

%% Plotting
figure(1);
subplot(421);
hold on;
plot(t,xl(:,1),'r--');
for i = 1:N
    plot(t,squeeze(x(:,i,1)));
end
hold off;
xlabel('$time, s$','Interpreter','latex');
ylabel('$x$','Interpreter','latex');
set(gca,'fontname','Times New Roman');
subplot(423);
hold on;
plot(t,xl(:,2),'r--');
for i = 1:N
    plot(t,squeeze(x(:,i,2)));
end
hold off;
xlabel('$time, s$','Interpreter','latex');
ylabel('$\dot{x}$','Interpreter','latex');
set(gca,'fontname','Times New Roman');
subplot(422);
hold on;
plot(t,xl(:,3),'r--');
for i = 1:N
    plot(t,squeeze(x(:,i,3)));
end
hold off;
xlabel('$time, s$','Interpreter','latex');
ylabel('$\vartheta$','Interpreter','latex');
set(gca,'fontname','Times New Roman');
subplot(424);
hold on;
plot(t,xl(:,4),'r--');
for i = 1:N
    plot(t,squeeze(x(:,i,4)));
end
hold off;
xlabel('$time, s$','Interpreter','latex');
ylabel('$\dot{\vartheta}$','Interpreter','latex');
set(gca,'fontname','Times New Roman');
subplot(426);
hold on;
plot(t,sum(r,2),'r--');
for i = 1:N
    plot(t(1:end-1),R(:,i));
end
hold off;
xlabel('$time, s$','Interpreter','latex');
ylabel('$\sum_{i} r_i$','Interpreter','latex');
set(gca,'fontname','Times New Roman');
subplot(428);
hold on;
plot(t(1:end-1),sum(G_compare,2),'r--');
for i = 1:N
    plot(t(1:end-1),G(:,i));
end
hold off;
xlabel('$time, s$','Interpreter','latex');
ylabel('$\sum_{i} \Phi_i(x,u)$','Interpreter','latex');
set(gca,'fontname','Times New Roman');
subplot(425);
hold on;
for i = 1:N
    plot(t,u(:,i));
end
xlabel('$time, s$','Interpreter','latex');
ylabel('$u$','Interpreter','latex');
set(gca,'fontname','Times New Roman');
subplot(427);
hold on;
for i = 1:N
    plot(t,sum(squeeze(theta(:,i,:)),2));
end
xlabel('$time, s$','Interpreter','latex');
ylabel('$\sum \theta_i\; \forall\; i \in \{1,\cdots,N\}$','Interpreter','latex');
set(gca,'fontname','Times New Roman');

% % Left eigenvector and Learning rate
% figure(2);
% subplot(211);
% hold on;
% plot([0, t(end)],[v_true', v_true'],'r--','linewidth',1.5);
% for i = 1:N
%     plot(t,squeeze(v(:,i,i)));
% end
% hold off;
% xlabel('$time, s$','Interpreter','latex');
% ylabel('$v$','Interpreter','latex');
% set(gca,'fontname','Times New Roman');
% subplot(212);
% plot(t(1:end-1),alpha);
% xlabel('$time, s$','Interpreter','latex');
% ylabel('$\alpha$','Interpreter','latex');
% set(gca,'fontname','Times New Roman');
%% Functions

% Cart-Pole Dynamics
function xp = f(x,u)
    global L m M g dt
    xddot = (u - m*sin(x(3))*(L*x(4)^2 + g*cos(x(3))))...
            /(M + m*(1 + cos(x(3))^2));
    thetaddot = (cos(x(3))*xddot + g*sin(x(3)))/L;
    
    xdot = [x(2), xddot, x(4), thetaddot];
    
    xp = x + dt*xdot;
end

% Linear Dynamics (linearized around x = [0 0 0 0])
function xp = h(x,u)
    global L m M g dt
    A = [0, 1, 0, 0;
        0, 0, -m*g/(M+2*m), 0;
        0, 0, 0, 1;
        0, 0, -g/L*(m/(M+2*m)-1), 0];
    
    B = [0, 1/(M+2*m), 0, 1/L/(M+2*m)];
    
    xdot = x*A' + u*B;
    
    xp = x + dt*xdot;
end

% Stable Linear Control
function K = linear_control(p)
    global L m M g
    A = [0, 1, 0, 0;
        0, 0, -m*g/(M+2*m), 0;
        0, 0, 0, 1;
        0, 0, -g/L*(m/(M+2*m)-1), 0];
    
    B = [0, 1/(M+2*m), 0, 1/L/(M+2*m)]';

    K = place(A, B, p);
end

% Find Adjacencies
function adj = find_adjacencies(A)
    global N
    adj = {};
    for i = 1:N
        adj{i} = find(A(i,:));
    end
end

% Left eigenvector estimate
function vp = left_eig(v,adj)
    global N d
    v = squeeze(v);
    vp = zeros(N);
    for i = 1:N
        ids = adj{i};
        for j = 1:length(ids)
            vp(i,:) = vp(i,:) + 1/(1+d(i))*(v(ids(j),:) - v(i,:));
        end
        vp(i,:) = vp(i,:) + v(i,:);
    end
end

% Radial Basis function vector
function z = Phi(x,u,xl,adj)
    global N sigma sigmal Q Ql mu mul b
    z = zeros(N,16);
    x = squeeze(x);
    for i = 1:N
        ids = adj{i};
        sum1 = zeros(1,4);
        sum2 = zeros(1,4);
        for j = 1:length(ids)
            sum1 = sum1 + (x(i,:) - x(ids(j),:)).^2;
            sum2 = sum2 + x(i,:) - x(ids(j),:);
        end
        
        sum3 = b(i)*(x(i,:) - xl).^2;
        sum4 = b(i)*(x(i,:) - xl);
        
        prod = gaussian(sum2,sigma(i,:),Q,mu(i,:));
        prod1 = gaussian(sum4,sigmal(i,:),Ql,mul(i,:));
        
        z(i,:) = [sum3.*prod1, u(i)*sum4.*prod1, sum1.*prod,...
                                    u(i)*sum2.*prod];
    end
end

% Find Action
function u = action(x,xl,adj,theta)
    global N b sigma sigmal Q Ql mu mul U action_space
    x = squeeze(x);
    theta = squeeze(theta);
    u = zeros(1,N);
    for i = 1:N
        ids = adj{i};
        sum2 = zeros(1,4);
        for j = 1:length(ids)
            sum2 = sum2 + x(i,:) - x(ids(j),:);
        end
        
        sum4 = b(i)*(x(i,:) - xl);
        
        prod = gaussian(sum2,sigma(i,:),Q,mu(i,:));
        prod1 = gaussian(sum4,sigmal(i,:),Ql,mul(i,:));
        
        u(i) = -1/2*sum(sum4.*prod1.*theta(i,5:8) + ...
                                sum2.*prod.*theta(i,13:16))/U(i,i);
        u(i) = max(min(u(i),action_space(2)),action_space(1));
    end
end
% Gaussian curve
function z = gaussian(x,sig,Q,mu)
    z = exp(-(x.*x - mu)./sig.^2)*Q;
end

% Cost Function
function r = cost(x,u,xl,adj)
    global N Q Ql U b
    x = squeeze(x);
    r = zeros(1,N);
    for i = 1:N
        ids = adj{i};
        for j = 1:length(ids)
            r(i) = r(i) + (x(i,:) - x(ids(j),:))*Q*(x(i,:) - x(ids(j),:))';
        end
        r(i) = r(i) + b(i)*(x(i,:) - xl)*Ql*(x(i,:) - xl)';
    end
    r = r + u.^2*U;
end

% Estimation
function [R, G, P] = estimate(R,G,P,r,u,phi_prev,phi,phi_next,theta,adj,w)
    global N A d U
    w = squeeze(w);
    for i = 1:N
        ids = adj{i};
        R_sum = 0;
        G_sum = 0;
        P_sum = 0;
        for j = 1:length(ids)
            R_sum = R_sum + A(i,j)*(R(j) - R(i));
            G_sum = G_sum + A(i,j)*(G(j) - G(i));
            P_sum = P_sum + A(i,j)*(P(j) - P(i));
        end
        
        R(i) = R(i) + 1/(1+d(i))*R_sum + 1/w(i,i)*(r(2,i) - r(1,i));
        G(i) = G(i) + 1/(1+d(i))*G_sum + 1/w(i,i)*(phi_next(i,:)...
            *squeeze(theta(2,i,:)) - phi(i,:)*squeeze(theta(1,i,:))+ ...
            U(i,i)*(u(3,i).^2 - u(2,i).^2));
        P(i) = P(i) + 1/(1+d(i))*P_sum + 1/w(i,i)*(phi(i,:)*...
            squeeze(theta(2,i,:))- phi_prev(i,:)*squeeze(theta(1,i,:))+ ...
             U(i,i)*(u(2,i).^2 - u(1,i).^2));
    end
end




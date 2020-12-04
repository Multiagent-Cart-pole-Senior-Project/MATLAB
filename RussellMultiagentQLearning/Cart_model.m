% Nonlinear Model of cart-pole
% Jing Wang, Nov. 2020

function xdot = Cart_model(t, xx)

% Model parameters
M = 0.5;
m = 0.2;
b = 0.1;
l = 0.3;
% I = 0.006;
I =0;
g = 9.8; 


%global control;

control = xx(5);
xdot(1) = xx(2);

% Nonlinear one
xdot(2) = (-m^2*l^2*cos(xx(1))*sin(xx(1))*xx(2)^2 + b*m*l*cos(xx(1))*xx(4) + m*g*l*sin(xx(1))*(M+m) ...
    -m*l*cos(xx(1))*control)/((M+m)*I + M*m*l^2+m^2*l^2*(1-(cos(xx(1)))^2));

% linear one
%xdot(2) = (b*m*l*xx(4) + m*g*l*(M+m)*xx(1) - m*l*control)/((M+m)*I + M*m*l^2);

xdot(3) = xx(4);

% Nonlinear one
xdot(4) = (m*l*sin(xx(1))*(I+m*l^2)*xx(2)^2-b*(I+m*l^2)*xx(4) - m^2*l^2*g*sin(xx(1))*cos(xx(1)) ...
    + (I+m*l^2)*control)/((M+m)*I + M*m*l^2+m^2*l^2*(1-(cos(xx(1)))^2));

% linear one
%xdot(4) = (-b*(I+m*l^2)*xx(4) - m^2*l^2*g*(xx(1)) + (I+m*l^2)*control)/((M+m)*I + M*m*l^2);

xdot(5) = 0;

xdot = xdot';

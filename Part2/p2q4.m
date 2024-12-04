% Initialize Parameters
clear; close all; clc;

% Time and system parameters
dt = 0.1;              % Sampling time
time = 0:dt:10;        % Simulation time
num_steps = length(time);
vg = 2;                % UGV velocity [m/s]
phi_g = -pi/18;        % UGV steering angle [rad]
va = 12;               % UAV velocity [m/s]
wa = pi/25;            % UAV angular velocity [rad/s]
L = 0.5;               % UGV wheelbase [m]

% State and measurement dimensions
n = 6;  % State dimension (xi_g, eta_g, theta_g, xi_a, eta_a, theta_a)
p = 5;  % Measurement dimension

% Covariance Matrices
Q_actual = diag([0.05, 0.05, 0.005, 0.05, 0.05, 0.005]); % True process noise
Q_KF = diag([0.1, 0.1, 0.01, 0.1, 0.1, 0.01]);          % KF guessed process noise
R_actual = diag([0.1, 0.01, 0.1, 0.1, 0.1]);            % True measurement noise
R_KF = R_actual;                                        % KF uses accurate R

% Initial covariance and perturbation
P_0 = diag([0.5, 0.5, 0.05, 0.5, 0.5, 0.05]);
perturbation = [0.1; 0.1; 0.05; 0.1; 0.1; 0.05];

% Define Nominal State Trajectory
xi_g_nom = @(t) (1 / (2 * tan(pi / 18))) * (20 * tan(pi / 18) + 1 - cos(4 * tan(phi_g) * t));
eta_g_nom = @(t) (1 / (2 * tan(pi / 18))) * sin(4 * tan(phi_g) * t);
theta_g_nom = @(t) pi / 2 + 4 * tan(phi_g) * t;
xi_a_nom = @(t) (1 / pi) * (300 - 60 * t - 300 * cos(pi / 25 * t));
eta_a_nom = @(t) (300 / pi) * sin(pi / 25 * t);
theta_a_nom = @(t) wrapToPi(-pi / 2 + pi / 25 * t);

nominal_state = @(t) [xi_g_nom(t); eta_g_nom(t); theta_g_nom(t); ...
                      xi_a_nom(t); eta_a_nom(t); theta_a_nom(t)];

% Nonlinear dynamics
f = @(x) [vg * cos(x(3));
          vg * sin(x(3));
          (vg / L) * tan(phi_g);
          va * cos(x(6));
          va * sin(x(6));
          wa];

% Nonlinear measurement model
h = @(x) [wrapToPi(atan2(x(5) - x(2), x(4) - x(1)) - x(3));  % Relative bearing UAV from UGV
          wrapToPi(atan2(x(2) - x(5), x(1) - x(4)) - x(6));  % Relative bearing UGV from UAV
          sqrt((x(5) - x(2))^2 + (x(4) - x(1))^2);           % Range between UAV and UGV
          x(4);                                              % UAV x-position (GPS)
          x(5)];                                             % UAV y-position (GPS)

% Jacobians
A = @(x) [0, 0, -vg*sin(x(3)), 0, 0, 0;
          0, 0,  vg*cos(x(3)), 0, 0, 0;
          0, 0,  0,             0, 0, 0;
          0, 0,  0,             0, 0, -va*sin(x(6));
          0, 0,  0,             0, 0,  va*cos(x(6));
          0, 0,  0,             0, 0,  0];

H = @(x) [-(x(5) - x(2)) / ((x(5) - x(2))^2 + (x(4) - x(1))^2), ...
           (x(4) - x(1)) / ((x(5) - x(2))^2 + (x(4) - x(1))^2), -1, ...
           (x(5) - x(2)) / ((x(5) - x(2))^2 + (x(4) - x(1))^2), ...
          -(x(4) - x(1)) / ((x(5) - x(2))^2 + (x(4) - x(1))^2), 0;
           (x(4) - x(1)) / ((x(5) - x(2))^2 + (x(4) - x(1))^2), ...
           (x(5) - x(2)) / ((x(5) - x(2))^2 + (x(4) - x(1))^2), 0, ...
          -(x(4) - x(1)) / ((x(5) - x(2))^2 + (x(4) - x(1))^2), ...
          -(x(5) - x(2)) / ((x(5) - x(2))^2 + (x(4) - x(1))^2), -1;
          (x(4) - x(1)) / sqrt((x(5) - x(2))^2 + (x(4) - x(1))^2), ...
          (x(5) - x(2)) / sqrt((x(5) - x(2))^2 + (x(4) - x(1))^2), 0, ...
          -(x(4) - x(1)) / sqrt((x(5) - x(2))^2 + (x(4) - x(1))^2), ...
          -(x(5) - x(2)) / sqrt((x(5) - x(2))^2 + (x(4) - x(1))^2), 0;
           0, 0, 0, 1, 0, 0;
           0, 0, 0, 0, 1, 0];

function [truth_history, measurements] = simulate_trajectory(time, Q_actual, R_actual, nominal_state, f, h, perturbation)

    % Dimensions
    n = size(Q_actual, 1); % State dimension
    p = size(R_actual, 1); % Measurement dimension
    num_steps = length(time); % Number of time steps

    % Initialize storage
    truth_history = zeros(n, num_steps);
    measurements = zeros(p, num_steps);

    % Initial true state
    truth = nominal_state(0) + perturbation;
    truth_history(:, 1) = truth;

    % Simulate trajectory
    for k = 1:num_steps - 1
        % Ground Truth Propagation
        process_noise = mvnrnd(zeros(n, 1), Q_actual)';
        truth = truth + (time(k + 1) - time(k)) * f(truth) + process_noise;
        truth_history(:, k + 1) = truth;

        % Generate Noisy Measurement
        measurement_noise = mvnrnd(zeros(p, 1), R_actual)';
        measurements(:, k + 1) = h(truth) + measurement_noise;
    end
end

% Main Script
[truth_history, measurements] = simulate_trajectory(time, Q_actual, R_actual, nominal_state, f, h, perturbation);

% Initialize NEES and NIS
N = 400; % Number of Monte Carlo runs
nees = zeros(num_steps - 1, N);
nis = zeros(num_steps - 1, N);

% Monte Carlo Loop
for j = 1:N
    % Initialize the state estimate and covariance
    x_hat = nominal_state(0) + perturbation; % Initial state estimate
    P = P_0; % Initial covariance

    % Loop through all time steps
    for k = 1:num_steps - 1
        % Ground Truth at time step k
        truth = truth_history(:, k);

        % Measurement at time step k
        measurement = measurements(:, k);

        % --- Prediction Step ---
        F = eye(n) + dt * A(x_hat); % Linearized state transition matrix
        x_hat = x_hat + dt * f(x_hat); % Predicted state estimate
        P = F * P * F' + Q_KF; % Predicted covariance

        % --- Measurement Update ---
        H_k = H(x_hat); % Linearized measurement matrix
        innovation = measurement - h(x_hat); % Innovation (measurement residual)
        S = H_k * P * H_k' + R_KF; % Innovation covariance
        K = P * H_k' / S; % Kalman gain
        x_hat = x_hat + K * innovation; % Updated state estimate
        P = (eye(n) - K * H_k) * P; % Updated covariance

        % --- NEES and NIS Calculation ---
        state_error = truth - x_hat; % State estimation error
        nees(k, j) = state_error' / P * state_error; % NEES statistic
        nis(k, j) = innovation' / S * innovation; % NIS statistic
    end
end

% Average NEES and NIS
mean_nees = mean(nees, 2);
mean_nis = mean(nis, 2);

% Confidence Bounds
alpha = 0.05;
nees_bounds = [chi2inv(alpha / 2, n), chi2inv(1 - alpha / 2, n)] / N;
nis_bounds = [chi2inv(alpha / 2, p), chi2inv(1 - alpha / 2, p)] / N;

% Plot NEES
figure;
plot(time(1:num_steps - 1), mean_nees, 'b', 'LineWidth', 1.5);
hold on;
yline(nees_bounds(1), 'r--', 'Lower Bound');
yline(nees_bounds(2), 'r--', 'Upper Bound');
title('NEES Test');
xlabel('Time [s]');
ylabel('NEES');
grid on;

% Plot NIS
figure;
plot(time(1:num_steps - 1), mean_nis, 'b', 'LineWidth', 1.5);
hold on;
yline(nis_bounds(1), 'r--', 'Lower Bound');
yline(nis_bounds(2), 'r--', 'Upper Bound');
title('NIS Test');
xlabel('Time [s]');
ylabel('NIS');
grid on;

function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices. 
% 
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% Setup some useful variables
m = size(X, 1);
         
% You need to return the following variables correctly 
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

% Expand 'y' output into a matrix of single values
eye_matrix = eye(num_labels);
y_matrix = eye_matrix(y, :);

%%%%%%%%%%%%%%%%%%%%%%%%%% FEEDFORWARD %%%%%%%%%%%%%%%%%%%%%%%%%%%
%======================= Input to Hidden =========================
% Add ones to the input layer
a1 = [ones(m, 1) X] % 3 * m;

% Combine first set of weights and values
z2 = a1 * Theta1' % 4 * m

% Calculate the activation
a2 = sigmoid(z2) % 4 * m;

%======================= Hidden to Output ========================
% Add ones to the previous activation layer
a2 = [ones(m, 1) a2] % 5 * m;

% Combine with the second set of weights
z3 = a2 * Theta2' % 16 x 4;

% Calculate the second activation layer
a3 = sigmoid(z3) % 16 x 4;

% Compute non regularized cost
J = (1/m) * sum(sum(-y_matrix .* log(a3) - (1 - y_matrix) .* log(1 - a3)));

% Compute the regularized error
% From ex2 costFunctionReg.m
% Make sure to remove the bias terms from the 1st columns
reg_error = (lambda/(2*m)) * (sum(sum((Theta1(:,2:end)).^2)) + sum(sum((Theta2(:,2:end)).^2)));

% Add regularization error to cost
J = J + reg_error;

% %%%%%%%%%%%%%%%%%%%%%%%% BACKPROPAGATE %%%%%%%%%%%%%%%%%%%%%%%%%%%
%====================== Output to Hidden =========================
% Compute first sigma term at output layer
sigma_3 = a3 - y_matrix; % 16x4

sigma_2 = (sigma_3 * Theta2) .* [ones(m, 1) sigmoidGradient(z2)]; 
sigma_2 = sigma_2(:, 2:end);

delta_1 = (sigma_2' * a1);
delta_2 = (sigma_3' * a2);

p1 = (lambda/m)*[zeros(size(Theta1, 1), 1) Theta1(:, 2:end)];
p2 = (lambda/m)*[zeros(size(Theta2, 1), 1) Theta2(:, 2:end)];
Theta1_grad = delta_1./m + p1;
Theta2_grad = delta_2./m + p2;

% %====================== Hidden to Input ==========================
% % Compute 2nd delta term making sure to skip the first column of sigma
% sigma_2 = (sigma_3 * Theta2(:, 2:end)) .* ...
%     sigmoidGradient(z2); % 16x4
% 
% Theta1_grad = (1/m) * (sigma_3' * a2);
% Theta2_grad = (1/m) * (sigma_2' * a1);

% =========================================================================
% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];
end

function [J, grad] = linearRegCostFunction(X, y, theta, lambda)
%LINEARREGCOSTFUNCTION Compute cost and gradient for regularized linear 
%regression with multiple variables
%   [J, grad] = LINEARREGCOSTFUNCTION(X, y, theta, lambda) computes the 
%   cost of using theta as the parameter for linear regression to fit the 
%   data points in X and y. Returns the cost in J and the gradient in grad

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% Calculate hypothesis
H = X * theta;
% Compute J with regularization
J = (1 / (2 *m)) * sum((H - y).^2) + ... 
    (lambda / (2 * m) * sum(theta(2:end).^2));

% Compute the gradient
% Need to put grad(1) or else it crashes
grad(1)= X(:,1)' * (H - y) / m; 
grad(2:end) = X(:,2:end)' * (H - y) / m + theta(2:end) * lambda / m;

% =========================================================================
grad = grad(:);

end

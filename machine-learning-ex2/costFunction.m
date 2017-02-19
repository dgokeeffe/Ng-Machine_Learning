function [J, grad] = costFunction(theta, X, y)
%COSTFUNCTION Compute cost and gradient for logistic regression
%   J = COSTFUNCTION(theta, X, y) computes the cost of using theta as the
%   parameter for logistic regression and the gradient of the cost
%   w.r.t. to the parameters.

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% Calculate the hypothesis
H = sigmoid((X*theta));

% Cost term
J = (1 / m) * sum(-y' * log(H) - (1 - y)' * log(1 - H));

% Gradient decent simulatenous update
grad = (1 / m) * sum(X .* repmat((H - y), 1, size(X, 2)));

% =============================================================

end

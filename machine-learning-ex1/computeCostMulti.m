
function J = computeCostMulti(X, y, theta)
%COMPUTECOSTMULTI Compute cost for linear regression with multiple variables
%   J = COMPUTECOSTMULTI(X, y, theta) computes the cost of using theta as the
%   parameter for linear regression to fit the data points in X and y

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;

% Calculate the hypothesis
H = (X * theta);

% Get the sum of squares error
S = sum((H - y) .^ 2);

% The purpose of the 1/m term is to remove the dependency on the data 
% set size. The error magnitude is thereby independent of m. However, 
% the accuracy of the error estimate will increase as m increases, 
% assuming that data examples are selected randomly.
% When you take the derivative of your cost function, the square becomes a 
% 2*(expression) and the 1/2 cancels out the 2
J = S / (2 * m);

% =========================================================================

end

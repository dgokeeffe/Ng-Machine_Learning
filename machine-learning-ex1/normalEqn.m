function [theta] = normalEqn(X, y)
%NORMALEQN Computes the closed-form solution to linear regression 
%   NORMALEQN(X,y) computes the closed-form solution to linear 
%   regression using the normal equations.

theta = zeros(size(X, 2), 1);

% Do it all at once! Numerical solution like at uni
theta = (pinv(X' * X )) * X' * y;

% ============================================================

end

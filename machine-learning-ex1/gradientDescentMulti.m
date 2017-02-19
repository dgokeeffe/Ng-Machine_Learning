function [theta, J_history] = gradientDescentMulti(X, y, theta, alpha, num_iters)
%GRADIENTDESCENTMULTI Performs gradient descent to learn theta
%   theta = GRADIENTDESCENTMULTI(x, y, theta, alpha, num_iters) updates theta by
%   taking num_iters gradient steps with learning rate alpha

% Initialize some useful values
m = length(y); % number of training examples
J_history = zeros(num_iters, 1);

for iter = 1:num_iters

    % Store computed theta values for a simultanous update.
	Thetas = zeros(size(X, 2), 1);
    H = (X * theta - y)';  
	
	% Same computation as gradientDescent.m, but we must loop over all 
    % features.
	for i = 1:size(X, 2)
	    t = theta(i) - alpha * (1/m) * H * X(:, i);
		Thetas(i) = t;
	end
	
	theta = Thetas;
end

end

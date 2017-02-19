function p = predict(Theta1, Theta2, X)
%PREDICT Predict the label of an input given a trained neural network
%   p = PREDICT(Theta1, Theta2, X) outputs the predicted label of X given the
%   trained weights of a neural network (Theta1, Theta2)

% Useful values
m = size(X, 1);

% You need to return the following variables correctly 
p = zeros(size(X, 1), 1);

% ====================== YOUR CODE HERE ======================
% Instructions: Complete the following code to make predictions using
%               your learned neural network. You should set p to a 
%               vector containing labels between 1 to num_labels.
%
% Hint: The max function might come in useful. In particular, the max
%       function can also return the index of the max element, for more
%       information see 'help max'. If your examples are in rows, then, you
%       can use max(A, [], 2) to obtain the max for each row.
%

%===================== Input to Hidden ======================
% Add ones to the input layer
X = [ones(m, 1) X];

% Combine first set of weights and values
z1 = X * Theta1';

% Calculate the activation
a1 = sigmoid(z1);

%===================== Hidden to Output =====================
% Add ones to the previous activation layer
a1 = [ones(m, 1) a1];

% Combine with the second set of weights
z2 = a1 * Theta2';

% Calculate the second activation layer
a2 = sigmoid(z2);

% Return predictions like in one vs all
[~, p] = max(a2, [], 2);
% =========================================================================


end

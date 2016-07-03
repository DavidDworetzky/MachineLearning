function p = predict(Theta1, Theta2, X)
%PREDICT Predict the label of an input given a trained neural network
%   p = PREDICT(Theta1, Theta2, X) outputs the predicted label of X given the
%   trained weights of a neural network (Theta1, Theta2)

% Useful values
m = size(X, 1);
num_labels = size(Theta2, 1);
num_hidden = size(Theta1, 1);

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
%calculate input layer

%add vector of ones to X


X1 = [ones(m, 1), X];

z1 = X1 * transpose(Theta1);

%get hidden layer values
a1 = sigmoid(z1);

%augment a2 with bias unit and do multiplication and sigmoid transform
m = size(a1,1);
X2 = [ones(m, 1), a1];

z2 = X2 * transpose(Theta2);

a2 = sigmoid(z2);

%max vals, max indices are a function of o1
[MaxVals, MaxIndices] = max(a2, [], 2);
p = MaxIndices;

    



% =========================================================================


end

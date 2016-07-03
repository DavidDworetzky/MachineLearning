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

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta
%
% Note: grad should have the same dimensions as theta
%

%find the hypothesis, then do a vectorized sum of the logistic regression
%cost function
hypothesis = X * theta;
J = 1/m * sum( (-y .* log(sigmoid(hypothesis))) - ((1 - y) .* log(1 - sigmoid(hypothesis))));

%we compute this based off of the partial derivative term
dim = size(grad);
%disp(size(hypothesis))
%disp(size(y))
%disp(size(X))
%grad = (1/m) * sum(sigmoid(hypothesis) - y .* X);

count = zeros(dim(1), 1);
for i = 1:dim(1)
    for j = 1:m
        count(i, 1) = count(i, 1) + (sigmoid(X(j,:) * theta) - y(j, 1)) * X(j, i);
    end
end
%apply m scaling now
grad = count .* (1/m);








% =============================================================

end

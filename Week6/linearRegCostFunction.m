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

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost and gradient of regularized linear 
%               regression for a particular choice of theta.
%
%               You should set J to the cost and grad to the gradient.
%
%sum of squared errors
Cost =  ((1/ (2 * m)) * sum(((X * theta) - y) .^2));

%regularized cost function requires us to compute theta without the first
%element
newTheta = theta(2:length(theta));
Regularized = (lambda / (2 * m)) * sum(newTheta .^ 2);
J = Cost + Regularized;

%now calculate the gradient, based off of the partial derivative terms. 
h = X * theta;

G = (lambda/m) .* theta;
G(1) = 0; 
grad = ((1/m) .* X' * (h - y)) + G;


% =========================================================================

grad = grad(:);

end

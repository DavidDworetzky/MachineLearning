function J = computeCost(X, y, theta)
%COMPUTECOST Compute cost for linear regression
%   J = COMPUTECOST(X, y, theta) computes the cost of using theta as the
%   parameter for linear regression to fit the data points in X and y

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta
%               You should set J to the cost.

%the intuition here is that the result of our operation is 1/2 m , times
%the sum of the square difference. Since the estimate can be written as
%X theta - y, we can take this difference vector and take the
%sum of squares to determine the final cost function value
value = 1/(2 *m)  * sum( (X * theta - y) .^ 2);
J = value;





% =========================================================================

end

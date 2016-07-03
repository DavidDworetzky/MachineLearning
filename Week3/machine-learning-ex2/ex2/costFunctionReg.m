function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

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


dim = size(grad);

%vectorized for the first component of the sum

hypothesis = X * theta;
J = 1/m * sum( (-y .* log(sigmoid(hypothesis))) - ((1 - y) .* log(1 - sigmoid(hypothesis))));

LamCost =0;

for i = 2 :dim(1)
   LamCost = LamCost + (lambda / (2 * m)) * theta(i) ^ 2; 
end
J = J + LamCost;


%we compute this based off of the partial derivative term... including
%lambda for regularization

count = zeros(dim(1), 1);
for i = 1:dim(1)
    for j = 1:m
        %lambda term is only added for terms greater than 1
        if i > 1
            lam = ((lambda / m) * theta(i));
        else
            lam = 0;
        end
        count(i, 1) = count(i, 1) + (sigmoid(X(j,:) * theta) - y(j, 1)) * X(j, i) + lam;
    end
end
%apply m scaling now
grad = count .* (1/m);




% =============================================================

end

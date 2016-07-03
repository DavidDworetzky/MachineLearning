function [theta, J_history] = gradientDescent(X, y, theta, alpha, num_iters)
%GRADIENTDESCENT Performs gradient descent to learn theta
%   theta = GRADIENTDESENT(X, y, theta, alpha, num_iters) updates theta by 
%   taking num_iters gradient steps with learning rate alpha

% Initialize some useful values
m = length(y); % number of training examples
J_history = zeros(num_iters, 1);

for iter = 1:num_iters

    % ====================== YOUR CODE HERE ======================
    % Instructions: Perform a single gradient step on the parameter vector
    %               theta. 
    %
    % Hint: While debugging, it can be useful to print out the values
    %       of the cost function (computeCost) and gradient here.
    %
    %save the value of theta to another vector
    temp = theta;
    sum1 = 0;
    sum2 = 0;
    %iterate over our sum
    for i = 1: m 
        %theta0 + theta1X - y
        sum1 = sum1 + temp(1,1) + (temp(2,1) * X(i, 2)) - y(i);
        %theta0 + theta1X - y * X
        sum2 = sum2 + ((temp(1,1) + (temp(2,1) * X(i, 2)) - y(i)) * X(i, 2));
    end
    %store values of descent values for theta0, theta1
    d1 = alpha * 1/m * sum1;
    d2 = alpha * 1/m * sum2;
    %temp updates with descent 1 and descent 2
    temp(1, 1) = temp(1,1) - d1;
    temp(2, 1) = temp(2,1) - d2;
    
    %update theta
    theta(1,1) = temp(1,1);
    theta(2,1) = temp(2,1);
       
    %alternative vectorization
    %theta(1,1) = temp(1,1) - ( (alpha * 1/m ) * sum(X * theta - y));
    %theta(2,1) = temp(2,1) - ( (alpha * 1/m ) * sum(X * theta - y .* X(:,2)));


    % ============================================================

    % Save the cost J in every iteration    
    J_history(iter) = computeCost(X, y, theta);

end

end

function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices. 
% 
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));


% Setup some useful variables
m = size(X, 1);

%some displays for viewing our inputs 
%disp('Theta 1 is')
%disp(Theta1)
%disp('Theta 2 is')
%disp(Theta2)
%disp('m is')
%disp(m)
 
% You need to return the following variables correctly 
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m
%
% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. You should return the partial derivatives of
%         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
%         Theta2_grad, respectively. After implementing Part 2, you can check
%         that your implementation is correct by running checkNNGradients
%
%         Note: The vector y passed into the function is a vector of labels
%               containing values from 1..K. You need to map this vector into a 
%               binary vector of 1's and 0's to be used with the neural network
%               cost function.
%
%         Hint: We recommend implementing backpropagation using a for-loop
%               over the training examples if you are implementing it for the 
%               first time.
%
% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%

%Y = eye(num_labels)(y,:);
%a1 = [ones(m,1), X];
%a2 = sigmoid(Theta1 * a1');
%a2 = [ones(1, size(a2, 2)); a2];
%h = sigmoid(Theta2 * a2);

%J = (1/m) * sum(-Y .* log(h)' - (1 - Y) .* log(1 - h)');



%number of possible labels
k = num_labels;
%append array of ones to X to get new X (bias units)
X = [ones(m,1) X];
%the matrix consisting of our labels can be expressed as the identity
%matrix, indexed by y
Y = eye(k);
Y = Y(y, :);

%layer 1
a1=X;
%z2 comes from X * Theta1 weights
z2=Theta1 * a1';
a2=sigmoid(z2);

%layer 2
a2 = [ones(1, size(a2, 2)); a2];
z3=Theta2 * a2;
% Sigmoid function converts to p between 0 to 1
h=sigmoid(z3);


%compute our cost
JSum = 0;
for i = 1:m
    Yi = Y(i, :);
    Hxi = h(:, i);
    JSum = JSum + (-Yi * log(Hxi)) - ((1-Yi)*(log(1-Hxi)));
end
JSum = (1/m) * JSum;

%now add our regularization terms... based off of the coefficients for
%theta 1 and theta 2

%first, take theta 1 and 2 and remove the bias units
theta1b = Theta1(:, 2:end);
theta2b = Theta2(:, 2:end);

%the regularization terms are the sum of our theta(i)(j) terms squared
reg1 = sum(sum(theta1b .^ 2));
reg2 = sum(sum(theta2b .^ 2));

%now multiply by lambda / 2 * m
Reg = (lambda / (2 * m)) * (reg1 + reg2);
JSum = JSum + Reg;

%our cost is our sum plus our regularization term
J = JSum;

%PART 2 - BACKPROP%
%these are the accumulators for our gradient values
delta_accum_1 = zeros(size(Theta1));
delta_accum_2 = zeros(size(Theta2));
for t = 1 : m
%for all training set examples t -> M, | M is the size of our training set

% 1>> Set the input layer’s values (a(1)) to the t-th training example x(t). 
%Perform a feedforward pass
	a1 = X(t,:);  
	z2 = a1 * Theta1';
	a2 = [1 sigmoid(z2)];
	z3 = a2 * Theta2';
	a3 = sigmoid(z3);
    
    %create our output vector
	yi = zeros(1,k);
	yi(y(t)) = 1;
	
% 2>>For each output unit k in layer 3 (the output layer), set
% ?(3) k = (a(3) k ?yk), where yk ? {0,1}
	delta3 = a3 - yi;

% 3>> For the hidden layer l = 2, set ?(2) =?(2)T ?(3).?g0(z(2)) 
	delta2 = delta3 * Theta2 .* sigmoidGradient([1 z2]);
% 4>>  Accumulate the gradient from this example
	delta_accum_1 = delta_accum_1 + delta2(2:end)' * a1;
	delta_accum_2 = delta_accum_2 + delta3' * a2;

% 5>>  Obtain the (unregularized) gradient for the neural network cost function 
% by dividing the accumulated gradients by 1 m: 

Theta1_grad = delta_accum_1 / m;
Theta2_grad = delta_accum_2 / m;
end

%now add regularization terms to gradients
Theta1_grad(:, 2:input_layer_size+1) = Theta1_grad(:, 2:input_layer_size+1) + lambda / m * Theta1(:, 2:input_layer_size+1);
Theta2_grad(:, 2:hidden_layer_size+1) = Theta2_grad(:, 2:hidden_layer_size+1) + lambda / m * Theta2(:, 2:hidden_layer_size+1);
    
    
    


















% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end

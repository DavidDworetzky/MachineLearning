function [C, sigma] = dataset3Params(X, y, Xval, yval)
%EX6PARAMS returns your choice of C and sigma for Part 3 of the exercise
%where you select the optimal (C, sigma) learning parameters to use for SVM
%with RBF kernel
%   [C, sigma] = EX6PARAMS(X, y, Xval, yval) returns your choice of C and 
%   sigma. You should complete this function to return the optimal C and 
%   sigma based on a cross-validation set.
%

% You need to return the following variables correctly.
C = 1;
sigma = 0.3;

% ====================== YOUR CODE HERE ======================
% Instructions: Fill in this function to return the optimal C and sigma
%               learning parameters found using the cross validation set.
%               You can use svmPredict to predict the labels on the cross
%               validation set. For example, 
%                   predictions = svmPredict(model, Xval);
%               will return the predictions on the cross validation set.
%
%  Note: You can compute the prediction error using 
%        mean(double(predictions ~= yval))
%







% =========================================================================

%create a results matrix
results = eye(64,3);
error = 0;
%for our possible values of c, and sigma
for C_Sample = [0.01 0.03 0.1 0.3 1, 3, 10 30]
    for sigma_Sample = [0.01 0.03 0.1 0.3 1, 3, 10 30]
        %increment the error count
        error = error + 1;
        model = svmTrain(X, y, C_Sample, @(x1, x2) gaussianKernel(x1, x2, sigma_Sample));
        
        predictions = svmPredict(model, Xval);
        prediction_error = mean(double(predictions ~= yval));

        results(error,:) = [C_Sample, sigma_Sample, prediction_error];     
    end
end

sorted = sortrows(results, 3); % sort our matrix by columns 

%C is the optimal result
C = sorted(1,1);
%sigma is the optimal sigma
sigma = sorted(1,2);




end

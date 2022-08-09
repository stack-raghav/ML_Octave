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

a = theta;
hypo = sigmoid(X*theta);
Ja = 1/m.*((-1.*y'*log(hypo)) - (1-y')*log(1.-hypo));
e = ones(size(theta)) - eye(size(theta));
Jb = ((lambda/(2*m)).*(theta).^2)'*e;
J = Ja + Jb;

theta(1) = 0;

Ga = 1/m.*(X'*(hypo-y));
Gb = (lambda/m).*theta;
grad = Ga + Gb;
% =============================================================

end

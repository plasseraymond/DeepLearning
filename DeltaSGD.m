% DeltaSGD.m
% Raymond Plasse
% E.g. 1
% 8/29/2024

function W = DeltaSGD(W,X,D)

    alpha = 0.9;                    % between 0 and 1
    
    N = 4;                          % loop through all training data points
    for k = 1:N
    
        x = X(k,:)';                % grab first row of input matrix     
        d = D(k);                   % grab correct output for first input
    
        v = W * x;                  % determine activation function input
        y = Sigmoid(v);             % calculate activation function output
    
        e = d - y;                  % identify error
        delta = y * (1-y) * e;      % assign delta based on Eq 2.7
    
        dW = alpha * delta * x;     % delta rule based on Eq 2.7
    
        W(1) = W(1) + dW(1);        % update weights based on Eq 2.7
        W(2) = W(2) + dW(2);
        W(3) = W(3) + dW(3);
    
    end
end

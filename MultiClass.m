% MultiClass.m
% Raymond Plasse
% E.g. 6
% 9/27/2024

function [W1, W2] = MultiClass(W1, W2, X, D)
    
    alpha = 0.9;

    N = 5;
    for k = 1:N

        x = reshape(X(:, :, k), 25, 1); % turns kth image of 5x5x5 3-D matrix into 25x1 column vector
        d = D(k, :)';

        v1 = W1 * x;
        y1 = Sigmoid(v1);
        v = W2 * y1;
        y = Softmax(v); % calls Softmax() function as defined literally

        e = d - y;
        delta = e; % according to CE-driven learning rule, delta & error are equal

        e1 = W2' * delta;
        delta1 = y1 .* (1-y1) .* e1; % back-prop algorithm applied to hidden layer

        dW1 = alpha * delta1 * x';
        W1 = W1 + dW1; % update weights

        dW2 = alpha * delta * y1';
        W2 = W2 + dW2; % update weights

    end
end

% RealMultiClass.m
% Raymond Plasse
% E.g. 6
% 9/27/2024

clear; clc; close all;

TestMultiClass; % test the neural network first then introduce contaminated data

X = zeros(5, 5, 5);

X(:, :, 1) = [ 0 0 1 1 0;
               0 0 1 1 0;
               0 1 0 1 0;
               0 0 0 1 0;
               0 1 1 1 0 ];

X(:, :, 2) = [ 1 1 1 1 0;
               0 0 0 0 1;
               0 1 1 1 0;
               1 0 0 0 1;
               1 1 1 1 1 ];

X(:, :, 3) = [ 1 1 1 1 0;
               0 0 0 0 1;
               0 1 1 1 0;
               1 0 0 0 1;
               1 1 1 1 0 ];

X(:, :, 4) = [ 0 1 1 1 0;
               0 1 0 0 0;
               0 1 1 1 0;
               0 0 0 1 0;
               0 1 1 1 0 ];

X(:, :, 5) = [ 0 1 1 1 1;
               0 1 0 0 0;
               0 1 1 1 0;
               0 0 0 1 0;
               1 1 1 1 0 ];

N = 5;
for k = 1:N

    x = reshape(X(:, :, k), 25, 1);
    v1 = W1 * x;
    y1 = Sigmoid(v1);
    v = W2 * y1;
    y = Softmax(v)

end

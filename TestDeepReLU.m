% TestDeepReLU.m
% Raymond Plasse
% E.g. 7
% 10/10/2024

clear; clc; close all;

X = zeros(5, 5, 5);

X(:, :, 1) = [ 0 1 1 0 0;
               0 0 1 0 0;
               0 0 1 0 0;
               0 0 1 0 0;
               0 1 1 1 0 ];

X(:, :, 2) = [ 1 1 1 1 0;
               0 0 0 0 1;
               0 1 1 1 0;
               1 0 0 0 0;
               1 1 1 1 1 ];

X(:, :, 3) = [ 1 1 1 1 0;
               0 0 0 0 1;
               0 1 1 1 0;
               0 0 0 0 1;
               1 1 1 1 0 ];

X(:, :, 4) = [ 0 0 0 1 0;
               0 0 1 1 0;
               0 1 0 1 0;
               1 1 1 1 1;
               0 0 0 1 0 ];

X(:, :, 5) = [ 1 1 1 1 1;
               1 0 0 0 0;
               1 1 1 1 0;
               0 0 0 0 1;
               1 1 1 1 0 ];

D = [ 1 0 0 0 0;
      0 1 0 0 0;
      0 0 1 0 0;
      0 0 0 1 0;
      0 0 0 0 1 ];

W1 = 2 * rand(20, 25) - 1; % sensitivity of ReLU() to initial weight values
W2 = 2 * rand(20, 20) - 1;
W3 = 2 * rand(20, 20) - 1;
W4 = 2 * rand(5, 20) - 1;

for epoch = 1:10000

    [W1, W2, W3, W4] = DeepReLU(W1, W2, W3, W4, X, D);

end

N = 5;
for k = 1:N

    x = reshape(X(:, :, k), 25, 1);
    v1 = W1 * x;
    y1 = ReLU(v1);

    v2 = W2 * y1;
    y2 = ReLU(v2);

    v3 = W3 * y2;
    y3 = ReLU(v3);

    v = W4 * y3;
    y = Softmax(v)

end
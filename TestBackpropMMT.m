% TestBackpropMMT.m
% Raymond Plasse
% E.g. 4
% 9/5/2024

clear; clc; close all;

X = [ 0 0 1;
      0 1 1;
      1 0 1;
      1 1 1;
    ];

D = [ 0
      1
      1
      0
    ];

W1 = 2 * rand(4,3) - 1;
W2 = 2 * rand(1,4) - 1;

for epoch = 1:10000             % train 10,000 times

    [W1, W2] = BackpropMMT(W1, W2, X, D);

end

N = 4;
for k = 1:N                     % multi-layer network solves XOR problem

    x = X(k,:)';
    v1 = W1 * x;
    y1 = Sigmoid(v1);
    v = W2 * y1;
    y = Sigmoid(v)

end

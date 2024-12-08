% TestDeltaBatch.m
% Raymond Plasse
% E.g. 2
% 8/29/2024

clear; clc; close all;

X = [ 0 0 1;
      0 1 1;
      1 0 1;
      1 1 1;
    ];

D = [ 0
      0
      1
      1
    ];

W = 2 * rand(1,3) - 1;

for epoch = 1:40000             % train 40,000 times (learns more slowly)

    W = DeltaBatch(W,X,D);

end

N = 4;
for k = 1:N                     % inference (output of model)

    x = X(k,:)';
    v = W * x;
    y = Sigmoid(v)

end

% DeepFilterClassTest.m
% Raymond Plasse
% FA24 Independent Study
% 12/10/2024

% This script trains a deep neural network (i.e., a neural network with
% more than one hidden layer) to classify four different types of digital
% audio filters (LPF, HPF, BPF, BSF). The training data consist of an
% impulse signal convolved with one of the four filter types. The inferred
% model uses the Rectified Linear Unit (ReLU) activation function in its
% hidden layers and the Softmax activation function for the output layer
% with the categorical cross-entropy loss function.

clear; clc; close all;

% array variables
Fs = 48000;
order = 10;
X = zeros(Fs+order,4);
x = [1 ; zeros(Fs-1,1)];
N = length(X);

% randomizer reset control
rng(3);

% input (4 classes)
b1 = fir1(order,0.5); % LPF
X(:,1) = conv(x,b1);

b2 = fir1(order,0.5,'high'); % HPF
X(:,2) = conv(x,b2);

b3 = fir1(order,[0.25 0.75]); % BPF
X(:,3) = conv(x,b3);

b4 = fir1(order,[0.25 0.75],'stop'); % BSF
X(:,4) = conv(x,b4);

% correct output
D = [1 0 0 0;
     0 1 0 0;
     0 0 1 0;
     0 0 0 1];

% initial weight values
W1 = 2 * rand(12, Fs+order) - 1;
W2 = 2 * rand(12, 12) - 1;
W3 = 2 * rand(12, 12) - 1;
W4 = 2 * rand(4, 12) - 1;

% train
for epoch = 1:10000

    [W1, W2, W3, W4] = DeepReLU(W1, W2, W3, W4, X, D);

end

% inference
N = 4;
for k = 1:N

    x = X(:,k);
    v1 = W1 * x;
    y1 = ReLU(v1);

    v2 = W2 * y1;
    y2 = ReLU(v2);

    v3 = W3 * y2;
    y3 = ReLU(v3);

    v = W4 * y3;
    y = Softmax(v) % model predicts filter type, amount of certainty displayed in command window

end

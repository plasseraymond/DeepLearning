% Sigmoid.m
% Raymond Plasse
% E.g. 1
% 8/29/2024

function y = Sigmoid(x)

    y = 1 ./ (1 + exp(-x));     % element-wise for vector input

end
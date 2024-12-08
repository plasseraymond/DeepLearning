% Softmax.m
% Raymond Plasse
% E.g. 6
% 9/27/2024

function y = Softmax(x)

    ex = exp(x);
    y = ex / sum(ex);
    
end
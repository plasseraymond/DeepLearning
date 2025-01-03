% Dropout.m
% Raymond Plasse
% E.g. 8
% 10/23/2024

function ym = Dropout(y, ratio)

    [m, n] = size(y);
    ym = zeros(m, n);

    num = round(m*n*(1-ratio));
    idx = randperm(m*n, num);
    ym(idx) = 1 / (1-ratio);

end
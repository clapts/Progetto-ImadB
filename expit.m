function [x] = expit(x)
    x=1./(1+exp(-x));
end
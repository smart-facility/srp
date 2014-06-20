
function [num_non1] = computenode(wight)
num_non1 = 0;
for i  = 1:size(wight,2)
    if (abs(wight(:,i))>0.0001) % the neuron  exists
        num_non1 = num_non1 + 1;
    end
end
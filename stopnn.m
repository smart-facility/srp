
function [stop]=stopnn(e_va)
stop = 0; %need to stop next generation;
T = 6;
index =  size(e_va,2);
e_va_temp =[];
if index >= T
    for i = (index-T+1):index
        e_va_temp = [e_va_temp e_va(i)];
    end

    if min(e_va_temp) == e_va_temp(1)
        stop = 1;
    end
end
    
 



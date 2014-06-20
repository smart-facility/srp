
function res = cr(C_true, testT)
scores    = C_true;
min_thres = min(scores);
max_thres = max(scores);
thres     = linspace(min_thres, max_thres, 101);i    = 0;
cdr  = zeros(size(thres));
res = 0;
for thres_value = thres
    i = i + 1;
    z = (scores >= thres_value); 
    for j= 1:size(z,2) 
        if z(1,j)==testT(1,j)
            cdr(i) = cdr(i)+1; 
        end
    end

    cdr(i) = cdr(i)/size(z,2);
end

res = max(cdr);
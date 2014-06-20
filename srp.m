function srp()
clc
clear
[P,T,validP,validT,testP,testT] = load_data;

ex_num = 1;
f_tr=[];
f_cr = [];   f_n_num = [];     f_time=[]; f_w_num = [];
    
for iter = 1:ex_num
    %num. of hidden neuron
    k = 128;
    R_mx = minmax(P);
    net = newff(R_mx,[k,size(T,1)],{'logsig','purelin'},'trainrp');
    net.trainParam.epochs = 500;
    net = train(net,P,T);  
    Y = sim(net,P);                  
    A1 = repmat(net.b{1},1 , size(P,2)); %A = GBest. repmat(X, m, n) repeats the matrix X in m rows by n columns.
    B1 = net.iw{1,1} * P + A1;
    B = logsig(B1);
    
    A2 = repmat(net.b{2},1 , size(B,2));
    C1 = net.lw{2,1} * B + A2;
    C = purelin(C1);
    
    % compute the error before pruning:
    % disp(strcat(['error before pruning:  ', num2str(mse(T-C))]));
    % Y = sim(net,validP);
    % disp(strcat(['error before pruning:  ', num2str(mse(validT-Y))]));
    % Y = sim(net,testP);
    % disp(strcat(['error before pruning:  ', num2str(mse(testT-Y))]));
    
%% prune 

    numi = 1;
    e_va=[];e_tr=[];
    e_time = [];    e_num = [];    
    stop = 0; index_c=1;
    
    
    a1=clock;% record process time
    while(stop == 0)                
        %prune
        A2 = repmat(net.b{2},1 , size(B,2));
        weight2 = (mmv(B',(T - A2)',numi))';                   
        %%valid sets      
        A1 = repmat(net.b{1},1 , size(validP,2)); %A = GBest. repmat(X, m, n) repeats the matrix X in m rows by n columns.
        B21 = net.iw{1,1} * validP + A1;
        B2 = logsig(B21);        
        A2 = repmat(net.b{2},1 , size(validT,2));
        C21 = weight2 * B2 + A2;
        C2 = purelin(C21);        
        
        %%train sets
        tA1 = repmat(net.b{1},1 , size(P,2)); %A = GBest. repmat(X, m, n) repeats the matrix X in m rows by n columns.
        tB21 = net.iw{1,1} * P + tA1;
        tB2 = logsig(tB21);        
        tA2 = repmat(net.b{2},1 , size(T,2));
        tC21 = weight2 * tB2 + tA2;
        tC2 = purelin(tC21);

        % save the network        
        res =1- cr(C2,validT);
        e_va = [e_va res];
        if size(e_va,2)==1 || e_va(size(e_va,2)) < min( e_va(1,1:size(e_va,2)-1) )
            best_wight2= weight2;
            index_c = size(e_va,2);
        end
        e_tr = [e_tr mse(T - tC2)];
        % check the stopping criterion        
        stop=stopnn(e_va);        
        
        numi = numi + 1;

    end %while  
    
    time=etime(clock,a1);
    e_time = time;
    

    % test
    A1 = repmat(net.b{1},1 , size(testP,2)); %A = GBest. repmat(X, m, n) repeats the matrix X in m rows by n columns.
    A2 = repmat(net.b{2},1 , size(testT,2));
    
    B_true = net.iw{1,1} * testP + A1;
    B1_true = logsig(B_true);     
    C_true = best_wight2 * B1_true + A2;
    C_true = purelin(C_true);
     
    num_non1 = 0;
    
    for i  = 1:size(best_wight2,2)
        if (find(best_wight2(:,i)~=0)~=0) % the neuron  exists
            num_non1 = num_non1 + 1;
        end
    end
    
    num_non = 0;
    for i  = 1:size(best_wight2,2)
        if (find(best_wight2(:,i)~=0)~=0) % the neuron  exists            
            for j  = 1:size(net.iw{1,1},2)                
                if net.iw{1,1}(i,j)~=0                        
                    num_non = num_non + 1;                                                                
                end
            end
            
            for j  = 1:size(best_wight2,1)
                if best_wight2(j,i)~=0                        
                    num_non = num_non + 1;                                                                
                end
            end
        end
    end
            
    
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% if classification
    res = cr(C_true, testT);
    f_cr=[f_cr res*100.0];
    f_tr=[f_tr e_tr(index_c(1,1))];
    f_time=[f_time e_time];
    f_n_num = [f_n_num num_non1]; 
    f_w_num = [f_w_num num_non]; 

  
    
end %for iter = 1:ex_num
 
disp(['Training accuracy is: ',num2str( 1 - mean(f_tr)   )]);
disp(['Weights: ',num2str( mean( f_w_num )    )]);
disp(['Nodes: ',num2str( mean( f_n_num )    )]);
disp(['Test accuracy: ',num2str( mean( f_cr )    )]);
disp(['Time is: ',num2str( mean(f_time)   )]);






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



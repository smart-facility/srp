function main2()
clear; clc; close all;

ex_num = 1;
num_nodes = 10;

r_nodes=[];r_time=[];r_acc=[];

for iter = 1: ex_num
    
    NetNode='H'; HiddenLayer = eval('NetNode')
    OutNode='L'; OutLayer = eval('OutNode')

    for i = 1:num_nodes,
        HiddenLayer = strcat(HiddenLayer,NetNode);
        OutLayer = strcat(OutLayer,'-');
    end

    NetDef = [HiddenLayer
              OutLayer];



    W1 = [];  % Weights to hidden layer 
    W2 = [];  % Weights to output
    trparms = settrain;      % Set training parameters to default values
    trparms=settrain(trparms,'maxiter',10,'eta',0.000001);
    % prepar for the training data
    [P,T,validP,validT,testP,testT] = load_data;

    % ----- Back propagation (Batch) -----
    [W1,W2,PI_vector,iter]=incbp(NetDef,W1,W2,P,T,trparms);


    % ----- pruning 1-----
    a1=clock;% record process time
    [theta_data,PI_vector,FPE_vector,PI_test_vec,deff_vec,pvec]=...
                                 obdprune(NetDef,W1,W2,P,T,trparms,[],validP,validT);
    time=etime(clock,a1);

    % -----------  Validate Network  -----------
    [Y_output,E,PI] = nneval(NetDef,W1,W2,testP,testT,'noplots');
    
    % -----------  Record  -----------
    r_nodes=[r_nodes computenode(W2)];
    r_time=[r_time time];
    r_acc=[r_acc cr(Y_output, testT)];
    
end


% -----------  Display  -----------
disp(['obsprune: Left nodes: ',num2str( mean(r_nodes)  )]);
disp(['obsprune: Test accuracy: ',num2str(  mean(r_acc) )]);
disp(['obsprune: Time is: ',num2str(  mean(r_time)  )]);





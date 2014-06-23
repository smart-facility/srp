function [P,T,validP,validT,testP,testT]=cancer()

total_sample = [
0.2 0.1 0.1 0.1 0.2 0.1 0.2 0.1 0.1 1 0
0.2 0.1 0.1 0.1 0.2 0.1 0.3 0.1 0.1 1 0
0.5 0.1 0.1 0.1 0.2 0.1 0.2 0.1 0.1 1 0
0.5 0.4 0.6 0.8 0.4 0.1 0.8 1 0.1 0 1
0.5 0.3 0.3 0.1 0.2 0.1 0.2 0.1 0.1 1 0
0.2 0.3 0.1 0.1 0.3 0.1 0.1 0.1 0.1 1 0
0.3 0.5 0.7 0.8 0.8 0.9 0.7 1 0.7 0 1
1 0.5 0.6 1 0.6 1 0.7 0.7 1 0 1
1 0.9 0.8 0.7 0.6 0.4 0.7 1 0.3 0 1
0.4 0.1 0.1 0.1 0.2 0.1 0.3 0.1 0.1 1 0
0.5 0.1 0.1 0.1 0.2 0.1 0.3 0.1 0.1 1 0
0.8 1 1 0.1 0.3 0.6 0.3 0.9 0.1 0 1
0.1 0.1 0.3 0.1 0.2 0.1 0.2 0.1 0.1 1 0
0.1 0.1 0.1 0.2 0.1 0.1 0.1 0.1 0.1 1 0
0.3 0.4 0.5 0.2 0.6 0.8 0.4 0.1 0.1 0 1
0.4 0.3 0.3 0.1 0.2 0.1 0.3 0.3 0.1 1 0
0.3 0.3 0.2 0.1 0.3 0.1 0.3 0.6 0.1 1 0
0.2 0.1 0.1 0.1 0.2 0.1 0.1 0.1 0.1 1 0
0.1 0.1 0.1 0.1 0.2 0.1 0.2 0.1 0.1 1 0
0.1 0.1 0.1 0.1 0.2 0.1 0.1 0.1 0.1 1 0
0.8 1 1 1 0.5 1 0.8 1 0.6 0 1
0.8 0.7 0.4 0.4 0.5 0.3 0.5 1 0.1 0 1
0.1 0.1 0.1 0.1 0.1 0.1 0.2 0.1 0.1 1 0
0.2 0.1 0.1 0.1 0.2 0.1 0.1 0.1 0.1 1 0
1 0.8 0.8 0.4 1 1 0.8 0.1 0.1 0 1
0.5 0.1 0.1 0.2 0.2 0.1 0.2 0.1 0.1 1 0
0.3 0.1 0.1 0.1 0.2 0.1 0.2 0.1 0.2 1 0
0.3 0.1 0.1 0.1 0.2 0.1 0.3 0.1 0.1 1 0
0.5 0.3 0.3 0.3 0.6 1 0.3 0.1 0.1 0 1
0.4 0.8 0.6 0.3 0.4 1 0.7 0.1 0.1 0 1
0.4 0.3 0.2 0.1 0.3 0.1 0.2 0.1 0.1 1 0
0.1 0.1 0.1 0.1 0.2 0.1 0.1 0.1 0.1 1 0
0.8 0.5 0.5 0.5 0.2 1 0.4 0.3 0.1 0 1
0.3 0.3 0.5 0.2 0.3 1 0.7 0.1 0.1 0 1
0.2 0.3 0.1 0.1 0.2 0.1 0.2 0.1 0.1 1 0
0.3 0.1 0.1 0.1 0.2 0.5 0.1 0.1 0.1 1 0
0.3 0.2 0.1 0.1 0.2 0.2 0.3 0.1 0.1 1 0
0.2 0.7 1 1 0.7 1 0.4 0.9 0.4 0 1
0.1 0.1 0.2 0.1 0.2 0.1 0.2 0.1 0.1 1 0
0.1 0.1 0.1 0.1 0.1 0.1 0.2 0.1 0.1 1 0
0.5 1 1 1 1 0.2 1 1 1 0 1
0.4 0.1 0.2 0.1 0.2 0.1 0.2 0.1 0.1 1 0
0.1 0.1 0.1 0.1 0.2 0.1 0.1 0.1 0.1 1 0
0.5 0.1 0.1 0.1 0.2 0.1 0.2 0.1 0.1 1 0
0.3 0.7 0.7 0.4 0.4 0.9 0.4 0.8 0.1 0 1
0.1 0.1 0.1 0.1 0.1 0.1 0.3 0.1 0.1 1 0
0.4 0.1 0.1 0.1 0.2 0.1 0.2 0.1 0.1 1 0
0.6 0.8 0.7 0.5 0.6 0.8 0.8 0.9 0.2 0 1
1 0.7 0.7 0.4 0.5 1 0.5 0.7 0.2 0 1
0.1 0.1 0.1 0.1 0.2 0.1 0.1 0.1 0.1 1 0
0.8 1 0.4 0.4 0.8 1 0.8 0.2 0.1 0 1
0.1 0.1 0.1 0.1 0.2 0.1 0.2 0.1 0.1 1 0
0.1 0.1 0.1 0.1 0.2 0.1 0.1 0.1 0.8 1 0
0.5 0.1 0.1 0.1 0.2 0.1 0.1 0.1 0.1 1 0
0.1 0.1 0.4 0.1 0.2 0.1 0.2 0.1 0.1 1 0
0.5 0.1 0.1 0.1 0.2 0.1 0.2 0.1 0.1 1 0
0.3 0.1 0.1 0.1 0.2 0.1 0.2 0.1 0.1 1 0
0.1 0.1 0.1 0.1 0.2 0.1 0.2 0.1 0.1 1 0
0.1 0.1 0.1 0.1 0.1 0.35 0.1 0.1 0.1 1 0
0.8 1 1 0.8 0.7 1 0.9 0.7 0.1 0 1
1 0.5 0.8 1 0.3 1 0.5 0.1 0.3 0 1
1 1 1 0.2 1 1 0.5 0.3 0.3 0 1
1 0.8 0.8 0.2 0.8 1 0.4 0.8 1 0 1
1 0.5 0.5 0.3 0.6 0.7 0.7 1 0.1 0 1
0.1 0.1 0.1 0.1 0.2 0.1 0.2 0.1 0.1 1 0
0.2 0.1 0.1 0.2 0.2 0.1 0.1 0.1 0.1 1 0
0.1 0.1 0.2 0.2 0.2 0.1 0.3 0.1 0.1 1 0
0.5 0.1 0.1 0.1 0.2 0.1 0.3 0.1 0.1 1 0
0.5 0.1 0.1 0.1 0.2 0.35 0.3 0.1 0.1 1 0
0.4 0.2 0.3 0.5 0.3 0.8 0.7 0.6 0.1 0 1
0.2 0.1 0.1 0.1 0.1 0.1 0.1 0.1 0.1 1 0
1 1 1 1 0.6 1 0.8 0.1 0.5 0 1
0.8 0.6 0.5 0.4 0.3 1 0.6 0.1 0.1 0 1
0.4 0.1 0.1 0.1 0.2 0.2 0.3 0.2 0.1 1 0
0.5 0.1 0.1 0.6 0.3 0.1 0.1 0.1 0.1 1 0
0.1 0.1 0.1 0.1 0.2 0.1 0.2 0.1 0.1 1 0
0.3 0.1 0.1 0.1 0.2 0.1 0.2 0.1 0.1 1 0
0.3 0.1 0.1 0.1 0.2 0.1 0.2 0.1 0.1 1 0
0.3 0.1 0.1 0.1 0.1 0.1 0.1 0.1 0.1 1 0
0.5 0.1 0.3 0.1 0.2 0.1 0.2 0.1 0.1 1 0
0.3 0.4 0.4 1 0.5 0.1 0.3 0.3 0.1 0 1
0.1 0.1 0.1 0.1 0.2 0.1 0.1 0.1 0.1 1 0
0.7 0.8 0.7 0.2 0.4 0.8 0.3 0.8 0.2 0 1
1 0.5 0.7 0.3 0.3 0.7 0.3 0.3 0.8 0 1
0.1 0.1 0.1 0.1 0.2 0.1 0.2 0.1 0.1 1 0
0.2 0.5 0.7 0.6 0.4 1 0.7 0.6 0.1 0 1
0.9 0.8 0.8 0.5 0.6 0.2 0.4 1 0.4 0 1
0.5 1 1 1 0.4 1 0.5 0.6 0.3 0 1
0.3 0.1 0.1 0.1 0.2 0.1 0.3 0.1 0.1 1 0
0.1 0.1 0.1 0.1 0.2 0.1 0.1 0.1 0.1 1 0
0.5 0.5 0.5 0.8 1 0.8 0.7 0.3 0.7 0 1
0.1 0.1 0.1 0.1 0.2 0.1 0.1 0.1 0.1 1 0
0.4 0.1 0.1 0.1 0.2 0.1 0.1 0.1 0.1 1 0
0.2 0.1 0.1 0.1 0.2 0.1 0.3 0.1 0.1 1 0
0.1 0.2 0.2 0.1 0.2 0.1 0.2 0.1 0.1 1 0
0.4 0.3 0.1 0.1 0.2 0.1 0.4 0.8 0.1 1 0
0.3 0.3 0.2 0.6 0.3 0.3 0.3 0.5 0.1 1 0
0.3 1 0.8 0.7 0.6 0.9 0.9 0.3 0.8 0 1
0.8 0.3 0.4 0.9 0.3 1 0.3 0.3 0.1 0 1
1 0.8 0.4 0.4 0.4 1 0.3 1 0.4 0 1
0.1 0.1 0.1 0.1 0.2 0.1 0.1 0.1 0.1 1 0
0.8 0.4 0.6 0.3 0.3 0.1 0.4 0.3 0.1 1 0
0.5 0.6 0.6 0.8 0.6 1 0.4 1 0.4 0 1
0.1 0.1 0.1 0.1 0.2 0.1 0.3 0.1 0.1 1 0
0.1 0.1 0.1 0.2 0.2 0.1 0.2 0.1 0.1 1 0
0.1 0.1 0.1 0.1 0.2 0.2 0.2 0.1 0.1 1 0
0.5 0.1 0.3 0.1 0.2 0.1 0.1 0.1 0.1 1 0
0.8 0.3 0.5 0.4 0.5 1 0.1 0.6 0.2 0 1
0.5 1 1 0.3 0.8 0.1 0.5 1 0.3 0 1
1 1 1 0.7 1 1 0.8 0.2 0.1 0 1
0.4 0.2 0.2 0.1 0.2 0.1 0.2 0.1 0.1 1 0
0.2 0.3 0.1 0.1 0.5 0.1 0.1 0.1 0.1 1 0
0.5 0.2 0.2 0.2 0.2 0.1 0.1 0.1 0.2 1 0
0.4 0.1 0.1 0.1 0.2 0.1 0.1 0.1 0.1 1 0
0.5 0.1 0.1 0.1 0.2 0.2 0.2 0.1 0.1 1 0
0.3 1 0.7 0.8 0.5 0.8 0.7 0.4 0.1 0 1
0.3 0.1 0.1 0.1 0.2 0.1 0.1 0.1 0.1 1 0
0.5 0.4 0.5 0.1 0.8 0.1 0.3 0.6 0.1 1 0
0.5 0.1 0.1 0.1 0.1 0.1 0.3 0.1 0.1 1 0
0.5 0.1 0.1 0.1 0.2 0.1 0.2 0.1 0.1 1 0
0.5 1 0.6 0.1 1 0.4 0.4 1 1 0 1
0.5 0.1 0.1 0.1 0.3 0.2 0.2 0.2 0.1 1 0
0.8 0.4 0.4 0.1 0.2 0.9 0.3 0.3 0.1 0 1
0.5 0.1 0.1 0.1 0.2 0.1 0.3 0.1 0.2 1 0
0.4 0.8 0.8 0.5 0.4 0.5 1 0.4 0.1 0 1
0.9 0.1 0.2 0.6 0.4 1 0.7 0.7 0.2 0 1
0.8 0.5 0.6 0.2 0.3 1 0.6 0.6 0.1 0 1
0.1 0.1 0.1 0.1 0.1 0.1 0.3 0.1 0.1 1 0
0.3 0.1 0.1 0.1 0.2 0.3 0.3 0.1 0.1 1 0
0.4 1 0.8 0.5 0.4 0.1 1 0.1 0.1 0 1
0.2 0.1 0.1 0.1 0.2 0.1 0.3 0.1 0.1 1 0
0.5 0.1 0.3 0.1 0.2 0.1 0.3 0.1 0.1 1 0
1 0.3 0.5 0.1 1 0.5 0.3 1 0.2 0 1
0.4 0.6 0.5 0.6 0.7 0.35 0.4 0.9 0.1 1 0
0.3 0.1 0.1 0.2 0.3 0.4 0.1 0.1 0.1 1 0
0.5 0.2 0.2 0.2 0.2 0.1 0.2 0.2 0.1 1 0
0.5 0.5 0.5 0.6 0.3 1 0.3 0.1 0.1 0 1
0.5 0.6 0.7 0.8 0.8 1 0.3 1 0.3 0 1
0.6 1 1 1 0.8 1 1 1 0.7 0 1
0.4 0.2 0.2 0.1 0.2 0.1 0.2 0.1 0.1 1 0
0.3 0.2 0.2 0.2 0.2 0.1 0.4 0.2 0.1 1 0
0.5 0.1 0.1 0.6 0.3 0.1 0.2 0.1 0.1 1 0
0.7 0.8 0.3 0.7 0.4 0.5 0.7 0.8 0.2 0 1
0.1 0.1 0.1 0.1 0.2 0.1 0.2 0.1 0.2 1 0
0.8 0.4 0.4 0.5 0.4 0.7 0.7 0.8 0.2 1 0
0.5 0.6 0.5 0.6 1 0.1 0.3 0.1 0.1 0 1
0.7 0.9 0.4 1 1 0.3 0.5 0.3 0.3 0 1
0.3 0.3 0.2 0.2 0.3 0.1 0.1 0.2 0.3 1 0
0.2 0.1 0.1 0.1 0.2 0.1 0.1 0.1 0.1 1 0
0.3 0.1 0.1 0.1 0.2 0.1 0.3 0.1 0.1 1 0
0.2 0.1 0.1 0.1 0.2 0.1 0.1 0.1 0.1 1 0
0.5 0.1 0.1 0.1 0.2 0.1 0.1 0.1 0.1 1 0
0.5 0.1 0.1 0.1 0.2 0.1 0.3 0.2 0.1 1 0
0.5 1 0.8 1 0.8 1 0.3 0.6 0.3 0 1
0.5 0.1 0.1 0.1 0.2 0.1 0.1 0.1 0.1 1 0
0.4 0.1 0.1 0.1 0.2 0.1 0.2 0.1 0.1 1 0
0.4 0.1 0.1 0.1 0.2 0.1 0.2 0.1 0.1 1 0
0.8 0.7 0.8 0.5 1 1 0.7 0.2 0.1 0 1
0.1 0.1 0.1 0.1 0.1 0.35 0.2 0.1 0.1 1 0
0.1 0.1 0.1 0.1 0.2 0.1 0.2 0.3 0.1 1 0
0.2 0.1 0.1 0.1 0.2 0.1 0.1 0.1 0.1 1 0
0.5 0.1 0.1 0.2 0.2 0.2 0.3 0.1 0.1 1 0
0.3 0.1 0.1 0.3 0.8 0.1 0.5 0.8 0.1 1 0
0.2 0.1 0.1 0.1 0.2 0.1 0.2 0.1 0.1 1 0
0.7 0.6 0.3 0.2 0.5 1 0.7 0.4 0.6 0 1
0.4 1 0.4 0.7 0.3 1 0.9 1 0.1 0 1
0.1 0.1 0.1 0.1 0.2 0.1 0.3 0.1 0.1 1 0
0.1 0.1 0.1 0.1 0.2 0.1 0.1 0.1 0.1 1 0
0.1 0.1 0.1 0.1 1 0.1 0.1 0.1 0.1 1 0
0.4 0.1 0.1 0.1 0.2 0.1 0.3 0.1 0.1 1 0
0.5 0.3 0.4 0.1 0.4 0.1 0.3 0.1 0.1 1 0
1 0.5 0.5 0.6 0.8 0.8 0.7 0.1 0.1 0 1
1 1 1 0.4 0.8 0.1 0.8 1 0.1 0 1
0.8 1 1 1 0.6 1 1 1 1 0 1
0.3 0.1 0.1 0.1 0.2 0.1 0.1 0.1 0.1 1 0
0.6 1 0.5 0.5 0.4 1 0.6 1 0.1 0 1
0.8 0.3 0.8 0.3 0.4 0.9 0.8 0.9 0.8 0 1
0.3 0.1 0.1 0.1 0.2 0.1 0.2 0.1 0.1 1 0
1 0.4 0.3 0.1 0.3 0.3 0.6 0.5 0.2 0 1
0.3 0.1 0.1 0.1 0.2 0.2 0.3 0.1 0.1 1 0
0.3 0.1 0.3 0.1 0.3 0.4 0.1 0.1 0.1 1 0
0.2 0.1 0.1 0.1 0.2 0.1 0.1 0.1 0.5 1 0
0.8 0.2 0.1 0.1 0.5 0.1 0.1 0.1 0.1 1 0
0.4 0.1 0.1 0.1 0.2 0.1 0.2 0.1 0.1 1 0
1 0.3 0.5 0.4 0.3 0.7 0.3 0.5 0.3 0 1
0.1 0.1 0.1 0.1 0.2 0.1 0.1 0.1 0.1 1 0
0.1 0.1 0.1 0.1 0.2 1 0.3 0.1 0.1 1 0
0.1 0.1 0.1 0.1 0.2 0.1 0.2 0.1 0.1 1 0
1 0.4 0.3 0.2 0.3 1 0.5 0.3 0.2 0 1
0.2 0.5 0.3 0.3 0.6 0.7 0.7 0.5 0.1 0 1
0.3 0.1 0.1 0.1 0.2 0.1 0.1 0.1 0.1 1 0
0.4 0.1 0.1 0.1 0.2 0.1 0.2 0.1 0.1 1 0
0.6 0.6 0.7 1 0.3 1 0.8 1 0.2 0 1
0.2 0.1 0.1 0.1 0.1 0.1 0.2 0.1 0.1 1 0
0.3 0.3 0.2 0.1 0.2 0.3 0.3 0.1 0.1 1 0
0.3 0.1 0.1 0.1 0.2 0.1 0.2 0.1 0.1 1 0
0.4 0.4 0.4 0.4 0.6 0.5 0.7 0.3 0.1 1 0
0.2 0.1 0.1 0.1 0.2 0.1 0.1 0.1 0.1 1 0
0.1 0.1 0.3 0.2 0.2 0.1 0.3 0.1 0.1 1 0
0.5 0.1 0.2 1 0.4 0.5 0.2 0.1 0.1 1 0
0.2 0.1 0.1 0.1 0.2 0.1 0.1 0.1 0.1 1 0
0.4 0.1 0.1 0.1 0.2 0.1 0.1 0.1 0.1 1 0
0.3 0.1 0.4 0.1 0.2 0.1 0.1 0.1 0.1 1 0
0.3 0.1 0.1 0.1 0.2 0.1 0.2 0.1 0.1 1 0
0.1 0.1 0.3 0.1 0.2 0.1 0.1 0.1 0.1 1 0
0.3 0.1 0.1 0.1 0.2 0.1 0.1 0.1 0.1 1 0
0.3 0.1 0.1 0.1 0.2 0.1 0.3 0.1 0.1 1 0
0.3 0.1 0.2 0.1 0.2 0.1 0.2 0.1 0.1 1 0
0.1 0.1 0.1 0.3 0.2 0.1 0.1 0.1 0.1 1 0
0.3 0.2 0.2 0.3 0.2 0.1 0.1 0.1 0.1 1 0
0.5 0.3 0.3 0.1 0.3 0.3 0.3 0.3 0.3 0 1
0.6 0.6 0.6 0.5 0.4 1 0.7 0.6 0.2 0 1
0.1 0.1 0.1 0.1 0.5 0.1 0.3 0.1 0.1 1 0
0.5 0.8 0.8 0.8 0.5 1 0.7 0.8 0.1 0 1
0.1 0.1 0.1 0.1 0.2 0.1 0.2 0.1 0.1 1 0
0.5 0.1 0.1 0.1 0.2 0.1 0.2 0.2 0.1 1 0
0.4 0.5 0.5 1 0.4 1 0.7 0.5 0.8 0 1
0.3 0.1 0.1 0.4 0.3 0.1 0.2 0.2 0.1 1 0
1 0.8 1 1 0.6 0.1 0.3 0.1 1 0 1
0.6 0.2 0.1 0.1 0.1 0.1 0.7 0.1 0.1 1 0
0.1 0.6 0.8 1 0.8 1 0.5 0.7 0.1 0 1
0.4 0.1 0.1 0.1 0.2 0.1 0.3 0.1 0.1 1 0
0.9 0.5 0.5 0.2 0.2 0.2 0.5 0.1 0.1 0 1
0.3 0.1 0.1 0.1 0.1 0.1 0.2 0.1 0.1 1 0
0.3 0.6 0.6 0.6 0.5 1 0.6 0.8 0.3 0 1
0.2 0.1 0.1 0.1 0.1 0.1 0.3 0.1 0.1 1 0
0.1 0.1 0.1 0.1 0.2 0.1 0.3 0.1 0.1 1 0
1 1 0.9 0.3 0.7 0.5 0.3 0.5 0.1 0 1
0.4 0.1 0.1 0.1 0.2 0.1 0.2 0.1 0.1 1 0
0.3 0.1 0.1 0.1 0.2 0.1 0.2 0.1 0.1 1 0
0.1 0.1 0.1 0.1 0.3 0.1 0.1 0.1 0.1 1 0
0.3 0.1 0.1 0.1 0.2 0.1 0.2 0.3 0.1 1 0
0.1 0.1 0.1 0.1 0.2 0.1 0.2 0.1 0.1 1 0
1 0.3 0.4 0.5 0.3 1 0.4 0.1 0.1 0 1
0.3 0.1 0.1 0.3 0.2 0.1 0.1 0.1 0.1 1 0
0.7 0.4 0.4 0.3 0.4 1 0.6 0.9 0.1 0 1
0.8 0.8 0.8 0.1 0.2 0.35 0.6 1 0.1 0 1
0.1 0.1 0.1 0.1 0.2 0.1 0.3 0.1 0.1 1 0
0.3 0.1 0.2 0.1 0.2 0.1 0.2 0.1 0.1 1 0
0.5 1 1 1 0.6 1 0.6 0.5 0.2 0 1
0.6 0.8 0.7 0.8 0.6 0.8 0.8 0.9 0.1 0 1
0.3 0.1 0.1 0.1 0.2 0.1 0.2 0.1 0.1 1 0
0.6 0.3 0.3 0.5 0.3 1 0.3 0.5 0.3 1 0
0.5 0.1 0.1 0.1 0.2 0.2 0.3 0.3 0.1 1 0
0.8 0.6 0.7 0.3 0.3 1 0.3 0.4 0.2 0 1
0.5 0.3 0.3 0.2 0.3 0.1 0.3 0.1 0.1 1 0
0.1 0.1 0.1 0.1 0.2 0.1 0.2 0.1 0.1 1 0
0.1 0.1 0.1 0.1 0.2 0.1 0.3 0.1 0.1 1 0
0.4 0.1 0.1 0.1 0.2 0.1 0.1 0.1 0.1 1 0
0.2 0.1 0.1 0.1 0.3 0.1 0.2 0.1 0.1 1 0
0.1 0.1 0.3 0.1 0.1 0.1 0.2 0.1 0.1 1 0
0.8 1 0.8 0.8 0.4 0.8 0.7 0.7 0.1 0 1
0.7 0.1 0.2 0.3 0.2 0.1 0.2 0.1 0.1 1 0
0.5 0.1 0.1 0.1 0.2 0.1 0.2 0.1 0.1 1 0
0.1 0.1 0.1 0.1 0.2 0.1 0.3 0.1 0.1 1 0
0.1 0.1 0.1 0.1 0.2 0.1 0.1 0.1 0.1 1 0
0.1 0.1 0.1 0.1 0.1 0.1 0.1 0.3 0.1 1 0
0.6 1 1 0.2 0.8 1 0.7 0.3 0.3 0 1
0.5 0.1 0.2 0.1 0.2 0.1 0.1 0.1 0.1 1 0
0.7 0.5 0.6 1 0.4 1 0.5 0.3 0.1 0 1
0.3 0.1 0.1 0.1 0.2 0.35 0.3 0.1 0.1 1 0
0.3 0.1 0.1 0.1 0.2 0.1 0.2 0.1 0.1 1 0
0.5 0.3 0.5 0.5 0.3 0.3 0.4 1 0.1 0 1
0.1 0.1 0.1 0.1 0.2 0.1 0.2 0.1 0.1 1 0
0.3 0.1 0.1 0.1 0.2 0.1 0.3 0.1 0.1 1 0
0.5 0.7 0.9 0.8 0.6 1 0.8 1 0.1 0 1
1 0.4 0.2 0.1 0.3 0.2 0.4 0.3 1 0 1
0.8 1 1 0.8 0.5 1 0.7 0.8 0.1 0 1
0.8 1 0.3 0.2 0.6 0.4 0.3 1 0.1 0 1
1 0.1 0.1 0.1 0.2 1 0.5 0.4 0.1 0 1
0.2 0.1 0.1 0.1 0.2 0.1 0.3 0.1 0.1 1 0
0.3 0.6 0.4 1 0.3 0.3 0.3 0.4 0.1 0 1
0.3 0.1 0.1 0.1 0.2 0.2 0.7 0.1 0.1 1 0
0.3 0.3 0.1 0.1 0.2 0.1 0.1 0.1 0.1 1 0
0.3 0.1 0.1 0.1 0.2 0.1 0.1 0.1 0.1 1 0
0.7 0.4 0.7 0.4 0.3 0.7 0.7 0.6 0.1 0 1
0.1 0.1 0.1 0.1 0.2 0.1 0.1 0.1 0.1 1 0
0.9 1 1 0.1 1 0.8 0.3 0.3 0.1 0 1
0.8 0.7 0.8 0.7 0.5 0.5 0.5 1 0.2 0 1
0.4 0.2 0.4 0.3 0.2 0.2 0.2 0.1 0.1 1 0
1 0.6 0.4 0.1 0.3 0.4 0.3 0.2 0.3 0 1
0.7 0.4 0.6 0.4 0.6 0.1 0.4 0.3 0.1 0 1
0.1 0.1 0.1 0.1 0.1 0.1 0.3 0.1 0.1 1 0
0.1 0.1 0.1 0.1 0.2 0.1 0.3 0.1 0.1 1 0
0.1 0.1 0.1 0.1 0.2 0.1 0.2 0.1 0.1 1 0
0.5 0.1 0.3 0.1 0.2 0.1 0.2 0.1 0.1 1 0
0.1 0.1 0.1 0.1 0.2 0.1 0.3 0.1 0.1 1 0
1 0.4 0.7 0.2 0.2 0.8 0.6 0.1 0.1 0 1
0.5 0.7 0.4 0.1 0.6 0.1 0.7 1 0.3 0 1
0.3 0.1 0.1 0.1 0.2 0.1 0.1 0.1 0.1 1 0
0.5 0.3 0.4 0.1 0.8 1 0.4 0.9 0.1 0 1
0.1 0.1 0.1 0.1 0.2 0.1 0.1 0.1 0.1 1 0
0.3 0.1 0.1 0.1 0.2 0.1 0.2 0.1 0.1 1 0
0.6 0.9 0.7 0.5 0.5 0.8 0.4 0.2 0.1 1 0
0.2 0.1 0.1 0.1 0.2 0.1 0.2 0.2 0.1 1 0
1 1 1 1 1 0.1 0.8 0.8 0.8 0 1
0.3 0.1 0.1 0.1 0.2 0.1 0.1 0.1 0.1 1 0
0.4 0.1 0.2 0.1 0.2 0.1 0.3 0.1 0.1 1 0
0.5 0.1 0.1 0.1 0.2 0.1 0.2 0.1 0.1 1 0
0.6 1 1 1 1 1 0.8 1 1 0 1
1 1 0.8 0.6 0.4 0.5 0.8 1 0.1 0 1
0.5 0.3 0.3 0.3 0.2 0.3 0.4 0.4 0.1 0 1
0.1 0.1 0.3 0.1 0.2 0.1 0.1 0.1 0.1 1 0
0.5 0.3 0.1 0.2 0.2 0.1 0.2 0.1 0.1 1 0
0.1 0.1 0.3 0.1 0.2 0.35 0.2 0.1 0.1 1 0
1 0.3 0.6 0.2 0.3 0.5 0.4 1 0.2 0 1
0.5 0.1 0.1 0.3 0.2 0.1 0.1 0.1 0.1 1 0
0.1 0.1 0.1 0.1 0.2 0.5 0.5 0.1 0.1 1 0
0.1 0.3 0.3 0.2 0.2 0.1 0.7 0.2 0.1 1 0
0.1 0.2 0.2 0.1 0.2 0.1 0.1 0.1 0.1 1 0
0.5 0.1 0.2 0.1 0.2 0.1 0.1 0.1 0.1 1 0
0.1 0.1 0.1 0.1 0.1 0.1 0.2 0.1 0.1 1 0
0.3 0.1 0.1 0.1 0.2 0.1 0.1 0.1 0.1 1 0
0.7 0.5 0.6 0.3 0.3 0.8 0.7 0.4 0.1 0 1
0.1 0.3 0.1 0.2 0.2 0.2 0.5 0.3 0.2 1 0
0.4 0.1 0.1 0.1 0.3 0.1 0.1 0.1 0.1 1 0
0.3 0.1 0.1 0.1 0.2 0.1 0.1 0.1 0.1 1 0
0.7 0.2 0.4 0.1 0.6 1 0.5 0.4 0.3 0 1
0.1 0.1 0.1 0.1 0.2 0.1 0.2 0.1 0.1 1 0
1 1 1 1 0.3 1 1 0.6 0.1 0 1
0.5 0.4 0.4 0.5 0.7 1 0.3 0.2 0.1 1 0
0.4 0.1 0.1 0.3 0.1 0.5 0.2 0.1 0.1 0 1
1 1 1 0.6 0.8 0.4 0.8 0.5 0.1 0 1
1 1 0.8 1 0.6 0.5 1 0.3 0.1 0 1
0.3 0.1 0.1 0.1 0.2 0.1 0.3 0.1 0.1 1 0
0.5 0.1 0.1 0.1 0.2 0.1 0.2 0.1 0.1 1 0
0.4 0.1 0.1 0.1 0.3 0.1 0.2 0.2 0.1 1 0
0.2 0.1 0.1 0.2 0.2 0.1 0.3 0.1 0.1 1 0
0.1 0.1 0.1 0.1 0.2 0.1 0.3 0.1 0.1 1 0
0.5 0.1 0.1 0.2 0.1 0.1 0.2 0.1 0.1 1 0
0.1 0.1 0.1 0.1 0.2 0.3 0.3 0.1 0.1 1 0
0.1 0.1 0.1 0.1 0.2 0.1 0.1 0.1 0.1 1 0
1 0.4 0.6 0.4 0.5 1 0.7 0.1 0.1 0 1
0.8 0.8 0.9 0.4 0.5 1 0.7 0.8 0.1 0 1
0.8 1 1 0.7 1 1 0.7 0.3 0.8 0 1
0.5 0.7 0.7 0.1 0.5 0.8 0.3 0.4 0.1 1 0
1 1 1 0.1 0.6 0.1 0.2 0.8 0.1 0 1
0.2 0.1 0.1 0.1 0.2 0.1 0.3 0.1 0.1 1 0
1 0.3 0.3 1 0.2 1 0.7 0.3 0.3 0 1
0.6 0.8 0.8 0.1 0.3 0.4 0.3 0.7 0.1 1 0
0.8 1 1 1 0.8 1 1 0.7 0.3 0 1
0.3 0.1 0.1 0.1 0.2 0.35 0.3 0.1 0.1 1 0
0.1 0.1 0.1 0.1 0.2 0.1 0.1 0.1 0.1 1 0
0.5 0.1 0.1 0.1 0.2 0.1 0.2 0.2 0.1 1 0
0.1 0.1 0.1 0.1 0.2 0.4 0.1 0.1 0.1 1 0
0.5 0.4 0.6 0.7 0.9 0.7 0.8 1 0.1 0 1
0.1 0.1 0.1 0.2 0.2 0.1 0.3 0.1 0.1 1 0
0.4 0.1 0.1 0.1 0.2 0.1 0.3 0.1 0.1 1 0
0.4 0.1 0.1 0.1 0.2 0.1 0.1 0.1 0.1 1 0
0.8 0.2 0.4 0.1 0.5 0.1 0.5 0.4 0.4 0 1
0.1 0.2 0.3 0.1 0.2 0.1 0.1 0.1 0.1 1 0
1 0.7 0.7 0.6 0.4 1 0.4 0.1 0.2 0 1
0.1 0.1 0.1 0.1 0.2 0.1 0.3 0.1 0.1 1 0
0.5 0.3 0.2 0.1 0.3 0.1 0.1 0.1 0.1 1 0
1 1 1 1 1 1 0.4 1 1 0 1
0.1 0.1 0.1 0.1 0.2 0.2 0.1 0.1 0.1 1 0
0.5 0.1 0.2 0.1 0.2 0.1 0.2 0.1 0.1 1 0
0.4 0.4 0.4 0.2 0.2 0.3 0.2 0.1 0.1 1 0
0.6 0.2 0.3 0.1 0.2 0.1 0.1 0.1 0.1 1 0
0.8 0.4 1 0.5 0.4 0.4 0.7 1 0.1 0 1
0.1 0.2 0.3 0.1 0.2 0.1 0.3 0.1 0.1 1 0
0.3 0.2 0.2 0.1 0.2 0.1 0.2 0.3 0.1 1 0
1 0.5 0.7 0.4 0.4 1 0.8 0.9 0.1 0 1
0.6 0.3 0.3 0.3 0.3 0.2 0.6 0.1 0.1 1 0
0.1 0.1 0.1 0.1 0.2 0.1 0.1 0.1 0.1 1 0
1 1 1 0.8 0.6 0.8 0.7 1 0.1 0 1
0.1 0.1 0.1 0.1 0.2 0.5 0.1 0.1 0.1 1 0
1 0.5 1 0.3 0.5 0.8 0.7 0.8 0.3 0 1
0.1 0.1 0.1 0.1 0.1 0.1 0.1 0.1 0.1 1 0
0.8 0.9 0.9 0.5 0.3 0.5 0.7 0.7 0.1 0 1
0.1 0.1 0.1 0.1 0.2 0.1 0.1 0.1 0.1 1 0
0.5 0.1 0.1 0.1 0.2 0.1 0.2 0.1 0.1 1 0
0.3 0.3 0.5 0.2 0.3 1 0.7 0.1 0.1 0 1
0.1 0.2 0.1 0.3 0.2 0.1 0.2 0.1 0.1 1 0
0.4 0.1 0.3 0.3 0.2 0.1 0.1 0.1 0.1 1 0
0.4 0.1 0.1 0.1 0.2 0.3 0.1 0.1 0.1 1 0
0.7 0.5 1 1 1 1 0.4 1 0.3 0 1
0.2 0.1 0.1 0.1 0.2 0.1 0.2 0.1 0.1 1 0
0.3 0.2 0.2 0.2 0.2 0.1 0.3 0.2 0.1 1 0
0.3 0.1 0.1 0.1 0.1 0.1 0.2 0.1 0.1 1 0
0.5 0.3 0.1 0.1 0.2 0.1 0.1 0.1 0.1 1 0
0.8 0.8 0.9 0.6 0.6 0.3 1 1 0.1 0 1
0.3 0.1 0.1 0.1 0.2 0.1 0.2 0.2 0.1 1 0
1 0.5 0.5 0.6 0.3 1 0.7 0.9 0.2 0 1
0.4 0.1 0.1 0.1 0.2 0.1 0.3 0.6 0.1 1 0
1 0.4 0.5 0.4 0.3 0.5 0.7 0.3 0.1 0 1
0.5 0.3 0.5 0.1 0.8 1 0.5 0.3 0.1 0 1
0.1 0.1 0.1 0.1 0.2 0.1 0.3 0.1 0.1 1 0
0.7 0.8 0.8 0.7 0.3 1 0.7 0.2 0.3 0 1
0.2 0.1 0.1 0.1 0.2 0.5 0.1 0.1 0.1 1 0
0.6 0.6 0.6 0.9 0.6 0.35 0.7 0.8 0.1 1 0
0.4 0.5 0.5 0.8 0.6 1 1 0.7 0.1 0 1
0.1 0.1 0.1 0.1 0.2 0.1 0.2 0.1 0.1 1 0
0.3 0.1 0.1 0.1 0.2 0.1 0.2 0.1 0.1 1 0
1 1 1 0.3 1 1 0.9 1 0.1 0 1
0.8 1 0.5 0.3 0.8 0.4 0.4 1 0.3 0 1
0.5 0.1 0.3 0.1 0.2 0.1 0.2 0.1 0.1 1 0
0.5 0.4 0.4 0.9 0.2 1 0.5 0.6 0.1 0 1
0.5 0.5 0.5 0.2 0.5 1 0.4 0.3 0.1 0 1
0.3 0.1 0.1 0.1 0.3 0.2 0.1 0.1 0.1 1 0
1 0.7 0.7 0.3 0.8 0.5 0.7 0.4 0.3 0 1
0.4 0.8 0.7 1 0.4 1 0.7 0.5 0.1 0 1
0.5 0.1 0.1 0.1 0.2 0.1 0.1 0.1 0.1 1 0
0.3 0.4 0.5 0.3 0.7 0.3 0.4 0.6 0.1 1 0
0.6 0.1 0.1 0.3 0.2 0.1 0.1 0.1 0.1 1 0
0.1 0.1 0.1 0.1 0.2 0.1 0.3 0.1 0.1 1 0
0.3 0.1 0.1 0.1 0.2 0.1 0.2 0.1 0.1 1 0
0.3 0.1 0.1 0.1 0.2 0.4 0.1 0.1 0.1 1 0
0.5 0.4 0.6 0.6 0.4 1 0.4 0.3 0.1 0 1
0.4 0.1 0.1 0.1 0.2 0.1 0.2 0.1 0.1 1 0
0.5 0.1 0.2 0.1 0.2 0.1 0.3 0.1 0.1 1 0
0.3 0.2 0.1 0.1 0.2 0.1 0.2 0.2 0.1 1 0
0.1 0.1 0.1 0.1 0.2 0.1 0.1 0.1 0.1 1 0
0.4 0.8 0.6 0.4 0.3 0.4 1 0.6 0.1 0 1
0.5 0.2 0.2 0.2 0.3 0.1 0.1 0.3 0.1 1 0
1 0.6 0.6 0.2 0.4 1 0.9 0.7 0.1 0 1
0.3 0.2 0.2 0.1 0.4 0.3 0.2 0.1 0.1 1 0
0.6 1 0.2 0.8 1 0.2 0.7 0.8 1 0 1
1 0.4 0.4 1 0.2 1 0.5 0.3 0.3 0 1
0.1 0.1 0.1 0.3 0.2 0.1 0.1 0.1 0.1 1 0
0.1 0.1 0.1 0.1 0.1 0.1 0.1 0.1 0.1 1 0
0.3 0.1 0.1 0.1 0.2 0.1 0.2 0.1 0.1 1 0
0.1 0.1 0.1 0.1 0.2 0.1 0.2 0.1 0.1 1 0
0.1 0.1 0.1 0.1 0.2 0.1 0.1 0.1 0.1 1 0
1 0.4 0.3 1 0.3 1 0.7 0.1 0.2 0 1
0.1 0.1 0.1 0.1 0.2 0.1 0.3 0.1 0.1 1 0
0.3 0.1 0.1 0.3 0.2 0.1 0.2 0.1 0.1 1 0
0.1 0.1 0.1 0.1 0.2 0.1 0.1 0.1 0.1 1 0
0.5 0.1 0.1 0.1 0.2 0.1 0.1 0.1 0.1 1 0
0.4 0.1 0.4 0.1 0.2 0.1 0.1 0.1 0.1 1 0
0.8 0.8 0.7 0.4 1 1 0.7 0.8 0.7 0 1
0.1 0.1 0.2 0.1 0.3 0.35 0.1 0.1 0.1 1 0
0.5 0.1 0.1 0.1 0.2 0.1 0.3 0.1 0.1 1 0
0.6 0.1 0.1 0.1 0.2 0.1 0.3 0.1 0.1 1 0
0.6 0.3 0.4 0.1 0.5 0.2 0.3 0.9 0.1 0 1
0.4 0.1 0.1 0.3 0.2 0.1 0.3 0.1 0.1 1 0
0.2 0.1 0.1 0.1 0.2 0.1 0.1 0.1 0.1 1 0
0.5 0.1 0.1 0.1 0.2 0.1 0.3 0.1 0.1 1 0
0.5 0.8 0.7 0.7 1 1 0.5 0.7 0.1 0 1
0.6 0.1 0.3 0.2 0.2 0.1 0.1 0.1 0.1 1 0
0.1 0.1 0.1 0.1 0.4 0.3 0.1 0.1 0.1 1 0
0.2 0.1 0.3 0.2 0.2 0.1 0.2 0.1 0.1 1 0
0.5 0.3 0.2 0.4 0.2 0.1 0.1 0.1 0.1 1 0
0.6 0.1 0.3 0.1 0.4 0.5 0.5 1 0.1 0 1
0.4 0.1 0.1 0.1 0.2 0.1 0.3 0.1 0.1 1 0
0.2 0.1 0.1 0.1 0.2 0.1 0.2 0.1 0.1 1 0
0.1 0.1 0.1 0.1 0.2 0.1 0.3 0.1 0.1 1 0
0.2 0.1 0.1 0.1 0.2 0.1 0.3 0.1 0.1 1 0
0.7 0.6 1 0.5 0.3 1 0.9 1 0.2 0 1
0.2 0.1 0.1 0.1 0.2 0.1 0.1 0.1 0.1 1 0
0.8 1 1 0.8 0.6 0.9 0.3 1 1 0 1
0.1 0.1 0.1 0.3 0.1 0.3 0.1 0.1 0.1 1 0
0.3 0.1 0.1 0.1 0.2 0.1 0.1 0.1 0.1 1 0
0.1 0.1 0.1 0.1 0.2 0.1 0.2 0.1 0.1 1 0
1 0.4 0.6 0.1 0.2 1 0.5 0.3 0.1 0 1
0.1 0.1 0.1 0.1 0.1 0.1 0.3 0.1 0.1 1 0
0.6 0.1 0.1 0.1 0.1 0.1 0.1 0.1 0.1 1 0
0.1 0.1 0.1 0.1 0.3 0.2 0.2 0.1 0.1 1 0
0.4 0.1 0.1 0.3 0.2 0.1 0.3 0.1 0.1 1 0
0.3 0.1 0.1 0.1 0.2 0.1 0.3 0.1 0.1 1 0
0.1 0.1 0.1 0.1 0.1 0.1 0.1 0.1 0.1 1 0
0.5 0.1 0.1 0.1 0.2 0.1 0.3 0.1 0.1 1 0
0.2 0.1 0.1 0.1 0.2 0.1 0.2 0.1 0.1 1 0
1 0.3 0.3 0.1 0.2 1 0.7 0.6 0.1 0 1
0.1 0.1 0.1 0.1 0.2 0.1 0.2 0.1 0.1 1 0
0.5 0.2 0.3 0.1 0.6 1 0.5 0.1 0.1 0 1
0.4 0.7 0.8 0.3 0.4 1 0.9 0.1 0.1 0 1
0.3 0.2 0.1 0.2 0.2 0.1 0.3 0.1 0.1 1 0
0.1 0.1 0.1 0.1 0.2 0.1 0.3 0.1 0.1 1 0
0.3 0.1 0.1 0.1 0.2 0.1 0.3 0.1 0.1 1 0
0.1 0.1 0.1 0.1 0.2 0.1 0.3 0.2 0.1 1 0
0.1 0.1 0.1 0.1 0.2 0.1 0.2 0.1 0.1 1 0
0.2 0.1 0.1 0.1 0.2 0.1 0.2 0.1 0.1 1 0
0.4 0.2 0.1 0.1 0.2 0.1 0.2 0.1 0.1 1 0
0.5 0.2 0.2 0.2 0.2 0.2 0.3 0.2 0.2 1 0
0.4 0.1 0.1 0.2 0.2 0.1 0.1 0.1 0.1 1 0
0.5 0.3 0.4 0.3 0.4 0.5 0.4 0.7 0.1 1 0
0.4 0.1 0.1 0.2 0.2 0.1 0.2 0.1 0.1 1 0
0.3 0.1 0.2 0.1 0.2 0.1 0.3 0.1 0.1 1 0
0.7 0.3 0.2 1 0.5 1 0.5 0.4 0.4 0 1
0.4 0.1 0.1 0.1 0.2 0.1 0.3 0.2 0.1 1 0
0.1 0.1 0.1 0.1 0.1 0.1 0.1 0.1 0.1 1 0
0.7 0.5 0.3 0.7 0.4 1 0.7 0.5 0.5 0 1
0.1 0.1 0.1 0.1 0.2 0.1 0.1 0.1 0.1 1 0
0.8 0.7 0.5 1 0.7 0.9 0.5 0.5 0.4 0 1
0.5 0.1 0.2 0.1 0.2 0.1 0.3 0.1 0.1 1 0
0.3 0.1 0.1 0.1 0.1 0.1 0.2 0.1 0.1 1 0
0.3 0.1 0.1 0.1 0.2 0.1 0.2 0.1 0.1 1 0
0.5 0.3 0.3 0.4 0.2 0.4 0.3 0.4 0.1 0 1
1 0.6 0.6 0.3 0.4 0.5 0.3 0.6 0.1 0 1
0.2 0.1 0.1 0.1 0.2 0.1 0.3 0.1 0.1 1 0
0.1 0.1 0.1 0.1 0.1 0.1 0.3 0.1 0.1 1 0
0.5 1 1 0.5 0.4 0.5 0.4 0.4 0.1 0 1
0.4 0.1 0.1 0.1 0.2 0.1 0.3 0.1 0.1 1 0
0.4 0.1 0.1 0.1 0.2 0.1 0.1 0.2 0.1 1 0
0.4 0.1 0.1 0.1 0.2 0.1 0.1 0.1 0.1 1 0
0.3 0.1 0.1 0.1 0.2 0.1 0.2 0.1 0.1 1 0
0.6 0.1 0.1 0.1 0.2 0.1 0.3 0.1 0.1 1 0
0.8 0.4 0.7 0.1 0.3 1 0.3 0.9 0.2 0 1
0.8 1 1 1 0.7 0.5 0.4 0.8 0.7 0 1
1 0.6 0.4 0.3 1 1 0.9 1 0.1 0 1
1 0.8 0.7 0.4 0.3 1 0.7 0.9 0.1 0 1
1 0.8 1 0.1 0.3 1 0.5 0.1 0.1 0 1
0.8 0.7 0.6 0.4 0.4 1 0.5 0.1 0.1 0 1
0.8 0.2 0.3 0.1 0.6 0.3 0.7 0.1 0.1 0 1
0.1 0.1 0.1 0.1 0.2 0.1 0.1 0.1 0.1 1 0
0.5 0.1 0.1 0.1 0.2 0.1 0.1 0.1 0.1 1 0
0.1 0.1 0.1 0.1 0.1 0.1 0.2 0.1 0.1 1 0
0.4 0.1 0.1 0.1 0.2 0.1 0.1 0.1 0.1 1 0
0.5 0.1 0.3 0.3 0.2 0.2 0.2 0.3 0.1 1 0
1 1 1 1 0.7 1 0.7 1 0.4 0 1
0.4 0.1 0.2 0.1 0.2 0.1 0.2 0.1 0.1 1 0
0.1 0.5 0.8 0.6 0.5 0.8 0.7 1 0.1 0 1
0.4 0.1 0.1 0.1 0.1 0.1 0.2 0.1 0.1 1 0
0.1 0.1 0.1 0.1 0.2 0.1 0.1 0.1 0.1 1 0
0.5 0.5 0.7 0.8 0.6 1 0.7 0.4 0.1 0 1
0.1 0.1 0.1 0.1 0.1 0.1 0.2 0.1 0.1 1 0
0.4 0.1 0.1 0.1 0.2 0.1 0.3 0.1 0.1 1 0
0.4 0.1 0.1 0.1 0.2 0.1 0.3 0.1 0.1 1 0
0.1 0.1 0.1 0.1 0.2 0.5 0.1 0.1 0.1 1 0
1 0.9 0.7 0.3 0.4 0.2 0.7 0.7 0.1 0 1
0.2 0.1 0.1 0.1 0.2 0.1 0.3 0.1 0.1 1 0
0.5 0.8 0.9 0.4 0.3 1 0.7 0.1 0.1 0 1
0.1 0.1 0.1 0.3 0.2 0.3 0.1 0.1 0.1 1 0
0.4 0.1 0.1 0.1 0.2 0.1 0.1 0.1 0.1 1 0
0.4 0.2 0.1 0.1 0.2 0.1 0.1 0.1 0.1 1 0
0.5 0.4 0.3 0.1 0.2 0.35 0.2 0.3 0.1 1 0
0.4 0.1 0.1 0.1 0.2 0.3 0.2 0.1 0.1 1 0
0.1 0.1 0.1 0.1 0.2 0.1 0.3 0.1 0.1 1 0
1 0.6 0.3 0.6 0.4 1 0.7 0.8 0.4 0 1
0.5 0.2 0.2 0.2 0.1 0.1 0.2 0.1 0.1 1 0
0.3 0.1 0.4 0.1 0.2 0.35 0.3 0.1 0.1 1 0
0.5 1 1 0.6 1 1 1 0.6 0.5 0 1
0.4 0.1 0.1 0.1 0.2 0.1 0.2 0.1 0.1 1 0
0.1 0.1 0.3 0.1 0.2 0.1 0.2 0.1 0.1 1 0
0.1 0.1 0.1 0.2 0.1 0.3 0.1 0.1 0.7 1 0
0.5 0.1 0.1 0.1 0.2 0.1 0.2 0.2 0.1 1 0
0.3 0.1 0.1 0.1 0.2 0.1 0.2 0.1 0.1 1 0
0.3 0.1 0.1 0.1 0.3 0.1 0.2 0.1 0.1 1 0
1 0.2 0.2 0.1 0.2 0.6 0.1 0.1 0.2 0 1
0.5 1 1 1 1 1 1 0.1 0.1 0 1
0.5 0.1 0.1 0.3 0.2 0.1 0.1 0.1 0.1 1 0
0.4 0.1 0.1 0.1 0.2 0.1 0.2 0.1 0.1 1 0
0.5 0.1 0.1 0.3 0.4 0.1 0.3 0.2 0.1 1 0
0.7 0.6 0.4 0.8 1 1 0.9 0.5 0.3 0 1
0.2 0.1 0.1 0.1 0.2 0.1 0.2 0.1 0.1 1 0
0.5 0.2 0.3 0.4 0.2 0.7 0.3 0.6 0.1 0 1
0.9 1 1 0.1 1 0.8 0.3 0.3 0.1 0 1
0.4 0.1 0.2 0.1 0.2 0.1 0.2 0.1 0.1 1 0
0.6 0.5 0.5 0.8 0.4 1 0.3 0.4 0.1 0 1
1 0.4 0.4 0.6 0.2 1 0.2 0.3 0.1 0 1
0.1 0.1 0.1 0.1 0.2 0.1 0.3 0.1 0.1 1 0
1 0.8 0.8 0.2 0.3 0.4 0.8 0.7 0.8 0 1
1 0.4 0.4 1 0.6 1 0.5 0.5 0.1 0 1
0.8 0.6 0.4 0.3 0.5 0.9 0.3 0.1 0.1 0 1
0.7 0.6 0.6 0.3 0.2 1 0.7 0.1 0.1 0 1
0.2 0.1 0.2 0.1 0.2 0.1 0.3 0.1 0.1 1 0
0.3 0.1 0.1 0.1 0.2 0.1 0.1 0.1 0.1 1 0
0.2 0.1 0.1 0.1 0.2 0.1 0.3 0.1 0.1 1 0
0.9 0.9 1 0.3 0.6 1 0.7 1 0.6 0 1
0.3 0.1 0.1 0.1 0.2 0.1 0.3 0.2 0.1 1 0
0.1 0.1 0.1 0.1 0.2 0.1 0.3 0.1 0.1 1 0
0.5 1 1 1 0.5 0.2 0.8 0.5 0.1 0 1
1 1 1 0.8 0.6 0.1 0.8 0.9 0.1 0 1
0.3 0.1 0.1 0.1 0.3 0.1 0.2 0.1 0.1 1 0
0.5 0.8 0.8 1 0.5 1 0.8 1 0.3 0 1
0.1 0.1 0.3 0.1 0.2 0.1 0.1 0.1 0.1 1 0
0.4 0.1 0.1 0.1 0.2 0.1 0.3 0.2 0.1 1 0
0.6 1 0.7 0.7 0.6 0.4 0.8 1 0.2 0 1
0.3 0.1 0.1 0.1 0.1 0.1 0.2 0.1 0.1 1 0
1 1 1 0.7 0.9 1 0.7 1 1 0 1
0.5 0.3 0.2 0.8 0.5 1 0.8 0.1 0.2 0 1
1 1 0.6 0.3 0.3 1 0.4 0.3 0.2 0 1
0.3 0.1 0.1 0.1 0.2 0.1 0.1 0.1 0.1 1 0
0.9 0.4 0.5 1 0.6 1 0.4 0.8 0.1 0 1
0.1 0.1 0.1 0.1 0.1 0.35 0.2 0.1 0.1 1 0
0.4 0.1 0.1 0.1 0.2 0.1 0.1 0.1 0.1 1 0
0.8 0.3 0.3 0.1 0.2 0.2 0.3 0.2 0.1 1 0
0.8 0.4 0.4 0.1 0.6 1 0.2 0.5 0.2 0 1
0.1 0.1 0.1 0.1 0.2 0.1 0.2 0.1 0.1 1 0
0.3 0.1 0.1 0.1 0.1 0.1 0.2 0.1 0.1 1 0
0.2 0.3 0.4 0.4 0.2 0.5 0.2 0.5 0.1 0 1
0.3 0.2 0.1 0.1 0.1 0.1 0.2 0.1 0.1 1 0
0.3 0.3 0.6 0.4 0.5 0.8 0.4 0.4 0.1 0 1
0.1 0.1 0.1 0.1 0.2 0.35 0.2 0.1 0.1 1 0
0.5 0.2 0.1 0.1 0.2 0.1 0.1 0.1 0.1 1 0
0.1 0.1 0.1 0.1 0.2 0.1 0.1 0.1 0.1 1 0
0.1 0.1 0.1 0.1 0.2 0.1 0.1 0.1 0.1 1 0
0.3 0.1 0.3 0.1 0.2 0.35 0.2 0.1 0.1 1 0
1 1 1 1 0.5 1 1 1 0.7 0 1
0.5 0.1 0.2 0.1 0.2 0.1 0.3 0.1 0.1 1 0
0.3 0.1 0.1 0.1 0.2 0.1 0.2 0.1 0.1 1 0
0.4 0.1 0.1 0.1 0.1 0.1 0.2 0.1 0.1 1 0
0.3 0.1 0.1 0.1 0.2 0.1 0.2 0.1 0.1 1 0
0.4 0.1 0.3 0.1 0.2 0.1 0.2 0.1 0.1 1 0
1 1 1 0.3 1 0.8 0.8 0.1 0.1 0 1
0.4 0.1 0.1 0.3 0.1 0.1 0.2 0.1 0.1 1 0
0.3 0.1 0.1 0.1 0.2 0.1 0.3 0.1 0.1 1 0
0.5 0.1 0.1 0.1 0.1 0.1 0.1 0.1 0.1 1 0
0.1 0.1 0.1 0.1 0.2 0.1 0.3 0.1 0.1 1 0
0.5 0.2 0.4 0.1 0.1 0.1 0.1 0.1 0.1 1 0
0.2 0.3 0.2 0.2 0.2 0.2 0.3 0.1 0.1 1 0
0.5 1 1 0.3 0.7 0.3 0.8 1 0.2 0 1
0.8 0.4 0.5 0.1 0.2 0.35 0.7 0.3 0.1 0 1
0.9 1 1 1 1 1 1 1 0.1 0 1
0.4 0.4 0.2 0.1 0.2 0.5 0.2 0.1 0.2 1 0
0.7 0.2 0.4 0.1 0.3 0.4 0.3 0.3 0.1 0 1
0.4 0.1 0.1 0.2 0.2 0.1 0.2 0.1 0.1 1 0
0.8 0.6 0.4 1 1 0.1 0.3 0.5 0.1 0 1
0.5 0.1 0.1 0.1 0.2 0.1 0.1 0.1 0.1 1 0
0.7 0.4 0.5 1 0.2 1 0.3 0.8 0.2 0 1
0.5 0.2 0.2 0.4 0.2 0.4 0.1 0.1 0.1 1 0
0.3 0.2 0.2 0.3 0.2 0.3 0.3 0.1 0.1 1 0
0.5 0.1 0.1 0.1 0.2 0.1 0.3 0.1 0.1 1 0
0.6 0.1 0.3 0.1 0.2 0.1 0.3 0.1 0.1 1 0
0.3 0.1 0.1 0.1 0.2 0.1 0.3 0.1 0.1 1 0
1 0.4 0.5 0.5 0.5 1 0.4 0.1 0.1 0 1
0.5 0.1 0.1 0.3 0.2 0.1 0.1 0.1 0.1 1 0
0.6 0.1 0.1 0.1 0.2 0.1 0.3 0.1 0.1 1 0
0.4 0.1 0.1 0.3 0.2 0.1 0.1 0.1 0.1 1 0
0.3 0.1 0.1 0.1 0.2 0.1 0.2 0.1 0.1 1 0
0.5 1 1 0.8 0.5 0.5 0.7 1 0.1 0 1
1 0.6 0.5 0.8 0.5 1 0.8 0.6 0.1 0 1
0.2 0.1 0.1 0.1 0.3 0.1 0.2 0.1 0.1 1 0
0.4 0.1 0.1 0.1 0.2 0.1 0.3 0.1 0.1 1 0
0.1 0.1 0.2 0.1 0.2 0.1 0.2 0.1 0.1 1 0
0.1 0.3 0.1 0.1 0.2 0.1 0.2 0.2 0.1 1 0
0.2 0.2 0.2 0.1 0.1 0.1 0.7 0.1 0.1 1 0
0.9 0.8 0.8 0.9 0.6 0.3 0.4 0.1 0.1 0 1
0.5 0.1 0.4 0.1 0.2 0.1 0.3 0.2 0.1 1 0
0.1 0.1 0.1 0.1 0.1 0.1 0.3 0.1 0.1 1 0
0.5 0.1 0.1 0.1 0.2 0.1 0.1 0.1 0.1 1 0
1 1 0.7 0.8 0.7 0.1 1 1 0.3 0 1
0.2 0.1 0.1 0.2 0.2 0.1 0.3 0.1 0.1 1 0
0.1 0.2 0.1 0.3 0.2 0.1 0.1 0.2 0.1 1 0
0.5 0.3 0.6 0.1 0.2 0.1 0.1 0.1 0.1 1 0
0.1 0.1 0.1 0.1 0.1 0.1 0.3 0.1 0.1 1 0
0.2 0.1 0.1 0.1 0.2 0.1 0.1 0.1 0.1 1 0
0.3 0.1 0.1 0.3 0.1 0.1 0.3 0.1 0.1 1 0
0.5 1 1 0.9 0.6 1 0.7 1 0.5 0 1
0.1 0.1 0.1 0.1 0.2 0.1 0.3 0.1 0.1 1 0
0.1 0.1 0.1 0.1 0.2 0.1 0.2 0.1 0.1 1 0
0.5 0.6 0.6 0.2 0.4 1 0.3 0.6 0.1 0 1
0.3 0.1 0.2 0.2 0.2 0.1 0.1 0.1 0.1 1 0
0.9 0.6 0.9 0.2 1 0.6 0.2 0.9 1 0 1
0.5 0.7 1 0.6 0.5 1 0.7 0.5 0.1 0 1
0.6 1 1 0.2 0.8 1 0.7 0.3 0.3 0 1
0.9 1 1 1 1 0.5 1 1 1 0 1
0.6 0.5 0.4 0.4 0.3 0.9 0.7 0.8 0.3 0 1
1 1 1 0.8 0.2 1 0.4 0.1 0.1 0 1
0.1 0.2 0.3 0.1 0.2 0.1 0.2 0.1 0.1 1 0
0.9 0.5 0.5 0.4 0.4 0.5 0.4 0.3 0.3 0 1
0.1 0.1 0.1 0.1 0.2 0.1 0.3 0.1 0.1 1 0
0.3 0.1 0.1 0.2 0.2 0.1 0.1 0.1 0.1 1 0
0.5 0.1 0.1 0.3 0.2 0.1 0.1 0.1 0.1 1 0
0.4 0.1 0.2 0.1 0.2 0.1 0.1 0.1 0.1 1 0
0.8 1 1 1 0.6 1 1 1 0.1 0 1
0.4 0.2 0.1 0.1 0.2 0.2 0.3 0.1 0.1 1 0
0.2 0.1 0.1 0.1 0.2 0.1 0.2 0.1 0.1 1 0
0.7 0.3 0.4 0.4 0.3 0.3 0.3 0.2 0.7 0 1
0.1 0.1 0.1 0.1 0.2 0.1 0.3 0.1 0.1 1 0
0.6 0.1 0.1 0.3 0.2 0.1 0.1 0.1 0.1 1 0
0.3 0.1 0.1 0.1 0.2 0.1 0.2 0.1 0.1 1 0
0.7 0.8 0.7 0.6 0.4 0.3 0.8 0.8 0.4 0 1
0.8 0.7 0.8 0.2 0.4 0.2 0.5 1 0.1 0 1
0.1 0.1 0.1 0.1 0.2 0.1 0.3 0.1 0.1 1 0
0.5 0.1 0.1 0.1 0.2 0.1 0.3 0.1 0.1 1 0
0.5 0.1 0.1 0.1 0.2 0.1 0.1 0.1 0.1 1 0
0.6 1 1 1 0.4 1 0.7 1 0.1 0 1
1 0.4 0.3 1 0.4 1 1 0.1 0.1 0 1
0.4 0.1 0.1 0.1 0.2 0.1 0.1 0.1 0.1 1 0
0.4 0.6 0.6 0.5 0.7 0.6 0.7 0.7 0.3 0 1
0.3 0.1 0.1 0.2 0.2 0.1 0.1 0.1 0.1 1 0
0.1 0.2 0.2 0.1 0.2 0.1 0.1 0.1 0.1 1 0
0.5 0.8 0.4 1 0.5 0.8 0.9 1 0.1 0 1
0.5 0.4 0.6 1 0.2 1 0.4 0.1 0.1 0 1
0.8 0.7 0.8 0.5 0.5 1 0.9 1 0.1 0 1
0.3 0.1 0.1 0.1 0.2 0.5 0.5 0.1 0.1 1 0
0.3 0.1 0.2 0.1 0.2 0.1 0.2 0.1 0.1 1 0
0.1 0.1 0.1 0.1 0.2 0.1 0.1 0.1 0.1 1 0
0.9 0.5 0.8 0.1 0.2 0.3 0.2 0.1 0.5 0 1
0.5 0.1 0.1 0.1 0.2 0.1 0.2 0.1 0.1 1 0
0.5 0.1 0.1 0.1 0.2 0.1 0.2 0.1 0.1 1 0
0.5 0.2 0.1 0.1 0.2 0.1 0.3 0.1 0.1 1 0
0.5 0.1 0.1 0.4 0.2 0.1 0.3 0.1 0.1 1 0
0.1 0.1 0.2 0.1 0.2 0.2 0.4 0.2 0.1 1 0
0.9 0.7 0.7 0.5 0.5 1 0.7 0.8 0.3 0 1
0.6 0.1 0.1 0.1 0.2 0.1 0.2 0.1 0.1 1 0
0.3 1 0.3 1 0.6 1 0.5 0.1 0.4 0 1
0.1 0.1 0.1 0.1 0.2 0.1 0.1 0.1 0.1 1 0
0.1 0.4 0.3 1 0.4 1 0.5 0.6 0.1 0 1
0.2 0.1 0.1 0.2 0.3 0.1 0.2 0.1 0.1 1 0
0.4 0.1 0.1 0.1 0.2 0.1 0.2 0.1 0.1 1 0
0.6 0.3 0.2 0.1 0.3 0.4 0.4 0.1 0.1 0 1
0.5 0.1 0.2 0.1 0.2 0.1 0.1 0.1 0.1 1 0
0.7 0.5 0.6 1 0.5 1 0.7 0.9 0.4 0 1
0.6 1 1 1 0.8 1 0.7 1 0.7 0 1
0.5 0.7 1 1 0.5 1 1 1 0.1 0 1
0.1 0.1 0.1 0.1 0.2 0.1 0.2 0.1 0.1 1 0
];




%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

input_d = 9;
output_d = 1;
P_num = 350;
validP_num = 175; 
testP_num = 174;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% random select

% rPerm = randperm(size(total_sample,1));
% 
% P = total_sample(rPerm(1:P_num), 1:1:input_d);
% T = total_sample(rPerm(1:P_num), (input_d+1):1:(input_d+output_d));
% 
% validP = total_sample(rPerm((P_num+1):size(rPerm,2)), 1:1:input_d);
% validT = total_sample(rPerm((P_num+1):size(rPerm,2)), (input_d+1):1:(input_d+output_d));

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% select by the squence

P = total_sample(1:1:P_num, 1:1:input_d);
T = total_sample(1:1:P_num, (input_d+1):1:(input_d+output_d));

validP = total_sample((P_num+1):1:(P_num+validP_num), 1:1:input_d);
validT = total_sample((P_num+1):1:(P_num+validP_num), (input_d+1):1:(input_d+output_d));

total_num = size(total_sample,1);
testP = total_sample((total_num):-1:(total_num-testP_num), 1:1:input_d);
testT = total_sample((total_num):-1:(total_num-testP_num), (input_d+1):1:(input_d+output_d));

P = P'; T = T';
validP = validP'; validT = validT';
testP = testP'; testT = testT';

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


% for i = 1:size(P,2)
%     [Pn(:,i),minp,maxp,Tn,mint,maxt]=premnmx(P(:,i),T);
% end
% 
% P = Pn;
% T = Tn;
% 
% for i = 1:size(validP,2)
%     [validPn(:,i),minp,maxp,Tn1,mint,maxt]=premnmx(validP(:,i),validT);
% end
% 
% validP = validPn;
% validT = Tn1;
% 
% 
% for i = 1:size(testP,2)
%     [testPn(:,i),minp,maxp,Tn2,mint,maxt]=premnmx(testP(:,i),testT);
% end
% 
% testP = testPn;
% testT = Tn2;

% [n,d] = size(P);
% A = mean(P);
% B = std(P);
% P1 = P - repmat(A,n,1);
% P = P1.*repmat(1./B,n,1);
% 
% 
% [n,d] = size(validP);
% validP1 = validP - repmat(A,n,1);
% validP = validP1.*repmat(1./B,n,1);








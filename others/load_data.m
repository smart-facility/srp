function [P,T,validP,validT,testP,testT]=cancer1()

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
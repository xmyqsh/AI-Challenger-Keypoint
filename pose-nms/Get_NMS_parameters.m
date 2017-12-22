clear all;
validation_ground_A=importdata('./NMS_data/validation_ground_A.mat');
validation_ground_B=importdata('./NMS_data/validation_ground_B.mat');
validation_ground=importdata('./NMS_data/validation_ground_new.mat');
det_validation_A=importdata('./NMS_data/det_validation_A.mat');
det_validation_B=importdata('./NMS_data/det_validation_B.mat');
det_validation=importdata('./NMS_data/det_validation.mat');
delta1 = 0.01; 
delta2 = 2.08;
mu = 2.08; 
gamma = 3;
var=[delta1,delta2,mu,gamma];
it_times=1000;
[nag_log_min,var,min_test] = SA_float_general( validation_ground_A,det_validation_A,var,it_times);
tic;
delta1=var(1);
delta2=var(2);
mu=var(3);
gamma=var(4);
[ det_new ] = del_nms( det_validation_B,delta1,mu,delta2,gamma );
temp=cal_mAP( validation_ground_B,det_new );
sum=0;
for i=1:1:10
    sum(i)=length(find(temp>0.5+(i-1)*0.05))/length(temp);
end
final_oks=mean(sum)
toc;


rand_test=randperm(29993);
delta1 = 0.05; 
gamma =2;
del_score_thr=0.214;
delta2 = 1;
mu = 2.2; 
%gamma =12.5;
mu1 =2;
mask_thr=0.2;
mu2=2;
dis_thr=2;
%var=[delta1,mu,delta2,gamma,mu1,mask_thr,mu2,dis_thr];
%var=[delta1,gamma,del_score_thr,delta2,mu,mu2,dis_thr];
var=importdata('var.mat');
it_times=30000;
%[nag_log_min,var,min_test] = SA_float_general( validation_ground_B,det_validation_B,var,it_times);
tic;
delta1=var(1);
gamma=var(2);
del_score_thr=var(3);
delta2=1000*var(4);
mu=var(5);

%mu1=var(5);
%mask_thr=var(6);
mu2=var(6);
dis_thr=1;
final_oks=[];
for k=1:1:1
    %gamma=gamma+0.1;
    %delta1=delta1*2;
    %delta1 =delta1 +0.01;
   % mu=mu+0.1;
    %[ det_new ] = del_nms( det_validation,delta1,mu,delta2,gamma );
    %[ det_new ] = del_nms_box_IoU( det_validation(rand_test(1:5000)),delta1,mu,delta2,gamma,mu2,dis_thr,del_score_thr );
    %temp=cal_mAP( validation_ground_new(rand_test(1:5000)),det_new );
    %[ det_new ] = del_nms_box_IoU( det_validation,delta1,mu,delta2,gamma,mu2,dis_thr,del_score_thr );
    %temp=cal_mAP( validation_ground_new,det_new );
    temp=cal_mAP( validation_ground_new,det_validation );
    sum1=0;
    for i=1:1:10
        sum1(i)=length(find(temp>0.5+(i-1)*0.05))/length(temp);
    end
    final_oks(k)=mean(sum1)
    toc;
end
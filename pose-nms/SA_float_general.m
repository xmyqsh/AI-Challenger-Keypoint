function [nag_log_min,var,min_test] = SA_float_general( validation_ground_A,det_validation_A,var,it_times)
var(1)=1*rand(1);
%var(2)=5*rand(1);
var(2)=20*rand(1);
var(3)=0.3*rand(1);
var(4)=rand(1);
var(5)=4*rand(1);
var(6)=8*rand(1);
%var(7)=10*rand(1);
%var(8)=10*rand(1);
nag_log_min=0;
var_new=var;
k=1;
change_flag=0;
T=2^12;
del_obj = @(delta1,mu,delta2,gamma,mu2,dis_thr,del_score_thr)...
        del_nms_box_IoU( det_validation_A,delta1,mu,delta2,gamma,mu2,dis_thr,del_score_thr ); 
cal_obj = @(det_new)...
        cal_mAP( validation_ground_A,det_new );

while(k<=it_times&&change_flag<=it_times)
            %func=(y-var_float1*phi)*(y-var_float1*phi)'+lambda*norm(var_float1);
            
            delta1 = var_new(1); 
            gamma = var_new(2);
            del_score_thr=var_new(3);
            delta2 = 1000*var_new(4);
            mu = var_new(5); 
        
           % mu1=var_new(5);
           % mask_thr=var_new(6);
            mu2=var_new(6);
            dis_thr=var_new(6);
            %[ det_new ] = del_nms_mask( det_validation_A,delta1,mu,delta2,gamma,mu1 );
            %temp=cal_mAP( validation_ground_A,det_new );
            [ det_new ] = del_obj(delta1,mu,delta2,gamma,mu2,dis_thr,del_score_thr );
            temp=cal_obj(det_new );
            sum=0;
            for i=1:1:10
                sum(i)=length(find(temp>0.5+(i-1)*0.05))/length(temp);
            end
            func=-mean(sum)
            %func=norm(-2*(y-var_float1*phi)*phi');
            if func<nag_log_min
                nag_log_min=func;
                change_flag=0;
                var=var_new
                 %if mod(k,3)==0
                  %      T=T*0.75;
                 %end
                func_1=-func
            else 
                if rand(1)<= exp(-1*(func-nag_log_min)/T)
                    nag_log_min=func;
                    change_flag=0;
                    var=var_new
                    if mod(k,1)==0
                       T=T*0.85;
                    end
                    func_1=-func
                end
            end
    min_test(k)=-nag_log_min;
    k=k+1;
    change_flag=change_flag+1;
    
    %var_new=change_bool(var);
    %var_new=change_bool1(var,var_num,left,right);
    var_new=change_float(var,k);
end
end

function [ var_float ] = bool2float( var_bool,num,left,right )
    for i=1:1:num
        var_float(i)=sub_bool2float(var_bool(1+(left+right)*(i-1):(left+right)*(i)),left,right);
    end
end

function [sub_var_float]= sub_bool2float(sub_var_bool,left,right)
    sub_var_float=0;
    for i=1:1:left
     sub_var_float=sub_var_float+sub_var_bool(i)*2^(left-i);
    end
    for i=1:1:right
     sub_var_float=sub_var_float+sub_var_bool(left+i)*2^(-i);
    end
end

function [ var_bool_new ] = change_bool( var_bool )
  var_bool_new=var_bool;
  num=length(var_bool);
  pos=fix(num*rand(1))+1;
      var_bool_new(pos)=1-var_bool(pos);
end

function [ var_bool_new ] = change_bool1( var_bool,num,left,right )
  var_bool_new=var_bool;
  pos=fix(num*rand(1))+1;
  if rand(1)>0.5
        var_bool_new((left+right)*(pos-1)+1:(left+right)*(pos-1)+left)=randi([0 1],1,left);
  else
        pos1=fix(right*rand(1)+1);
        var_bool_new((left+right)*(pos-1)+left+pos1)=1-var_bool((left+right)*(pos-1)+left+pos1);
  end
end

function [ var_random ] = randprob( num,prob )
    for i=1:1:num
        var_random(i)=0;
    end
    for i=1:1:prob
        var_random(i)=1;
    end
    var_random=var_random(randperm(numel(var_random)));
end
function [ var_float_new ] = change_float( var_float,k )
  var_float_new=var_float;
  num=length(var_float);
  num=3;
  pos=fix(num*rand(1))+1;
  if k<=50
    temp_rand=0.1;
  elseif k<=100
    temp_rand=0.2;
  elseif k<=200
    temp_rand=0.4;
  elseif k<=500
     temp_rand=0.6;
  elseif k<=1000
     temp_rand=0.8;
  else
     temp_rand=0.8;
  end
  if pos==1
      var_float_new(pos)= temp_rand*var_float(pos)+(1-temp_rand)*1*rand(1);
  elseif pos==2
      var_float_new(pos)= temp_rand*var_float(pos)+(1-temp_rand)*20*rand(1);
  elseif pos==3
      var_float_new(pos)= temp_rand*(var_float(pos))+(1-temp_rand)*0.3*rand(1);
      var_float_new(pos)=max(var_float_new(pos),0.1);
  elseif pos==4
      var_float_new(pos)= temp_rand*var_float(pos)+(1-temp_rand)*1*rand(1);
  elseif pos==5
      var_float_new(pos)= temp_rand*var_float(pos)+(1-temp_rand)*4*rand(1);
  elseif pos==6
      var_float_new(pos)= temp_rand*var_float(pos)+(1-temp_rand)*8*rand(1);
  elseif pos==7
      var_float_new(pos)= temp_rand*var_float(pos)+(1-temp_rand)*10*rand(1);
  elseif pos==8
      var_float_new(pos)= temp_rand*var_float(pos)+(1-temp_rand)*10*rand(1);
  end
end
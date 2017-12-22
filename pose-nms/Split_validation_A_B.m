clear all;
validation_ground=importdata('./NMS_data/validation_ground.mat'); 
%det_validation=importdata('det_nms_struct.mat');
det_validation=importdata('./NMS_data/det_validation.mat');
det_validation= det_validation(1:end-1);
%% if det is not struct
%for i=1:1:length(det_nms_cell)
 %    det_validation(i)=det_nms_cell{i};
%end


%% Generete new validation according to det_validation
det_len=length(det_validation);
val_len=length(validation_ground);
validation_ground_new=validation_ground(1:det_len);
tic
k=0;
for i=1:1:det_len
    temp_image=det_validation(i).image_id;
    for j=1:1:val_len
        if strcmp(temp_image,validation_ground(j).image_id)==1
             validation_ground_new(i)=validation_ground(j);
             k=k+1;
             break;
        end
    end
end
toc

human_num=zeros(1,length(validation_ground_new));

for i=1:1:length(validation_ground_new)
    temp=length(fieldnames(validation_ground_new(i).human_annotations));
    human_num(i)=temp;
end
for i=1:1:max(human_num)
    temp=find(human_num==i);
    person(i).index=temp;
    %temp1=floor(length(temp)/2);
    person_A(i).index=temp(1:floor(length(temp)/2));
    person_B(i).index=temp(floor(length(temp)/2)+1:end);
end
temp_A=person_A(1).index;
temp_B=person_B(1).index;
for i=2:1:max(human_num)
   temp_A=union(temp_A,person_A(i).index);
   temp_B=union(temp_B,person_B(i).index);
end
validation_ground_A=validation_ground_new(temp_A);
validation_ground_B=validation_ground_new(temp_B);
det_validation_A=det_validation(temp_A);
det_validation_B=det_validation(temp_B);




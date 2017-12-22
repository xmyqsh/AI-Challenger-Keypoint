function [ Oks ] = cal_mAP( validation_ground_A,det_validation_A )
%UNTITLED4 �˴���ʾ�йش˺�����ժҪ
%   �˴���ʾ��ϸ˵��


for i=1:1:length(validation_ground_A)
    temp_validation=length(fieldnames(validation_ground_A(i).keypoint_annotations));
    validation_num(i)=temp_validation;
    temp_det=length(fieldnames(det_validation_A(i).keypoint_annotations));
    det_num(i)=temp_det;
    Oks(i)=0;
end

parfor i=1:1:length(validation_ground_A)
     temp_validation_filed=fieldnames(validation_ground_A(i).keypoint_annotations);
     temp_det_filed=fieldnames(det_validation_A(i).keypoint_annotations);
     temp_oks=zeros(validation_num(i),det_num(i)); 
     for j=1:1:validation_num(i)
        for k=1:1:det_num(i)
            temp_oks(j,k)=cal_oks(getfield(validation_ground_A(i).keypoint_annotations,temp_validation_filed{j}),...
                getfield(det_validation_A(i).keypoint_annotations,temp_det_filed{k}),getfield(validation_ground_A(i).human_annotations,temp_validation_filed{j}));
        end
     end
     %Oks(i)=sum(max(temp_oks'))/max(validation_num(i),det_num(i));
     Oks(i)=sum(max(temp_oks'))/validation_num(i);
end

end

function [ oks ] = cal_oks( validation,det,box )
%UNTITLED5 �˴���ʾ�йش˺�����ժҪ
%   �˴���ʾ��ϸ˵��
%theta=2*[0.01388152, 0.01515228, 0.01057665, 0.01417709, 0.01497891, 0.01402144, 0.03909642, 0.03686941, 0.01981803,0.03843971, 0.03412318, 0.02415081, 0.01291456, 0.01236173];
theta=2*[0.079, 0.072, 0.01057665, 0.079, 0.072, 0.01402144, 0.062, 0.087, 0.089,0.062, 0.087, 0.089, 0.01291456, 0.01236173];
 %0.026, 0.025, 0.035, 0.079, 0.072, 0.062, 0.107, 0.087, & 0.089 
 %for the nose, eyes, ears, shoulders, elbows, wrists, hips, knees, & ankles, respectively.
x1=box(1);
y1=box(2);
x2=box(3);
y2=box(4);
s=(x2-x1)*(y2-y1);
v=validation(3:3:end);
x_val=validation(1:3:end);
y_val=validation(2:3:end);
x_det=det(1:3:end);
y_det=det(2:3:end);
d2=(x_det-x_val).^2+(y_det-y_val).^2;
oks=exp(-d2./(2*theta.^2*s));
oks=mean(oks(find(v==1)));
if isnan(oks)==1
   oks=0;
end

end




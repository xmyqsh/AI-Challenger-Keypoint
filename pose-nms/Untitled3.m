
%det_new=det_validation;
for i=1:1:length(det_new )
    temp_det=length(fieldnames(det_new(i).keypoint_annotations));
    det_new_num(i)=temp_det;
end
sum(det_new_num)
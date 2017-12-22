%% txt 2 struct
clear all;
det_val_txt=importdata('./NMS_data/gen_validation.txt');
k=1;
change_flag=1;
for i=1:1:1000
     temp_det{i}=strcat('human',num2str(i));
end
total_num=0;
for i =1:1:length(det_val_txt.data)
    if change_flag==1
        det_validation(k).image_id=det_val_txt.textdata(i);
        det_validation(k).keypoint_annotations=[];
        det_validation(k).human_annotations=[];
        index=1;
        change_flag=0;
        empty_flag=1;
        best_score=0;
        best_index=[];
    end
    if det_val_txt.data(i,5)>=0.10
        det_validation(k).human_annotations=setfield(det_validation(k).human_annotations,temp_det{index}...
            ,det_val_txt.data(i,1:4));
        det_validation(k).keypoint_annotations=setfield(det_validation(k).keypoint_annotations,temp_det{index}...
            ,det_val_txt.data(i,6:end));
        index=index+1;
        total_num=total_num+1;
        empty_flag=0;
    else
        if det_val_txt.data(i,5)>best_score
            best_score=det_val_txt.data(i,5);
            best_index=i;
           % index=index+1;
        end
    end
    if i~=length(det_val_txt.data)
        if strcmp(det_val_txt.textdata(i),det_val_txt.textdata(i+1))==0
                if empty_flag==1;
                    det_validation(k).human_annotations=setfield(det_validation(k).human_annotations,temp_det{index}...
                        ,det_val_txt.data(best_index,1:4));
                    det_validation(k).keypoint_annotations=setfield(det_validation(k).keypoint_annotations,temp_det{index}...
                        ,det_val_txt.data(best_index,6:end-1));
                    total_num=total_num+1;
                end
                k=k+1;
                change_flag=1;
        end
    end
end

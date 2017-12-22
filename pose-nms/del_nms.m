function [ det_new ] = del_nms( det_validation_A,delta1,mu,delta2,gamma )
%UNTITLED2 此处显示有关此函数的摘要
%   此处显示详细说明

for i =1:1:length(det_validation_A)
    det_new(i).image_id=det_validation_A(i).image_id;
    det_new(i).keypoint_annotations=[];
    det_new(i).human_annotations=[];
    while length(fieldnames(det_validation_A(i).keypoint_annotations))>0
        temp_det=fieldnames(det_validation_A(i).keypoint_annotations);
        score=[];
        for j=1:1:length(temp_det)
            temp= getfield(det_validation_A(i).keypoint_annotations,temp_det{j});
            score(j)=mean(temp(3:3:end));
        end
        [temp,index]=max(score);
        det_new(i).human_annotations=setfield(det_new(i).human_annotations,temp_det{index}...
            ,getfield(det_validation_A(i).human_annotations,temp_det{index}));
        det_new(i).keypoint_annotations=setfield(det_new(i).keypoint_annotations,temp_det{index}...
            ,getfield(det_validation_A(i).keypoint_annotations,temp_det{index}));
        k=1;
        del_index=[];
        for j=1:1:length(temp_det)
              
                if j~= index
                    human1_keypoint=getfield(det_validation_A(i).keypoint_annotations,temp_det{index});
                    human2_keypoint=getfield(det_validation_A(i).keypoint_annotations,temp_det{j});
                    human1_box=getfield(det_validation_A(i).human_annotations,temp_det{index});
                    if get_simlar_dis( human1_keypoint,human2_keypoint,human1_box,delta1,delta2,mu )>=gamma
                         del_index(k)=j;
                         k=k+1;
                    end
                end
        end
        det_validation_A(i).keypoint_annotations=rmfield(det_validation_A(i).keypoint_annotations,temp_det{index});
        for j=1:1:length(del_index)
            det_validation_A(i).keypoint_annotations=rmfield(det_validation_A(i).keypoint_annotations,temp_det{del_index(j)});
        end
        
    end

end

end

function [ output ] = get_simlar_dis( human1_keypoint,human2_keypoint,human1_box,delta1,delta2,mu  )
%UNTITLED8 此处显示有关此函数的摘要
%   此处显示详细说明
    %human1_keypoint=getfield(det_validation_A(i).keypoint_annotations,temp_det{index});
    %human2_keypoint=getfield(det_validation_A(i).keypoint_annotations,temp_det{j});
    %human1_box=getfield(det_validation_A(i).human_annotations,temp_det{index})
    x1=human1_box(1);
    y1=human1_box(2);
    x2=human1_box(3);
    y2=human1_box(4);
    threhold=sqrt((x2-x1)*(y2-y1)/10)/0.1;
    x_1=human1_keypoint(1:3:end);
    y_1=human1_keypoint(2:3:end);
    c_1=human1_keypoint(3:3:end);
    x_2=human2_keypoint(1:3:end);
    y_2=human2_keypoint(2:3:end);
    c_2=human2_keypoint(3:3:end);
    d2=(x_1-x_2).^2+(y_1-y_2).^2;
    v=find(d2<=threhold);
    Hsim=sum(exp(-d2/delta2));
    temp1=tanh(c_1/delta1).*tanh(c_2/delta1);
    Ksim=sum(temp1(v));
    output=Ksim+mu* Hsim;
end


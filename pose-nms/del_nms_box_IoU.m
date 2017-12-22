function [ det_new ] = del_nms_box_IoU( det_validation_A,delta1,mu,delta2,gamma,mu2,dis_thr,del_score_thr )
%UNTITLED2 此处显示有关此函数的摘要
%   此处显示详细说明
for i =1:1:length(det_validation_A)
    %det_new(i).image_id=det_validation_A(i).image_id;
    det_new(i).image_id=[];
    det_new(i).keypoint_annotations=[];
    det_new(i).human_annotations=[];
end

parfor i =1:1:length(det_validation_A)
    det_new(i).image_id=det_validation_A(i).image_id;
   % temp_mask=[1,2,3,4,5,6,7,8,10,11,13,14];
    %det_new(i).keypoint_annotations=[];
    %det_new(i).human_annotations=[];
    while length(fieldnames(det_validation_A(i).keypoint_annotations))>0
        temp_det=fieldnames(det_validation_A(i).keypoint_annotations);
        score=[];
        %mask=[];
        for j=1:1:length(temp_det)
            temp= getfield(det_validation_A(i).keypoint_annotations,temp_det{j});
            temp1=temp(3:3:end);
            %score(j)=mean(temp1(temp_mask));
            score(j)=mean(temp(3:3:end));
            %temp_score=temp(3:3:end);
            %score(j,:)=temp_score;
            %temp_mask=find(temp_score>mask_thr);
         %   if length(temp_mask)>length(mask)
          %      mask=temp_mask;
          %  end
            %score(j)=mean(temp_score(temp_mask));
        end
        %score=mean(score(:,mask),2);
       
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
                    human2_box=getfield(det_validation_A(i).human_annotations,temp_det{j});
                    if get_simlar_dis_mask( human1_keypoint,human2_keypoint,human1_box,human2_box,delta1,delta2,mu,mu2,dis_thr )>=gamma ...
                            || mean(human2_keypoint(3:3:end))<=del_score_thr
                         del_index(k)=j;
                         k=k+1;
                    %end
                    %if mean(human2_keypoint(3:3:end))<=0.186
                     %    del_index(k)=j;
                      %   k=k+1;
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

function [ output ] = get_simlar_dis_mask( human1_keypoint,human2_keypoint,human1_box,human2_box,delta1,delta2,mu,mu2,dis_thr  )
%UNTITLED8 此处显示有关此函数的摘要
%   此处显示详细说明
    %%
    %human1_keypoint=getfield(det_validation_A(i).keypoint_annotations,temp_det{index});
    %human2_keypoint=getfield(det_validation_A(i).keypoint_annotations,temp_det{j});
    %human1_box=getfield(det_validation_A(i).human_annotations,temp_det{index})
    
    %% 
    theta=2*[0.01388152, 0.01515228, 0.01057665, 0.01417709, 0.01497891, 0.01402144, 0.03909642, 0.03686941, 0.01981803,0.03843971, 0.03412318, 0.02415081, 0.01291456, 0.01236173];
    %theta_inverse_normal= (1/theta.^2)/min(1/theta.^2);
    theta_inverse=[7.9323,  6.6576,   13.6640,    7.6050,   6.8126,   7.7748,   1.0000,   ...
       1.1245,    3.8918,    1.0345,    1.3127,    2.6207,    9.1646,   10.0027];
    % theta_inverse_sqrt= sqrt(theta_inverse)/max
    theta_inverse_sqrt=[ 0.7619,    0.6980,    1.0000,    0.7460,    0.7061,    0.7543,    0.2705,   ...
        0.2869,    0.5337,    0.2752,    0.3100,    0.4379,    0.8190,    0.8556];
   % theta_inverse=1./theta_inverse;
    x1_l=human1_box(1);
    y1_l=human1_box(2);
    x1_r=human1_box(3);
    y1_r=human1_box(4);

    %threhold=sqrt((x1_r-x1_l)*(y1_r-y1_l)/10)/dis_thr;
    s=(x1_r-x1_l)*(y1_r-y1_l);
    threhold=-1*s*2*theta.^2*log(0.5);
    x_1=human1_keypoint(1:3:end);
    y_1=human1_keypoint(2:3:end);
    c_1=human1_keypoint(3:3:end);
    x_2=human2_keypoint(1:3:end);
    y_2=human2_keypoint(2:3:end);
    c_2=human2_keypoint(3:3:end);
    d2=(x_1-x_2).^2+(y_1-y_2).^2;
    %v=find(d2<=threhold);
    v=d2<=threhold;
    Hsim=sum(exp(-d2/delta2));
    %Hsim1=exp(-d2/delta2);
    %Hsim=sum(exp((-d2.*theta_inverse)/delta2));
    temp1=tanh(c_1/delta1).*tanh(c_2/delta1);
    Ksim=sum(temp1(v));
    %mask_thr=0.2;
    %mu1=10;
    %% mask similar
     %mask_c1=find(c_1<=mask_thr);
     %mask_c2=find(c_2<=mask_thr);
     %mask_IOU=length(intersect(mask_c1,mask_c2))/length(union(mask_c1,mask_c2));
     %mask_IOU=NaN;
     %bboxA=[human1_box(1),human1_box(2),abs(human1_box(3)-human1_box(1)),abs(human1_box(4)-human1_box(2))];
     %bboxB=[human2_box(1),human2_box(2),abs(human2_box(3)-human2_box(1)),abs(human2_box(4)-human2_box(2))];
     %box_IoU=bboxOverlapRatio(bboxA,bboxB);
     box_IoU=boxoverlap(human1_box,human2_box);
     %if isnan(mask_IOU)==0
      % output=Ksim+mu*Hsim+mu1*mask_IOU;
      % output=Hsim+mu1*mask_IOU;
     %else
       %output=Ksim;

       %output=Hsim1(v);
     %end
     %% output
     %output=Ksim+mu*Hsim;
     %output=Ksim;
     output=Ksim+mu* Hsim+mu2*box_IoU;
     %output=Ksim+mu2*box_IoU;
     %end
end

function o = boxoverlap(a, b)
% Compute the symmetric intersection over union overlap between a set of
% bounding boxes in a and a single bounding box in b.
%
% a  a matrix where each row specifies a bounding box
% b  a matrix where each row specifies a bounding box

% AUTORIGHTS
% -------------------------------------------------------
% Copyright (C) 2011-2012 Ross Girshick
% Copyright (C) 2008, 2009, 2010 Pedro Felzenszwalb, Ross Girshick
% 
% This file is part of the voc-releaseX code
% (http://people.cs.uchicago.edu/~rbg/latent/)
% and is available under the terms of an MIT-like license
% provided in COPYING. Please retain this notice and
% COPYING if you use this file (or a portion of it) in
% your project.
% -------------------------------------------------------

o = cell(1, size(b, 1));
for i = 1:size(b, 1)
    x1 = max(a(:,1), b(i,1));
    y1 = max(a(:,2), b(i,2));
    x2 = min(a(:,3), b(i,3));
    y2 = min(a(:,4), b(i,4));

    w = x2-x1+1;
    h = y2-y1+1;
    inter = w.*h;
    aarea = (a(:,3)-a(:,1)+1) .* (a(:,4)-a(:,2)+1);
    barea = (b(i,3)-b(i,1)+1) * (b(i,4)-b(i,2)+1);
    % intersection over union overlap
    o{i} = inter ./ (aarea+barea-inter);
    % set invalid entries to 0 overlap
    o{i}(w <= 0) = 0;
    o{i}(h <= 0) = 0;
end

o = cell2mat(o);

end


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Author: Gao Chenzhong
%   1. National Key Laboratory of Science and Technology on Space-Born Intelligent Information Processing
%   2. School of Information and Electronics, Beijing Institute of Technology (BIT), Beijing 100081, China
%   3. Beijing Institute of Technology, Zhuhai (ZHBIT), Guangdong 519088, China
% Contact: gao-pingqi@qq.com

% Pure HIMO cross-arbitrary-modality image matching demo.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
close all; clc; clear;
addpath('functions', 'func_Math', 'func_HIMO')
save_path = '.\save_image\';

%% Parameters
int_flag = 1;  % Is there any obvious intensity difference (cross-modal), yes:1, no:0
rot_flag = 1;  % Is there any obvious rotation difference
scl_flag = 0;  % Is there any obvious scale difference
par_flag = 0;  % Do you want parallel computing in multi-scale strategy
trans_form = 'affine'; % What spatial transform model do you need
                       % similarity, affine, projective / perspective / homography
out_form   = 'union';  % What image pair output form do you need
                       % reference, union, inter
chg_scale= 1;  % Do you want the resolution of sensed image be changed to the reference
Is_flag  = 1;  % Do you want the visualization of matching results
I3_flag  = 1;  % Overlap form output
I4_flag  = 1;  % Mosaic form output
save_flag= 0;  % Save output files

nOctaves = 3;    % Gaussian pyramid octave number, default: 3 or 6
nLayers  = 2;    % Gaussian pyramid  layer number, default: 2 or 1
G_resize = 1.2;  % Gaussian pyramid downsampling ratio, default: 2 or 1.2
G_sigma  = 1.6;  % Gaussian blurring standard deviation, default: 1.6
key_type = 'PC-ShiTomasi'; % What kind of feature point do you want as the keypoint
                           % Harris, ShiTomasi, PC-Harris, PC-ShiTomasi, detector-free
kps_thr  = 0.01; % Keypoints response threshold, default: 0.01
kps_R    = 1;    % Local non-maximum suppression radius, default: 2
kps_N    = 5000; % Keypoints number threshold, default: 5000
patchsize= 72;   % PolarP patchsize, default: 72 or 96
NBA      = 12;   % PolarP location division, default: 12
NBO      = 12;   % PolarP orientation division, default: 12
Error    = 5;    % Outlier removal pixel loss, default: 5 or 3
K        = 1;    % Outlier removal repetition times

%% Image input and preprocessing
if ~exist('file1','var'), file1 = []; end
if ~exist('file2','var'), file2 = []; end
[image_1,file1,DataInfo1] = Readimage(file1);
[image_2,file2,DataInfo2] = Readimage(file2);
[~,resample1] = Deal_Extreme(image_1,64,512,0); %resample1 = 1;
[~,resample2] = Deal_Extreme(image_2,64,512,0); %resample2 = 1;
[I1_s,I1] = Preproscessing(image_1,resample1,[]); figure,imshow(I1_s); drawnow
[I2_s,I2] = Preproscessing(image_2,resample2,[]); figure,imshow(I2_s); drawnow

%% Start
if par_flag && isempty(gcp('nocreate')), parpool(maxNumCompThreads); end  % Start parallel computing, time needed
fprintf('\n** Image matching starts ...\n'); t = []; warning off

%% Build HIMO feature pyramids
tic,[IMOm_pyr1,IMOo_pyr1,DoF_1] = Build_Himo_Pyramid(I1,...
    nOctaves,nLayers,G_resize,G_sigma,patchsize,NBA,int_flag,key_type);
    t(1)=toc; fprintf([' Done: HIMO extraction of reference image, time cost: ',num2str(t(1)),'s\n']);
%     Display_Pyramid(IMOm_pyr1,'IMO magnitude pyramid of reference image',0);
%     Display_Pyramid(IMOo_pyr1,'IMO orientation pyramid of reference image',0);
%     figure,imagesc(DoF_1), axis off, colormap(turbo); drawnow

tic,[IMOm_pyr2,IMOo_pyr2,DoF_2] = Build_Himo_Pyramid(I2,...
    nOctaves,nLayers,G_resize,G_sigma,patchsize,NBA,int_flag,key_type);
    t(2)=toc; fprintf([' Done: HIMO extraction of sensed image, time cost: ',num2str(t(2)),'s\n']);
%     Display_Pyramid(IMOm_pyr2,'IMO magnitude pyramid of sensed image',0);
%     Display_Pyramid(IMOo_pyr2,'IMO orientation pyramid of sensed image',0);
%     figure,imagesc(DoF_2), axis off, colormap(turbo); drawnow

%% Keypoints detection
ratio = sqrt((size(I1,1)*size(I1,2))/(size(I2,1)*size(I2,2)));
if ratio>=1
    r2 = kps_R; r1 = round(kps_R*ratio);
else
    r1 = kps_R; r2 = round(kps_R/ratio);
end

tic,kpts_1 = Detect_Himo_Keypoint(I1,DoF_1,6,kps_thr,r1,kps_N,nOctaves,G_resize,key_type);
    t(3)=toc; fprintf([' Done: Keypoints detection of reference image, time cost: ',num2str(t(3)),'s\n']); clear DoF_1
    figure,imshow(I1_s), hold on, plot(kpts_1(:,1),kpts_1(:,2),'r.'); 
    title(['Reference image —— ',num2str(size(kpts_1,1)),' keypoints']); drawnow

tic,kpts_2 = Detect_Himo_Keypoint(I2,DoF_2,6,kps_thr,r2,kps_N,nOctaves,G_resize,key_type);
    t(4)=toc; fprintf([' Done: Keypoints detection of sensed image, time cost: ',num2str(t(4)),'s\n']); clear DoF_2
    figure,imshow(I2_s), hold on, plot(kpts_2(:,1),kpts_2(:,2),'r.');
    title(['Sensed image —— ',num2str(size(kpts_2,1)),' keypoints']); drawnow

%% Keypoints description and matching (Multiscale strategy)
tic,[cor1,cor2] = Multiscale_Strategy(kpts_1,kpts_2,IMOm_pyr1,IMOm_pyr2,IMOo_pyr1,IMOo_pyr2,...
    patchsize,NBA,NBO,G_resize,Error,K,trans_form,rot_flag,scl_flag,par_flag,I1_s,I2_s);
    t(5)=toc; fprintf([' Done: Keypoints description and matching, time cost: ',num2str(t(5)),'s\n']); clear IMOm_pyr1 IMOm_pyr2 IMOo_pyr1 IMOo_pyr2
%     cor1 = data{1}*resample1; cor2 = data{2}*resample2;  % load results
    matchment = Show_Matches(I1_s,I2_s,cor1,cor2,0);

%% Done
T=num2str(sum(t)); fprintf(['* Done image matching, total time: ',T,'s\n']);

%% Image transformation (Geography enable)
tic,[I1_r,I2_r,I1_rs,I2_rs,I3,I4,t_form,pos] = Transformation(image_1,image_2,...
    cor1/resample1,cor2/resample2,trans_form,out_form,chg_scale,Is_flag,I3_flag,I4_flag);
    t(6)=toc; fprintf([' Done: Image tranformation, time cost: ',num2str(t(6)),'s\n']);
    figure,imshow(I3); title('Overlap Form'); drawnow
    figure,imshow(I4); title('Mosaic Form'); drawnow

if ~save_flag, return; end

%% Save results
Date = datestr(now,'yyyy-mm-dd_HH-MM-SS__'); tic
Imwrite({cor1/resample1; cor2/resample2; t_form}, [save_path,Date,'0 correspond','.mat']);
if exist('matchment','var') && ~isempty(matchment) && isvalid(matchment)
    saveas(matchment, [save_path,Date,'0 Matching Result','.png']);
end
if strcmpi(out_form,'reference')
%     Imwrite(image_1, [save_path,Date,'1 Reference Image','.tif'],GeoInfo1,DataInfo1);
    if Is_flag
        [I1_s,~] = Preproscessing(image_1,1,[]); 
        Imwrite(I1_s, [save_path,Date,'3 Reference Image Show','.png']);
    end
else
%     Imwrite(I1_r , [save_path,Date,'1 Reference Image','.tif'],GeoInfo1,DataInfo1);
    Imwrite(I1_rs, [save_path,Date,'3 Reference Image Show','.png']);
end
% Imwrite(I2_r , [save_path,Date,'2 Registered Image','.tif'],GeoInfo2,DataInfo1);
Imwrite(I2_rs, [save_path,Date,'4 Registered Image Show','.png']);
Imwrite(I3   , [save_path,Date,'5 Overlap of results','.png']);
Imwrite(I4   , [save_path,Date,'6 Mosaic of results','.png']);
t(7)=toc; disp([' Matching results are saved at ', save_path,', time cost: ',num2str(t(7)),'s']);
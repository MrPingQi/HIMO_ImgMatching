%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Author: Gao Chenzhong
% Contact: gao-pingqi@qq.com
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [cor1,cor2] = Multiscale_Strategy(kps_1,kps_2,IMOm_pyr1,IMOm_pyr2,IMOo_pyr1,IMOo_pyr2,...
    patch_size,NBA,NBO,G_resize,Error,K,trans_form,rot_flag,scl_flag,par_flag,aaa,bbb)

%% Multiscale procedure initialization
kps_1 = [kps_1(:,1:2),(1:size(kps_1,1))'];  % give idx
kps_2 = [kps_2(:,1:2),(1:size(kps_2,1))'];  % give idx
[nOctaves,nLayers] = size(IMOo_pyr1);
keypoints_1 = cell(nOctaves,1); img_size1 = zeros(nOctaves,2);
keypoints_2 = cell(nOctaves,1); img_size2 = zeros(nOctaves,2);
for octave=1:nOctaves
    kps_1t = round(kps_1(:,1:2)./G_resize^(octave-1));
    kps_2t = round(kps_2(:,1:2)./G_resize^(octave-1));
    [~,idx_1,~] = unique(kps_1t,'rows');
    [~,idx_2,~] = unique(kps_2t,'rows');
    keypoints_1{octave} = [kps_1t(idx_1,:), kps_1(idx_1,:)];
    keypoints_2{octave} = [kps_2t(idx_2,:), kps_2(idx_2,:)];
    img_size1(octave,:) = size(IMOo_pyr1{octave,1});
    img_size2(octave,:) = size(IMOo_pyr2{octave,1});
end
clear kps_1t kps_2t
matches = cell(nOctaves,nLayers,nOctaves,nLayers);
confidence = zeros(nOctaves,nLayers,nOctaves,nLayers);

%% Multiscale descriptor and matching procedure --- version 1 (parallel computing allowed)
if par_flag
% 描述符并行提取，索引统一化
nScales = nOctaves*nLayers;
descriptor1 = cell(nScales,1);
descriptor2 = cell(nScales,1);
% for k=1:nScales
parfor k=1:nScales
    octave = ceil(k/nLayers);
    layer = k-(octave-1)*nLayers;
    descriptor1{k} = GPolar_Descriptor(...
        IMOm_pyr1{octave,layer}, IMOo_pyr1{octave,layer}, keypoints_1{octave},...
        patch_size, NBA, NBO, rot_flag);  % [xt,yt,x,y,idx,orient,...des...]
    descriptor2{k} = GPolar_Descriptor(...
        IMOm_pyr2{octave,layer}, IMOo_pyr2{octave,layer}, keypoints_2{octave},...
        patch_size, NBA, NBO, rot_flag);  % [xt,yt,x,y,idx,orient,...des...]
end

% 描述符索引重分配
descriptors_1 = cell(nOctaves,nLayers);
descriptors_2 = cell(nOctaves,nLayers);
for k=1:nScales
    octave = ceil(k/nLayers);
    layer = k-(octave-1)*nLayers;
    descriptors_1{octave,layer} = descriptor1{k}(:,[3,4,1,2,6,5,7:end]);
    descriptors_2{octave,layer} = descriptor2{k}(:,[3,4,1,2,6,5,7:end]);  % [x,y,xt,yt,orient,idx,...des...]
end
clear descriptor1 descriptor2

% 并行匹配索引初始化
if scl_flag
    nScales = nOctaves*nLayers*nOctaves*nLayers;
    k = 1:nScales;
    Octave2 = ceil(k/(nOctaves*nLayers*nLayers));
    Octave1 = mod(ceil(k/(nLayers*nLayers))-1,nOctaves)+1;
else
    nScales = nOctaves*nLayers*nLayers;
    k = 1:nScales;
    Octave2 = ceil(k/(nLayers*nLayers));
    Octave1 = Octave2;
end
Layer2 = mod(ceil(k/nLayers)-1,nLayers)+1;
Layer1 = mod(k-1,nLayers)+1;

% 并行匹配，索引统一化
tmatches = cell(nScales,1);
tconfidence = zeros(nScales,1);
% for k=1:nScales
parfor k=1:nScales
    [tmatches{k},tconfidence(k)] = Match_Keypoint(...
        descriptors_1{Octave1(k),Layer1(k)}, ...
        descriptors_2{Octave2(k),Layer2(k)}, Error, K, trans_form, []);
end
clear descriptors_1 descriptors_2

% 匹配索引重分配
for k=1:nScales
    matches{   Octave1(k),Layer1(k),Octave2(k),Layer2(k)} = tmatches{k};
    confidence(Octave1(k),Layer1(k),Octave2(k),Layer2(k)) = tconfidence(k);
end
clear tmatches tconfidence
end


%% Multiscale descriptor and matching procedure --- version 2 (new iterative optimization)
if ~par_flag
descriptors_1 = cell(nOctaves,nLayers); idx_1 = [];
descriptors_2 = cell(nOctaves,nLayers); idx_2 = [];
ref = {[],[]};

for octave2=1:nOctaves
% for octave2=nOctaves:-1:1
    kps_2 = keypoints_2{octave2};
    
    for octave1=1:nOctaves
%     for octave1=nOctaves:-1:1
        if ~scl_flag, octave1 = octave2; end
        kps_1 = keypoints_1{octave1};
        
        for layer2=1:nLayers
            descriptor2 = descriptors_2{octave2,layer2};
            if isempty(descriptor2)
                kps_2 = sample_out(kps_2,idx_2,5);  % filter out matched
                if size(kps_2,1)<3, continue; end
                descriptor2 = GPolar_Descriptor(...
                    IMOm_pyr2{octave2,layer2}, IMOo_pyr2{octave2,layer2}, kps_2,...
                    patch_size, NBA, NBO, rot_flag);  % [xt,yt,x,y,idx,orient,...des...]
                descriptor2 = descriptor2(:,[3,4,1,2,6,5,7:end]);  % [x,y,xt,yt,orient,idx,...des...]
            else
                descriptor2 = sample_out(descriptor2,idx_2,6);  % filter out matched
            end
            descriptors_2{octave2,layer2} = descriptor2;
            
            for layer1=1:nLayers
                descriptor1 = descriptors_1{octave1,layer1};
                if isempty(descriptor1)
                    kps_1 = sample_out(kps_1,idx_1,5);  % filter out matched
                    if size(kps_1,1)<3, continue; end
                    descriptor1 = GPolar_Descriptor(...
                        IMOm_pyr1{octave1,layer1}, IMOo_pyr1{octave1,layer1}, kps_1,...
                        patch_size, NBA, NBO, rot_flag);  % [xt,yt,x,y,idx,orient,...des...]
                    descriptor1 = descriptor1(:,[3,4,1,2,6,5,7:end]);  % [x,y,xt,yt,orient,idx,...des...]
                else
                    descriptor1 = sample_out(descriptor1,idx_1,6);  % filter out matched
                end
                descriptors_1{octave1,layer1} = descriptor1;
                
                [match, confidence(octave1,layer1,octave2,layer2)] = ...
                    Match_Keypoint(descriptor1,descriptor2,Error,K,trans_form,ref,aaa,bbb);
                matches{octave1,layer1,octave2,layer2} = match;
                
                if size(match,1)>50
%                     idx_1 = unique([idx_1; match(:, 6)]);
%                     idx_2 = unique([idx_2; match(:,12)]);
%                     ref = mean(match(:,[1:2,7:8]),1);
                    idx_1 = unique(match(:, 6));
                    idx_2 = unique(match(:,12));
                    ref{1} = match(:,1: 6);
                    ref{2} = match(:,7:12);
                end
            end
        end
        if ~scl_flag, break; end
    end
end
end
clear descriptors_1 descriptors_2 descriptor1 descriptor2


%% Optimizing
Confidence = zeros(nOctaves,nOctaves);
Matches = cell(nOctaves,nOctaves);
for octave1=1:nOctaves
    for octave2=1:nOctaves
        matches_t = [];
        for layer1=1:nLayers
            for layer2=1:nLayers
                matches_t = [matches_t; matches{octave1,layer1,octave2,layer2}];
%                 if size(matches{octave1,layer1,octave2,layer2},1)>3
%                     Show_Matches(LMOo_pyr1{octave1,layer1},...
%                                  LMOo_pyr2{octave2,layer2},...
%                                  matches{octave1,layer1,octave2,layer2}(:,3:4),...
%                                  matches{octave1,layer1,octave2,layer2}(:,9:10),0);
%                     title(['octave1 = ',num2str(octave1),...
%                         '  layer1 = ',num2str(layer1),...
%                         '  octave2 = ',num2str(octave2),...
%                         '  layer2 = ',num2str(layer2)]); drawnow
%                 end
            end
        end
        if size(matches_t,1)>20
            [~,idx,~] = unique(matches_t(:,1:2),'rows');
            matches_t = matches_t(idx,:);
            [~,idx,~] = unique(matches_t(:,7:8),'rows');
            matches_t = matches_t(idx,:);
        end
        if size(matches_t,1)>20
            Matches{octave1,octave2} = matches_t;
            Confidence(octave1,octave2) = size(matches_t,1);
        end
    end
end
[max_O1,max_O2] = find(Confidence==max(max(Confidence)));

MMatches = [];
for i = 1-min(max_O1,max_O2):min(nOctaves-max_O1,nOctaves-max_O2)
    matches_t = Matches{max_O1+i,max_O2+i};
    if size(matches_t,1)>3, MMatches = [MMatches; matches_t]; end
end
[~,idx,~] = unique(MMatches(:,1:2),'rows');
MMatches = MMatches(idx,:);
[~,idx,~] = unique(MMatches(:,7:8),'rows');
MMatches = MMatches(idx,:);

%% One last outlier removal
% iter = 10000;
iter = 800;
NCMs = zeros(K,1); indexPairs = cell(K,1);
for k = 1:K
    [~,~,indexPairs{k}] = Outlier_Removal(MMatches(:,1:6),...
                                          MMatches(:,7:end),Error,iter,trans_form,[]);
    NCMs(k) = sum(indexPairs{k});
end
[~,maxIdx] = max(NCMs);
indexPairs = indexPairs{maxIdx};
MMatches = MMatches(indexPairs,:);
cor1 = MMatches(:,1:6); cor2 = MMatches(:,7:end);



function samples = sample_out(samples,id_list,loc)
% 假设samples是一个二维数组，id_list是要查找的编号数组
% samples(:,loc)是要查找的第loc列，保存了编号
if isempty(samples) || isempty(id_list) || loc>size(samples,2), return; end
% 找到samples的第loc列中属于id_list的所有行
rows_to_delete = ismember(samples(:, loc), id_list);
% 删除这些行
samples(rows_to_delete, :) = [];
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Author: Gao Chenzhong
% Contact: gao-pingqi@qq.com
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [matches, num_keys, conf] = Match_Keypoint(descriptor_1,descriptor_2,Error,K,trans_form,ref,aaa,bbb)
kps1 = descriptor_1(:,1:6); des1 = descriptor_1(:,7:end); clear descriptor_1
kps2 = descriptor_2(:,1:6); des2 = descriptor_2(:,7:end); clear descriptor_2

%% Match the keypoints
[indexPairs,conf] = matchFeatures(des1, des2, 'MaxRatio',1, 'MatchThreshold',100);
[~,uniqueIdx] = unique(indexPairs(:,2),'rows');
indexPairs = indexPairs(uniqueIdx,:);
num_keys = size(indexPairs,1);
if(num_keys<20), num_keys = 0; matches = []; return; end
cor1 = kps1(indexPairs(:,1),:);
cor2 = kps2(indexPairs(:,2),:);
conf = conf(indexPairs(:,1),:);

%% Remove incorrect matches
NCMs = zeros(K,1); indexPairs = cell(K,1);
for k = 1:K
    [~,~,indexPairs{k}] = Outlier_Removal(cor1,cor2,Error,800,trans_form,ref);
    NCMs(k) = sum(indexPairs{k});
end
[num_keys,maxIdx] = max(NCMs);
if(num_keys<4), num_keys = 0; matches = []; return; end
indexPairs = indexPairs{maxIdx};

% ccc = size(cor1,1);
% ddd = indexPairs(1:ccc);
% matchment = Show_Matches(aaa,bbb,cor1(ddd,:),cor2(ddd,:),0);
% if exist('matchment','var') && ~isempty(matchment) && isvalid(matchment)
%     saveas(matchment, ['.\save_image\',datestr(now,'yyyy-mm-dd_HH-MM-SS__'),'0 Matching Result','.png']);
% end

if ~isempty(ref)
    cor1 = [cor1; ref{1}];
    cor2 = [cor2; ref{2}];
end
cor1 = cor1(indexPairs,:);
cor2 = cor2(indexPairs,:);
matches = [cor1,cor2];

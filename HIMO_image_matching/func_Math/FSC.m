function [solution,rmse,cor1,cor2] = FSC(cor1,cor2,t_form,error,max_iter,ref)

% Delete duplicate point pair
% [~,i,~] = unique(cor1(:,1:2),'rows','first');
% cor1 = cor1(i,:); cor2 = cor2(i,:);
% [~,i,~] = unique(cor2(:,1:2),'rows','first');
% cor1 = cor1(i,:); cor2 = cor2(i,:);

if size(cor1,1)<20, solution = []; rmse = []; cor1 = []; cor2 = []; return; end

M0 = size(cor1,1);
if ~exist('ref','var'), ref = []; end
if ~isempty(ref)
    if ~isempty(ref{1})
        cor1 = [cor1; ref{1}(:,1:2)];
        cor2 = [cor2; ref{2}(:,1:2)];
    end
end
M = size(cor1,1);

proj_flag = 0; t_form = lower(t_form);
if contains(t_form, {'project','perspect','homo'}), t_form = 'projective'; end
switch lower(t_form)
    case 'similarity',  n = 2; total_iter = M*(M-1)/2;
    case 'affine',      n = 3; total_iter = M*(M-1)*(M-2)/(2*3);
    case 'projective',  n = 4; total_iter = M*(M-1)*(M-2)/(2*3); proj_flag = 1;
    otherwise, error('Unknown t_form');
end

% if ~exist('max_iter','var') || isempty(max_iter), max_iter = 10000; end
if ~exist('max_iter','var') || isempty(max_iter), max_iter = 800; end
total_iter = min(total_iter, max_iter);  % Algorithm iteration num

match1_xy = [cor1(:,1:2)'; ones(1,M)];
match2_xy = [cor2(:,1:2)'; ones(1,M)];
match2_xy_12 = match2_xy(1:2,:);  % 相似/仿射用
match2_test = cor2(:,1:2)';
error = error^2;
most_consensus_number = 0;
inlier_save = 1:M0;


%%
% rand('seed',0);
for i = 1:total_iter
%     if size(unique(cor1_new(:,1:2),'rows'),1)<20 || ...
%        size(unique(cor2_new(:,1:2),'rows'),1)<20
%         solution = []; rmse = []; cor1_new = []; cor2_new = []; return
%     end
    
    idx = randperm(M, n);
    cor1t = cor1(idx,1:2);
    cor2t = cor2(idx,1:2);
    [H,~] = LSM(cor1t,cor2t,t_form);
	solution = [H(1),H(2),H(5);
                H(3),H(4),H(6);
                H(7),H(8),1  ];
    
    t_match1_xy = solution*match1_xy;
    if ~proj_flag  % for similarity and affine 
        diff_match_xy = t_match1_xy(1:2,:)-match2_xy_12;
    else            % for perspective
        t_match1_xy = t_match1_xy(1:2,:) ./ repmat(t_match1_xy(3,:),2,1);
        diff_match_xy = t_match1_xy-match2_test;
    end
    diff_match_xy = sum(diff_match_xy.^2,1);
    inlier = diff_match_xy<error;
    consensus_num = sum(inlier);
     
    if(consensus_num>most_consensus_number)
        most_consensus_number = consensus_num;
        inlier_save = inlier;
    end
end
inlier_save((M0+1):end) = false;

if sum(inlier_save)<20, solution = []; rmse = []; cor1 = []; cor2 = []; return; end

cor1 = cor1(inlier_save,:);
cor2 = cor2(inlier_save,:);

% % Delete duplicate point pair
% [~,i,~] = unique(cor1(:,1:2),'rows','first');
% cor1 = cor1(i,:); cor2 = cor2(i,:);
% [~,i,~] = unique(cor2(:,1:2),'rows','first');
% cor1 = cor1(i,:); cor2 = cor2(i,:);

[H,rmse] = LSM(cor1(:,1:2),cor2(:,1:2),t_form);
solution = [H(1),H(2),H(5);
            H(3),H(4),H(6);
            H(7),H(8),1  ];
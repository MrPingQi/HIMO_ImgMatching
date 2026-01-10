%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Author: Gao Chenzhong
% Contact: gao-pingqi@qq.com
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function descriptor = GPolar_Descriptor(MOM_m, MOM_o, kps, ...
    patch_size, NBA, NBO, rot_flag)

if isempty(kps), descriptor = []; return; end

if isempty(MOM_m), MOM_m = ones(size(MOM_o)); end

W = floor(patch_size/2);  % 窗半径
X = -W : W;  % 邻域x坐标
Y = -W : W;  % 邻域y坐标
[XX,YY] = meshgrid(X,Y);
Wcircle = (XX.^2 + YY.^2) < (W+1)^2;  % 圆形窗

% Rho area divide
rr1 = W^2/(2*NBA+1); rr2 = rr1*(NBA+1);
Rho_0 = XX.^2+YY.^2;
Rho_0(Rho_0<=rr1)             = 1;
Rho_0(Rho_0>rr1 & Rho_0<=rr2) = 2;
Rho_0(Rho_0>rr2)              = 3;
Rho_0 = Rho_0 .* Wcircle - 1;  % 三个区域：0、1、2

% Theta area divide
Theta_0 = atan2(YY,XX) + pi;  % 取值范围：[0,2pi]
if ~rot_flag
    Theta = mod(floor(Theta_0*NBA/pi/2),NBA)+1;  % 取值范围：[1,NBA]
end

% Deeper feature parameters
w_ratio1 = sqrt(rr1/rr2);
w_ratio2 = sqrt(rr1)/W;
c_idx = [2:NBA,1];
NBA2 = NBA/2;
NBA3 = NBA*3/4;

%% 基准方向 Base direction
% kps_o记录特征点原始坐标，kps记录特征点在当前尺度下坐标，需根据代码索引正确更改顺序
if rot_flag
    kps = Base_Direction(kps,MOM_m,MOM_o,W,NBO);
else
    kps = [kps,zeros(size(kps,1),1)];
end

%% 描述符 GPolar
% descriptor = zeros(size(kps,1), (2+3*NBA)*NBO);  % descriptor (size:(1+3×s+1)×o)
for k = 1:size(kps,1)
    x = kps(k,1); x1 = max(1,x-W); x2 = min(x+W,size(MOM_o,2)); xx_idx = (W+x1-x+1):(W+x2-x+1);
    y = kps(k,2); y1 = max(1,y-W); y2 = min(y+W,size(MOM_o,1)); yy_idx = (W+y1-y+1):(W+y2-y+1);
    Rho = Rho_0(yy_idx, xx_idx);
    weight = MOM_m(y1:y2, x1:x2);
    
    % Rotation invariance
    if rot_flag
        orient = kps(k,end);
        angle_bin = mod(floor((MOM_o(y1:y2, x1:x2)-orient) * NBO/pi), NBO) + 1;  % 取值范围：[1,NBO]
        Theta = mod(floor((Theta_0(yy_idx, xx_idx)-orient)*NBA/pi/2), NBA) + 1;  % 取值范围：[1,NBA]
    else
        angle_bin = floor(MOM_o(y1:y2, x1:x2) * NBO/pi) + 1;  % 取值范围：[1,NBO]
    end
    
    % Feature histogram
    feat_center = zeros(NBO,1);
    feat_outer = zeros(NBO,NBA,3);
    for xx = 1:size(Rho,2)
        for yy = 1:size(Rho,1)
            Rho_t = Rho(yy,xx);
            if Rho_t<0 || Rho_t>2, continue; end
            Theta_t = Theta(yy,xx);
            angle_t = angle_bin(yy,xx);
            if Rho_t==0
                feat_center(angle_t) = feat_center(angle_t) + weight(yy,xx);
            else
                feat_outer(angle_t,Theta_t,Rho_t) = ...
                    feat_outer(angle_t,Theta_t,Rho_t) + weight(yy,xx);
            end
        end
    end
    
    % Blurring
%     feat_outer(:,:,1:2) = imfilter(feat_outer(:,:,1:2), H, 'circular');
    
    % Deeper feature
    feat_outer(:,:,3) = ( ...
        (feat_outer(:,:,1) + feat_outer(:,c_idx,1)) * w_ratio1 + ...
        (feat_outer(:,:,2) + feat_outer(:,c_idx,2)) * w_ratio2 ) / 2 + ...
        feat_center /3 /2;
    feat_all = mean(feat_outer(:,:,3),2);
    
    feat_skip = [mean(feat_outer(:,1:2:NBA-1,1)*w_ratio1+feat_outer(:,2:2:NBA,2)*w_ratio2,2);
                 mean(feat_outer(:,1:2:NBA-1,2)*w_ratio2+feat_outer(:,2:2:NBA,1)*w_ratio1,2)]/2;
    
    % Inversion dealing
    if rot_flag
        des_H1 = feat_outer(:,1:NBA2,:);
        des_H2 = feat_outer(:,NBA2+1:NBA,:);
%         V = var(des_H1,0,'all') - var(des_H2,0,'all');  % old OCD
        V = sum((var(des_H1,0,1) - var(des_H2,0,1)) >= 0, 'all') - NBA3;  % new OCD
        if V>0, feat_outer = cat(2,des_H2,des_H1); end
    end
    
    % Descriptor vectors
%     des = [feat_center; feat_outer(:)];
%     des = [feat_center; feat_outer(:); feat_all];
    des = [feat_center; feat_outer(:); feat_skip(:); feat_all];
    if k==1
        descriptor = zeros(size(kps,1), size(des,1));  % descriptor (size:(1+3×s+1)×o)
    end
    descriptor(k,:) = des;
end
descriptor = [kps, descriptor];
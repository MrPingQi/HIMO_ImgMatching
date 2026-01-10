%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Author: Gao Chenzhong
% Contact: gao-pingqi@qq.com
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [LMOm_pyr,LMOo_pyr,DoF] = Build_Himo_Pyramid(I,...
    nOctaves,nLayers,G_resize,G_sigma,patch_size,NBA,int_flag,key_type)

DoF_flag = ~contains(lower(key_type), 'free');
sig = Get_Gaussian_Scale(G_sigma,nLayers);
W = floor(patch_size/2);
r = sqrt(W^2/(2*NBA+1));

%% Co-occurancy filtering
sigma_s = 5;     % 空间高斯权重标准差
sigma_oc = 1.6;  % 共现矩阵高斯平滑标准差
% ratio = 2^(1/3);
% for i=1:3, sigma_oc = sigma_oc*ratio; end

%% Build pyramids
[M,N,~] = size(I); I = I * 255 / mean(I(:)) / 2;
% img_pyr = cell(nOctaves,nLayers);  % 不需要记录图像金字塔
% DoF_pyr = cell(nOctaves,nLayers);  % 不需要记录差分金字塔
LMOm_pyr = cell(nOctaves,nLayers);
LMOo_pyr = cell(nOctaves,nLayers);
DoF = ones(M,N); norm_f = 2/pi;
for octave = 1:nOctaves
    I = CoOcurFilter(I, sigma_s, sigma_oc)*0.25 + I*0.75;
    I_t = imresize(I,1/G_resize^(octave-1),'bicubic');
    for layer = 1:nLayers
        I_t = Gaussian_Scaling(I_t, layer, sig(layer));
        [LMOm_pyr{octave,layer}, ...
         LMOo_pyr{octave,layer},map_curr] = Major_Orientation_Map(I_t,1,r,4,int_flag);
        
        if DoF_flag
            map_curr(:,:,5) = LMOo_pyr{octave,layer};
            if layer==1
                if octave>1, map_prev(:,:,5) = LMOo_pyr{octave-1,end}; end
            else
                map_prev(:,:,5) = LMOo_pyr{octave,layer-1};
            end
            if layer~=1 || octave~=1
                % 只关心差值，不判断正负。根据[0,pi)限制，方向角度最大差值为pi/2，超过的部分按反向处理
                DoF = DoF .* prod(abs(abs(imresize(map_curr,[M,N]) - ...
                                          imresize(map_prev,[M,N])) * norm_f - 1), 3).^(1/5);  % Feature maps
                DoF = DoF .* rescale( abs(imresize(I_t     ,[M,N]) - I), 0.9, 1);  % CoF feature
            end
            map_prev = map_curr;
        end
        
%         if layer==octave+nLayers-nOctaves
%             I_t = (I_t-min(I_t(:))) ./ (max(I_t(:))-min(I_t(:)));
%             Imwrite(I_t,['img',num2str(octave),'.png']);
%         end
    end
end



function sig = Get_Gaussian_Scale(sigma,numLayers)
sig = zeros(1,numLayers);
sig(1) = sigma;  % 认为第一个图像尺度就是σ
if numLayers<2, return; end
k = 2^(1.0/(numLayers-1));
for i = 2:1:numLayers
    sig_prev = k^(i-2)*sigma;
    sig_curr = k*sig_prev;
    sig(i) = sqrt(sig_curr^2-sig_prev^2);
end



function I_t = Gaussian_Scaling(I_t,layer,sig)
if(layer>1)
    window_size = round(3*sig);
    window_size = 2*window_size+1;
    w = fspecial('gaussian',[window_size,window_size],sig);
    I_t = imfilter(I_t,w,'replicate');
end
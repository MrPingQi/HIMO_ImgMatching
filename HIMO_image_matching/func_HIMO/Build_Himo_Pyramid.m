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

%% Build pyramids
[M,N,~] = size(I); I = I * 255 / median(I(I~=0),'all') / 2;
LMOm_pyr = cell(nOctaves,nLayers);
LMOo_pyr = cell(nOctaves,nLayers);
DoF = ones(M,N); n_scale = nOctaves*nLayers; norm_f = 2/pi;
for octave = 1:nOctaves
    I = CoOcurFilter(I, sigma_s, sigma_oc)*0.25 + I*0.75;
    I_t = imresize(I,1/G_resize^(octave-1),'bicubic');
    for layer = 1:nLayers
        I_t = Gaussian_Scaling(I_t, layer, sig(layer));
        [LMOm_pyr{octave,layer}, ...
         LMOo_pyr{octave,layer},map_curr] = Intrinsic_Major_Orientation(I_t,1,r,4,int_flag);
        
        if DoF_flag
            n_map = size(map_curr,3)+1;
            map_curr(:,:,n_map) = LMOo_pyr{octave,layer};
            if layer==1
                if octave>1
                map_prev(:,:,n_map) = LMOo_pyr{octave-1,  end}; end
            else
                map_prev(:,:,n_map) = LMOo_pyr{octave,layer-1};
            end
            if layer~=1 || octave~=1
                % 只关心差值，不判断正负。根据[0,pi)限制，方向角度最大差值为pi/2，超过的部分按反向处理
                DoF = DoF .* prod(abs(abs(imresize(map_curr,[M,N]) - ...
                                          imresize(map_prev,[M,N])) * norm_f - 1), 3).^(6/n_scale);  % Feature maps
                DoF = DoF .*     log( abs(imresize(     I_t,[M,N]) - I) + 1);  % CoF feature
            end
            map_prev = map_curr;
        end
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
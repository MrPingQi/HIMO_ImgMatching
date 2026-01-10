%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Author: Gao Chenzhong
% Contact: gao-pingqi@qq.com
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [magnitude,orientation,maps] = Major_Orientation_Map(I,R1,R2,s,int_flag)
% LMO: Local Major Orientation
[M,N] = size(I);
maps = zeros(M,N,4);

hx1 = [-1,0,1;-2,0,2;-1,0,1];  % Odd-Sobel算子
hy1 = [-1,-2,-1;0,0,0;1,2,1];
hx2 = [-1,2,-1;-2,4,-2;-1,2,-1]*2/3;  % Even-Sobel算子
hy2 = [-1,-2,-1;2,4,2;-1,-2,-1]*2/3;
Gx1 = imfilter(I, hx1, 'replicate');
Gx2 = imfilter(I, hx2, 'replicate') .* ((Gx1 >= 0) * 2 - 1);
Gx = Gx1 .* 1i + Gx2; clear Gx1 Gx2
Gy1 = imfilter(I, hy1, 'replicate');
Gy2 = imfilter(I, hy2, 'replicate') .* ((Gy1 >= 0) * 2 - 1);
Gy = Gy1 .* 1i + Gy2; clear Gy1 Gy2
% Gx = zeros(M,N); Gy = zeros(M,N);

%% LogGabor feature maps
Ns = 4; No = 6;
EO = LogGabor(I,Ns,No,3,1.6,0.75);  % minWaveLength = 3, mult = 1.6, sigmaOnf = 0.75
angle = pi*(0:No-1)/No;
angle_cos = cos(angle);
angle_sin = sin(angle);
for j=1:No
    for i=1:Ns
        direct  = (imag(EO{i,j}) >= 0) * 2 - 1;
        Gx = Gx - (imag(EO{i,j}) * 1i...
                +  real(EO{i,j}) .* direct) * angle_cos(j)*(Ns-i+1);  % 注意图像与matlab矩阵坐标关系，y轴是反的
        Gy = Gy + (imag(EO{i,j}) * 1i...
                +  real(EO{i,j}) .* direct) * angle_sin(j)*(Ns-i+1);
    end
end

Gx1 = imag(Gx); Gx2 = real(Gx); clear Gx
Gy1 = imag(Gy); Gy2 = real(Gy); clear Gy

maps(:,:,1) = mod(atan2(Gy1,Gx1),pi);
maps(:,:,2) = mod(atan2(Gy2,Gx2),pi);

% * for ablation *
% orientation1 = mod(atan2(Gy1,Gx1),pi);  % 取值范围：[-pi,pi] ——> [0,pi)
% magnitude1 = sqrt(sqrt(Gx1.^2+Gy1.^2)); clear Gsx1 Gsy1
% orientation2 = mod(atan2(Gy2,Gx2),pi);  % 取值范围：[-pi,pi] ——> [0,pi)
% magnitude2 = sqrt(sqrt(Gx2.^2+Gy2.^2)); clear Gsx2 Gsy2

%% Averaging Squared LogGabor (ASLG) feature maps
W = floor(R2);  % 窗半径
dx = -W : W;  % 邻域x坐标
dy = -W : W;  % 邻域y坐标
[dx,dy] = meshgrid(dx,dy);
Wcircle = ((dx.^2 + dy.^2) < (W+1)^2);  % 圆形窗
Patchsize = 2*W+1;

if s==1
    h = fspecial('gaussian',[Patchsize,Patchsize], R1/6);
else
    step = (R2-R1)/(s-1);
    h = zeros(Patchsize,Patchsize);
    for i=0:s-1
        sigma = (R1+step*i)/6;
        h = h + fspecial('gaussian',[Patchsize,Patchsize], sigma);
    end
end
h = h.*Wcircle;

Gxx1 = imfilter(Gx1.*Gx1, h, 'replicate');
Gyy1 = imfilter(Gy1.*Gy1, h, 'replicate');
Gxy1 = imfilter(Gx1.*Gy1, h, 'replicate'); clear Gx1 Gy1
Gsx1 = Gxx1-Gyy1;                          clear Gxx1 Gyy1
Gsy1 = 2*Gxy1;                             clear Gxy1

Gxx2 = imfilter(Gx2.*Gx2, h, 'replicate');
Gyy2 = imfilter(Gy2.*Gy2, h, 'replicate');
Gxy2 = imfilter(Gx2.*Gy2, h, 'replicate'); clear Gx2 Gy2
Gsx2 = Gxx2-Gyy2;                          clear Gxx2 Gyy2
Gsy2 = 2*Gxy2;                             clear Gxy2

%% Odd/Even-ASLG feature maps
orientation1 = atan2(Gsy1,Gsx1)/2 + pi/2;  % 取值范围：[-pi,pi] ——> [0,pi]
magnitude1 = Gsx1.^2+Gsy1.^2; clear Gsx1 Gsy1
orientation2 = atan2(Gsy2,Gsx2)/2 + pi/2;  % 取值范围：[-pi,pi] ——> [0,pi]
magnitude2 = Gsx2.^2+Gsy2.^2; clear Gsx2 Gsy2

% * for ablation *
% orientation = orientation1; magnitude = magnitude1;
% orientation = orientation2; magnitude = magnitude2;
% magnitude = ones(M,N);
% return

%% Odd/Even coupling to MOM feature
idx = ceil((sign(magnitude1-magnitude2)+0.1)/2);
orientation = idx.*orientation1 + (1-idx).*orientation2;  % 耦合！
maps(:,:,3) = orientation1; clear orientation1
maps(:,:,4) = orientation2; clear orientation2
orientation = mod(orientation,pi);  % 取值范围：[0,pi] ——> [0,pi)
if int_flag
    scale = 6;
    W = floor(scale/2);  % 窗半径; window radius
    [dx,dy] = meshgrid(-W : W,-W : W);
    Wcircle = (dx.^2 + dy.^2) < (W+1)^2;  % 圆形窗; circular window
    h = fspecial('gaussian',[scale+1,scale+1], scale/6) .* Wcircle;
    Gxy = imfilter(magnitude1.*magnitude2, h, 'replicate');
    Gxx = imfilter(magnitude1.*magnitude1, h, 'replicate'); clear magnitude1
    Gyy = imfilter(magnitude2.*magnitude2, h, 'replicate'); clear magnitude2
    magnitude = (Gxx.*Gyy - Gxy.^2) ./ (Gxx + Gyy + eps); clear Gxx Gyy Gxy
    
    thr = 0.1;
%     thr = 10000;
    magnitude(magnitude<=thr) = 0;
    magnitude(magnitude> thr) = 1;
    
%     figure(9), bond = 20;
%     imshow(magnitude(1+bond:end-bond,1+bond:end-bond,:)); pause(1)
%     Imwrite(magnitude(1+bond:end-bond,1+bond:end-bond,:),[datestr(now,'yyyy-mm-dd_HH-MM-SS__'),'.png']);
%     figure(10), bond = 20;
%     imagesc(1-magnitude(1+bond:end-bond,1+bond:end-bond,:)); axis off;
%     set(gcf, 'Units', 'pixels', 'Position', [0, 0, 512, 512]);
%     set(gca, 'Units', 'normalized', 'Position', [0, 0, 1, 1]);
%     saveas(gcf, 'W.png');
else
    magnitude = idx.*magnitude1 + (1-idx).*magnitude2;
    magnitude = sqrt(sqrt(magnitude));
end
% magnitude = ones(M,N);

% * for ablation *
% hx = [-1,0,1;-2,0,2;-1,0,1];  % 一阶梯度 Sobel算子
% hy = [-1,-2,-1;0,0,0;1,2,1];
% Gx1 = imfilter(I, hx, 'replicate');
% Gy1 = imfilter(I, hy, 'replicate');
% [M,N] = size(I); clear I
% magnitude = sqrt(Gx1.^2+Gy1.^2);
% orientation = mod(mod(atan2(Gy1,Gx1),pi),pi);
% orientation = mod(mod(atan2(Gy1,Gx1),2*pi),2*pi);
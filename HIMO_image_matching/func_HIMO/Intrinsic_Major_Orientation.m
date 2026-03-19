%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Author: Gao Chenzhong
% Contact: gao-pingqi@qq.com
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [magnitude,orientation,maps] = Major_Orientation_Map(I,R1,R2,s,int_flag)
% IMO: Intrinsic Major Orientation
[M,N] = size(I);
maps = zeros(M,N,4);

%% Deep-Shallow Orientation Extraction
h_o = [-1,0, 1;-2,0, 2;-1,0, 1];      %  Odd-Sobel算子
h_e = [-1,2,-1;-2,4,-2;-1,2,-1]*2/3;  % Even-Sobel算子
Gx1 = imfilter(I, h_o , 'replicate');
Gx2 = imfilter(I, h_e , 'replicate') .* ((Gx1 >= 0) * 2 - 1);
Gx = Gx1 .* 1i + Gx2; clear Gx1 Gx2
Gy1 = imfilter(I, h_o', 'replicate');
Gy2 = imfilter(I, h_e', 'replicate') .* ((Gy1 >= 0) * 2 - 1);
Gy = Gy1 .* 1i + Gy2; clear Gy1 Gy2

Ns = 4; No = 6;
EO = LogGabor(I,Ns,No,3,1.6,0.75);  % minWaveLength = 3, mult = 1.6, sigmaOnf = 0.75
angle = pi*(0:No-1)/No;
angle_cos = cos(angle);
angle_sin = sin(angle);
for j=1:No
    for i=1:Ns
        direct = (imag(EO{i,j}) >= 0) * 2 - 1;
        E = imag(EO{i,j}) * 1i + real(EO{i,j}) .* direct;
        Gx = Gx - E * angle_cos(j) * (Ns-i+1);  % 注意图像与matlab矩阵坐标关系，y轴是反的
        Gy = Gy + E * angle_sin(j) * (Ns-i+1);
    end
end
Gx = cat(3, imag(Gx), real(Gx));
Gy = cat(3, imag(Gy), real(Gy));

maps(:,:,1:2) = mod(atan2(Gy,Gx),pi);

%% Multi-Scale Orientation Weighting
W = floor(R2);  % 窗半径
dx = -W : W;  % 邻域x坐标
dy = -W : W;  % 邻域y坐标
[dx,dy] = meshgrid(dx,dy);
Wcircle = (dx.^2 + dy.^2) < (W+1)^2;  % 圆形窗
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

Gxx = imfilter(Gx.*Gx, h, 'replicate');
Gyy = imfilter(Gy.*Gy, h, 'replicate');
Gxy = imfilter(Gx.*Gy, h, 'replicate'); clear Gx Gy
Gsx = Gxx-Gyy;                          clear Gxx Gyy
Gsy = 2*Gxy;                            clear Gxy

orientation = atan2(Gsy,Gsx)/2 + pi/2;  % 取值范围：[-pi,pi] ——> [0,pi]
magnitude = Gsx.^2+Gsy.^2; clear Gsx Gsy

maps(:,:,3:4) = orientation;

%% Odd/Even Orientation Coupling
idx = magnitude(:,:,1) >= magnitude(:,:,2);
orientation = idx.*orientation(:,:,1) + (1-idx).*orientation(:,:,2);  % 耦合！
orientation = mod(orientation,pi);  % 取值范围：[0,pi] ——> [0,pi)
if int_flag
    scale = 6;
    W = floor(scale/2);  % 窗半径; window radius
    [dx,dy] = meshgrid(-W : W,-W : W);
    Wcircle = (dx.^2 + dy.^2) < (W+1)^2;  % 圆形窗; circular window
    h = fspecial('gaussian',[scale+1,scale+1], scale/6) .* Wcircle;
    Gxy = imfilter(magnitude(:,:,1).*magnitude(:,:,2), h, 'replicate');
    Gxx = imfilter(magnitude(:,:,1).^2, h, 'replicate');
    Gyy = imfilter(magnitude(:,:,2).^2, h, 'replicate');
    magnitude = (Gxx.*Gyy - Gxy.^2) ./ (Gxx + Gyy + eps); clear Gxx Gyy Gxy
    
    thr = 0.1;
    magnitude(magnitude<=thr) = 0;
    magnitude(magnitude> thr) = 1;
else
    magnitude = idx.*magnitude1 + (1-idx).*magnitude2;
    magnitude = magnitude.^(1/4);
end
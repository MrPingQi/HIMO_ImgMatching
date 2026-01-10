%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Author: Gao Chenzhong
% Contact: gao-pingqi@qq.com
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function value = Harris(img, scale)

hx = [-1,0,1;-2,0,2;-1,0,1];  % 一阶梯度 Sobel算子; first-order gradient Sobel operator
hy = hx';
Gx = imfilter(img, hx, 'replicate');
Gy = imfilter(img, hy, 'replicate');

W = floor(scale/2);  % 窗半径; window radius
dx = -W : W;         % 邻域x坐标; neighborhood x-coordinate
dy = -W : W;         % 邻域y坐标; neighborhood y-coordinate
[dx,dy] = meshgrid(dx,dy);
Wcircle = (dx.^2 + dy.^2) < (W+1)^2;  % 圆形窗; circular window
h = fspecial('gaussian',[scale+1,scale+1], scale/6) .* Wcircle;
Gxx = imfilter(Gx.*Gx, h, 'replicate');
Gyy = imfilter(Gy.*Gy, h, 'replicate');
Gxy = imfilter(Gx.*Gy, h, 'replicate');

value = (Gxx.*Gyy - Gxy.^2) ./ (Gxx + Gyy + eps);
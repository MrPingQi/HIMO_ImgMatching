function img=CoOcurFilter(img, sigma_s,sigma_oc)

img_min = min(img(:));
if (img_min < 0)
    img = img - img_min;
end

if isfloat(img)
    img = int32(round(img));
end

coc_mat = GetCoOcurMat(img, sigma_oc);
win_size = 3 * sigma_s;
win_halfsize = ceil(win_size/2);
win_size = win_halfsize*2 + 1;
[xx, yy] = meshgrid(-win_halfsize:win_halfsize, -win_halfsize:win_halfsize);
w_s = exp(-(xx.^2 + yy.^2)/(2*sigma_s^2));
w_s = w_s(:);

img_pad = padarray(img, [win_halfsize, win_halfsize], 0, 'both');
[nrow, ncol] = size(img_pad);

for i = 1:(nrow - win_size + 1)
    for j = 1:(ncol - win_size + 1)
        img_sub = img_pad(i:i+win_size-1, j:j+win_size-1);
        w_oc = coc_mat(img_sub(:)+1, img_sub(win_halfsize+1, win_halfsize+1)+1);
        w = w_oc .* w_s;
        img(i,j) = sum(double(img_sub(:)) .* w) / (sum(w)+eps);
    end
end
img = double(img);
end


function coc_mat = GetCoOcurMat(img, sigma_oc)
img = int32(img);
gray_max = max(img(:));
coc_mat = zeros(gray_max+1, gray_max+1);
[H, W] = size(img);

for dx = -1:1
    for dy = -1:1
        if dx==0 && dy==0
            continue
        end
        for i=1:H
            for j=1:W
                x2 = i + dx;
                y2 = j + dy;
                if x2 < 1 || x2 > H || y2 < 1 || y2 > W
                    continue
                end
                g1 = img(i,j);
                g2 = img(x2,y2);
                coc_mat(g1+1, g2+1) = coc_mat(g1+1, g2+1) + 1;
            end
        end
    end
end

coc_mat = coc_mat / sum(coc_mat(:));

if sigma_oc > 0
    hsize = 2*ceil(3*sigma_oc)+1;
    h = fspecial('gaussian', hsize, sigma_oc);
    coc_mat = imfilter(coc_mat, h, 'replicate');
    coc_mat = coc_mat / sum(coc_mat(:));
end
end
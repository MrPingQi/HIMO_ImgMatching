function S_dcm = calc_DCM(img,points)
NCM = size(points,1);
if NCM<4, S_dcm = 0; return; end

try
    D = pdist2(points,points);
    D(1:NCM+1:end) = inf;
    [dist,~] = min(D);
    S_uniform = sum(dist)/NCM/max(dist);

    [M,N,~] = size(img);
    idx = convhull(points(:,1),points(:,2));
    hullArea = polyarea(points(idx,1), points(idx,2));
    S_area = hullArea/(M*N);

    S_dcm = log10(S_area * S_uniform * NCM + 1);
    
catch
    S_dcm = 0;
end
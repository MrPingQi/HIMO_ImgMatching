function [H,rmse] = LSM(cor1,cor2,change_form)
warning off

% This function gets the transformation parameters
% based on two sets of matches using the least square method
match1_xy = cor1(:,[1 2]);
match2_xy = cor2(:,[1 2]);  % generate the initial point set randomly

A = [];  % Ax = b;
% A = [x1,y1,x1,y1；x1,y1,x1,y1;......xn,yn,xn,yn；xn,yn,xn,yn]
for i = 1:size(match1_xy,1)
    A = [A;repmat(match1_xy(i,:),2,2)];
end
% B = [1 1 0 0;0 0 1 1....1 1 0 0;0 0 1 1]
B = [1 1 0 0;0 0 1 1]; B = repmat(B,size(match1_xy,1),1);
% A = [x1,y1,0,0;0,0,x1,y1;.....xn,yn,0,0;0,0,xn,yn]

A = A.*B;
% B = [1,0;0,1;....1,0;0,1]
B = [1 0;0 1]; B = repmat(B,size(match1_xy,1),1);
% A = [x1,y1,0,0,1,0;0,0,x1,y1,0,1;.....xn,yn,0,0,1,0;0,0,xn,yn,0,1]
A = [A B];
% 若match2_xy是M*2矩阵，那么t_match2_xy是2*M矩阵
% t_match2_xy = [1,2;3,4]，那么b = [1,3,2,4]'的列向量
t_match2_xy = match2_xy'; b = t_match2_xy(:);

% change_form = lower(change_form);
% if contains(change_form, {'project','perspect','homo'}), change_form = 'projective'; end

if strcmpi(change_form,'affine')
    [Q,R] = qr(A);
    H = R\(Q'*b);
    H(7:8) = 0;
    % parameters = A\b;
    N = size(cor1,1);
    match1_test = cor1(:,[1 2]);match2_test = cor2(:,[1 2]);
    M = [H(1) H(2); H(3) H(4)];
    match1_test_trans = M*match1_test'+repmat([H(5); H(6)],1,N);
    match1_test_trans = match1_test_trans';
    test = match1_test_trans-match2_test;
    rmse = sqrt(sum(sum(test.^2))/N);
    
elseif strcmpi(change_form,'projective')
    % Perspective transformation model
    % [u'*w,v'*w,w]' = [u,v,w]' = [a1,a2,a5;
    %                              a3,a4,a6;
    %                              a7,a8, 1]*[x,y,1]'
    % [u',v']' = [x,y,0,0,1,0,-u'x,-u'y;
    %             0,0,x,y,0,1,-v'x,-v'y]*[a1,a2,a3,a4,a5,a6,a7,a8]'
    % 即，Y = A*X
    % 生成[-x1,-y1;-x1,-y1.....;-xm,-ym;-xm,-ym]
    temp_1 = [];
    for i = 1:size(match1_xy,1)
        temp_1 = [temp_1; repmat(match1_xy(i,:),2,1)];
    end
    temp_1 = -temp_1;
    
    % 生成[-u1',-u1';-v1',-v1';....;-um',-um';-vm'，-vm']
    temp_2 = repmat(b,1,2);
    temp = temp_1.*temp_2;
    A = [A temp];
    [Q,R] = qr(A);
    H = R\(Q'*b);
    N = size(cor1,1);
    match1_test = match1_xy';
    match1_test = [match1_test;ones(1,N)];
    M = [H(1),H(2),H(5);...
         H(3),H(4),H(6);...
         H(7),H(8),1];
    match1_test_trans = M*match1_test;
    match1_test_trans_12 = match1_test_trans(1:2,:);
    match1_test_trans_3 = match1_test_trans(3,:);
    match1_test_trans_3 = repmat(match1_test_trans_3,2,1);
    match1_test_trans = match1_test_trans_12./match1_test_trans_3;
    match1_test_trans = match1_test_trans';
    match2_test = match2_xy;
    test = match1_test_trans-match2_test;
    rmse = sqrt(sum(sum(test.^2))/N);
    
elseif strcmpi(change_form,'similarity')
    % [x, y,1,0;
    %  y,-x,0,1]*[a,b,c,d]' = [u,v]
    A = [];
    for i = 1:size(match1_xy,1)
        A = [A; match1_xy(i,:),1,0; match1_xy(i,2),-match1_xy(i,1),0,1];
    end
    [Q,R] = qr(A);
    H = R\(Q'*b);
    % parameters = A\b;
    H(7:8) =  0;
    H(5:6) =  H(3:4);
    H(3)   = -H(2);
    H(4)   =  H(1);
    
    N = size(cor1,1);
    match1_test = cor1(:,[1 2]); match2_test = cor2(:,[1 2]);
    M = [H(1) H(2); H(3) H(4)];
    match1_test_trans = M*match1_test'+repmat([H(5); H(6)],1,N);
    match1_test_trans = match1_test_trans';
    test = match1_test_trans-match2_test;
    rmse = sqrt(sum(sum(test.^2))/N);
end
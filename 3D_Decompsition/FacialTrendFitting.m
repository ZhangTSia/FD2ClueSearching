function [appearance] = FacialTrendFitting(vertex, tri, tex, norm, valid_bin, img, lambda)

% t = zeros(size(vertex));
% t(1,valid_bin) = 1;
% DrawTextureHead(vertex, tri, t);

max_iterations = 2;

[height, width, nChannels] = size(img);
nver = size(vertex,2);

pt2d = vertex([1,2], :);
pt2d(1,:) = min(max(pt2d(1,:),1),width);
pt2d(2,:) = min(max(pt2d(2,:),1),height);
pt2d = round(pt2d);
ind = sub2ind([height,width], pt2d (2,:), pt2d(1,:));

tex_pixel = zeros(size(tex));
for i = 1:nChannels
    temp = img(:,:,i);
    tex_pixel(i,:) = double(temp(ind));
end


%% Shperical Harmonic Basis
harmonic_dim = 9;
nx = norm(1,:)'; ny = norm(2,:)'; nz = norm(3,:)';
harmonic = zeros(size(vertex,2), harmonic_dim);

harmonic(:,1) = sqrt(1/(4*pi)) * ones(size(vertex,2),1);
harmonic(:,2) = sqrt(3/(4*pi)) * nx;
harmonic(:,3) = sqrt(3/(4*pi)) * ny;
harmonic(:,4) = sqrt(3/(4*pi)) * nz;
harmonic(:,5) = 1/2 * sqrt(3/(4*pi)) * (2*nz.*nz - nx.*nx - ny.*ny);
harmonic(:,6) = 3 * sqrt(5/(12*pi)) * (ny.*nz);
harmonic(:,7) = 3 * sqrt(5/(12*pi)) * (nx.*nz);
harmonic(:,8) = 3 * sqrt(5/(12*pi)) * (nx.*ny);
harmonic(:,9) = 3/2 * sqrt(5/(12*pi)) * (nx.*nx - ny.*ny);


A = []; Y = [];
for i = 1:nChannels
    temp = harmonic .* repmat(tex(i,:)', 1, harmonic_dim);
    A = [A; temp(valid_bin,:)];
    Y = [Y; tex_pixel(i,valid_bin)'];
end

light = ones(3,1);

% get light
for i = 1:nChannels
    XX = tex(i, valid_bin)';
    YY = tex_pixel(i, valid_bin)';
    light(i) = (XX' * YY) / (XX' * XX);
end

nver_valid = length(find(valid_bin));
Regular_Matrix = lambda * eye(harmonic_dim);

for i = 1:max_iterations
    
    % 1. get harmonic coefficients
    Y_c = Y;
    for j = 1:nChannels
        Y_c((j-1)*nver_valid+1 : (j-1)*nver_valid + nver_valid) = Y_c((j-1)*nver_valid+1 : (j-1)*nver_valid + nver_valid) / light(j);
    end
    
    % solve the Y_current = A * alpha
    left = A' * Y_c;
    right = A' * A + Regular_Matrix;
    
    alpha = right \ left;
    
    % 2. get light coefficients
    
    for j = 1:nChannels
        Y_c = Y((j-1)*nver_valid+1 : (j-1)*nver_valid + nver_valid);
        A_c = A((j-1)*nver_valid+1 : (j-1)*nver_valid + nver_valid, :) * alpha;
        light(j) = (A_c' * Y_c) / (A_c' * A_c);
    end

end

appearance = zeros(size(tex));
for j = 1:nChannels
    temp = (harmonic .* repmat(tex(j,:)', 1, harmonic_dim)) * alpha * light(j);
    appearance(j,:) = temp';
end

appearance = min(max(appearance,0),1);

end

function [facial_amb_dir_ctex_img, facial_amb_dir_ctex_uv , facial_itex_img, facial_amb_ctex_img, facial_amb_ctex_uv, facial_dir_itex_img , alpha_harmonic, alpha_harmonic_amb, alphas_tex, alphas_tex_amb] = FacialTrendFitting(vertex, tri, mu_tex, w_tex, sigma_tex, norm, valid_bin, img, UV, facial_detail_distribute)
[height, width, nChannels] = size(img);
sigma_tex = ones(size(sigma_tex)) ./ sigma_tex;

vertex_img = vertex;
vertex_img(2,:) = height + 1 - vertex_img(2,:);

lambda = 5e-3;
% t = zeros(size(vertex));
% t(1,valid_bin) = 1;
% DrawTextureHead(vertex, tri, t);

valid_bin1 = [valid_bin(:), valid_bin(:), valid_bin(:)]';
valid_bin1 = valid_bin1(:);
w_tex_valid = w_tex(valid_bin1,:);
mu_tex_valid = mu_tex(valid_bin1);

max_iterations = 2;


pt2d = vertex([1,2], :);
pt2d(2,:) = height + 1 - pt2d(2,:);
pt2d(1,:) = min(max(pt2d(1,:),1),width);
pt2d(2,:) = min(max(pt2d(2,:),1),height);
pt2d = round(pt2d);
ind = sub2ind([height,width], pt2d (2,:), pt2d(1,:));

tex_pixel = zeros(size(vertex));
for i = 1:nChannels
    temp = img(:,:,i);
    tex_pixel(i,:) = double(temp(ind));
end

%% 1. under full light
% Shperical Harmonic Basis
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


Hs = []; Ys = []; Ys1 = [];
for i = 1:nChannels
    Hs{i} = harmonic(valid_bin,:);
    Ys{i} = tex_pixel(i,valid_bin)';
    Ys1 = [Ys1; tex_pixel(i,valid_bin)'];
end


alphas = [];
alpha_tex = zeros(size(w_tex_valid,2), 1);

for i = 1:max_iterations
    
    common_tex = mu_tex_valid + w_tex_valid * alpha_tex;
    common_tex = reshape(common_tex, 3, length(common_tex)/3);
    % 1. get harmonic coefficients
    for c = 1:nChannels
        % solve the Y_current = A * alpha
        H = Hs{c};
        Y = Ys{c};
        
        left = H .* repmat(common_tex(c,:)', 1, harmonic_dim);
        right = Y;
        alpha = left \ right;
        
        alphas{c} = alpha;
    end
    
    H = [];
    for c = 1:nChannels
        H = [H; harmonic(valid_bin,:) * alphas{c}];
    end
    
    left = w_tex_valid .* repmat(H, 1, size(w_tex_valid,2));
    right = Ys1 - mu_tex_valid.* H;
    right = left' * right;
    left = left' * left + eye(size(left,2)) * diag(sigma_tex) * lambda;
    alpha_tex = left \ right;
    alpha_tex = alpha_tex(:);
end

%alpha_tex(:) = 0;
common_tex = mu_tex + w_tex * alpha_tex;
common_tex = reshape(common_tex, 3, length(common_tex)/3);
common_tex = min(max(common_tex,0),1);
%DrawTextureHead(vertex, tri, face_tex);
facial_amb_dir_ctex = zeros(size(common_tex));

for c = 1:nChannels
    temp = (harmonic * alphas{c}) .* common_tex(c,:)';
    %id_tex(c,:) = tex_pixel(c,:)' ./ (harmonic * alphas{c}) - common_tex(c,:)';
    facial_amb_dir_ctex(c,:) = temp';
end

facial_amb_dir_ctex = min(max(facial_amb_dir_ctex,0),1);
[facial_amb_dir_ctex_img, temp] = Mex_ZBuffer(vertex_img, tri, facial_amb_dir_ctex, img);
facial_itex_img = img - facial_amb_dir_ctex_img;

[facial_amb_dir_ctex_uv, tri_ind] = Mex_ZBuffer(UV, tri, facial_amb_dir_ctex, zeros(256,256,3));


%% 2. under amb light
harmonic_dim = 1;
alphas_amb = zeros(3,1);
alpha_tex_amb = zeros(size(w_tex_valid,2), 1);

Hs = []; Ys = []; Ys1 = [];
for i = 1:nChannels
    Hs{i} = harmonic(valid_bin,1);
    Ys{i} = tex_pixel(i,valid_bin)';
    Ys1 = [Ys1; tex_pixel(i,valid_bin)'];
end

for i = 1:max_iterations
    
    common_tex = mu_tex_valid + w_tex_valid * alpha_tex_amb;
    common_tex = reshape(common_tex, 3, length(common_tex)/3);
    % 1. get harmonic coefficients
    for c = 1:nChannels
        % solve the Y_current = A * alpha
        H = Hs{c};
        Y = Ys{c};
        
        left = H .* repmat(common_tex(c,:)', 1, harmonic_dim);
        right = Y;
        alpha = left \ right;
        
        alphas_amb(c) = alpha;
    end
    
    H = [];
    for c = 1:nChannels
        H = [H; harmonic(valid_bin,1) * alphas_amb(c)];
    end
    
    left = w_tex_valid .* repmat(H, 1, size(w_tex_valid,2));
    right = Ys1 - mu_tex_valid.* H;
    right = left' * right;
    left = left' * left + eye(size(left,2)) * diag(sigma_tex) * lambda;
    alpha_tex_amb = left \ right;
    alpha_tex_amb = alpha_tex_amb(:);
end

common_tex_amb = mu_tex + w_tex * alpha_tex_amb;
common_tex_amb = reshape(common_tex_amb, 3, length(common_tex_amb)/3);
common_tex_amb = min(max(common_tex_amb,0),1);

facial_amb_ctex = zeros(size(common_tex_amb));
%id_tex = zeros(size(common_tex));

for c = 1:nChannels
    temp = harmonic(:,1) * alphas_amb(c) .* common_tex_amb(c,:)';
    facial_amb_ctex(c,:) = temp';
end

facial_amb_ctex = min(max(facial_amb_ctex,0),1);
[facial_amb_ctex_img, temp] = Mex_ZBuffer(vertex_img, tri, facial_amb_ctex, img);
facial_dir_itex_img = img - facial_amb_ctex_img;

[facial_amb_ctex_uv, tri_ind] = Mex_ZBuffer(UV, tri, facial_amb_ctex, zeros(256,256,3));

% subplot(3,3,1);
% imshow(img);
% subplot(3,3,2);
% imshow(img);
% subplot(3,3,3);
% imshow(img);
% subplot(3,3,4);
% imshow(facial_amb_dir_ctex_img);
% subplot(3,3,5);
% imshow(facial_amb_dir_ctex_uv);
% subplot(3,3,6);
% imshow(facial_itex_img);
% subplot(3,3,7);
% imshow(facial_amb_ctex_img);
% subplot(3,3,8);
% imshow(facial_amb_ctex_uv);
% subplot(3,3,9);
% imshow(facial_dir_itex_img);

alpha_harmonic = alphas;
alpha_harmonic_amb = alphas_amb;
alphas_tex = alpha_tex;
alphas_tex_amb = alpha_tex_amb;

end

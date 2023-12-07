function Components = Decompose3D(vertex, tri, mu_tex, w_tex, sigma_tex, norm, valid_bin, keypoints, img, UV, facial_detail_distribute)

std_size = 256;

Components = [];


[height, width, nChannels] = size(img);
sigma_tex = ones(size(sigma_tex)) ./ sigma_tex;

facial_itex_min = facial_detail_distribute.facial_itex_min;
facial_itex_max = facial_detail_distribute.facial_itex_max;
facial_dir_itex_min = facial_detail_distribute.facial_dir_itex_min;
facial_dir_itex_max = facial_detail_distribute.facial_dir_itex_max;
facial_dir_min = -0.6;
facial_dir_max = 0.6;

vertex_img = vertex;
vertex_img(2,:) = height + 1 - vertex_img(2,:);
pts_box = vertex_img(1:2, keypoints);
bbox = [min(pts_box(1,:)), min(pts_box(2,:)), max(pts_box(1,:)), max(pts_box(2,:))];


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

% [cropped_img] = crop(img, bbox, std_size);
cropped_img = imresize(img, [std_size, std_size]);
Components.dir_itex_res = cropped_img;

%%%%%%%%%% 1. under full light %%%%%%%%%%%%%%%
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

%% components: dir
alphas_only_dir = [];
for c = 1:nChannels
    tex_this = common_tex(c,valid_bin)';
    amb_this = mean(Ys{c} ./ tex_this);
    tex_in_amb = tex_this * amb_this;
    
    H = Hs{c}(:,2:end);
    Y = Ys{c} - tex_in_amb;
        
    left = H .* repmat(tex_this, 1, harmonic_dim-1);
    right = Y;
    alpha = left \ right;
        
    alphas_only_dir{c} = alpha;
end


facial_dir = zeros(size(common_tex));
for c = 1:nChannels
    temp = (harmonic(:,2:end) * alphas_only_dir{c});
    %id_tex(c,:) = tex_pixel(c,:)' ./ (harmonic * alphas{c}) - common_tex(c,:)';
    facial_dir(c,:) = temp';
end

facial_dir = (facial_dir - facial_dir_min)/(facial_dir_max - facial_dir_min);
facial_dir = min(facial_dir,1);
facial_dir = max(facial_dir,0);
%DrawTextureHead(vertex, tri, facial_dir);
[facial_dir_uv, tri_ind] = Mex_ZBuffer(UV, tri, facial_dir, zeros(256,256,3));
%imshow(facial_dir_uv);
Components.dir = facial_dir_uv;

%% components: itex
facial_amb_dir_ctex = zeros(size(common_tex));
for c = 1:nChannels
    temp = (harmonic * alphas{c}) .* common_tex(c,:)';
    %id_tex(c,:) = tex_pixel(c,:)' ./ (harmonic * alphas{c}) - common_tex(c,:)';
    facial_amb_dir_ctex(c,:) = temp';
end
facial_amb_dir_ctex = min(max(facial_amb_dir_ctex,0),1);
facial_itex = tex_pixel - facial_amb_dir_ctex;
[facial_itex_uv, tri_ind] = Mex_ZBuffer(UV, tri, facial_itex, zeros(256,256,3));
Components.itex = facial_itex_uv - facial_itex_min / (facial_itex_max - facial_itex_min);

%% components: dir+res: w/o itex
[facial_amb_dir_ctex_img, temp] = Mex_ZBuffer(vertex_img, tri, facial_amb_dir_ctex, img);
% [cropped_img] = crop(facial_amb_dir_ctex_img, bbox, std_size);
cropped_img = imresize(facial_amb_dir_ctex_img, [std_size, std_size]);
Components.dir_res = cropped_img;

%%%%%%%%%%% 2. under amb light %%%%%%%%%%%%%
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
    
    common_tex_amb = mu_tex_valid + w_tex_valid * alpha_tex_amb;
    common_tex_amb = reshape(common_tex_amb, 3, length(common_tex_amb)/3);
    % 1. get harmonic coefficients
    for c = 1:nChannels
        % solve the Y_current = A * alpha
        H = Hs{c};
        Y = Ys{c};
        
        left = H .* repmat(common_tex_amb(c,:)', 1, harmonic_dim);
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


%% components: res
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
[facial_amb_ctex_img, valid_map] = Mex_ZBuffer(vertex_img, tri, facial_amb_ctex, img);
% [cropped_img] = crop(facial_amb_ctex_img, bbox, std_size);
cropped_img = imresize(facial_amb_ctex_img, [std_size, std_size]);
Components.res = cropped_img;

%% components: itex+res (w/o dir)
tex = common_tex + facial_itex;
facial_amb_tex = zeros(size(tex));
for c = 1:nChannels
    temp = harmonic(:,1) * alphas_amb(c) .* tex(c,:)';
    facial_amb_tex(c,:) = temp';
end
facial_amb_tex = min(facial_amb_tex,1);
facial_amb_tex = max(facial_amb_tex,0);
[facial_amb_tex_img, valid_map] = Mex_ZBuffer(vertex_img, tri, facial_amb_tex, img);

% [cropped_img] = crop(facial_amb_tex_img, bbox, std_size);
cropped_img = imresize(facial_amb_tex_img, [std_size, std_size]);
Components.itex_res = cropped_img;

%% components: itex+dir
[facial_amb_ctex_uv, tri_ind] = Mex_ZBuffer(UV, tri, facial_amb_ctex, zeros(256,256,3));
[img_uv, tri_ind] = Mex_ZBuffer(UV, tri, tex_pixel, zeros(256,256,3));
facial_dir_itex_uv = img_uv - facial_amb_ctex_uv;
Components.dir_itex = (facial_dir_itex_uv - facial_dir_itex_min) / (facial_dir_itex_max - facial_dir_itex_min);



end

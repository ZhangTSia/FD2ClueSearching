%load('Model_BFM_PCA.mat');
%load('Model_BFM_Exp.mat');
load('Modelplus_face_bin.mat');
load('BFM_UV');
std_size = 256;
UV(:,1) = UV(:,1) * (std_size-1) + 1;
UV(:,2) = UV(:,2) * (std_size-1) + 1;
UV(:,2) = std_size + 1 - UV(:,2);
UV = [UV, zeros(size(UV,1),1)]';

path = 'D:\Database\FaceAlignment\DAFL\data\AFW\';
des_path = 'results\';
filelist = dir([path '*.mat']);

mu = mu_shape + mu_exp;
tex = reshape(mu_tex, 3, length(mu_tex)/3) / 255;
w_tex = w_tex(:, 1:20);
sigma_tex = sigma_tex(1:20);

facial_itex_img_all = [];
facial_dir_itex_img_all = [];

for fi = 1:length(filelist)
tic
    filename = filelist(fi).name(1:end-4);
     load([path filename '.mat']);
    %filename = 'AFW_815038_1_0';
    img = double(img) / 255;
    [height, width, nChannels] = size(img);
   
    
    vertex = mu + w * Shape_Para + w_exp * Exp_Para;
    vertex = reshape(vertex, 3, length(vertex)/3);
    [phi, gamma, theta, t3d, f] = ParaMap_Pose(Pose_Para);
    R = RotationMatrix(phi, gamma, theta);
    ProjectVertex = f * R * vertex + repmat(t3d, 1, size(vertex,2));
    
    norm = NormDirection(ProjectVertex, tri);
    visibility = norm(3,:) < 0;

    [facial_amb_dir_ctex_img, facial_amb_dir_ctex_uv , facial_itex_img, facial_amb_ctex_img, facial_amb_ctex_uv, facial_dir_itex_img , alpha_harmonic, alpha_harmonic_amb, alphas_tex, alphas_tex_amb] = facial_detail_distribute_aid(ProjectVertex, tri, mu_tex/255, w_tex/255, sigma_tex, norm, face_front_bin(:) & visibility(:), img, UV);
    
    facial_itex_pixels = [];
    facial_dir_itex_pixels = [];
    for c = 1:3
        temp = facial_itex_img(:,:,c);
        temp = temp(temp~=0)';
        rand_ind = randperm(length(temp));
        temp = temp(rand_ind(1:1000));
        
        facial_itex_pixels = [facial_itex_pixels; temp];
        
        temp = facial_dir_itex_img(:,:,c);
        temp = temp(temp~=0)';
        rand_ind = randperm(length(temp));
        temp = temp(rand_ind(1:1000));
        facial_dir_itex_pixels = [facial_dir_itex_pixels; temp];
    end
    
    facial_itex_img_all = [facial_itex_img_all, facial_itex_pixels];
    facial_dir_itex_img_all = [facial_dir_itex_img_all, facial_dir_itex_pixels];
    
    
    
%     imwrite(facial_amb_dir_ctex_img, [des_path 'facial_amb_dir_ctex_img\' filename '.jpg']);
%     imwrite(facial_amb_dir_ctex_uv, [des_path 'facial_amb_dir_ctex_uv\' filename '.jpg']);
%     imwrite(facial_itex_img, [des_path 'facial_itex_img\' filename '.jpg']);
%     
%     imwrite(facial_amb_ctex_img, [des_path 'facial_amb_ctex_img\' filename '.jpg']);
%     imwrite(facial_amb_ctex_uv, [des_path 'facial_amb_ctex_uv\' filename '.jpg']);
%     imwrite(facial_dir_itex_img, [des_path 'facial_dir_itex_img\' filename '.jpg']);
%     
%     save([des_path 'meta\' filename '.mat'], 'alpha_harmonic', 'alpha_harmonic_amb', 'alphas_tex', 'alphas_tex_amb');
    
toc
end

% facial_itex_mean = mean(facial_itex_img_all, 2);
% facial_itex_std = std(facial_itex_img_all, 0, 2);
% facial_itex_min = facial_itex_mean - 5 * facial_itex_std;
% facial_itex_max = facial_itex_mean + 5 * facial_itex_std;
% 
% facial_dir_itex_mean = mean(facial_dir_itex_img_all, 2);
% facial_dir_itex_std = std(facial_dir_itex_img_all, 0, 2);
% facial_dir_itex_min = facial_dir_itex_mean - 5 * facial_dir_itex_std;
% facial_dir_itex_max = facial_dir_itex_mean + 5 * facial_dir_itex_std;

facial_itex_mean = mean(facial_itex_img_all(:));
facial_itex_std = std(facial_itex_img_all(:));
facial_itex_min = facial_itex_mean - 3 * facial_itex_std;
facial_itex_max = facial_itex_mean + 3 * facial_itex_std;

facial_dir_itex_mean = mean(facial_dir_itex_img_all(:));
facial_dir_itex_std = std(facial_dir_itex_img_all(:));
facial_dir_itex_min = facial_dir_itex_mean - 3 * facial_dir_itex_std;
facial_dir_itex_max = facial_dir_itex_mean + 3 * facial_dir_itex_std;

save('facial_detail_distribute.mat', 'facial_itex_mean', 'facial_itex_std', 'facial_dir_itex_mean', 'facial_dir_itex_std', 'facial_itex_min', 'facial_itex_max', 'facial_dir_itex_min', 'facial_dir_itex_max');


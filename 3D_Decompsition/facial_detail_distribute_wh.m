load('Model_BFM_PCA.mat');
load('Model_BFM_Exp.mat');
load('Modelplus_face_bin.mat');
load('BFM_UV');
std_size = 256;
UV(:,1) = UV(:,1) * (std_size-1) + 1;
UV(:,2) = UV(:,2) * (std_size-1) + 1;
UV(:,2) = std_size + 1 - UV(:,2);
UV = [UV, zeros(size(UV,1),1)]';

path = '../../dataset/ffpp/new_idea/';
cls = ['manipulated' '_sequences/'];
type = 'NeuralTextures/';
des_path = 'results/';
data_path = [path cls type 'c23/3d/'];
filelist = dir(data_path);

mu = mu_shape + mu_exp;
tex = reshape(mu_tex, 3, length(mu_tex)/3) / 255;
w_tex = w_tex(:, 1:20);
sigma_tex = sigma_tex(1:20);

facial_itex_img_all = [];
facial_dir_itex_img_all = [];

for i = 1:length(filelist)
    if(isequal(filelist(i).name,'.')||...
       isequal(filelist(i).name,'..')||...
       ~filelist(i).isdir)
           continue;
    end
    mat_list = dir([data_path filelist(i).name '/*.mat']); 
    for j =1:length(mat_list)
        if isequal(mat_list(j).name, '0000.mat')
            tic;
            mat_file = [data_path filelist(i).name '/' mat_list(j).name];
            load(mat_file);
            
            img = double(img) / 255;
            img_rgb = cat(3,img(:,:,3),img(:,:,2),img(:,:,1));
            [height, width, nChannels] = size(img_rgb);
            
            ProjectVertex = vertex;
%             ProjectVertex(2,:) = height + 1 - ProjectVertex(2,:);
            norm = NormDirection(ProjectVertex, tri);
            visibility = norm(3,:) < 0;

            [facial_amb_dir_ctex_img, facial_amb_dir_ctex_uv , facial_itex_img, facial_amb_ctex_img, facial_amb_ctex_uv, facial_dir_itex_img , alpha_harmonic, alpha_harmonic_amb, alphas_tex, alphas_tex_amb] = facial_detail_distribute_aid(ProjectVertex, tri, mu_tex/255, w_tex/255, sigma_tex, norm, face_front_bin(:) & visibility(:), img_rgb, UV);

            facial_itex_pixels = [];
            facial_dir_itex_pixels = [];
            
            sign_ = 0;
            for c = 1:3
                temp = facial_itex_img(:,:,c);
                temp = temp(temp~=0)';
                if length(temp) < 1000
                    sign_ = 1;
                end
            end
            
            if sign_ == 1
                continue
            end
            
            for c = 1:3
                temp = facial_itex_img(:,:,c);
                temp = temp(temp~=0)';
                if length(temp) < 1000
                    continue
                end
                rand_ind = randperm(length(temp));
                temp = temp(rand_ind(1:1000));

                facial_itex_pixels = [facial_itex_pixels; temp];

                temp = facial_dir_itex_img(:,:,c);
                temp = temp(temp~=0)';
                rand_ind = randperm(length(temp));
                temp = temp(rand_ind(1:1000));
                facial_dir_itex_pixels = [facial_dir_itex_pixels; temp];
            end
            toc;
            facial_itex_img_all = [facial_itex_img_all, facial_itex_pixels];
            facial_dir_itex_img_all = [facial_dir_itex_img_all, facial_dir_itex_pixels];
        end
    end
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

save('facial_detail_distribute_wh.mat', 'facial_itex_mean', 'facial_itex_std', 'facial_dir_itex_mean', 'facial_dir_itex_std', 'facial_itex_min', 'facial_itex_max', 'facial_dir_itex_min', 'facial_dir_itex_max');


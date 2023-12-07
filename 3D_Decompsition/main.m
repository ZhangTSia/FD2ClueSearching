load('Model_BFM_PCA.mat');
load('Model_BFM_Exp.mat');
load('Modelplus_face_bin.mat');
load('BFM_UV');

std_size = 256;
UV(:,1) = UV(:,1) * (std_size-1) + 1;
UV(:,2) = UV(:,2) * (std_size-1) + 1;
UV(:,2) = std_size + 1 - UV(:,2);
UV = [UV, zeros(size(UV,1),1)]';

load('facial_detail_distribute_wh.mat');
facial_detail_distribute.facial_itex_min = facial_itex_min;
facial_detail_distribute.facial_itex_max = facial_itex_max;
facial_detail_distribute.facial_dir_itex_min = facial_dir_itex_min;
facial_detail_distribute.facial_dir_itex_max = facial_dir_itex_max;

path = '../data/';
data_path = [path '/vertex/'];
filelist = dir(data_path);

mu = mu_shape + mu_exp;
tex = reshape(mu_tex, 3, length(mu_tex)/3) / 255;
w_tex = w_tex(:, 1:20);
sigma_tex = sigma_tex(1:20);

for i = 1:length(filelist)
%     if i < 552
%         continue
%     end
    if(isequal(filelist(i).name,'.')||...
       isequal(filelist(i).name,'..')||...
       ~filelist(i).isdir)
           continue;
    end
    if strcmp(type, 'Face2Face/') && strcmp(filelist(i), '654_648')
        continue
    end
    mat_list = dir([data_path filelist(i).name '/*.mat']); 
    for j =1:length(mat_list)
        tic;
        mat_file = [data_path filelist(i).name '/' mat_list(j).name];
        load(mat_file);

        img_rgb = double(imread(strrep(strrep(mat_file, 'vertex', 'face'), '.mat', '.png'))) / 255;
        [height, width, nChannels] = size(img_rgb);

        ProjectVertex = vertex;
        ProjectVertex(2,:) = height + 1 - ProjectVertex(2,:);
        norm = NormDirection(ProjectVertex, tri);
        visibility = norm(3,:) < 0;

        Components = Decompose3D(ProjectVertex, tri, mu_tex/255, w_tex/255, sigma_tex, norm, face_front_bin(:) & visibility(:), keypoints, img_rgb, UV, facial_detail_distribute);

        field = fieldnames(Components);
        for k = 1:length(field)
            name_i = field{k};
            comp = getfield(Components, name_i);
            des_name = [strrep(mat_file(1:end-4), 'vertex', name_i) '.png'];
            [filepath, name, ext] = fileparts(des_name);
            if ~exist(filepath, 'dir')
                mkdir(filepath);
            end
            imwrite(comp, des_name);
        end
        
        toc;
    end
end

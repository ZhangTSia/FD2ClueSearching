function [result] = Mex_FaceFrontalizationFilling(img, corres_map)
 % From matlab index to C index
corres_map = corres_map - 1;
                                      
[result] = Mex_FaceFrontalizationFillingC(img, size(img,2), size(img,1), size(img,3), corres_map);

end



%% Matlab Ver.
% function [des_img, sym_img] = Frontalization(img, mask, tri_ind, ...
%                                             all_vertex_src, all_vertex_ref, all_tri, bg_tri_num, ...
%                                             valid_tri_half, vertex_length, tri_length, ...
%                                             sym_tri_list)
%                                         
% des_img = zeros(size(img)); % the des image with invisible pixel set to 0
% sym_img = zeros(size(img)); % the full image filling with face symmetry
% 
% for y = 1:height
%     for x = 1:width
%         corres_tri = tri_ind(y,x);
%         if(corres_tri == 0)
%             des_img(y,x,:) = 0;
%             continue;
%         end
%         
%         % filling the des_img
%         pt1 = all_vertex_ref([1,2], all_tri(1, corres_tri));
%         pt2 = all_vertex_ref([1,2], all_tri(2, corres_tri));
%         pt3 = all_vertex_ref([1,2], all_tri(3, corres_tri));
%         pt = [x;y];
%         weight = position2weight(pt, pt1, pt2, pt3);
%         
%         pt1 = all_vertex_src([1,2], all_tri(1, corres_tri));
%         pt2 = all_vertex_src([1,2], all_tri(2, corres_tri));
%         pt3 = all_vertex_src([1,2], all_tri(3, corres_tri));
%         
%         pt_src = pt1 * weight(1) + pt2 * weight(2) + pt3 * weight(3);
% 
%         des_pixel = BilinearInterpolation(img, pt_src(1), pt_src(2));
%         
%         % visibility test
%         if(mask(y,x)==1)         
%         	% invisible pixel
%             des_img(y,x,1) = 0;
%         else
%             des_img(y,x,:) = des_pixel;
%         end
%      
%         
%         % the sym_img
%         sym_img(y,x,:) = des_pixel;
%         if((corres_tri > bg_tri_num) && ~valid_tri_half(corres_tri - bg_tri_num))   
%             %sym_tri = FindSym(corres_tri-size(tri_in,2), symlist_tri) + size(tri_in,2);
%             sym_tri = FindSym(corres_tri - bg_tri_num, sym_tri_list) + bg_tri_num;
%             pt1 = all_vertex_src([1,2], all_tri(1, sym_tri));
%             pt2 = all_vertex_src([1,2], all_tri(2, sym_tri));
%             pt3 = all_vertex_src([1,2], all_tri(3, sym_tri));
%             pt_sym = (pt1 + pt2 + pt3)/3;
% 
%             sym_pixel = BilinearInterpolation(img, pt_sym(1), pt_sym(2));
%             sym_img(y,x,:) = sym_pixel;
%         end
%     end
% end
% 
% 
% end


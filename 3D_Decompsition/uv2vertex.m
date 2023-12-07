function [vertex] = uv2vertex(uv_map, UV)
[height, width, nChannels] = size(uv_map);

UV(:,1) = UV(:,1) * (width-1) + 1;
UV(:,2) = UV(:,2) * (height-1) + 1;
UV(:,2) = height + 1 - UV(:,2);
UV = round(UV);

ind = sub2ind([height, width], UV(:,2), UV(:,1));

vertex = zeros(3, size(UV,1));

for c = 1:3
    temp = uv_map(:,:,c);
    vertex(c,:) = temp(ind);
end


end


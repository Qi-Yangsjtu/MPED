function score=SPED_score_v(pc_ori,pc_dis,pc_fast,k,distance_type, color_type, T)
if k<1
    error('the numebr of neighbors should be bigger than 1!');
end
center_coordinate = pc_fast.Location;
source_coordinate = pc_ori.Location;
target_coordinate = pc_dis.Location;

center_color = single(pc_fast.Color);
source_color = single(pc_ori.Color);
target_color = single(pc_dis.Color);
%% Neighborhood establish
% idx_source: size(center)*10, idex of neighbor, dit_source:  size(center)*10, Euclidea_distance of neighbor to center  
[idx_source, dit_source] = knnsearch( source_coordinate, center_coordinate, 'k', k,  'distance', 'euclidean');
[idx_target, dit_target] = knnsearch( target_coordinate, center_coordinate, 'k', k,  'distance', 'euclidean');

%% Potential energy of each neighborhood
center_mass = center_color;
%neighbor source mass is a matrix size(center)*10*3
idx_source_T=idx_source';
idx_target_T=idx_target';
dit_source_T=dit_source';
dit_target_T=dit_target';
neighbor_source_mass = source_color(idx_source_T(:),:);
neighbor_target_mass = target_color(idx_target_T(:),:);
neighbor_source_coordinate = source_coordinate(idx_source_T(:),:);
neighbor_target_coordinate = target_coordinate(idx_target_T(:),:);
dis_square_source = (dit_source_T(:)).^2; 
dis_square_target = (dit_target_T(:)).^2; 
%spatial filed

% sigma: size(center)*1
sigma = mean(dit_source.^2,2);
sigma_rep = repmat(sigma,1,k)';
sigma_r = sigma_rep(:)+T;
if k==1
    g_source = 1;
    g_target = 1;
else
    g_source = 1./(1+exp(-dis_square_source./sigma_r));
    g_target = 1./(1+exp(-dis_square_target./sigma_r));
end
center_mass_rep = reshape(repmat(center_mass(:)',k,[]),[],3);
source_mass_dif = abs(neighbor_source_mass-center_mass_rep);
target_mass_dif = abs(neighbor_target_mass-center_mass_rep);
if color_type == 'RGB'
    source_mass_dif = 1*source_mass_dif(:,1)+2*source_mass_dif(:,2)+1*source_mass_dif(:,3)+1;
    target_mass_dif = 1*target_mass_dif(:,1)+2*target_mass_dif(:,2)+1*target_mass_dif(:,3)+1;
elseif color_type == 'GCM'|color_type == 'YUV'
    source_mass_dif = 6*source_mass_dif(:,1)+1*source_mass_dif(:,2)+1*source_mass_dif(:,3)+1;
    target_mass_dif = 6*target_mass_dif(:,1)+1*target_mass_dif(:,2)+1*target_mass_dif(:,3)+1;
else
    ('Wrong color type! Please use RGB, YUV or GCM!');
end
center_coordinate_rep = reshape(repmat(center_coordinate(:)',k,[]),[],3);
% distance between center and neighbor
source_coordinate_dif = neighbor_source_coordinate - center_coordinate_rep;
target_coordinate_dif = neighbor_target_coordinate - center_coordinate_rep;
if distance_type=='1-norm'
    source_distance_dif = sum(abs(source_coordinate_dif),2);
    target_distance_dif = sum(abs(target_coordinate_dif),2);
elseif distance_type=='2-norm'
    source_distance_dif = dis_square_source;
    target_distance_dif = dis_square_target;
else
    error('Wrong distance type! Please use 1-norm or 2-norm!');
end
    energy_source = sum(source_mass_dif.* g_source.*source_distance_dif);
    energy_target = sum(target_mass_dif.* g_target.*target_distance_dif);
score = similarity(energy_source,energy_target,T);

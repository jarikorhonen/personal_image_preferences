% Script for extracting features for CEED2016 database images
% Run in the root directory where CEED2013 database is installed
%
f = fopen('.\\ceed2016_features_x.csv','w+');

i=1;
for set=1:30

    %filepath = sprintf('g:\\Mendeley_dataset\\original_images\\img%d.bmp',set)
    %orig_features = ComputeImageFeatures(filepath);
    
    for im=1:6        
        filepath = sprintf('.\\enhanced_images\\img%d-%d.bmp',set,im)
        disp(['Extracting features from image ' filepath]);        
        features = compute_image_features(filepath);    
        for j=1:length(features)-1
            fprintf(f, '%1.5f,', features(j));
        end
        fprintf(f, '%1.5f', features(end));
        fprintf(f, '\n'); 
        i = i + 1;
    end
end

fclose(f);
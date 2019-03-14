% Script for extracting features for CID2013 database images
% Run in the root directory where CID2013 database is installed
%
set = 6; co = 5; de=11;

cs = [1 2 3 4 5 6
      1 2 3 4 5 6
      1 2 3 4 5 6
      1 2 3 4 5 6
      1 2 3 5 6 8
      1 2 3 6 7 8];
  
ds = [14 13 13 13 12 14];
roman = {'I','II','III','IV','V','VI'};

i=1;
for set=1:6
    feature_path = sprintf('.\\cid2013_features_dataset_%d.csv',set);
    f = fopen(feature_path,'w+');
    for c=cs(set,1:6)
        for d=1:ds(set)
            path = sprintf('.\\IS%d\\co%d\\',set,c);
            filepath = sprintf('%sIS_%s_C0%d_D%02d.jpg',path,roman{set},c,d);
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
end



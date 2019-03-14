function features = compute_image_features(filename)

    % Read image from file
    image = cast(imread(filename),'double')./255.0;
    
    % Initializations
    [height,width,~] = size(image);
    yuv_image = rgb2ycbcr(image);
    mono_image = yuv_image(:,:,1);
    lab_image = rgb2lab(image);
    
    % Apply Sobel filter
    mask = zeros(height,width);
    mask(3:end-2,3:end-2)=1;
    H = [1 2 1; 0 0 0; -1 -2 -1]./8;
    sob_image(:,:,1) = imfilter(mono_image(:,:),H).*mask;
    sob_image(:,:,2) = imfilter(mono_image(:,:),H').*mask;
    sob_image(:,:,3) = sqrt(sob_image(:,:,1).^2+sob_image(:,:,2).^2);
    
    % Find bright saturated areas
    [a,b,sat_image1] = measure_saturation(mono_image,1);
    [c,d,sat_image2] = measure_saturation(mono_image,0);
    sat_image = max(sat_image1, sat_image2);
    saturation = [a b c d];  
    
    % Spatial activity related features
    spatial = measure_spatial_activity(sob_image, sat_image);

    % Noise features
    noisiness = measure_noise(mono_image, sat_image, lab_image) ;  
        
    % Sharpness related features
    sharpness = measure_sharpness(sob_image);
                                                 
    % Contrast related features
    contrast = measure_contrast_exposure(lab_image);
    
    % Colorfulness related features
    color = measure_colorfulness(image);

    % Make the feature vector
    features = [spatial saturation noisiness sharpness contrast color];
end


% This function computes the saturation (bright or dark)
function [len,num,segs] = measure_saturation(image, is_bright)

    [height,width] = size(image);

    lens = [];
    num = 0;    
    
    segs = zeros(height,width);
    
    if (is_bright==1 && max(max(image))>0.9) || ...
       (is_bright==0 && min(min(image))<0.1)     
    
        segs = seg_loop(image,segs,3,3,0.05, is_bright);
        for i=1:max(max(segs))
            len = length(find(segs==i));
            if len<50
                segs(find(segs==i))=0;
            else
                lens = [lens len];
                num = num + 1;
            end
        end 
        segs(find(segs>0))=1;
    end
    
    len = sum(lens)/(width*height);
    if num > 0
        num = len / num;
    end

end

% This function is used for segmentation by measure_saturation
function segim = seg_loop(image, segs, wh, ww, interval, is_bright)

    [height,width] = size(image);

    segim = segs;
    
    maxi = max(max(image));
    mini = min(min(image));
    
    for i=1:height-wh+1
        for j=1:width-ww+1
            if (is_bright == 1 && ...
              min(min(image(i:i+wh-1,j:j+ww-1)))>maxi-interval) || ...
              (is_bright == 0 && ...
              max(max(image(i:i+wh-1,j:j+ww-1)))<mini+interval)
            
                maxsg = max(max(segim(i:i+wh-1,j:j+ww-1)));
                if maxsg>0
                    segs_temp = reshape(segim(i:i+wh-1,j:j+ww-1),wh*ww,1);
                    minsg=min(segs_temp(find(segs_temp>0)));
                    segim(i:i+wh-1,j:j+ww-1)=minsg;
                    if minsg<maxsg
                        segim(find(segim==maxsg))=minsg;
                    end
                else
                    segim(i:i+wh-1,j:j+ww-1)=max(max(segim(:,:)))+1;
                end
            end
        end
    end

end

% This function is used to compute noise related features
function out = measure_noise(mono_image, sat_im, lab_image)
    
    [height,width] = size(mono_image);

    new_im = zeros(height, width, 3);

    nonsat_pix = 0;
    noise_pix = 0;
    noise_int = [];
    
    for i=5:height-4
        for j=5:width-4
            if sat_im(i,j)==0
                surr_pix = mono_image(i-2:i+2,j-2:j+2);
                surr_pix = surr_pix(:);
                surr_pix = [surr_pix(1:12); surr_pix(14:25)];
                if (mono_image(i,j)>max(surr_pix) || ...
                    mono_image(i,j)<min(surr_pix))
                    surr_pix = mono_image(i-3:i+3,j-3:j+3);
                    if std2(surr_pix)<0.05
                        new_im(i,j,2) = 1;
                        pix_diff = sqrt( ...
                            mean(mean((lab_image(i-2:i+2,j-2:j+2,1)-...
                                 lab_image(i,j,1)).^2)) + ...
                            mean(mean((lab_image(i-2:i+2,j-2:j+2,2)-...
                                 lab_image(i,j,2)).^2)) + ...
                            mean(mean((lab_image(i-2:i+2,j-2:j+2,3)-...
                                 lab_image(i,j,3)).^2)));
                        noise_int = [noise_int pix_diff/100]; 
                        noise_pix = noise_pix + 1;
                    end
                end
                nonsat_pix = nonsat_pix + 1;
            end
        end
    end

    noise_intensity = 0;
    noise_variance = 0;
    
    if nonsat_pix > 0 && noise_pix > 0
        noise_intensity = noise_pix / nonsat_pix;
        noise_variance = std(noise_int);
    end
    
    out = [noise_intensity noise_variance];
       
end

% This function is used to compute spatial activity features
function res = measure_spatial_activity(sobel_image, sat_image)
    
    [height,width,~] = size(sobel_image);
    
    sob_strength = sobel_image(:,:,3);
       
    sob_dists = zeros(1,height*width);
    sob_dists2 = zeros(height*width,2);
    sob_str = zeros(1,height*width);
    sumstr = 0;
    
    n = 0;
    for i=1:height
        for j=1:width
            if sat_image(i,j)==0
                if sob_strength(i,j)<0.01
                    sob_strength(i,j)=0;
                end
                sumstr = sumstr + sob_strength(i,j);
                if sob_strength(i,j) > 0
                    n = n + 1;
                    sob_str(n) = sob_strength(i,j);
                    sob_dists(n) = sqrt((i/height-0.5)^2+(j/width-0.5)^2);
                    sob_dists2(n,1) = i/height-0.5;
                    sob_dists2(n,2) = j/width-0.5;                   
                end
            end
        end
    end  
    
    sob_str = sob_str(1:n);
    
    res = 0;
    if ~isempty(sob_str)>0
        res = std2(sob_strength);      
    end

end

% This function is used to compute contrast and exposure related features
function out = measure_contrast_exposure(lab_image)

    a=0;
    b=0;
    
    [height,width,~] = size(lab_image);
    yuv_int = floor(lab_image(:,:,1));
    
    %sat_image = sat_image(:);
    %yuv_int2 = yuv_int(sat_image(:)==0);
    yuv_int2 = yuv_int(:);
    cumu_err = 0;
    cumu_tar = 0;
    if ~isempty(yuv_int2)
        for i=0:100
            cumu_tar = cumu_tar + 1/100;
            cumu_err = cumu_err + (sum(yuv_int2<=i)/length(yuv_int2) - ...
                       cumu_tar)/100;
        end
        a = (cumu_err+1.0)/2.0;
    else
        a = 1;
    end
    
    % Compute contrast
    block_means = [];
    local_contrasts = [];
    for i=1:16:height-15
        for j=1:16:width-15
            pixels = lab_image(i:i+15,j:j+15,1)./100;
            pixels = pixels(:);
            block_means = [block_means mean(pixels)];
            pixels = sort(pixels,'descend');
            local_contrasts = [local_contrasts ...
                mean(pixels(1:5))-mean(pixels(end-4:end))];
        end
    end
    blk_means = sort(block_means,'descend');
    b = sum(blk_means(1:3))/6 + sum(blk_means(end-2:end))/6;
    
    out = [a b];
end
    

% This function is used to compute sharpness related features
function out = measure_sharpness(sob_im)
    
    im_s_h = sob_im(:,:,1);
    im_s_v = sob_im(:,:,2);
    im_s = sob_im(:,:,3);
    
    [height,width] = size(im_s_h);
    bl_size = 16;
    conv_list = [];
    
    blur_im = zeros(height,width);
    edge_strong = [];
    edge_all = [];
    
    conv_cube = [];
    blurvals = [];
    
    n_blks = 0;
    
    conv_val_tot = zeros(17);
    for y=floor(bl_size/2):bl_size:height-ceil(3*bl_size/2)
        for x=floor(bl_size/2):bl_size:width-ceil(3*bl_size/2)
            
            n_blks = n_blks + 1;
            
            conv_val = zeros(17);
            for i=0:6
                for j=0:6
                    if i==0 || j==0 || i==j
                        weight_h = 1;
                        weight_v = 1;
                        if i~=0 || j~=0
                            weight_h = abs(i)/(abs(i)+abs(j));
                            weight_v = abs(j)/(abs(i)+abs(j));
                        end
                        diff_h = (im_s_h(y+i:y+bl_size+i,   ...
                                         x+j:x+bl_size+j).* ...
                                  im_s_h(y:y+bl_size,       ...
                                         x:x+bl_size));
                        diff_v = (im_s_v(y+i:y+bl_size+i,   ...
                                         x+j:x+bl_size+j).* ...
                                  im_s_v(y:y+bl_size,       ...
                                         x:x+bl_size));
                        conv_val(i+9,j+9) = weight_h*(mean(diff_h(:)))+ ...
                                            weight_v*(mean(diff_v(:)));
                    end
                end
            end
            blur_im(y:y+bl_size-1,x:x+bl_size-1)=0.5;
            edge_all =  [edge_all conv_val(9,9)];
            if conv_val(9,9)>0.0001
                edge_strong =  [edge_strong conv_val(9,9)];
                conv_val=conv_val./conv_val(9,9);
                conv_val_tot = conv_val_tot + conv_val;

                new_conv_v = [];
                for i=1:6
                    new_conv_v = [new_conv_v sum(sum(conv_val(9-i:9+i,...
                                                            9-i:9+i)))- ...
                                             sum(sum(conv_val(10-i:8+i, ...
                                                            10-i:8+i)))];
                end
                if new_conv_v(1)>0
                    new_conv_v=new_conv_v./new_conv_v(1);
                end

                conv_list = [conv_list; new_conv_v];
                conv_cube(:,:,1)=conv_val;
                blurvals = [blurvals std2(im_s(y:y+bl_size, x:x+bl_size))];

                blur_im(y:y+bl_size-1,x:x+bl_size-1) = ...
                                  0.5 + mean(new_conv_v(2:6))/5;
            end
        end
    end

    % Find the sharpest blocks
    blurs_sharp = [];
    blurs_blur = [];
    if ~isempty(blurvals)    
        for i=1:length(blurvals)
            if blurvals(i)>mean(blurvals)
                conv_val_tot = + conv_val_tot + conv_cube(:,:,1);
            end
        end
    end
   
    new_conv_v = zeros(1,9);
    if ~isempty(edge_strong)>0
        if length(conv_list(:,1))>1
            new_conv_v = mean(conv_list);
        else
            new_conv_v = conv_list;
        end
    end 
       
    out = [mean(new_conv_v(4:6)) mean(new_conv_v(2:3))];
end

% This function is used to compute colorfulness related features
function res = measure_colorfulness(rgb_im)

    col_max = max(rgb_im,[],3);
    col_min = min(rgb_im,[],3);
    diff = col_max-col_min;
    diff = sort(diff, 'descend');
    res = mean(diff(1:floor(0.05*end)));
end
    
    
    
    
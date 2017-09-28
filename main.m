% References:
% 1) sort_nat.m function: http://www.mathworks.com/matlabcentral/fileexchange/10959-sort-nat--natural-order-sort
% 2) https://www.mathworks.com/help/nnet/ref/feedforwardnet.html
% 3) Eugene Li: ME 597 Course Notes
% ----------------------------------------------------------------------- %

% Close all figures ; clear output
clear;
clc;
close all;

% Simulation Initializations
dt = 0.1;
Tf = 10;
T = 0:dt:Tf;

% Load (all) Image Data
cd 'imageClassify';
files = dir('*.png');
cd ..;
file_names = extractfield(files, 'name');
file_names_ordered = sort_nat(squeeze(file_names(1,:)));
for i=1:length(file_names_ordered)
   images(i, :, :) = imread(strcat('imageClassify/', char(file_names_ordered(1,i))));
end

% Calculate Pixel Density for each image
for k=1:length(images)
    pixel_count = 0;
    total_count = 0;
    current_image = squeeze(images(k,:,:));
    for i=1:length(current_image)
       for j=1:length(current_image)
           if current_image(i,j) ~= 0
               pixel_count = pixel_count + 1;
           end
           total_count = total_count + 1;
       end
    end
    pixel_density(k) = pixel_count / total_count;
end

% Create Neural Net
net = feedforwardnet(10);

% Create Training Data
training_data_size = 550;
for i=1:training_data_size
   
    % Group 1: Pixel Density <= 10%
    if pixel_density(i) <= 0.1
        training_data(i,1) = pixel_density(i);
    % Group 2: Pixel Density >= 15%
    elseif pixel_density(i) >= 0.15
        training_data(i,2) = pixel_density(i);
    % Group 3: 10% < Pixel Density < 15%
    else
        training_data(i,3) = pixel_density(i);
    end
end

% Train Neural Net
training_input = pixel_density(1:training_data_size);
training_data = training_data'; % Need to match input dimensions exactly
net = train(net, training_input, training_data);

% Test Neural Net
test_input = pixel_density;
output = sim(net, test_input);

% Clean data and plot
hist_count = zeros(1,3);
for i=1:3
    for j=1:length(output(i,:))
        if abs(output(i,j)) < 0.02
            output(i,j) = 0;
        else
            hist_count(:,i) = hist_count(:,i) + 1;
        end
    end
end
figure();
bar(hist_count);
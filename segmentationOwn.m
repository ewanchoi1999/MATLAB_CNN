datasetDir = fullfile('data_for_moodle/');
imageDir = fullfile(datasetDir, 'images_256');
labelDir = fullfile(datasetDir, 'labels_256');

imSize = [256 256 3];
numClasses = 2;
classNames = ["flower", "background"];
labelIDs   = [1 3];

imds = imageDatastore(imageDir);
pxds = pixelLabelDatastore(labelDir,classNames,labelIDs);
trainingData = pixelLabelImageDatastore(imds,pxds);

% dataset splitting function
function [imdsTrain, imdsVal, imdsTest, pxdsTrain, pxdsVal, pxdsTest] = splitDataSet(imds, pxds);
rng(0);
numFiles = numpartitions(imds);
shuffledIndices = randperm(numFiles);
numTrain = round(0.60 * numFiles);
trainingIdx = shuffledIndices(1:numTrain);
numVal = round(0.20 * numFiles);
valIdx = shuffledIndices(numTrain+1:numTrain+numVal);
testIdx = shuffledIndices(numTrain+numVal+1:end);
imdsTrain = subset(imds,trainingIdx);
imdsVal = subset(imds,valIdx);
imdsTest = subset(imds,testIdx);
pxdsTrain = subset(pxds,trainingIdx);
pxdsVal = subset(pxds,valIdx);
pxdsTest = subset(pxds,testIdx);
end

%Augmenting
function data = augmentImageAndLabel(data, xTrans, yTrans)
for i = 1:size(data,1)

    tform = randomAffine2d(...
        XReflection=true,...
        XTranslation=xTrans, ...
        YTranslation=yTrans);

    rout = affineOutputView(size(data{i,1}), tform, BoundsStyle='centerOutput');

    data{i,1} = imwarp(data{i,1}, tform, OutputView=rout);
    data{i,2} = imwarp(data{i,2}, tform, OutputView=rout);

end
end

[imdsTrain, imdsVal, imdsTest, pxdsTrain, pxdsVal, pxdsTest] = splitDataSet(imds,pxds);
dsVal = combine(imdsVal,pxdsVal);
dsTrain = combine(imdsTrain,pxdsTrain);

xTrans = [-10 10];
yTrans = [-10 10];
dsTrain = transform(dsTrain, @(data)augmentImageAndLabel(data,xTrans,yTrans));

trainingData = pixelLabelImageDatastore(imds,pxds);
tbl = countEachLabel(pxds) 


numFilters = 64;
filterSize = 5;


layers = [
    imageInputLayer([256 256 3], 'Normalization', 'zerocenter', 'Name', 'input')

    % Encoder
    convolution2dLayer(filterSize, numFilters, 'Padding', 'same', 'Name', 'conv1')
    batchNormalizationLayer('Name', 'bn1')
    reluLayer('Name', 'relu1')
    maxPooling2dLayer(2, 'Stride', 2, 'Name', 'maxpool1')

    convolution2dLayer(filterSize, numFilters * 2, 'Padding', 'same', 'Name', 'conv2')
    batchNormalizationLayer('Name', 'bn2')
    reluLayer('Name', 'relu2')
    maxPooling2dLayer(2, 'Stride', 2, 'Name', 'maxpool2')

    convolution2dLayer(filterSize, numFilters * 4, 'Padding', 'same', 'Name', 'conv3')
    batchNormalizationLayer('Name', 'bn3')
    reluLayer('Name', 'relu3')
    maxPooling2dLayer(2, 'Stride', 2, 'Name', 'maxpool3')

    % Segmentation branch
    transposedConv2dLayer(4, numFilters * 2, 'Stride', 2, 'Cropping', 1, 'Name', 'deconv1_seg')
    convolution2dLayer(filterSize, numFilters * 2, 'Padding', 'same', 'Name', 'conv4_seg')
    batchNormalizationLayer('Name', 'bn4_seg')
    reluLayer('Name', 'relu4_seg')

    transposedConv2dLayer(4, numFilters * 4, 'Stride', 2, 'Cropping', 1, 'Name', 'deconv2_seg')
    convolution2dLayer(filterSize, numFilters, 'Padding', 'same', 'Name', 'conv5_seg')
    batchNormalizationLayer('Name', 'bn5_seg')
    reluLayer('Name', 'relu5_seg')

    transposedConv2dLayer(4, numClasses, 'Stride', 2, 'Cropping', 1, 'Name', 'deconv3_seg')
    sigmoidLayer('Name', 'sigmoid_edge')
    pixelClassificationLayer('Classes', classNames, 'Name', 'pixelclass_seg')

];



opts = trainingOptions('adam', ...
    LearnRateSchedule="piecewise",...
    LearnRateDropPeriod=6,...
    LearnRateDropFactor=0.1,...
    Shuffle = "every-epoch" ,...
    InitialLearnRate=1e-4, ...
    MaxEpochs=8, ... 
    MiniBatchSize =64 ,...
    Plots= "training-progress",...
    ValidationData=dsVal)

net = trainNetwork(dsTrain, layers, opts)

testImage = readimage(imdsTest, 5);
segmentationTest = semanticseg(testImage,net, Classes=classNames); 
segmentationTestOverlay = labeloverlay(testImage,segmentationTest);
imshow(segmentationTestOverlay);

pxdsResults = semanticseg(imdsTest,net,Classes=classNames,WriteLocation=tempdir);
metrics = evaluateSemanticSegmentation(pxdsResults, pxdsTest);
metrics.ClassMetrics;

save('segmentationownnet.mat', 'net');


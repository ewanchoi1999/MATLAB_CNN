%Loading of data and preprocessing
datasetDir = fullfile('data_for_moodle/');
imageDir = fullfile(datasetDir, 'images_256');
labelDir = fullfile(datasetDir, 'labels_256');
%numFiles = 847;

% Using deeplabv3+
imSize = [256 256 3];
numClasses = 2;
classNames = ["flower", "background"];
labelIDs   = [1 3];
net = deeplabv3plus(imSize, numClasses, "resnet18");

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

tbl = countEachLabel(pxds)

%Balancing of classes
totalNumberOfPixels = sum(tbl.PixelCount);
frequency = tbl.PixelCount / totalNumberOfPixels;
zeroFreqIndices = frequency == 0;
frequency(zeroFreqIndices) = 1e-6;
classWeights = 1./frequency

%Loss function
function loss = modelLoss(Y,T,classWeights)
weights = dlarray(classWeights,"C");
mask = ~isnan(T);
T(isnan(T)) = 0;
loss = crossentropy(Y,T,weights,Mask=mask,NormalizationFactor="mask-included");
end


%Training Options
opts = trainingOptions('sgdm', ...
    LearnRateSchedule="piecewise",...
    LearnRateDropPeriod=6,...
    LearnRateDropFactor=0.1,...
    Momentum=0.9,...
    L2Regularization=0.005,...
    Shuffle="every-epoch",...
    InitialLearnRate= 1e-3, ...
    ValidationData=dsVal,...
    GradientThreshold=1 , ...
    MaxEpochs= 10, ...
    MiniBatchSize=64, ...
    Plots= "training-progress")

net = trainnet(dsTrain, net, @(Y,T) modelLoss(Y,T,classWeights),opts);

testImage = readimage(imdsTest, 5);
segmentationTest = semanticseg(testImage,net, Classes=classNames);
segmentationTestOverlay = labeloverlay(testImage,segmentationTest);
imshow(segmentationTestOverlay);

pxdsResults = semanticseg(imdsTest,net,Classes=classNames,WriteLocation=tempdir);
metrics = evaluateSemanticSegmentation(pxdsResults, pxdsTest);
metrics.ClassMetrics;

save("segmentationexistnet.mat", "net");


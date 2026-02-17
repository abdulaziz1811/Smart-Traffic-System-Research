function newdetections = detectionAddNoise(detections)

global sceneInfo

ignoreRegion = load(sceneInfo.ignorePath);
igrMap = zeros(sceneInfo.imgHeight, sceneInfo.imgWidth);
if(~isempty(ignoreRegion))
    numIgnore = size(ignoreRegion,1);
    for j = 1:numIgnore
        igrMap(ignoreRegion(j,2):min(sceneInfo.imgHeight,ignoreRegion(j,2)+ignoreRegion(j,4)),ignoreRegion(j,1):min(sceneInfo.imgWidth,ignoreRegion(j,1)+ignoreRegion(j,3))) = 1;
%         rectangle('Position', ignoreRegion(j,:),'LineWidth',4,'edgecolor','y');       
    end
end

leftId = [];
addDetections = [];
tic;
for fr = sceneInfo.frameNums
    if(mod(fr,200)==0)
        fr
    end
    detMap = zeros(sceneInfo.imgHeight, sceneInfo.imgWidth);
    idxDet = find(detections(:,5) == fr);
    numDetections = numel(idxDet);
    meanWidth = mean(detections(idxDet,3) - detections(idxDet,1));
    meanHeight = mean(detections(idxDet,4) - detections(idxDet,2));
    
    for j = 1:numDetections
        detMap(detections(idxDet(j),2):min(sceneInfo.imgHeight,detections(idxDet(j),1)), detections(idxDet(j),4):min(sceneInfo.imgWidth,detections(idxDet(j),3))) = 1;
    end
    imgMap = and(detMap == 1, igrMap == 1);
    intMap = createIntImg(imgMap);
    
    %% add noises
    keepNum = numDetections - round(sceneInfo.FNratio*numDetections);
    addNum = round(sceneInfo.FPratio*numDetections);
    keepId = randperm(numDetections, keepNum);
    leftId = cat(2, leftId, keepId);
    addPos = [];
    for j = 1:5*addNum
        x = min(round(sceneInfo.imgWidth*0.95),max(1,round(sceneInfo.imgWidth*rand)));
        y = min(round(sceneInfo.imgHeight*0.95),max(1,round(sceneInfo.imgHeight*rand)));
        w = min(round(1.25*meanWidth), max(round(0.75*meanWidth), round(rand*meanWidth)));
        h = min(round(1.25*meanHeight), max(round(0.75*meanHeight), round(rand*meanHeight)));
        tl = intMap(y, x);
        tr = intMap(y, min(sceneInfo.imgWidth,x+w));
        bl = intMap(min(sceneInfo.imgHeight,y+h), x);
        br = intMap(min(sceneInfo.imgHeight,y+h), min(sceneInfo.imgWidth,x+w));
        foreValue = tl + br - tr - bl;
        if(nnz(foreValue)/w*h<0.4)
            addPos = cat(1, addPos, [x, y, w, h, fr]);
        end
    end
    if(size(addPos, 1) > addNum)
        idx = randperm(size(addPos, 1), addNum);
        addPos = addPos(idx, :);
    end
    addDetections = cat(1, addDetections, addPos);
end 
toc;
% drop detections
newdetections = detections(leftId, 1:5);
% add detections
newdetections = cat(1, newdetections, addDetections);
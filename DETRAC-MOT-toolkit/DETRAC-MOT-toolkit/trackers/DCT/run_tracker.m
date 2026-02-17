function [stateInfo, speed] = run_tracker(curSequence, baselinedetections)
%% Discrete-Continuous Optimization for Multi-Target Tracking
global dcStartTime

%% seed random for deterministic results
rand('seed',1); %#ok<RAND>
randn('seed',1); %#ok<RAND>

%% declare global variables
global detections nPoints sceneInfo opt globiter
globiter = 0;
global LOG_allens LOG_allmets2d LOG_allmets3d %for debug output

% fill options struct
opt = getDCOptionsDemo();
opt.labelCost=      300;
opt.outlierCost=    200;
opt.unaryFactor=    1;
opt.pairwiseFactor= 100;
opt.goodnessFactor= 10;
opt.proxcostFactor= 0.0;
opt.nInitModels=    500;
opt.minCPs=         1;
opt.ncpsPerFrame=   1/50;   
opt.totalSupFactor= 2;
opt.meanDPFFactor=  2;
opt.meanDPFeFactor= 2;
opt.curvatureFactor=0;
opt.tau =           25;     % threshold (pixel) for spatio-temporal neighbors
opt.borderMargin =  250;   % (pixel) % distance for persistence
opt.print = false;
opt.display = false;

%% multi-object tracking
% fill scene info
sceneInfo = getSceneInfoDCDemo(curSequence);
frames = curSequence.frameNums;

if(opt.visOptim)  
    reopenFig('optimization'); 
end

%% load detections
[detections, nPoints] = parseDetections(baselinedetections, frames);

%% top image limit
sceneInfo.imTopLimit = min([detections(:).yi]);

dcStartTime = tic;
T = size(detections,2);                   % length of sequence
stateInfo.F = T; 
stateInfo.frameNums = frames;

%% put all detections into a single vector
alldpoints = createAllDetPoints(detections);

%% create spatio-temporal neighborhood graph
TNeighbors = getTemporalNeighbors(alldpoints);

%% init solution
% generate initial spline trajectories
mhs = getSplineProposals(alldpoints,opt.nInitModels,T);

%% get splines from EKF
for ekfexp = 1:5
    mhsekf = getSplinesFromEKF(fullfile('demo','ekf',sprintf('e%04d.mat',ekfexp)),frames,alldpoints,T);
    mhs = [mhs mhsekf];
end
nCurModels = length(mhs);
nInitModels = nCurModels;

%% set initial labeling to all outliers
nCurModels = length(mhs);
nLabels = nCurModels+1;
outlierLabel = nLabels;
labeling = nLabels*ones(1,nPoints); % all labeled as outliers

%% initialize labelcost
[splineGoodness, goodnessComp] = getSplineGoodness(mhs,1:opt.nInitModels,alldpoints,T);

% unary is constant to outlierCost
Dcost = opt.outlierCost * ones(nLabels,nPoints);
Scost = opt.pairwiseFactor-opt.pairwiseFactor*eye(nLabels);
Lcost = getLabelCost(mhs);

[inE, inD, inS, inL] = getGCO_Energy(Dcost, Scost, Lcost, TNeighbors, labeling);
bestE=inE; E=inE; D=inD; S=inS; L=inL;
printDCUpdate(stateInfo,mhs,[],0,0,0,D,S,L);

%% first plot
drawDCUpdate(mhs,1:length(mhs),alldpoints,0,outlierLabel,TNeighbors,frames);

nAddRandomModels = 10; % random models
nAddModelsOutliers = 10;
nAdded=0; nRemoved=0;

%% start energy minimization loop
itcnt = 0; % only count one discrete-continuous cycle as one iteration
iteachcnt = 0; % count each discrete and each continuous optimization step
used = [];
mhsafterrefit = [];
while 1
    oldN = length(mhs);
    for m = 1:length(mhs)
        if(~isempty(intersect(m,used)))
            mhs(m).lastused = 0; 
        else
            mhs(m).lastused = mhs(m).lastused+1;
        end
    end

    mhs_ = mhs;
    tokeep = find([mhs.lastused]<3);
    mhs = mhs(tokeep);   

    nRemoved = oldN-length(tokeep);
    nCurModels = length(mhs); nLabels = nCurModels+1; outlierLabel = nLabels;

    % old labeling
    l_ = labeling;
    E_=E; D_=D; S_=S; L_=L;

  %% relabel
    % minimize discrete Energy E(f), (Eq. 4)
    Dcost = getUnarySpline(nLabels,nPoints,mhs,alldpoints,opt.outlierCost,opt.unaryFactor,T);
    Lcost = getLabelCost(mhs);
    Scost = opt.pairwiseFactor-opt.pairwiseFactor*eye(nLabels);
    [E, D, S, L, labeling]=doAlphaExpansion(Dcost, Scost, Lcost, TNeighbors);

    % if new energy worse (or same), restore previous labeling and done
    if E >= bestE
        printMessage(2, 'Discrete Optimization did not find a lower energy\n');
        labeling = l_;
        mhs = mhsafterrefit;
        E = E_; D = D_; S = S_; L = L_;
        nCurModels=length(mhs); nLabels=nCurModels+1; outlierLabel=nLabels;
        used=setdiff(unique(labeling),outlierLabel); nUsed=numel(used);
        break;
    end

    % otherwise refit and adjust models
    bestE = E;
    itcnt = itcnt+1;
    iteachcnt = iteachcnt+1;
    outlierLabel = nLabels;
    used = setdiff(unique(labeling),outlierLabel); nUsed = numel(used);

    % print update
    drawDCUpdate(mhs,used,alldpoints,labeling,outlierLabel,TNeighbors,frames);
    [m2d, m3d] = printDCUpdate(stateInfo,mhs,used,nAdded,nRemoved,iteachcnt,D,S,L);
    LOG_allens(iteachcnt,:)=double([D S L]);LOG_allmets2d(iteachcnt,:)=m2d;LOG_allmets3d(iteachcnt,:)=m3d;
    % now refit models (Eq. 1)
    mhsbeforerefit = mhs;
    mhsusedbeforerefit = mhs(used);
    mhsnew = reestimateSplines(alldpoints,used,labeling,nLabels,mhs,Dcost,T);
    mhsafterrefit = mhsnew;
    Dcost = getUnarySpline(nLabels,nPoints,mhsnew,alldpoints,opt.outlierCost,opt.unaryFactor,T);
    Lcost = getLabelCost(mhsnew);
    Scost = opt.pairwiseFactor-opt.pairwiseFactor*eye(nLabels);
    h = setupGCO(nPoints,nLabels,Dcost,Lcost,Scost,TNeighbors);
    GCO_SetLabeling(h,labeling);
    [E, D, S, L] = GCO_ComputeEnergy(h);    
    GCO_Delete(h);
    mhs(used) = mhsnew(used);
    nCurModels = length(mhs);
    clear Scost Dcost Lcost
    iteachcnt = iteachcnt+1;
    % print update
    drawDCUpdate(mhs,1:length(mhs),alldpoints,0,outlierLabel,TNeighbors,frames);
    drawDCUpdate(mhs,used,alldpoints,labeling,outlierLabel,TNeighbors,frames);
    printDCUpdate(stateInfo,mhs,used,nAdded,nRemoved,iteachcnt,D,S,L);
    LOG_allens(iteachcnt,:)=double([D S L]);LOG_allmets2d(iteachcnt,:)=m2d;LOG_allmets3d(iteachcnt,:)=m3d;
    %% Expand the hypothesis space
    if(nCurModels<opt.maxModels)
        nModelsBeforeAdded = nCurModels;
     %% get random new proposals
        mhsnew = getSplineProposals(alldpoints,nAddRandomModels,T);
        mhs = [mhs mhsnew];
     %% get new proposals from outliers
        outlierPoints = find(labeling==outlierLabel); % indexes
        if length(outlierPoints)>4
            outlpts = selectPointsSubset(alldpoints,outlierPoints);
            mhsnew = getSplineProposals(outlpts,nAddRandomModels,T);
            mhs = [mhs mhsnew];
        end
     %% extend existing
        mhs = extendSplines(alldpoints,mhs,used,labeling,T,E);
     %% merge existing
        mhs = mergeSplines(alldpoints,mhs,used,labeling,T,E);
    end
    nCurModels = length(mhs); nLabels = nCurModels+1; outlierLabel = nLabels;
    nAdded = nCurModels-length(mhsbeforerefit);
end

% basically we are done
speed = stateInfo.F/toc(dcStartTime);

%% final plot
drawDCUpdate(mhs,used,alldpoints,labeling,outlierLabel,TNeighbors,frames);
stateInfo = getStateFromSplines(mhs(used), stateInfo);
stateInfo = postProcessState(stateInfo);
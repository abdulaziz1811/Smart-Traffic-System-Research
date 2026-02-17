function sceneInfo = getSceneInfoDCDemo(curSequence)
% fill all necessary information about the
% scene into the sceneInfo struct
%
% Required:
%   detfile         detections file (.idl or .xml)
%   frameNums       frame numbers (eg. frameNums=1:107)
%   imgFolder       image folder
%   imgFileFormat   format for images (eg. frame_%04d.jpg)
%   targetSize      approx. size of targets (default: 5 on image, 350 in 3d)
%
% Required for 3D Tracking only
%   trackingArea    tracking area
%   camFile         camera calibration file (.xml PETS format)
%
% Optional:
%   gtFile          file with ground truth bounding boxes (.xml CVML)
%   initSolFile     initial solution (.xml or .mat)
%   targetAR        aspect ratio of targets on image
%   bgMask          mask to bleach out the background
% 
% (C) Anton Andriyenko, 2012
%
% The code may be used free of charge for non-commercial and
% educational purposes, the only requirement is that this text is
% preserved within the derivative work. For any other purpose you
% must contact the authors for permission. This code may not be
% redistributed without written permission from the authors.

sceneInfo.imgFolder = curSequence.imgFolder;
sceneInfo.frameNums = curSequence.frameNums;
sceneInfo.imgFileFormat = curSequence.imgFileFormat;

% image dimensions
sceneInfo.imgHeight = curSequence.imgHeight;
sceneInfo.imgWidth = curSequence.imgWidth;

%% tracking area
% if we are tracking on the ground plane
% we need to explicitly secify the tracking area
% otherwise image = tracking area
sceneInfo.trackingArea=[1 sceneInfo.imgWidth 1 sceneInfo.imgHeight];   % tracking area

%% camera
cameraconffile = [];
sceneInfo.camFile = cameraconffile;

%% target size
sceneInfo.targetSize=sceneInfo.imgWidth/30;

%% ground truth
sceneInfo.gtAvailable=0;
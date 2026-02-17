function detectionEvaluation()

global options

detectorSet = options.detectorSet; % the detector set

% Debug Logs
curPath = pwd;
log_path = [curPath '/logs/'];
createPath(log_path); 

cd('./evaluation');
addpath(genpath('.'));
for idDet = 1:length(detectorSet) 
    detectorName = detectorSet{idDet};
    diary([log_path 'DETRAC_' detectorName '_detection_logs.txt']);    
    % Check formats
    detPath = [options.detPath detectorName '/'];
    [errorMsg, realList] = checkResultFormat(detPath, detectorName);
    if(~isempty(errorMsg))
        error(errorMsg);
    end
    % pre-processing the detection results
    preProcessDetectionResults(detectorName);
    % obtain detection PR scores
    if(options.printDetectionEval)
        detprFile = [options.detPath detectorName '/' detectorName '_detection_PR.txt'];
        detprLine = ['AP_DET_EVAL.exe ' detectorName ' ' detPath ' ' curPath '/sequences.txt ' detPath];
        evalexe(detprLine);
        if(exist(detprFile, 'file'))                
            APscore = showDetectionPRCurve(detprFile, detectorName);
            disp(['The AP score of the detector ' detectorName ' is ' num2str(roundn(APscore,-2)) '%.']);
        else
            error('no detection results!');
        end
    end
end
cd('../');

%% zip the detection results for DETRAC-Test
zipDetectionResults(detectorSet, realList);
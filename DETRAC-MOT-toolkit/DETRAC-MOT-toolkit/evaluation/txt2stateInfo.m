function stateInfo = txt2stateInfo(resultSavePath, frameNums)

stateInfo.F = numel(frameNums);
stateInfo.frameNums = frameNums;

left = load([resultSavePath '_LX.txt']);
top = load([resultSavePath '_LY.txt']);
w = load([resultSavePath '_W.txt']);
h = load([resultSavePath '_H.txt']);
xc = left + w/2;
yc = top + h/2;

% foot position
stateInfo.X = xc;      
stateInfo.Xi = xc;
stateInfo.Y = yc+h/2;
stateInfo.Yi = yc+h/2;
stateInfo.H = h;
stateInfo.W = w;
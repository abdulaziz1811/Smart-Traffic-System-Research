function metrics = printSeqEvaluation(seqName, gtInfo, stateInfo, folder)

global options

metrics = [];

if(options.printEvaluationForEachSeq)
    [metrics, metricsInfo] = CLEAR_MOT(gtInfo, stateInfo);    
    disp(['2D Evaluation of Sequence ' seqName ' (Detection Score Threshold=' folder '):']);
    printMetrics(metrics, metricsInfo, 1);
end
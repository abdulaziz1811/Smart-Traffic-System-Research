The DETRAC-MOT evaluation toolkit
======================

This is the official detection/tracking evaluation kit for the [DETection and tRACking (DETRAC) Benchmark](http://detrac-db.rit.albany.edu/), 
which contains the actual evaluation toolkit as a set of Matlab (Octave compatible) scripts, a readme documentation and several state-of-the-art tracking algorithms. 
You can find more informations from our paper: [arXiv](http://arxiv.org/abs/1511.04136). 

In addition, if you use the DETRAC benchmark or the evaluation toolkit, please cite the paper:
@article{DETRAC:CoRR:WenDCLCQLYL15,
  author    = {Longyin Wen and Dawei Du and Zhaowei Cai and Zhen Lei and Ming{-}Ching Chang and
               Honggang Qi and Jongwoo Lim and Ming{-}Hsuan Yang and Siwei Lyu},
  title     = { {DETRAC:} {A} New Benchmark and Protocol for Multi-Object Detection and Tracking},
  journal   = {arXiv CoRR},
  volume    = {abs/1511.04136},
  year      = {2015}
}     

Enquiries, Questions and Comments
--------------------------------

If you have any further enquiries, questions, or comments, please find the information on the website: [DETRAC homepage](http://detrac-db.rit.albany.edu/), 
or contact us: ua.detrack@gmail.com or lywen.cv.workbox@gmail.com. Besides, the QQ Group (387090533) is available to facilitate discussion. 
If you find any bug in the toolkit, please inform us.

You can also subscribe to the DETRAC [mailing list](http://detrac-db.rit.albany.edu/auth/register/) to receive news about the benchmark and important software updates.

Platform Support
----------------

The toolkit is written in Matlab with executable binaries as well as support for multiple versions of Matlab (at the moment it was tested with versions from 2013 to 2015).

The code should be run on Windows system. However, it is hard to verify this on all versions and system configurations. 
If you find there exists some problems to run it on your computer, please contact us as soon as possible.

Tracker Modules
-------

The tracker module contains several tracking related functions. For the proposed tracker, you should create a main function like below:

[stateInfo, speed] = run_tracker(curSequence, baselinedetections);

**Input**: There are two input variables available for the tracker:

* 'curSequence' - A cell containing the sequence information.

* 'baselinedetections' - A matrix containing comma-separated values that denote the detection bounding box and the detection score of the object.

**Output**: the results file will be produced after execution.

* 'stateInfo' - A struct containing the trajectory positions of the tacker (i.e., LX, LY, W, H).

* 'speed' - A variable indicating the running speed of the tracker. This is an additional measure.

If you want to submit the tracking results for comparison, you should put the code of the tracker into the path 'trackers/' of the toolkit at first. 

How to run your code
-------

1. Open the 'initialize_enviroment.m' function, and set the parameters as follows:
 a) make sure the images, detections and annotations are in the right path. 
 b) select the evaluation type,  i.e.,  Detection and Tracking.
 c) select the sequences for evaluation,  i.e.,  DETRAC-Train, DETRAC-Test, and DETRAC-Free.
 d) if you chose 'DETRAC-Free', edit the 'sequences.txt' file to select the evaluated sequences you want. 
2. Open the 'DETRAC_experiment.m' function, and input the name of the tracker if you chose 'Tracking'.
3. Run the 'DETRAC_experiment.m' function for evaluation.

More details are presented on the website http://detrac-db.rit.albany.edu/instructions. 

To ease your first encounter with the code, we provide several source codes of state-of-the-art trackers in the toolkit, including CEM, CMOT, DCT, FH2T, GOG, H2T, IHTILS and RMOT.
Specifically, the FH2T and RMOT methods are not included in the arXiv paper due to the time issue. 
If you have any further questions about the trackers, please contact to the original authors of the tracker.
# Title
Simple CNN Models in Terms of Size and Computational Complexity for Classification On Medical Images


## Context
Millions of imaging procedures are being done worldwide every week for diagnostic purposes which needs expertise knowledge for interpretation. But according to the world health organization(WHO), there is a shortage of 4.3 million physicians, globally.

Deep learning has the potential to use efficient methods to do the diagnosis. An efficient use of this information for diagnostic studies can reduce patients need to expose to radiation and invasion procedures thereby saving time and money

Over the past few years, there have been few breakthroughs achieved in medical imaging using Deep Learning. 

But medical datasets are fundamentally different from natural image datasets in many aspects. For instance, medical datasets are usually quite small consisting of few dozens to thousands of observations compared to the 1.2 million observations used on ImageNet. Most problems consist of predicting a few classes (typical 2 to 5) and not 1000 classes like ImageNet. Here is a question we are trying to answer.

Is it necessary to use over parameterized architectures that are harder to train for simpler problems? 


## Techniques
We start with two base architectures and we down-scale the models by depth, width and by replacing different kinds onc convolutions like grouped convolutions and depthwise-separable convolutions in order to achieve similar or better performance while saving on training time and memory.


## Datasets:
* MURA: It consists of 14,863 studies from 12,173 patients, with a total of 40,561 multi-view radiographic images. Each belongs to one of seven standard upper extremity radiographic study types: elbow, finger, forearm, hand, humerus, shoulder, and wrist. Each study is labeled as either 1 or 0 which represents normal and abnormal respectively.
  
* RSNA Brain Hemorrhage: It consists of 674K Brain CT scan images. The images can be labelled with 6 kinds of classes, 5 being different Hemorrhage types and 1 being normal. To make this problem similar to that of typical medical image dataset, total number of images are reduced to 30k for training and 30k for testing. Each study is labelled as 1 or 0 which represents the presence of a Hemorrhage or normal respectively 
  
* Chexpert:It consists of 224,316 chest radio-graphs from 65,240 patients labeled for 14 diseases as negative, positive or uncertain. From those classes, only 5 of them were analyzed: Atelectasis, Cardiomegaly, Consolidation, Edema and Pleural Effusion. The training set is down-sampled to 40k images with 5 labels to each observation. 




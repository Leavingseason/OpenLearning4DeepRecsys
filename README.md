# OpenLearning4DeepRecsys
We have implemented some popular and promising recommendation systems with deep learning techniques. We would like to open souce the code and hope it can help more people on the related topics, and at the same time improve our code quality.


Usage: Simply download the corresponding file, modify some lines according to your own configuration, then run "python xxxx.py". Currently we aim to provide the opportunity for communication in research area. Later we plan to build an integrated tool for off-the-shelf usage. So kindly let me know if you have any suggestions.

### DeepFM
https://arxiv.org/abs/1703.04247 DeepFM: A Factorization-Machine based Neural Network for CTR Prediction. We implement the model according to the paper. Some results (AUC) on the demo dataset: 
* Linear only: 0.669 
* FM only: 0.684   
* DNN only: 0.671  
* DeepFM: 0.692 .

Notice:  (1) Input format is the same as svmlight, feature index starts with 1. 
         (2) You have to provide the "field number" (a.k.a field_cnt in the source code) for the input feature file. All instances have exactly field_cnt fields.  Each field can be numerical type or categorical (one-hot) type.

### CCFNet
http://dl.acm.org/citation.cfm?id=3054207  Part of the paper "CCCFNet: A Content-Boosted Collaborative Filtering Neural Network for Cross Domain Recommender Systems".  The original code was written in c#. We re-implement the model in tensorflow for unification. The demo data is from MovieLens.

### Biased Matrix Factorization
BMF model in https://datajobs.com/data-science-repo/Recommender-Systems-%5BNetflix%5D.pdf . Demo data is Douban user-book sample dataset.

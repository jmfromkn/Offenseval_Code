# Offenseval_Code
 Code & Data for UNTLing @ SemEval 2020 Task 12: Offenseval 2
 
 Code will be cleaned and refactored to make it more efficient in future.
 
 Version information for used libaries:
 Python 3.6.0
 Spacy 2.2.3
 Scikit-Learn 0.22.1
 Pandas 0.25.3
 Visualizations in paper utilized matplotlib 3.1.2
 
 spaCy was used in this code to map and average vectors to the feature dataframe. This will require you to build basic models with spacy    using the word embeddings to use the nlp pipeline (see https://spacy.io/usage/vectors-similarity#converting)
 
 The Data Folder will include the hand-annotated development sets from the paper for Task A and Task B.  Full Training sets will not be included on this repository.

*Please modify path arguments in the files to match your paths
 If you simply want to run the full feature model and generate predictions: use TrainandPredict.py and TrainAndPredictTaskB.py as these will only require the filepaths of the training and test documents and will run the full process of training the model to predictions and output.
 
 For the feature ablations and threshold testing: First run ExportFeatures.py and ExportFeaturesTaskB.py to generate csv's of your train and test features to be stored.  From there, ensure that you set up the paths correctly in FeatureAblation.py and Thresholdtesting.py.  FeatureAblation.py will output a csv with predictions for each feature set (via the paper).  Thresholdtesting.py will run a simulation test on the development set in 0.01 threshold value increments to find the ideal threshold to discretize the confidence values to labels and output a csv with a corresponding threshold value and its F1 score on the development set.

# Naive-Bayes-Classifier
The class for naive Bayes classifier is implemented from the scratch in the file "naive_bayes.py".  

## Dependency
- math
- statistics  

## Benchmark
The naive Bayes classifier is applied to one example which predicts the probability of having heart disease given several nomial and numeric features. The features and their possible values are summarized in "heart_name.txt". The training data for the classifier are given in "heart_training_data.txt". The testing data are giben in "heart_testing_data.txt".   

The predictive and actual results for the first 5 testing examples from the naive Bayes classifier are summarized in the following table:

| Index of testing example | *P*(One has heart disease) |     Prediction   |   Actual result  |
| ------------------------ | -------------------------- | ---------------- | ---------------- |
|            1             |          0.001984          | No heart disease | No heart disease |
|            2             |          0.999579          |   Heart disease  |   Heart disease  |
|            3             |          0.000990          | No heart disease | No heart disease |
|            4             |          0.993271          |   Heart disease  |   Heart disease  |
|            5             |          0.772412          |   Heart disease  |   Heart disease  |

**To run naive Bayes classifier**
```
Python main_script.py
```

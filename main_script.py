import naive_bayes


#######################################Testing implementation using 'heart_training_data.txt'###################
#apply gaussian pdf to numeric features
naivebayes_gaussian = naive_bayes.naive_bayes()
with open('heart_name.txt', 'r') as file:
    classes = file.readline()
    naivebayes_gaussian.classes = [x.strip() for x in classes.split(',')]
    # add attributes
    for line in file:
        [attribute, values] = [x.strip() for x in line.split(':')]
        values = [x.strip() for x in values.split(',')]
        naivebayes_gaussian.attrValues[attribute] = values
naivebayes_gaussian.numAttributes = len(naivebayes_gaussian.attrValues.keys())
naivebayes_gaussian.attributes = list(naivebayes_gaussian.attrValues.keys())
with open('heart_training_data.txt', 'r') as file:
    for line in file:
        row = [x.strip() for x in line.split(' ')]
        if row != [] or row != [","]:
            naivebayes_gaussian.training_data.append(row)
with open('heart_testing_data.txt', 'r') as file:
    for line in file:
        row = [x.strip() for x in line.split(' ')]
        if row != [] or row != [","]:
            naivebayes_gaussian.testing_data.append(row)
#preprocess data for numeric features
naivebayes_gaussian.preprocessData()
#compute posterior probability
naivebayes_gaussian.predictProbility()
#print rsults
naivebayes_gaussian.printPrediction()

#######################################Testing implementation using 'heart_training_data.txt'###################
#apply discretization to numeric features
naivebayes_discretize = naive_bayes.naive_bayes()
naivebayes_discretize.continuousProcessing = 'discretize'
naivebayes_discretize.numBins = 500
with open('heart_name.txt', 'r') as file:
    classes = file.readline()
    naivebayes_discretize.classes = [x.strip() for x in classes.split(',')]
    # add attributes
    for line in file:
        [attribute, values] = [x.strip() for x in line.split(':')]
        values = [x.strip() for x in values.split(',')]
        naivebayes_discretize.attrValues[attribute] = values
naivebayes_discretize.numAttributes = len(naivebayes_discretize.attrValues.keys())
naivebayes_discretize.attributes = list(naivebayes_discretize.attrValues.keys())
with open('heart_training_data.txt', 'r') as file:
    for line in file:
        row = [x.strip() for x in line.split(' ')]
        if row != [] or row != [","]:
            naivebayes_discretize.training_data.append(row)
with open('heart_testing_data.txt', 'r') as file:
    for line in file:
        row = [x.strip() for x in line.split(' ')]
        if row != [] or row != [","]:
            naivebayes_discretize.testing_data.append(row)
#preprocess data for numeric features
naivebayes_discretize.preprocessData()
#compute posterior probability
naivebayes_discretize.predictProbility()
#print rsults
naivebayes_discretize.printPrediction()
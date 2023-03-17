import math
import statistics


class naive_bayes:
    #implement navie bayes classifier
    def __init__(self):
        #initializa
        self.training_data = []
        self.testing_data = []
        self.classes = []
        self.numAttributes = -1
        self.attrValues = {}
        self.attributes = []
        self.target_name = 'class'
        self.prediction = []
        #method specified to process numeric feature, can be 'gaussian' or 'discretize'
        self.continuousProcessing = 'gaussian'
        self.numBins = 1000                 #number of bins used if discretizing numeric feature

    def preprocessData(self):
        #convert numeric freature values to floating number
        for index, row in enumerate(self.training_data):
            for attr_index in range(self.numAttributes):
                if(not self.isAttrDiscrete(self.attributes[attr_index])):
                    self.training_data[index][attr_index] = float(self.training_data[index][attr_index])
        for index, row in enumerate(self.testing_data):
            for attr_index in range(self.numAttributes):
                if(not self.isAttrDiscrete(self.attributes[attr_index])):
                    self.testing_data[index][attr_index] = float(self.testing_data[index][attr_index])

    def isAttrDiscrete(self, attribute):
        #asses if the feature is discrete or numeirc
        if attribute not in self.attributes:
            raise ValueError('Attribute not listed')
        elif len(self.attrValues[attribute]) == 1 and self.attrValues[attribute][0] == 'continuous':
            return False
        else:
            return True

    def predictProbility(self):
        #compute posterior probability for each of testing data
        self.prediction = [self.posteriorProb(testing_row) for testing_row in self.testing_data]

    def posteriorProb(self, testing_row):
        #compute posterior probability for single testing example
        S = len(self.training_data)
        p_of_y = [0 for i in self.classes]
        for row in self.training_data:
            classIndex = list(self.classes).index(row[-1])
            p_of_y[classIndex] += 1
        p_of_y = [x/S for x in p_of_y]
        likelihood = p_of_y
        for i in range(len(self.classes)):
            for j in range(self.numAttributes):
                likelihood[i] *= self.condiProb(testing_row[j], self.attributes[j], self.classes[i])
        p_of_disease = likelihood[-1]/math.fsum(likelihood)
        return p_of_disease

    def condiProb(self, curAttributevalue, curAttributes, curClasses):
    #compute conditional probability for numeric features
        indexOfAttribute = self.attributes.index(curAttributes)
        curData = []
        if self.isAttrDiscrete(curAttributes):
            count_x = 0
            for row in self.training_data:
                if row[-1] == curClasses:
                    curData.append(row)
                    if row[indexOfAttribute] == curAttributevalue:
                        count_x += 1
            cond_prob = (count_x+1)/(len(curData)+len(self.attrValues[curAttributes]))
        elif self.continuousProcessing == 'gaussian':
            #use gaussian pdf to compute conditional probability
            curAttributevalue = float(curAttributevalue)
            for row in self.training_data:
                if row[-1] == curClasses:
                    curData.append(row)
            samples = [row[indexOfAttribute] for row in curData]
            mu = statistics.mean(samples)
            sig = statistics.stdev(samples)
            cond_prob = 1/math.sqrt(2*math.pi*sig**2)*math.exp(-(curAttributevalue-mu)**2/(2*sig**2))
        else:
            #discretize numeric feature with equal width bins and compute conditionsl probability
            curattribute = [row[indexOfAttribute] for row in self.training_data]
            lower_bound = math.floor(min(curattribute))
            upper_bound = math.ceil(max(curattribute))
            h = (upper_bound-lower_bound)/self.numBins
            indexofBins = math.ceil((curAttributevalue-lower_bound)/h)
            count_x = 0
            for row in self.training_data:
                if row[-1] == curClasses:
                    curData.append(row)
                    if row[indexOfAttribute]>=lower_bound+(indexofBins-1)*h and row[indexOfAttribute]<lower_bound+indexofBins*h:
                        count_x += 1
            cond_prob = (count_x + 1) / (len(curData) + self.numBins)
        return cond_prob

    def printPrediction(self):
        index = 0
        for prob in self.prediction:
            #print('%f\n'%(prob))
            index = index+1
            if prob>0.5:
                print('No.%d testing case: P(Y=heart disease)=%f' % (index, prob)+'     '+'Prediction: Heart disease\n')
            else:
                print('No.%d testing case: P(Y=heart disease)=%f' % (index, prob) + '     ' + 'Prediction: No heart disease\n')


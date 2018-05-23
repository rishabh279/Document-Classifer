# -*- coding: utf-8 -*-
"""
Created on Tue May 22 11:37:07 2018

@author: rishabh
"""

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer 
from sklearn.decomposition import TruncatedSVD
from sklearn.ensemble import VotingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
import numpy as np

def removeStopwords(w): 
    stopwords = ['all', 'just', 'being', 'over', 'both', 'through', 'yourselves', 'its', 'before', 'herself', 'had', 'should', 'to', 'only', 'under', 'ours', 'has', 'do', 'them', 'his', 'very', 'they', 'not', 'during', 'now', 'him', 'nor', 'did', 'this', 'she', 'each', 'further', 'where', 'few', 'because', 'doing', 'some', 'are', 'our', 'ourselves', 'out', 'what', 'for', 'while', 'does', 'above', 'between', 't', 'be', 'we', 'who', 'were', 'here', 'hers', 'by', 'on', 'about', 'of', 'against', 's', 'or', 'own', 'into', 'yourself', 'down', 'your', 'from', 'her', 'their', 'there', 'been', 'whom', 'too', 'themselves', 'was', 'until', 'more', 'himself', 'that', 'but', 'don', 'with', 'than', 'those', 'he', 'me', 'myself', 'these', 'up', 'will', 'below', 'can', 'theirs', 'my', 'and', 'then', 'is', 'am', 'it', 'an', 'as', 'itself', 'at', 'have', 'in', 'any', 'if', 'again', 'no', 'when', 'same', 'how', 'other', 'which', 'you', 'after', 'most', 'such', 'why', 'a', 'off', 'i', 'yours', 'so', 'the', 'having', 'once']
    return w in stopwords

file = open("trainingdata.txt", "rb")
raw_data = file.read().decode("latin1")
file.close()

documents = raw_data.split("\n")
documents = documents[1: ]
train = []
labels = []

for d in documents:
    d = d.split()
    if len(d)!=0:
        label = d[0]
        labels.append(int(label))
        #remove the stopwords
        train.append(' '.join([i for i in d[1:] if not removeStopwords(i)]))
    
def working(train,labels,testData):
    vector = CountVectorizer(input='content',ngram_range=(1,2))
    XTrainCounts = vector.fit_transform(train,)
    tf_transformer = TfidfTransformer(use_idf=True,).fit(XTrainCounts)
    XTraintf = tf_transformer.transform(XTrainCounts)
    
    svd = TruncatedSVD(n_components=50, random_state=9)
    xTrain = svd.fit_transform(XTraintf)
    
    classifier = KNeighborsClassifier(n_neighbors=8).fit(xTrain, labels)
    
    classifier2 = RandomForestClassifier().fit(xTrain, labels)
    
    classifier3 = VotingClassifier(estimators=[('kn', classifier), ('rf', classifier2)], voting='soft')# 91.1 point
    classifier3.fit(xTrain, labels)
    
    xTestCounts = vector.transform(testData)
    xTesttfidf = tf_transformer.transform(xTestCounts)
    xTest = svd.transform(xTesttfidf)
    return classifier3.predict(xTest)

def accuracy():
    trainingSet = train[ :4485]
    testingSet = train[4486: ]
    labelTrain=labels[:4485]
    labelTest=labels[4486: ] 
    predictions=working(trainingSet,labelTrain,testingSet)
    acc= np.mean(predictions==labelTest)
    print("Accuray is",acc*100,"%")   

    
#input from the user    
n = int(input())
a = []
for a_i in range(n):
    a_t = input()
    a.append(' '.join([i for i in a_t.split() if not removeStopwords(i)]))
        
predicted_labels = working(train,labels,a)
for l in predicted_labels:
    print (l)

#To check accuracy    
#accuracy()
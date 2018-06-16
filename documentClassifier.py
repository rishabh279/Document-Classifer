# -*- coding: utf-8 -*-
"""
Created on Tue May 22 11:37:07 2018

@author: rishabh
"""

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.ensemble import VotingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
import numpy as np

documents=[]
labels=[]   
train=[] 
with open("E:/RS/Applying for Machine Learning Jobs/Loktra/Document-Classification-master/trainingdata.txt", "r") as file:
    next(file)
    content=file.readlines()
    
for line in content:
    documents.append(line.strip('\n'))
        
for d in documents:
    if len(d)!=0:
        label=int(d[0])
        labels.append(label)
        train.append(d[1:])
        
def removeStopwords(w): 
    stopwords = ['all', 'just', 'being', 'over', 'both', 'through', 'yourselves', 'its', 'before', 'herself', 'had', 'should', 'to', 'only', 'under', 'ours', 'has', 'do', 'them', 'his', 'very', 'they', 'not', 'during', 'now', 'him', 'nor', 'did', 'this', 'she', 'each', 'further', 'where', 'few', 'because', 'doing', 'some', 'are', 'our', 'ourselves', 'out', 'what', 'for', 'while', 'does', 'above', 'between', 't', 'be', 'we', 'who', 'were', 'here', 'hers', 'by', 'on', 'about', 'of', 'against', 's', 'or', 'own', 'into', 'yourself', 'down', 'your', 'from', 'her', 'their', 'there', 'been', 'whom', 'too', 'themselves', 'was', 'until', 'more', 'himself', 'that', 'but', 'don', 'with', 'than', 'those', 'he', 'me', 'myself', 'these', 'up', 'will', 'below', 'can', 'theirs', 'my', 'and', 'then', 'is', 'am', 'it', 'an', 'as', 'itself', 'at', 'have', 'in', 'any', 'if', 'again', 'no', 'when', 'same', 'how', 'other', 'which', 'you', 'after', 'most', 'such', 'why', 'a', 'off', 'i', 'yours', 'so', 'the', 'having', 'once']
    return w in stopwords
        
def working(train,labels,testData):
    vector=TfidfVectorizer(input='content',stop_words='english',ngram_range=(1,2))
    XTraintf=vector.fit_transform(train)

    svd = TruncatedSVD(n_components=50, random_state=9)
    xTrain = svd.fit_transform(XTraintf)
    
    classifier = KNeighborsClassifier(n_neighbors=8).fit(xTrain, labels)
    
    classifier2 = RandomForestClassifier().fit(xTrain, labels)
    
    classifier3 = VotingClassifier(estimators=[('kn', classifier), ('rf', classifier2)], voting='soft')# 91.1 point
    classifier3.fit(xTrain, labels)
    
    XTesttf=vector.transform(testData)
    xTest=svd.transform(XTesttf)
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
Question 1)
The document classifier question is able to achieve an accuracy of 92.54%
Step 1:The documents are first preprocessed using the feature extraction techniques such as CountVectorizer,TfidfTransformer
Step 2:Then due to sparse matrix obtained from feature extraction, SVD is applied to reduce the dimensionality.
Step 3:Classifier such as KNeighborsClassifier, RandomForestClassifier are used to classify the documents according to labels. 
Step 4:The ensemble algorithm such as Voting Classifier is applied in order to obtain best results.
Step 5:Accuracy function is defined in order to obtain the acuracy

References:-
1)https://machinelearningmastery.com/prepare-text-data-machine-learning-scikit-learn/
2)http://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.CountVectorizer.html
3)https://www.quora.com/How-is-singular-value-decomposition-used-in-nlp
4)http://scikit-learn.org/stable/modules/generated/sklearn.decomposition.TruncatedSVD.html
5)http://scikit-learn.org/stable/modules/generated/sklearn.ensemble.VotingClassifier.html++

Question:
The answer obtained for question 2 is 8.5

References:-
1)https://stats.stackexchange.com/questions/22718/what-is-the-difference-between-linear-regression-on-y-with-x-and-x-with-y)
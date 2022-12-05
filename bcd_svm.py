from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score
from preprocess_data import optimize_preprocess
import numpy as np
import matplotlib.pyplot as plt

def main():
      train_data, train_label, test_data, test_label, init_centroids = optimize_preprocess()
      clf=SVC(kernel='rbf',C=1.0,random_state=1).fit(train_data,train_label)
      pred=clf.predict(test_data)
      score=accuracy_score(test_label,pred)
      CVscore=cross_val_score(clf,train_data,train_label,cv=4)
      mean=CVscore.mean()
      std=CVscore.std()
      print('\nSVM with RBF Kernel')
      print('Accuracy:',score)
      print('Cross Validation:',CVscore)
      print('Cross Validation Mean:',mean)
      print('Cross Validation Std:',std)

if __name__ == '__main__':
      main()
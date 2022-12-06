from sklearn.svm import SVC, LinearSVC
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score
from preprocess_data import optimize_preprocess
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

def getFigure(clf,data,y,C=0.8):   #创建网格，以绘制图表
      grid_length=0.02             #网格中的步长
      minX,maxX=float(min(data[:,0]))-1,float(max(data[:,0]))+1  
      minY,maxY=float(min(data[:,1]))-1,float(max(data[:,1]))+1  
      x_num,y_num=np.meshgrid(np.arange(minX,maxX,grid_length),np.arange(minY,maxY,grid_length))
      Z=clf.predict(np.c_[x_num.ravel(),y_num.ravel()])
      Z=Z.reshape(x_num.shape) 
      plt.contourf(x_num,y_num,Z,cmap=plt.cm.get_cmap('RdYlBu'),alpha=0.95)  #等高线函数：使用不同颜色划分区域
      plt.scatter(data[:,0],data[:,1],c=y,cmap=plt.cm.get_cmap('RdYlBu'))      #以离散点的形式绘制训练数据
      plt.xlabel("X")
      plt.ylabel("Y")
      plt.xlim(minX,maxX)
      plt.ylim(minY,maxY)
      plt.title('BCCD Test Dataset using SVM-RBF')

      # Creating legend with color box
      WBC = mpatches.Patch(color='r', label='WBC')
      RBC = mpatches.Patch(color='y', label='RBC')
      Platelets = mpatches.Patch(color='b', label='Platelets')
      plt.legend(handles=[WBC,RBC,Platelets])

      plt.show()

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

      #getFigure(clf,test_data[:200],test_label[:200])

if __name__ == '__main__':
      main()
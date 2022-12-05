from preprocess_data import optimize_preprocess
import numpy as np
import random
import matplotlib.pyplot as plt

def Euclidean_dist(point1,point2):
      return np.sqrt(np.square(point2[0]-point1[0])+np.square(point2[1]-point1[1]))

def update_centroid(classes):
      x,y=0,0
      for points in classes:
            x += points[0]
            y += points[1]
      x = x / len(classes)
      y = y / len(classes)
      new_centroid = [x,y]
      return new_centroid

def round_centroids(centroids_ls):
      return [[round(c[0],2),round(c[1],2)] for c in centroids_ls]

def kmeans(data,label,centroids):
      class1,class2,class3 = [],[],[]
      accurate_count1,accurate_count2,accurate_count3 = 0,0,0
      len1,len2,len3 = 0,0,0
      for i in range(len(data)):
            x = data[i][0]
            y = data[i][1]
            point = [x,y]

            if label[i] == 1:
                  len1 += 1
            elif label[i] == 2:
                  len2 += 1
            elif label[i] == 3:
                  len3 += 1

            dist1 = Euclidean_dist(centroids[0],point)
            dist2 = Euclidean_dist(centroids[1],point)
            dist3 = Euclidean_dist(centroids[2],point)
            min_dist = min(dist1,dist2,dist3)

            if min_dist == dist1:
                  class1.append(point)
                  centroids[0] = update_centroid(class1)
                  if label[i] == 1:
                        accurate_count1 += 1
            elif min_dist == dist2:
                  class2.append(point)
                  centroids[1] = update_centroid(class2)
                  if label[i] == 2:
                        accurate_count2 += 1
            elif min_dist == dist3:
                  class3.append(point)
                  centroids[2] = update_centroid(class3)
                  if label[i] == 3:
                        accurate_count3 += 1

      classes = [class1,class2,class3]
      sse = calculate_SSE(classes,centroids)
      accurate_count = accurate_count1 + accurate_count2 + accurate_count3
      total_accuracy = accurate_count / len(data)
      accuracy1 = accurate_count1 / len1
      accuracy2 = accurate_count2 / len2
      accuracy3 = accurate_count1 / len3
      accuracies = [accuracy1,accuracy2,accuracy3]
      centroids = round_centroids([centroids[0],centroids[1],centroids[2]])
      return total_accuracy,centroids,accuracies,sse,classes

def calculate_SSE(classes,centroids):
      sse = 0
      index = -1
      for cl in classes:
            index += 1
            for points in cl:
                  sse += np.square(Euclidean_dist(points,centroids[index]))
      return np.sqrt(sse)

def visualize(classes,centroids,train=False):  #可视化数据
      plt.xlabel('X')
      plt.ylabel('Y')
      X,Y,centers=[0,0,0],[0,0,0],centroids
      for points in classes[0]:
            WBC = plt.scatter(points[0],points[1],s=50,c='red')
            X[0] += points[0]
            Y[0] += points[1]
      for points in classes[1]:
            RBC = plt.scatter(points[0],points[1],s=50,c='purple')
            X[1] += points[0]
            Y[1] += points[1]
      for points in classes[2]:
            PLATELETS = plt.scatter(points[0],points[1],s=50,c='yellow')
            X[2] += points[0]
            Y[2] += points[1]
      for i in range(3):
            center=plt.scatter(centers[i][0],centers[i][1],s=50,c='blue')
      if train:
            title = 'BCCD Train Dataset K-Means Clustering'
      else:
            title = 'BCCD Test Dataset K-Means Clustering'
      plt.title(title)
      plt.legend([WBC,RBC,PLATELETS,center],['WBC','RBC','Platelets','Centroid'])
      plt.show()


def main(iteration,random_centroids=False):
      train_data, train_label, test_data, test_label, init_centroids = optimize_preprocess()
      print('\nK-MEANS')
      if not random_centroids:
            print('Initialize Centroids:',round_centroids(init_centroids))
      else:
            xmax,ymax = 200,200
            init_centroids = [[random.randint(0,xmax),random.randint(0,ymax)],[random.randint(0,xmax),random.randint(0,ymax)],[random.randint(0,xmax),random.randint(0,ymax)]]
      new_centroids = []
      for iter in range(iteration):
            if iter == 0:
                  cur_centroids = init_centroids
            else:
                  cur_centroids = new_centroids
            train_accuracy,new_centroids,cell_accuracies,sse,classes = kmeans(train_data,train_label,cur_centroids)
            print('\nITERATION',iter+1)
            print('Train Accuracy:',train_accuracy,'| SSE:',sse)
            print('Train Centroids:',new_centroids)
            print('WBC :',round(cell_accuracies[0],4),end=' | ')
            print('RBC ',round(cell_accuracies[1],4),end=' | ')
            print('Platelets :',round(cell_accuracies[2],4),end='')
            if iter == iteration-1:
                  visualize(classes,new_centroids,train=True)

            test_accuracy,new_centroids,cell_accuracies,sse,classes = kmeans(test_data,test_label,new_centroids)
            print('\nTest Accuracy:',test_accuracy,'| SSE:',sse)
            print('Test Centroids:',new_centroids)
            print('WBC :',round(cell_accuracies[0],4),end=' | ')
            print('RBC ',round(cell_accuracies[1],4),end=' | ')
            print('Platelets :',round(cell_accuracies[2],4),end='\n')
            if iter == iteration-1:
                  visualize(classes,new_centroids,train=False)

      




if __name__ == '__main__':
      #main(iteration = 10,random_centroids=True)
      main(iteration = 3,random_centroids=False)




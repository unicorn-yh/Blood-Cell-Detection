import pandas as pd
import shutil
import os
import glob
import xml.etree.ElementTree as ET
import numpy as np

train_df = pd.read_csv("BCCD/ImageSets/Main/trainval.txt", sep = " ", header=None)
val_df = pd.read_csv("BCCD/ImageSets/Main/test.txt", sep = " ", header=None)

def classify_data():
      # Training images & their annotations
      for path in train_df[0].tolist():
            image_path = os.path.join("BCCD/JPEGImages/", path + ".jpg")
            annotation_path = os.path.join("BCCD/Annotations/", path + ".xml")
            i_path = os.path.join("images/train", path + ".jpg")
            a_path = os.path.join("images/train", path + ".xml")
            shutil.copy2(image_path, i_path)
            shutil.copy2(annotation_path, a_path)

      # Test images & their annotations
      for path in val_df[0].tolist():
            image_path = os.path.join("BCCD/JPEGImages/", path + ".jpg")
            annotation_path = os.path.join("BCCD/Annotations/", path + ".xml")
            i_path = os.path.join("images/test", path + ".jpg")
            a_path = os.path.join("images/test", path + ".xml")
            shutil.copy2(image_path, i_path)
            shutil.copy2(annotation_path, a_path)

      print('PREPROCESS DATA')
      print('Train Set:',len(os.listdir("images/train")))
      print('Test Set:',len(os.listdir("images/test")))
      print('')

def xml_to_csv(path):
      xml_list = []
      for xml_file in glob.glob(path + '/*.xml'):
            tree = ET.parse(xml_file)
            root = tree.getroot()
            for member in root.findall('object'):
                  value = (root.find('filename').text,
                        int(root.find('size')[0].text),
                        int(root.find('size')[1].text),
                        member[0].text,
                        int(member[4][0].text),
                        int(member[4][1].text),
                        int(member[4][2].text),
                        int(member[4][3].text)
                        )
                  xml_list.append(value)
      column_name = ['filename', 'width', 'height', 'classes', 'xmin', 'ymin', 'xmax', 'ymax']
      xml_df = pd.DataFrame(xml_list, columns=column_name)
      return xml_df


def preprocess():
      classify_data()
      for directory in ['train','test']:
            image_path = os.path.join(os.getcwd(), 'images/{}'.format(directory))
            xml_df = xml_to_csv(image_path)
            xml_df.to_csv('data/{}.csv'.format(directory), index=None)


def define_label(train_set,test_set):
      label = []
      for i in train_set.classes:
            if i == 'WBC':
                  label.append(1)
            elif i == 'RBC':
                  label.append(2)
            else:
                  label.append(3)
      train_set['Label']= label

      label = []
      for i in test_set.classes:
            if i == 'WBC':
                  label.append(1)
            elif i == 'RBC':
                  label.append(2)
            else:
                  label.append(3)
      test_set['Label']= label


def get_mean_xy(train_set,test_set):
      x,y=[],[]
      xmin,xmax,ymin,ymax = [],[],[],[]
      data,label = [],[]
      index = -1
      sum1,sum2,sum3 = 0,0,0
      sum1y,sum2y,sum3y = 0,0,0
      count1,count2,count3 = 0,0,0

      '''TRAIN'''
      # get x
      for i in train_set.xmin:
            xmin.append(i)
      for i in train_set.xmax:
            xmax.append(i)   
      for i in range(len(xmin)):
            xtemp = xmax[i]-xmin[i]
            x.append(xtemp)

      # get y
      for i in train_set.ymin:
            ymin.append(i)
      for i in train_set.ymax:
            ymax.append(i)   
      for i in range(len(ymin)):
            ytemp = ymax[i]-ymin[i]
            y.append(ytemp)

      for i in range(len(x)):
            data.append([x[i],y[i]])

      for i in train_set.Label:
            index += 1
            label.append(i)
            if i == 1:
                  sum1 += x[index]
                  sum1y += y[index]
                  count1 += 1
            elif i == 2:
                  sum2 += x[index]
                  sum2y += y[index]
                  count2 += 1
            elif i == 3:
                  sum3 += x[index]
                  sum3y += y[index]
                  count3 += 1

      train_data = np.array(data)
      train_label = np.array(label)

      print('TRAIN AVERAGE X')
      print('1','WBC:',sum1/count1)
      print('2','RBC:',sum2/count2)
      print('3','Platelets:',sum3/count3)
      print('\nTRAIN AVERAGE Y')
      print('1','WBC:',sum1y/count1)
      print('2','RBC:',sum2y/count2)
      print('3','Platelets:',sum3y/count3)

      centroids = [[sum1/count1,sum1y/count1],[sum2/count2,sum2y/count2],[sum3/count3,sum3y/count3]]


      '''TEST'''
      x,y=[],[]
      xmin,xmax,ymin,ymax = [],[],[],[]
      data,label = [],[]
      index = -1
      sum1,sum2,sum3 = 0,0,0
      sum1y,sum2y,sum3y = 0,0,0
      count1,count2,count3 = 0,0,0

      # get x
      for i in test_set.xmin:
            xmin.append(i)
      for i in test_set.xmax:
            xmax.append(i)   
      for i in range(len(xmin)):
            xtemp = xmax[i]-xmin[i]
            x.append(xtemp)

      # get y
      for i in test_set.ymin:
            ymin.append(i)
      for i in test_set.ymax:
            ymax.append(i)   
      for i in range(len(ymin)):
            ytemp = ymax[i]-ymin[i]
            y.append(ytemp)

      for i in range(len(x)):
            data.append([x[i],y[i]])

      for i in test_set.Label:
            index += 1
            label.append(i)
            if i == 1:
                  sum1 += x[index]
                  sum1y += y[index]
                  count1 += 1
            elif i == 2:
                  sum2 += x[index]
                  sum2y += y[index]
                  count2 += 1
            elif i == 3:
                  sum3 += x[index]
                  sum3y += y[index]
                  count3 += 1

      test_data = np.array(data)  
      test_label = np.array(label)  

      print('\nTEST AVERAGE X')
      print('1','WBC:',sum1/count1)
      print('2','RBC:',sum2/count2)
      print('3','Platelets:',sum3/count3)
      print('\nTEST AVERAGE Y')
      print('1','WBC:',sum1y/count1)
      print('2','RBC:',sum2y/count2)
      print('3','Platelets:',sum3y/count3)

      return train_data, train_label, test_data, test_label, centroids


def optimize_preprocess():
      preprocess()
      train_set = pd.read_csv("data/train.csv")
      test_set = pd.read_csv("data/test.csv")
      define_label(train_set,test_set)
      train_data, train_label, test_data, test_label, init_centroids = get_mean_xy(train_set,test_set)
      return train_data, train_label, test_data, test_label, init_centroids







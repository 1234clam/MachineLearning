#################################################  
# kmeans: k-means cluster  
# Author : zouxy  
# Date   : 2013-12-25  
# HomePage : http://blog.csdn.net/zouxy09  
# Email  : zouxy09@qq.com  
#################################################  

from numpy import *

import time
import matplotlib.pyplot as plt

## step 1: load data
from src.Kmeans import kmeans, showCluster

print("step 1: load data...")
dataSet = []
fileIn = open('../data/testSet.txt')
#read each line in the file
for line in fileIn.readlines():
    linedata = []
    lineArr = line.strip().split(',')
    for data in lineArr:
        linedata.append(float(data));
    dataSet.append(linedata)
    #print(dataSet)

## step 2: clustering...  
print("step 2: clustering...")
dataSet = mat(dataSet)
k = 4
centroids, clusterAssment = kmeans(dataSet,k,1)

## step 3: show the result  
print("step 3: show the result...")
showCluster(dataSet, k, centroids, clusterAssment) 
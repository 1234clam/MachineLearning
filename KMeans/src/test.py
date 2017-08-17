#################################################  
# kmeans: k-means cluster  
# Author : August
# Date   : 2017-8-11
# HomePage : http://blog.csdn.net/redhatforyou
# Email  : 2980159638@qq.com
#################################################  

from numpy import *

import time
import matplotlib.pyplot as plt

## step 1: load data
from src.Kmeans import kmeans, showCluster

print("step 1: load data...")
dataSet = []
fileIn = open('../data/Iris.txt')
#read each line in the file
for line in fileIn.readlines():
    linedata = []
    lineArr = line.strip().split(',')
    for data in lineArr:
        linedata.append(float(data));
    dataSet.append(linedata)

print("step 2: clustering...")
dataSet = mat(dataSet)
k = 3
centroids, clusterAssment = kmeans(dataSet,k,1)

## step 3: show the result  
print("step 3: show the result...")
showCluster(dataSet, k, centroids, clusterAssment) 
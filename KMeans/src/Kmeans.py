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

# calculate Euclidean distance
def euclDistance(vector1, vector2):
    return sqrt(sum(power(vector2 - vector1, 2)))

# init centroids with random samples
def Random_initCentroids(dataSet, k):
    numSamples, dim = dataSet.shape
    centroids = zeros((k, dim))
    #随机生成聚类中心
    for i in range(k):
        index = int(random.uniform(0, numSamples))
        centroids[i, :] = dataSet[index, :]
    return centroids

def maxDis_initCentroids(dataSet,k):
    numSamples, dim = dataSet.shape
    centroids = zeros((k,dim));

    #随机生成初始聚类中心
    index = int(random.uniform(0,numSamples));
    centroids[0,:] = dataSet[index,:]
    len = 1
    for i in range(k-1):
        maxDistance(dataSet,centroids,len);
        len = len+1
    return centroids

def maxDistance(dataSet,centroids,numCentroids):
    numSamples,dim = dataSet.shape;
    maxDis = 0;
    maxindex = 0;
    for i in range(numSamples):
        distance = 0;
        for j in range(numCentroids):
            distance = euclDistance(dataSet[i,:],centroids[j,:]) + distance;
        if distance > maxDis:
            maxDis = distance;
            maxindex = i;
    centroids[numCentroids,:] = dataSet[maxindex,:]

# k-means cluster
def kmeans(dataSet, k,medthod):
    numSamples = dataSet.shape[0]
    # first column stores which cluster this sample belongs to,
    # second column stores the error between this sample and its centroid
    clusterAssment = mat(zeros((numSamples, 2)))
    clusterChanged = True

    ## 初始化聚类中心
    if medthod == 0:
        centroids = Random_initCentroids(dataSet, k)
    else:
        centroids = maxDis_initCentroids(dataSet, k)

    while clusterChanged:
        clusterChanged = False

        ## 对每一个点计算聚类中心
        for i in range(numSamples):
            minDist = 100000.0
            minIndex = 0

            ## 寻找最近的聚类中心
            for j in range(k):
                distance = euclDistance(centroids[j, :], dataSet[i, :])
                if distance < minDist:
                    minDist = distance
                    minIndex = j

            ##更新节点所属的簇判断迭代是否继续
            if clusterAssment[i, 0] != minIndex:
                clusterChanged = True
            clusterAssment[i, :] = minIndex, minDist ** 2
        ## 更新聚类中心
        for j in range(k):
            pointsInCluster = dataSet[nonzero(clusterAssment[:, 0].A == j)[0]]
            centroids[j, :] = mean(pointsInCluster, axis=0)
    print('Congratulations, cluster complete!')
    return centroids, clusterAssment


# show your cluster only available with 2-D data
def showCluster(dataSet, k, centroids, clusterAssment):
    numSamples, dim = dataSet.shape
    printCluster(dataSet, centroids, clusterAssment)
    if dim != 2:
        print("Sorry! I can not draw because the dimension of your data is not 2!")
        return 1

    mark = ['or', 'ob', 'og', 'ok', '^r', '+r', 'sr', 'dr', '<r', 'pr']
    if k > len(mark):
        print
        "Sorry! Your k is too large! please contact Zouxy"
        return 1

        # draw all samples
    for i in range(numSamples):
        markIndex = int(clusterAssment[i, 0])
        plt.plot(dataSet[i, 0], dataSet[i, 1], mark[markIndex])

    mark = ['Dr', 'Db', 'Dg', 'Dk', '^b', '+b', 'sb', 'db', '<b', 'pb']
    # draw the centroids
    for i in range(k):
        plt.plot(centroids[i, 0], centroids[i, 1], mark[i], markersize=12)
    plt.show()

def printCluster(dataSet, centroids, clusterAssment):
    numSample,dim = dataSet.shape
    i = 0;
    for a in centroids:
        print("Cluster ",i,"centroids is ",a);
        i = i + 1
    for i in range(numSample):
        print("the node ",dataSet[i]," belong to the cluster",clusterAssment[i],"the centroids is ",centroids[int(clusterAssment[i,0])]);
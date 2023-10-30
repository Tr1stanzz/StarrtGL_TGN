import numpy as np
import pandas as pd
import random
import torch
from collections import Counter

class Kmeans:
    
    def __init__(self, model, n_clusters, device, n_tolerance=50, t=5.0, use_inner_product=False):
        self.model = model
        self.n_clusters = n_clusters
        self.device = device
        self.t = t
        self.n_tolerance = n_tolerance
        self.use_inner_product = use_inner_product

    def calcDisMLP(self, dataset, centroids):
        dis_list = []
        instace_num = len(dataset)
        for k in range(instace_num):
            # inner_pro = np.matmul(centroids, np.array(data))
            # dist = np.exp(-inner_pro/t)
            # print(dataset[k].shape)
            data_nn = torch.cat([dataset[k].view(1,-1)] * self.n_clusters, dim=0)
            # print(data_nn.shape)
            # print(centroids.shape)
            dist = self.model.affinity_score(data_nn, centroids).squeeze(dim=0)
            dist = dist.cpu().numpy()
            dist = np.exp(-dist/self.t)
            dis_list.append(dist)
        dis_list = np.array(dis_list)
        return dis_list

    def calcDisInnerProduct(self, dataset, centroids):
        dis_list = []
        instace_num = len(dataset)
        for k in range(instace_num):
            data_nn = torch.cat([dataset[k].view(1,-1)] * self.n_clusters, dim=0)
            inner_pro = torch.sum(torch.mul(data_nn, centroids), dim=1)
            inner_pro = inner_pro.cpu().numpy()
            dist = np.exp(-inner_pro/self.t)
            dis_list.append(dist)
        dis_list = np.array(dis_list)
        return dis_list

    def classify(self, dataSet, centroids):
        # 计算样本到质心的距离
        if self.use_inner_product:
            clalist = self.calcDisInnerProduct(dataSet, centroids)
        else:
            clalist = self.calcDisMLP(dataSet, centroids)
        # 分组并计算新的质心
        minDistIndices = np.argmin(clalist, axis=1)    #axis=1 表示求出每行的最小值的下标
        if isinstance(dataSet, np.ndarray):
            dataSet_np = dataSet
        else:
            dataSet_np = dataSet.cpu().numpy()
        if len(minDistIndices.shape) > 1:
            minDistIndices = np.squeeze(minDistIndices,axis=1)
        newCentroids = pd.DataFrame(dataSet_np).groupby(minDistIndices).mean() #DataFramte(dataSet)对DataSet分组，groupby(min)按照min进行统计分类，mean()对分类结果求均值
        newCentroids = newCentroids.values
        if not isinstance(centroids, np.ndarray):
            centroids = centroids.cpu().numpy()
        changed = newCentroids - centroids
        newCentroids = torch.from_numpy(newCentroids).to(self.device)
        return changed, newCentroids

    # 使用k-means分类
    def fit(self, dataSet):
        # 随机取质心
        i = 0
        random.seed()
        centroids_idx = random.sample(range(len(dataSet)), self.n_clusters)
        # print(centroids_idx)
        centroids = dataSet[centroids_idx]
        # 更新质心 直到变化量全为0
        changed, newCentroids = self.classify(dataSet, centroids)
        while np.any(changed != 0) and i < self.n_tolerance:
            changed, newCentroids = self.classify(dataSet, newCentroids)
            i += 1
        centroids = newCentroids
        clalist = self.calcDisInnerProduct(dataSet, centroids) #调用欧拉距离
        minDistIndices = np.argmin(clalist, axis=1)
        return np.array(minDistIndices)
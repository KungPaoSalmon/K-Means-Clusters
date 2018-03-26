import pandas as pd
import numpy as np
import random



class kmeans:

    def __init__(self,k,max_iter=500,threshold=0.01):
        #k是聚类簇的个数
        #max_iter是算法最大的迭代次数
        #threshold是算法收敛的条件，评价系数小于threshold即停止算法
        self.k=k
        self.max_iter=max_iter
        self.threshold=threshold

    
    def dis_compute(self,sample,center):
        #dis_compute计算每个样本点到所有中心的距离
        #返回样本点所在簇，以及与簇中心的距离
        sample1=sample.as_matrix()
        center1=center.as_matrix()
        dis=pd.DataFrame((center1-sample1)*(center1-sample1))
        dis=dis.apply(axis=1,func=np.sum)
        label=np.argmax(dis)+1
        dis_label=np.max(dis)
        return [label,dis_label]
        
            
    def fit(self,data):
        #fit为kmeans算法实现
        #默认data是一个pandas.DataFrame文件
        
        
        #随机选取k个样本作为初始中心
        #center_row=random.sample(range(0,data.shape[0]),2) 
        center_row=random.sample(range(0,data.shape[0]),self.k) 
        center=data.iloc[center_row,:]      
        
        #迭代k means算法
        result_old=[]
        result=[]
        dis_old=None
        
        for i in range(0,self.max_iter):
            
            if i >= 1:
                result_old=result               
            
            #计算每个样本点到所有中心点的距离,并为其分类
            result=data.apply(axis=1,func=self.dis_compute,center=center)
            
            #计算评价系数
            if i >= 1:
                score_old=np.sum(result_old[1]*result_old[1])
                score_new=np.sum(result[1]*result[1])
                self.score=score_new
                #判断是否小于threshold并停止迭代 
                if  score_new - score_old < self.threshold:
                    break
            
            #获得新的中心点
            for i in range(0,2):
                center.iloc[i,:]=data[result[0]==i+1].apply(axis=0,func=np.mean)
            
            self.center=center
            self.label=result[0]
            self.label_dis=result[1]




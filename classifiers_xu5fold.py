# -*- coding: utf-8 -*-
"""
Created on Sat May 26 20:30:21 2018

@author: xuhaohao
"""

import scipy.io as sio
import numpy as np

def warn(*args, **kwargs): pass
import warnings
warnings.warn = warn

from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC, LinearSVC, NuSVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis





######SelectKBest#########C:\Users\xuhaohao\Desktop\研一春季学期\医学数据智能计算\project\常用纹理特征提取方法_MATLAB
#root='C:/Users/xuhaohao/Desktop/研一春季学期/医学数据智能计算/project/常用纹理特征提取方法_MATLAB_xu/xudata/addlbp/'
#features1=(sio.loadmat(root+'feature1.mat'))['feature1']#加载.mat 58*608

#root='D:/feature/3637/拓end/'
#features0=(sio.loadmat(root+'feature0.mat'))['feature0']#加载.mat 58*608
#features1=(sio.loadmat(root+'feature1.mat'))['feature1']#加载.mat 58*608

root='D:\\xin\\shao\\'
features0=(sio.loadmat(root+'t2.mat'))['t2']#加载.mat 58*608
features1=(sio.loadmat(root+'t21.mat'))['t21']

y1=np.zeros((3679,1))
y2=np.ones((5637,1))
y=np.vstack((y1,y2))#116*1
X=np.vstack((features0,features1))

from sklearn.model_selection import KFold

from sklearn.decomposition import PCA
from scipy import stats
m=np.zeros((2,5)); 
j=-1;
###PCA

i=0;

##########classifiers###############
    #classifiers = [
   # KNeighborsClassifier(3),

kf = KFold(n_splits=5,shuffle=True, random_state=1)
kf.get_n_splits(X)

for train_index, test_index in kf.split(X):    
        #print("TRAIN:", train_index, "TEST:", test_index)
    
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]
   # c0=np.argwhere(y_train==0)
   # c0=c0.T[0]#取第一列找出0的index
    
    #c1=np.argwhere(y_train==1)
   # c1=c1.T[0]#取第一列，找出1的index
   # cc0=X_train[c0]#找出训练样本中的0类和1类
    #cc1=X_train[c1]
   # t,p=stats.ttest_ind(X_train[c0],X_train[c1])
   # pindex1=np.argwhere(p<0.05)
    #X_train=np.squeeze(X_train[:,pindex1])
   # pp1=np.squeeze(pp)
    
    #yppp=y_train(pindex1)
    clf = SVC(kernel='linear',C=10).fit(X_train, y_train)#1##61.2 63.5 
    #a=clf.score(X_test, y_test);
    #print(a)
    #clf = SVC(kernel='poly',C=10).fit(X_train, y_train)
    #clf=KNeighborsClassifier(n_neighbors=20,weights='distance').fit(X_train, y_train)#knn
    #clf =LinearDiscriminantAnalysis().fit(X_train, y_train)#高####################
    #clf = NuSVC(probability=True).fit(X_train, y_train)#1########################
    #clf = GaussianNB(priors=[0.1,0.9]).fit(X_train, y_train)  #Naive Bayes 
    #clf = RandomForestClassifier(n_estimators=1000).fit(X_train, y_train)#1RF
    #clf = GradientBoostingClassifier(n_estimators=100).fit(X_train, y_train)
    #clf = AdaBoostClassifier().fit(X_train, y_train) #62.5  63.8
    a=clf.score(X_test, y_test);
    print(a)
    #clf = DecisionTreeClassifier().fit(X_train, y_train)#1
   # X_test=X_test(:,0)
    #X_test=np.squeeze(X_test[:,pindex1])
      # sss = clf.predict(X_test)
    #answer = clf.predict(X_test)
    #knn算法调参
    '''
        best_method=''
        best_score=0.0
        best_k=-1
        for method in ["uniform","distance"]:
            for k in range(1,100):
                clf=KNeighborsClassifier(n_neighbors=k)
                clf.fit(X_train, y_train)
                score=clf.score(X_test, y_test)
                if score>best_score:
                    best_score=score
                    best_k=k
                    best_method=method
        print("best_score:",best_score)
        print("best_k:",best_k)
        print("best_method:",best_method)
        '''
        
'''
        m[j][i]=best_score;
        i=i+1;
'''
    #n=np.mean(m);
    #print(n);     
    
'''    
    #########################
    #knn算法调参
    best_method=''
    best_score=0.0
    best_k=-1
    for method in ["uniform","distance"]:
        for k in range(1,100):
            clf=KNeighborsClassifier(n_neighbors=k)
            clf.fit(X_train, y_train)
            score=clf.score(X_test, y_test)
            if score>best_score:
                best_score=score
                best_k=k
                best_method=method
    print("best_score:",best_score)
    print("best_k:",best_k)
    print("best_method:",best_method)           
'''            
    
'''   
aa=sss
numz=len(list(aa))
a0=aa[0:738]
a1=aa[738:1879]
numz0=list(a0).count(0)
numz1=list(a1).count(1)
num0=len(list(a0))
num1=len(list(a1))
print(numz0/num0)
print(numz1/num1)
  '''  
        

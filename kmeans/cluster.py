import numpy as np
import math
from multiprocessing.dummy import Pool

class KMeans:
    
    def __init__(self, n_clusters=5, max_iter=100, shift_tolerance=0.05, thread_capacity=100):
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.shift_tolerance = shift_tolerance
        self.thread_capacity = thread_capacity
        self.inertia=0
        self.min_value=0
        self.max_value=0
        self.inertia_=[]
        self.cluster=[]
        self.centroids=[]
        self.delta_shift=[]   
    
    def init_centroids(self, data):
        self.centroids=[]
        delta = int(len(data)/self.n_clusters)
        for i in range(self.n_clusters):
            self.centroids.append(data[i*delta])
            
    def analyze_data(self, data):
        self.min_value=min(min(point[0] for point in data), min(point[1] for point in data))
        self.max_value=min(max(point[0] for point in data), max(point[1] for point in data))
        self.tolerance_degree = self.max_value*self.shift_tolerance
    
    def identify_cluster(self, x):
        distances=[]
        for i in range(self.n_clusters):
            distance = math.sqrt((x[0]-self.centroids[i][0])**2 + (x[1]-self.centroids[i][1])**2)
            distances.append(distance)
        distances=np.asarray(distances)
        self.cluster[distances.argmin()].append(x)
        
    def shift_centroid(self, i):
        x = (sum(point[0] for point in self.cluster[i]))/len(self.cluster[i])
        y = (sum(point[1] for point in self.cluster[i]))/len(self.cluster[i])
        delta = (math.sqrt((x-self.centroids[i][0])**2 + (y-self.centroids[i][1])**2))
        if(delta>=self.tolerance_degree):
            self.centroids[i]=[x,y]
            self.delta_shift.append(delta)
            
    def calculate_inertia(self, i):
        sum=0
        for x in self.cluster[i]:
            distance = math.sqrt((x[0]-self.centroids[i][0])**2 + (x[1]-self.centroids[i][1])**2)
            sum=sum+(distance**2)
        self.inertia_.append(sum)
    
    def fit_showDetails(self, data):
        print('')
        print('Initializing clustering :')
        print('-- Using Base Parameters :')
        print('\t- Number of clusters = '+str(self.n_clusters))
        print('\t- Shifting tolerance = '+str(self.shift_tolerance))
        print('\t- Maximum Thread P/M = '+str(self.thread_capacity))
        print('')
        
        self.analyze_data(data)
        print('-- Data analytics :')
        print('\t- Min value in dataset = '+str(self.min_value))
        print('\t- Max value in dataset = '+str(self.max_value))
        print('\t- Degree of Tolerance  = '+str(self.tolerance_degree))
        print('')
        
        self.init_centroids(data)
        print('-- Initial Centroids :')
        print(self.centroids)
        print('')        
        
        print('Clustering Initiated!')
        for i in range(self.max_iter):            
            self.cluster=[]
            for i in range(self.n_clusters):
                self.cluster.append([])
            
            pool = Pool(min(len(data),self.thread_capacity))
            pool.map(self.identify_cluster, data)
            pool.close()
            pool.join()
            
            self.delta_shift=[]
            
            pool = Pool(min(self.n_clusters,self.thread_capacity))
            pool.map(self.shift_centroid, range(self.n_clusters))
            pool.close()
            pool.join()
            
            if(len(self.delta_shift)<1):
                print('info: Centroid Shift occured within tolerance.'+' Delta Shift is Empty. ' 
                      'Completing clustering.')
                print('info: Iteration count = '+str(i))
                break
        
        print('-- Final Centroids :')
        print(self.centroids)
        print('')
        
        pool = Pool(min(self.n_clusters,self.thread_capacity))
        pool.map(self.calculate_inertia, range(self.n_clusters))
        pool.close()
        pool.join()
        self.inertia=sum(self.inertia_)
        print('-- Cluster Inertia = '+str(self.inertia))
        print('')
        
        print('')
        print('Clustering Completed!')
        
    def fit(self, data):
        self.analyze_data(data)
        self.init_centroids(data)
        for i in range(self.max_iter):            
            self.cluster=[]
            for i in range(self.n_clusters):
                self.cluster.append([])
            
            pool = Pool(min(len(data),self.thread_capacity))
            pool.map(self.identify_cluster, data)
            pool.close()
            pool.join()
            
            self.delta_shift=[]
            
            pool = Pool(min(self.n_clusters,self.thread_capacity))
            pool.map(self.shift_centroid, range(self.n_clusters))
            pool.close()
            pool.join()
            
            if(len(self.delta_shift)<1):
                break
        
        pool = Pool(min(self.n_clusters,self.thread_capacity))
        pool.map(self.calculate_inertia, range(self.n_clusters))
        pool.close()
        pool.join()
        self.inertia=sum(self.inertia_)
        
    def fit_onSingleThread(self, data):
        self.analyze_data(data)
        self.init_centroids(data)
        for i in range(self.max_iter):            
            self.cluster=[]
            for i in range(self.n_clusters):
                self.cluster.append([])
            
            for i in data:
                self.identify_cluster(i)
            
            self.delta_shift=[]
            
            for i in range(self.n_clusters):
                self.shift_centroid(i)
            
            if(len(self.delta_shift)<1):
                break
        
        for i in range(self.n_clusters):
            self.calculate_inertia(i)
      
        self.inertia=sum(self.inertia_)
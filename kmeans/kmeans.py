import pandas as pd
import matplotlib.pyplot as plt
import cluster as clt

import timeit

start = timeit.default_timer()

dataset = pd.read_csv('/home/neo/Desktop/kmeans/dataset.csv')
dataset = dataset.values

wcss=[]
for i in range(1,10):
    kmeans = clt.KMeans(n_clusters=i, shift_tolerance=0.02, thread_capacity=4)
    kmeans.fit(dataset)
    wcss.append(kmeans.inertia)
plt.plot(range(1,10),wcss)
plt.show()

kmeans = clt.KMeans(n_clusters=2, shift_tolerance=0.005, thread_capacity=4)
kmeans.fit_showDetails(dataset)

plt.scatter([x[0] for x in kmeans.cluster[0]], [x[1] for x in kmeans.cluster[0]], s=2, color='blue')
plt.scatter([x[0] for x in kmeans.cluster[1]], [x[1] for x in kmeans.cluster[1]], s=2, color='red')
plt.scatter([x[0] for x in kmeans.cluster[2]], [x[1] for x in kmeans.cluster[2]], s=2, color='cyan')
#plt.scatter([x[0] for x in kmeans.cluster[3]], [x[1] for x in kmeans.cluster[3]], s=2, color='pink')
#plt.scatter([x[0] for x in kmeans.cluster[4]], [x[1] for x in kmeans.cluster[4]], s=2, color='yellow')
plt.scatter([x[0] for x in kmeans.centroids] , [x[1] for x in kmeans.centroids], s=10, color='green')
plt.show()

stop = timeit.default_timer()

print('Time: ', stop - start)  
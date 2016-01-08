from sklearn import datasets
from sklearn.cluster import KMeans

from sklearn.metrics import silhouette_score, silhouette_samples

import matplotlib.pyplot as plt
import plotly.plotly as py

iris = datasets.load_iris()

# dimensions of X: 150x4 
X = iris.data[:]
X1 = iris.data[:,:2]
X2 = iris.data[2:]
X3 = iris.data[:, 1:3]
Y = iris.target

print "The iris dataset..."

plt.xlabel('Sepal Length')
plt.ylabel('Sepal Width')
plt.scatter(X1[:,0],X1[:,1])
plt.show()


plt.xlabel('Petal Length')
plt.ylabel('Petal Width')
plt.scatter(X2[:,0],X2[:,1])
plt.show()


print "Performing KMeans clustering with silhouette-score analysis on the iris dataset..."
print 

print "Clustering using all 4 dimensions..."

silhouette_avg = [] # Storing average silhoutte values for each clustering

i = 2
while i <= 7:
    kmeansClusterer = KMeans(n_clusters=i)
    clusterLabels = kmeansClusterer.fit_predict(X)
    silhouette_avg.append(silhouette_score(X, clusterLabels))
    print "The silhouette_avg(using all 4 dimensions) is: ", silhouette_avg[-1]
    # clusterLabels1 = kmeansClusterer.fit_predict(X1)
    # clusterLabels2 = kmeansClusterer.fit_predict(X2)
    # plt.scatter(X[:,1],clusterLabels)
    # plt.show()
    i+=1

print
print "By using silhoutte average values on kmeans, we see that kmeans with ", silhouette_avg.index(max(silhouette_avg))+2, " clusters does the best"

kmeansClusterer1 = KMeans(n_clusters=2)
cluster_labels = kmeansClusterer1.fit_predict(X)
sample_silhouette_values = silhouette_samples(X, cluster_labels) 

print sample_silhouette_values
                                                                               


x = [2,3,4,5,6,7]
plt.bar(x,silhouette_avg,width=1, color='blue')
plt.xlabel("Silhouette_avg")
plt.ylabel("Number of clusters")
plt.title("Silhoutte-avg vs Number of clusters")
fig = plt.gcf()
fig.show()
plot_url = py.plot_mpl(fig, filename='silhouette_avg')

x = range(150)
y = sample_silhouette_values


plt.bar(x,y,width=1, color='yellow')
plt.xlabel("Sample number in the iris dataset")
plt.ylabel("Sample Silhouette Coefficient Values")
plt.title("Individual silhoutte coefficient values")
fig = plt.gcf()
fig.show()
plot_url = py.plot_mpl(fig, filename='Sample silhouette values')

# print "Clustering using only the first 2 dimensions..."


# silhouette_avg = [] # Storing average silhoutte values for each clustering

# i = 2
# while i <= 4:
#     kmeansClusterer = KMeans(n_clusters=i)
#     # kmeansClusterer = kmeansClusterer.fit(X2)

#     clusterLabels = kmeansClusterer.fit_predict(X2)
#     # print kmeansClusterer.get_params()
#     # print clusterLabels
#     # print i, kmeansClusterer.cluster_centers_
#     silhouette_avg.append(silhouette_score(X2, clusterLabels))
#     print "The silhouette_avg is: ", silhouette_avg[-1]

#     i+=1

# print
# print "By using silhoutte average values on kmeans(dimensions of X: 150x2), we see that kmeans with ", silhouette_avg.index(max(silhouette_avg))+2, " clusters does the best"



# kmeansClusterer = KMeans(n_clusters=2)
# clusterLabels = kmeansClusterer.fit_predict(X)

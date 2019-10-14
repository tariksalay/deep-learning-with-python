#PART 6: K-MEANS (Completed)
import pandas as pd
data = pd.read_csv('cars.csv', delimiter=',', header=None, skiprows=1, names=['mpg','cylinders','cubicinches','hp','weightlbs','time-to-60','year','brand'])
#Nulls find
nulls = pd.DataFrame(data.isnull().sum().sort_values(ascending=False))
nulls.columns = ['Features']
nulls.index.name = 'Nulls count'
print(nulls)
#No nulls, good to go
#Handling non-numeric data
import numpy as np
x = data.select_dtypes(include=[np.number]).interpolate().dropna()
#Visualize data
import seaborn as sns
import matplotlib.pyplot as plt
sns.FacetGrid(data, hue='brand', size=4).map(plt.scatter, 'mpg', 'cylinders').add_legend()
plt.show()
sns.FacetGrid(data, hue='brand', size=4).map(plt.scatter, 'cubicinches', 'hp').add_legend()
plt.show()
sns.FacetGrid(data, hue='brand', size=4).map(plt.scatter, 'weightlbs', 'time-to-60').add_legend()
plt.show()

#Apply k-means algorithm
from sklearn.cluster import KMeans
wcss = []
for i in range(1, 9):
    km = KMeans(n_clusters=i, init='k-means++', max_iter=300, n_init=10, random_state=0)
    km.fit(x)
    wcss.append(km.inertia_)
#Visualize elbow method
import matplotlib.pyplot as plt
plt.plot(range(1, 9), wcss)
plt.title = 'The Elbow Method'
plt.xlabel = 'n-clusters'
plt.ylabel = 'wcss'
plt.show()

#Found k=2
km = KMeans(n_clusters=2)
from sklearn.metrics import silhouette_score
km.fit(x)
x_pred = km.predict(x)
print('Silhouette score for k=2:', silhouette_score(x, x_pred))
#Found k=4
km = KMeans(n_clusters=4)
from sklearn.metrics import silhouette_score
km.fit(x)
x_pred = km.predict(x)
print('Silhouette score for k=4:', silhouette_score(x, x_pred))

# >>> k=2 is better fit
# For this data, it is a bad idea if data is scaled and apply PCA due to the silhouette score is very low


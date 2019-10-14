import pandas as pd
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.preprocessing import StandardScaler
import seaborn as sns
sns.set(style='white', color_codes=True)
import warnings
warnings.filterwarnings("ignore")

data = pd.read_csv('./CC.csv')
#Check how many clusters => 7
print(data['TENURE'].value_counts())
#Look for nulls
nulls = pd.DataFrame(data.isnull().sum().sort_values(ascending=False)[:25])
nulls.columns = ['Null Count']
nulls.index.name = 'Feature'
print(nulls)
#MINIMUM_PAYMENTS and CREDIT_LIMIT have nulls, below replacing them by the mean
data.loc[(data['MINIMUM_PAYMENTS'].isnull()==True), 'MINIMUM_PAYMENTS'] = data['MINIMUM_PAYMENTS'].mean()
data.loc[(data['CREDIT_LIMIT'].isnull()==True), 'CREDIT_LIMIT'] = data['CREDIT_LIMIT'].mean()
#Deviding data into x, y where y is TENURE
x = data.iloc[:,1:]
print(x.shape)

#Elbow Method
#wcss:within-cluster sums of squares
wcss = []
#try 1-9 clusters
print()
for i in range(1, 10):
    kmeans = KMeans(n_clusters=i, init='k-means++', max_iter=300, n_init=10, random_state=0)
    kmeans.fit(x)
    #Save within-cluster sums of squares to the list
    wcss.append(kmeans.inertia_)

#Display the graph
print(wcss)
plt.plot(range(1, 10), wcss)
plt.title('the elbow method')
plt.xlabel('Number of Clusters')
plt.ylabel('WCSS')
plt.show()

#From the map, at k=3 seem like data slowly unchange => choose k=3
#Silhouette score
km = KMeans(n_clusters=3)
km.fit(x)
y_cluster_kmeans = km.predict(x)
score = metrics.silhouette_score(x, y_cluster_kmeans)
print()
print('Silhouette score for',3,'clusters',score)

###########################################################################
from sklearn import preprocessing
scaler = preprocessing.StandardScaler()
scaler.fit(x)
X_scaled_array = scaler.transform(x)
X_scaled = pd.DataFrame(X_scaled_array, columns = x.columns)

km = KMeans(n_clusters=3)
km.fit(X_scaled)
y_cluster_kmeans = km.predict(X_scaled)
from sklearn import metrics
score = metrics.silhouette_score(X_scaled, y_cluster_kmeans)
print('Silhouette score for',3,'clusters after scaled',score)
###########################################################################
#Apply PCA
#scale the features
scaler = StandardScaler()
scaler.fit(x)
x_scaler = scaler.transform(x)

#Apply KMeans to PCA, 10 is picked randomly
pca = PCA(4)
x_pca = pca.fit_transform(x_scaler)
#combine data into 10 columns
df = pd.DataFrame(data=x_pca)
# #Display PCA
# plt.scatter(df[0], df[1], alpha=.1, color='black')
# plt.xlabel('PCA 1')
# plt.ylabel('PCA 2')
# plt.show()

wcss=[]
for i in range(1, 8):
    kmeans = KMeans(n_clusters=i, init='k-means++', max_iter=300, n_init=10, random_state=0)
    kmeans.fit(x_pca)
    #Save within-cluster sums of squares to the list
    wcss.append(kmeans.inertia_)
print()
print(wcss)
plt.plot(range(1, 8), wcss)
plt.title('the elbow method')
plt.xlabel('Number of Clusters')
plt.ylabel('WCSS')
plt.show()

#From the graph, choose k=4
km = KMeans(n_clusters=4)
km.fit(x_pca)
y_cluster_kmeans = km.predict(x_pca)
score = metrics.silhouette_score(x_pca, y_cluster_kmeans)
print('Silhouette score after applying PCA',score)
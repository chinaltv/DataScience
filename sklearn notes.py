k = 1 # 假数字
X_train = [1,2,3] # 假数据
Y_train = [0,1,1] # 假数据


# KNN
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(algorithm = "auto", leaf_size = k, metric = "minkowski", metric_params = None, n_jobs = k, n_neighbors= k, p = k, weight = 'uniform')
knn.fit(X_train, Y_train)
y_pred = knn.predict()


from sklearn.ensemble import RandomForestClassifier
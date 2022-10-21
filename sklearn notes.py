k = 1 # 假数字
X_train = [1,2,3] # 假数据
Y_train = [0,1,1] # 假数据


# KNN
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors= k)
knn.fit(X_train, Y_train)

# KNN调参

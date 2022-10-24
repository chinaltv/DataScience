k = 1 # 假数字
X_train = [1,2,3] # 假数据
Y_train = [0,1,1] # 假数据


# KNN
from random import Random
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(algorithm = "auto", leaf_size = k, metric = "minkowski", metric_params = None, n_jobs = k, n_neighbors= k, p = k, weight = 'uniform')
knn.fit(X_train, Y_train)
y_pred = knn.predict()


from sklearn.ensemble import RandomForestClassifier
forest = RandomForestClassifier(n_estimators = 500, 
                                criterion = 'gini', 
                                max_depth = 7, 
                                min_samples_leaf = 1, 
                                min_weight_fraction_leaf = 0, # unmodified
                                max_features = 2,
                                max_leaf_nodes = 1,
                                min_impurity_decrease = 1,
                                bootstrap = ,
                                oob_score = ,
                                n_jobs = ,
                                random_state = ,
                                verbose = 0,
                                warm_start = False,
                                class_weight = ,
                                ccp_alpha = ,
                                max_samples = None)

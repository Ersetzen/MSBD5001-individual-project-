#Can not be run all the methods at same time
#Please run them respectively

import pandas as pd
import csv
from catboost import cv,Pool,CatBoostRegressor
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.preprocessing import MinMaxScaler, Normalizer, scale

# #importance
# corr_matrix = train_data.corr()
# corr_matrix.sort_values(["time"], ascending=False, inplace=True)
# corr_matrix.time

# #some data preprocessing
# train_data['bu'] = (train_data['bu']-train_data['bu'].mean())/train_data['bu'].std()
# train_data['cu'] = (train_data['cu']-train_data['cu'].mean())/train_data['cu'].std()
# train_data['uu4'] = (train_data['uu4']-train_data['uu4'].mean())/train_data['uu4'].std()
# train_data['h/k'] = (train_data['h/k']-train_data['h/k'].mean())/train_data['h/k'].std()
# train_data['i/j'] = (train_data['i/j']-train_data['i/j'].mean())/train_data['i/j'].std()
# train_data['UU'] = (train_data['UU']-train_data['UU'].mean())/train_data['UU'].std()
# train_data['max_iter'] = (train_data['max_iter']-train_data['max_iter'].mean())/train_data['max_iter'].std()
# train_data['n_samples'] = (train_data['n_samples']-train_data['n_samples'].mean())/train_data['n_samples'].std()
# train_data['n_features'] = (train_data['n_features']-train_data['n_features'].mean())/train_data['n_features'].std()
# train_data['n_classes'] = (train_data['n_classes']-train_data['n_classes'].mean())/train_data['n_classes'].std()
# train_data['n_clusters_per_class'] = (train_data['n_clusters_per_class']-train_data['n_clusters_per_class'].mean())/train_data['n_clusters_per_class'].std()
# train_data['n_informative'] = (train_data['n_informative']-train_data['n_informative'].mean())/train_data['n_informative'].std()
# train_data['1\s'] = (train_data['1\s']-train_data['1\s'].mean())/train_data['1\s'].std()
# train_data['flip_y'] = (train_data['flip_y']-train_data['flip_y'].mean())/train_data['flip_y'].std()

# test_data['bu'] = (test_data['bu']-test_data['bu'].mean())/test_data['bu'].std()
# test_data['cu'] = (test_data['cu']-test_data['cu'].mean())/test_data['cu'].std()
# test_data['uu4'] = (test_data['uu4']-test_data['uu4'].mean())/test_data['uu4'].std()
# test_data['h/k'] = (test_data['h/k']-test_data['h/k'].mean())/test_data['h/k'].std()
# test_data['i/j'] = (test_data['i/j']-test_data['i/j'].mean())/test_data['i/j'].std()
# test_data['UU'] = (test_data['UU']-test_data['UU'].mean())/test_data['UU'].std()
# test_data['max_iter'] = (test_data['max_iter']-test_data['max_iter'].mean())/test_data['max_iter'].std()
# test_data['n_samples'] = (test_data['n_samples']-test_data['n_samples'].mean())/test_data['n_samples'].std()
# test_data['n_features'] = (test_data['n_features']-test_data['n_features'].mean())/test_data['n_features'].std()
# test_data['n_classes'] = (test_data['n_classes']-test_data['n_classes'].mean())/test_data['n_classes'].std()
# test_data['n_clusters_per_class'] = (test_data['n_clusters_per_class']-test_data['n_clusters_per_class'].mean())/test_data['n_clusters_per_class'].std()
# test_data['n_informative'] = (test_data['n_informative']-test_data['n_informative'].mean())/test_data['n_informative'].std()
# test_data['1\s'] = (test_data['1\s']-test_data['1\s'].mean())/test_data['1\s'].std()
# test_data['flip_y'] = (test_data['flip_y']-test_data['flip_y'].mean())/test_data['flip_y'].std()


# method 1
#time is log time
X = pd.read_csv('/Users/levichen/Desktop/1ttrr.csv').astype('str')
y = pd.read_csv('/Users/levichen/Desktop/trainl2.csv')
Z = pd.read_csv('/Users/levichen/Desktop/1tt.csv').astype('str')

X_train, X_validation, y_train, y_validation = train_test_split(X, y, train_size=0.85)

cat_features = list(range(0, X.shape[1]))


params = {}
params['iterations'] = 100
params['custom_loss'] = 'AUC'
params['learning_rate'] = 0.12

cv_data = cv(
    params = params,
    pool = Pool(X, y, cat_features),
    fold_count=5,
    inverted=False,
    shuffle=True,
    partition_random_seed=0,
    plot=True,
    stratified=False,
    verbose=False
)
model = CatBoostRegressor(iterations=500,
                             learning_rate=0.03,
                             depth=6,
                             eval_metric='RMSE',
                            #  bagging_temperature = 0.9,
                             )

model.fit(X_train, y_train,
             eval_set=(X_validation,y_validation),
             cat_features=cat_features,
             use_best_model=True,
             logging_level='Silent')



print('Model is fitted: ' + str(model.is_fitted()))
print('Model params:')
print(model.get_params())
print(model.get_best_score())
R = model.predict(Z)
print(np.power(2.3,R))
pd.DataFrame(np.power(2.3,R), columns=['time']).to_csv('/Users/levichen/Desktop/sub1.csv',header='time',index_label='id')






# #method 2
#time is log time
from keras.models import Sequential
from keras.layers import Dense,Dropout
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
import keras
from keras import regularizers
from keras import optimizers

X = pd.read_csv('/Users/levichen/Desktop/2ttrr.csv')
y = pd.read_csv('/Users/levichen/Desktop/trainl2.csv')

X_train, X_validation, y_train, y_validation = train_test_split(X, y, train_size=0.8)
Z = pd.read_csv('/Users/levichen/Desktop/2tt.csv')

model = MLPRegressor(hidden_layer_sizes=(800,4), activation='relu', solver='adam', 
             alpha=0.01, batch_size='auto', learning_rate='adaptive', 
             learning_rate_init=0.001, power_t=0.4, max_iter=500, 
             shuffle=True, random_state=None, tol=0.001, verbose=False, 
             warm_start=False, momentum=0.2, nesterovs_momentum=True, early_stopping=False, 
             validation_fraction=0.2, beta_1=0.99, beta_2=0.9999, epsilon=1e-08, n_iter_no_change=100)
model.fit(X_train, y_train)
R = model.predict(Z)
print(np.power(2.3,R))
print(model.score(X_validation, y_validation))
pd.DataFrame(np.power(2.3,R), columns=['time']).to_csv('/Users/levichen/Desktop/sub2.csv',header='time',index_label='id')

#method 3
from keras.models import Sequential
from keras.layers import Dense,Dropout
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
import keras
from keras import regularizers
from keras import optimizers

X = pd.read_csv('/Users/levichen/Desktop/2ttrr.csv')
y = pd.read_csv('/Users/levichen/Desktop/trainl2.csv')
X_train, X_validation, y_train, y_validation = train_test_split(X, y, train_size=0.8)

model = Sequential()
model.add(Dense(600, input_dim=20, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(600, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(1))
model.compile(loss='mean_squared_error', optimizer=keras.optimizers.Adam(lr=0.001, \
            epsilon=None, decay=0.0001, amsgrad=False),metrics=['accuracy'])
#model.compile(loss='mean_squared_error',\
#    optimizer=SGD(lr=0.01, momentum=0.9, nesterov=True))
model.fit(X_train, y_train, epochs=600, verbose=0, batch_size=60)

Z = pd.read_csv('/Users/levichen/Desktop/2tt.csv')

loss_and_metrics = model.evaluate(X_validation, y_validation, batch_size=64)
R = model.predict(Z)
#yy = pd.read_csv('/Users/levichen/Desktop/submission_averaging_on_best3.csv')

print(loss_and_metrics)
print(np.power(2.3,R))
pd.DataFrame(np.power(2.3,R)).to_csv('/Users/levichen/Desktop/sub3.csv',header='time',index_label='id')


#choose the best performance result and final result is as following:
#result of method1 * 0.3 + result of method2(or3) * 0.7
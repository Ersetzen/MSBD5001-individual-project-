# MSBD5001-individual-project-
5001 individual project 

It's my MSBD5001 individual project, including codes(algorithm & data preprocessing), data(including training and test data after processing) and readme. My method combines MLP with catboost, and my result based on different weights of two methods. 


import pandas as pd
import csv
from catboost import cv,Pool,CatBoostRegressor
from sklearn.model_selection import train_test_split
import numpy
from sklearn.preprocessing import MinMaxScaler, Normalizer, scale
from keras.models import Sequential
from keras.layers import Dense,Dropout
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
import keras
from keras import regularizers
from keras import optimizers

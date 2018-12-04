# MSBD5001-individual-project-
5001 individual project 

It's my MSBD5001 individual project, including codes(algorithm & data preprocessing), data and readme. My method combines MLP with catboost, and my result based on different weight of two methods. 


import pandas as pd
import csv
from catboost import cv,Pool,CatBoostRegressor
from sklearn.model_selection import train_test_split
import numpy
from sklearn.preprocessing import MinMaxScaler, Normalizer, scale

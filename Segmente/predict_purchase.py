from datetime import datetime, timedelta,date
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

import keras
from keras.layers import Dense
from keras.models import Sequential
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping
from keras.utils import np_utils
from keras.layers import LSTM
from sklearn.model_selection import KFold, cross_val_score, train_test_split

#read the data in csv
df_sales = pd.read_csv('datasets/retail_sales.csv')

#convert date field from string to datetime
df_sales['date'] = pd.to_datetime(df_sales['date'])

#show first 10 rows
print(df_sales.head(10))
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
sales = pd.read_csv('datasets/retail_sales.csv')

#convert date field from string to datetime
sales['date'] = pd.to_datetime(sales['date'])

#show first 10 rows
print(sales.head(10))

#represent month in date field as its first day
sales['date'] = sales['date'].dt.year.astype('str') + '-' + sales['date'].dt.month.astype('str') + '-01'
sales['date'] = pd.to_datetime(sales['date'])
#groupby date and sum the sales
sales = sales.groupby('date').sales.sum().reset_index()
print('Total monthly Sales')
print(sales.head())

# plot monthly sales data
sns.lineplot(x=sales['date'], y=sales['sales'], data=sales, dashes=True)
sns.despine(left=True, bottom=True)
plt.title('Monthly Sales data')
plt.ylabel('Sales($)')
plt.xlabel('Month')
plt.show()

#create a new dataframe to model the difference
sales_diff = sales.copy()
#add previous sales to the next row
sales_diff['prev_sales'] = sales_diff['sales'].shift(1)
#drop the null values and calculate the difference
sales_diff = sales_diff.dropna()# very first row will contain Nan due to shift function
sales_diff['diff'] = (sales_diff['sales'] - sales_diff['prev_sales'])
print('Sales Movement')
print(sales_diff.head(10))
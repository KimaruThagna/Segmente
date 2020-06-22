from datetime import datetime, timedelta,date
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.formula.api as smf
import numpy as np
import keras
from keras.layers import Dense
from keras.models import Sequential
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping
from keras.utils import np_utils
from keras.layers import LSTM
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.preprocessing import MinMaxScaler
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

# plot monthly difference in sales data
sns.lineplot(x=sales_diff['date'], y=sales_diff['diff'], data=sales_diff, dashes=True)
sns.despine(left=True, bottom=True)
plt.title('Monthly Delta in Sales data')
plt.ylabel('Delta($)')
plt.xlabel('Month')
plt.show()

#create dataframe for transformation from time series to supervised
salessupervised = sales_diff.drop(['prev_sales'],axis=1)
#adding lags
for inc in range(1,13):
    field_name = 'lag_' + str(inc)
    salessupervised[field_name] = salessupervised['diff'].shift(inc)
#drop null values
salessupervised = salessupervised.dropna().reset_index(drop=True)
print(salessupervised)

# using adjusted r-squared to determine how well the featureset explains a label
# Import statsmodels.formula.api

# Define the regression formula
model = smf.ols(formula='diff ~ lag_1', data=salessupervised)
# Fit the regression
model_fit = model.fit()
# Extract the adjusted r-squared
regression_adj_rsq = model_fit.rsquared_adj
print(regression_adj_rsq)

salesmodel = salessupervised.drop(['sales','date'],axis=1)
#split train and test set
train_set, test_set = salesmodel[0:-6].values, salesmodel[-6:].values
print(train_set)

#apply Min Max Scaler
scaler = MinMaxScaler(feature_range=(-1, 1))
scaler = scaler.fit(train_set)
# reshape training set
train_set = train_set.reshape(train_set.shape[0], train_set.shape[1])
train_set_scaled = scaler.transform(train_set)
# reshape test set
test_set = test_set.reshape(test_set.shape[0], test_set.shape[1])
test_set_scaled = scaler.transform(test_set)

#train test split
X_train, y_train = train_set_scaled[:, 1:], train_set_scaled[:, 0:1]
X_train = X_train.reshape(X_train.shape[0], 1, X_train.shape[1])
X_test, y_test = test_set_scaled[:, 1:], test_set_scaled[:, 0:1]
X_test = X_test.reshape(X_test.shape[0], 1, X_test.shape[1])

#model definition
model = Sequential()
model.add(LSTM(4, batch_input_shape=(1, X_train.shape[1], X_train.shape[2]), stateful=True))
model.add(Dense(1))

model.compile(loss='mean_squared_error', optimizer='adam')
model.fit(X_train, y_train, nb_epoch=100, batch_size=1, verbose=1, shuffle=False)
y_pred = model.predict(X_test,batch_size=1)
print('predictions')
print(y_pred)

#reshape y_pred
y_pred = y_pred.reshape(y_pred.shape[0], 1, y_pred.shape[1])
#rebuild test set for inverse transform
pred_test_set = []
for index in range(0,len(y_pred)):
    print (np.concatenate([y_pred[index],X_test[index]],axis=1))
    pred_test_set.append(np.concatenate([y_pred[index],X_test[index]],axis=1))
#reshape pred_test_set
pred_test_set = np.array(pred_test_set)
pred_test_set = pred_test_set.reshape(pred_test_set.shape[0], pred_test_set.shape[2])
#inverse transform
pred_test_set_inverted = scaler.inverse_transform(pred_test_set)
print('Predictions inverted')
print(pred_test_set)

#create dataframe that shows the predicted sales
result_list = []
sales_dates = list(sales[-7:].date)
act_sales = list(sales[-7:].sales)
for index in range(0,len(pred_test_set_inverted)):
    result_dict = {}
    result_dict['pred_value'] = int(pred_test_set_inverted[index][0] + act_sales[index])
    result_dict['date'] = sales_dates[index+1]
    result_list.append(result_dict)
result = pd.DataFrame(result_list)
print('Predicted Sales')
print(result)

#merge with actual sales dataframe
sales_pred = pd.merge(sales,result,on='date',how='left')

sns.lineplot(x=sales_pred['date'], y=sales_pred['sales'], data=sales_pred, dashes=True)
sns.lineplot(x=sales_pred['date'], y=sales_pred['pred_value'], data=sales_pred, dashes=True)
sns.despine(left=True, bottom=True)
plt.title('Monthly Sales Data Prediction')
plt.ylabel('Sales($)')
plt.xlabel('Month')
plt.show()
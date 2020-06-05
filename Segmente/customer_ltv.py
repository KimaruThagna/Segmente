from datetime import date
import pandas as pd
from sklearn.metrics import classification_report,confusion_matrix
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.cluster import KMeans
import xgboost as xgb
from sklearn.model_selection import KFold, cross_val_score, train_test_split

'''
lifetimeValue = Gross Revenue-Total Cost
'''
#read data from csv and redo the data work we done before
transaction_data = pd.read_csv('datasets/OnlineRetail.csv')
transaction_data['InvoiceDate'] = pd.to_datetime(transaction_data['InvoiceDate'])
uk_users = transaction_data.query("Country=='United Kingdom'").reset_index(drop=True)

#3 month dataframe
transaction_data_3m = uk_users[(uk_users.InvoiceDate < date(2011,6,1)) & (uk_users.InvoiceDate >= date(2011,3,1))].reset_index(drop=True)
# 6 months, june to dec
transaction_data_6m = uk_users[(uk_users.InvoiceDate >= date(2011,6,1)) & (uk_users.InvoiceDate < date(2011,12,1))].reset_index(drop=True)

#create user_3m for assigning clustering
user_3m = pd.DataFrame(transaction_data_3m['CustomerID'].unique())
user_3m.columns = ['CustomerID']

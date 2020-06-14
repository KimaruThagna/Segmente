from datetime import date
import pandas as pd
from sklearn.metrics import classification_report,confusion_matrix
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
#ml libraries
from sklearn.cluster import KMeans
from sklearn.svm import SVC
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
import xgboost as xgb
from sklearn.model_selection import KFold, cross_val_score, train_test_split

transaction_data = pd.read_csv('datasets/OnlineRetail.csv')
transaction_data['InvoiceDate'] = pd.to_datetime(transaction_data['InvoiceDate'])
uk_users = transaction_data.query("Country=='United Kingdom'").reset_index(drop=True)
# use 6 months of customer purchase data to predict the next purchase date
transaction_6m = uk_users[(uk_users.InvoiceDate < date(2011,9,1)) & (uk_users.InvoiceDate >= date(2011,3,1))].reset_index(drop=True)
transaction_next = uk_users[(uk_users.InvoiceDate >= date(2011,9,1)) & (uk_users.InvoiceDate < date(2011,12,1))].reset_index(drop=True)
# user data
transaction_user = pd.DataFrame(transaction_6m['CustomerID'].unique())
transaction_user.columns = ['CustomerID']
#df with first purchase date after cutoff date
transaction_next_first_purchase = transaction_next.groupby('CustomerID').InvoiceDate.min().reset_index()
transaction_next_first_purchase.columns = ['CustomerID','MinPurchaseDate']
#df with last purchase date before cutoff date
transaction_last_purchase = transaction_6m.groupby('CustomerID').InvoiceDate.max().reset_index()
transaction_last_purchase.columns = ['CustomerID','MaxPurchaseDate']
#merge
transaction_purchase_dates = pd.merge(transaction_last_purchase,transaction_next_first_purchase,on='CustomerID',how='left')
#time delta in days:
transaction_purchase_dates['NextPurchaseDay'] = (transaction_purchase_dates['MinPurchaseDate'] - transaction_purchase_dates['MaxPurchaseDate']).dt.days
#merge with transaction_user 
transaction_user = pd.merge(transaction_user, transaction_purchase_dates[['CustomerID','NextPurchaseDay']],on='CustomerID',how='left')

#print transaction_user
print(transaction_user.head())

#fill NA values with 999
transaction_user = transaction_user.fillna(999)

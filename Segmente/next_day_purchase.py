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
tx_6m = uk_users[(uk_users.InvoiceDate < date(2011,9,1)) & (uk_users.InvoiceDate >= date(2011,3,1))].reset_index(drop=True)
tx_next = uk_users[(uk_users.InvoiceDate >= date(2011,9,1)) & (uk_users.InvoiceDate < date(2011,12,1))].reset_index(drop=True)

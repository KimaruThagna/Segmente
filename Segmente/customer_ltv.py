import pandas as pd
from sklearn.metrics import classification_report,confusion_matrix
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.cluster import KMeans
import xgboost as xgb
from sklearn.model_selection import KFold, cross_val_score, train_test_split

#read data from csv and redo the data work we done before
transaction_data = pd.read_csv('datasets/OnlineRetail.csv')


'''
lifetimeValue = Gross Revenue-Total Cost
'''
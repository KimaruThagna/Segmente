'''
Hypothesis
Group A(test) → Offer → Higher Retention
Group B → No offer → Lower Retention

'''
import pandas as pd
import numpy as np


#create own dataset
df_hv = pd.DataFrame()
df_hv['customer_id'] = np.array([count for count in range(20000)])
df_hv['segment'] = np.array(['high-value' for _ in range(20000)])
df_hv['group'] = 'control'
df_hv.loc[df_hv.index<10000,'group'] = 'test'

# purchase count distribution
df_hv.loc[df_hv.group == 'test', 'purchase_count'] = np.random.poisson(0.6, 10000)
df_hv.loc[df_hv.group == 'control', 'purchase_count'] = np.random.poisson(0.5, 10000)
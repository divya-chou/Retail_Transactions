import pandas as pd
import numpy as np
from matplotlib import pyplot
from datetime import timedelta
from tqdm import tqdm
from sklearn.model_selection import train_test_split

from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, auc,roc_auc_score, precision_recall_fscore_support, confusion_matrix, f1_score

raw_data = pd.read_csv(r'D:\Data\Data.csv')

raw_data = raw_data[raw_data['Customer ID'].notnull()]
# Cleaning the raw_data
raw_data['InvoiceDate_dt'] = pd.to_datetime(raw_data.InvoiceDate, format='%Y-%m-%d %H:%M:%S') 
raw_data['InvoiceDate_dt'].head()
#Shifiting time stamp by 9 days, because 
days = timedelta(9)
raw_data['InvoiceDate_dt'] = raw_data['InvoiceDate_dt'] - days

raw_data['month_wise'] = [ (dt.year*100) + (dt.month ) for dt in raw_data.InvoiceDate_dt]
raw_data['day_wise'] = [ (dt.year*10000) + (dt.month *100) + (dt.day) for dt in raw_data.InvoiceDate_dt]


raw_data['Q'] = raw_data['InvoiceDate_dt'].dt.quarter

raw_data['spend'] = raw_data.Price * raw_data.Quantity
raw_data['Postage'] = np.where(raw_data.StockCode == 'POST', 'Postage','No_Postage')
raw_data['UK'] = np.where(raw_data.Country == 'United Kingdom', 1,0)
raw_data['c'] = raw_data.Invoice.str[0]
raw_data.loc[raw_data.c != 'C','c'] = 'p'

all_months = list(raw_data.month_wise.unique())
start_month = [201105, 201106, 201107, 201108, 201109, 201110]
#end_month = [200911, 200912, 201001, 201002, 201003, 201004]
y_month = [201106, 201107, 201108, 201109, 201110, 201111]
#days_diff = [548, 365, 274, 183, 91, 60, 30]
months_diff = [18, 12, 9, 6, 3, 2, 1]


cust_df_all_cols = pd.DataFrame(columns = ['Customer ID', 'count_Invoice_all_18', 'min_Q_all_18', 'max_Q_all_18', 'avg_Q_all_18', 'min_spend_all_18', 'max_spend_all_18', 'avg_spend_all_18', 'count_Invoice_c_18', 'min_Q_c_18', 'max_Q_c_18', 'avg_Q_c_18', 'min_spend_c_18', 'max_spend_c_18', 'avg_spend_c_18', 'count_Invoice_p_18', 'min_Q_p_18', 'max_Q_p_18', 'avg_Q_p_18', 'min_spend_p_18', 'max_spend_p_18', 'avg_spend_p_18', 'Spend_Items_18', 'Spend_Postage_18', 'Post_Spend_ratio_18', 'count_Invoice_all_12', 'min_Q_all_12', 'max_Q_all_12', 'avg_Q_all_12', 'min_spend_all_12', 'max_spend_all_12', 'avg_spend_all_12', 'count_Invoice_c_12', 'min_Q_c_12', 'max_Q_c_12', 'avg_Q_c_12', 'min_spend_c_12', 'max_spend_c_12', 'avg_spend_c_12', 'count_Invoice_p_12', 'min_Q_p_12', 'max_Q_p_12', 'avg_Q_p_12', 'min_spend_p_12', 'max_spend_p_12', 'avg_spend_p_12', 'Spend_Items_12', 'Spend_Postage_12', 'Post_Spend_ratio_12', 'count_Invoice_all_9', 'min_Q_all_9', 'max_Q_all_9', 'avg_Q_all_9', 'min_spend_all_9', 'max_spend_all_9', 'avg_spend_all_9', 'count_Invoice_c_9', 'min_Q_c_9', 'max_Q_c_9', 'avg_Q_c_9', 'min_spend_c_9', 'max_spend_c_9', 'avg_spend_c_9', 'count_Invoice_p_9', 'min_Q_p_9', 'max_Q_p_9', 'avg_Q_p_9', 'min_spend_p_9', 'max_spend_p_9', 'avg_spend_p_9', 'Spend_Items_9', 'Spend_Postage_9', 'Post_Spend_ratio_9', 'count_Invoice_all_6', 'min_Q_all_6', 'max_Q_all_6', 'avg_Q_all_6', 'min_spend_all_6', 'max_spend_all_6', 'avg_spend_all_6', 'count_Invoice_c_6', 'min_Q_c_6', 'max_Q_c_6', 'avg_Q_c_6', 'min_spend_c_6', 'max_spend_c_6', 'avg_spend_c_6', 'count_Invoice_p_6', 'min_Q_p_6', 'max_Q_p_6', 'avg_Q_p_6', 'min_spend_p_6', 'max_spend_p_6', 'avg_spend_p_6', 'Spend_Items_6', 'Spend_Postage_6', 'Post_Spend_ratio_6', 'count_Invoice_all_3', 'min_Q_all_3', 'max_Q_all_3', 'avg_Q_all_3', 'min_spend_all_3', 'max_spend_all_3', 'avg_spend_all_3', 'count_Invoice_c_3', 'min_Q_c_3', 'max_Q_c_3', 'avg_Q_c_3', 'min_spend_c_3', 'max_spend_c_3', 'avg_spend_c_3', 'count_Invoice_p_3', 'min_Q_p_3', 'max_Q_p_3', 'avg_Q_p_3', 'min_spend_p_3', 'max_spend_p_3', 'avg_spend_p_3', 'Spend_Items_3', 'Spend_Postage_3', 'Post_Spend_ratio_3', 'count_Invoice_all_2', 'min_Q_all_2', 'max_Q_all_2', 'avg_Q_all_2', 'min_spend_all_2', 'max_spend_all_2', 'avg_spend_all_2', 'count_Invoice_c_2', 'min_Q_c_2', 'max_Q_c_2', 'avg_Q_c_2', 'min_spend_c_2', 'max_spend_c_2', 'avg_spend_c_2', 'count_Invoice_p_2', 'min_Q_p_2', 'max_Q_p_2', 'avg_Q_p_2', 'min_spend_p_2', 'max_spend_p_2', 'avg_spend_p_2', 'Spend_Items_2', 'Spend_Postage_2', 'Post_Spend_ratio_2', 'count_Invoice_all_1', 'min_Q_all_1', 'max_Q_all_1', 'avg_Q_all_1', 'min_spend_all_1', 'max_spend_all_1', 'avg_spend_all_1', 'count_Invoice_c_1', 'min_Q_c_1', 'max_Q_c_1', 'avg_Q_c_1', 'min_spend_c_1', 'max_spend_c_1', 'avg_spend_c_1', 'count_Invoice_p_1', 'min_Q_p_1', 'max_Q_p_1', 'avg_Q_p_1', 'min_spend_p_1', 'max_spend_p_1', 'avg_spend_p_1', 'Spend_Items_1', 'Spend_Postage_1', 'Post_Spend_ratio_1', 'next_month_purch'])


for i in range(0,len(start_month)):
    print (y_month[i], start_month[i])
    
    # defining base dataframe
    cust_df = pd.DataFrame()
    cust_df['Customer ID'] = raw_data['Customer ID'].unique()
    
    #calculating y label
    if(y_month[i] != 201111):
        purch_cust = raw_data.loc[(raw_data.month_wise == y_month[i]) & (raw_data.c =='p'),'Customer ID']. unique()
        
        cust_label=pd.DataFrame()
        cust_label['Customer ID'] = purch_cust
        label = [1 for i in purch_cust]
        cust_label['next_month_purch'] = label
    else:
        #separate treatment for month 201111(which will be y_pred month)
        purch_cust = raw_data['Customer ID']. unique()
        cust_label=pd.DataFrame()
        cust_label['Customer ID'] = purch_cust
        label = [1 for i in purch_cust]
        cust_label['next_month_purch'] = label
    
    idx_start_mon = all_months.index(start_month[i])
    
    data_in_range = raw_data[(raw_data['month_wise'] <= start_month[i])] 
    
    for mon in months_diff:
        idx_stop_mon = idx_start_mon - mon
        #print(mon, all_months[idx_start_mon], all_months[idx_stop_mon])
        data_mon_range = data_in_range[(data_in_range.month_wise <= all_months[idx_start_mon]) & (data_in_range.month_wise >= all_months[idx_stop_mon]) ]
        
        cid_gp_by = data_mon_range.groupby(['Customer ID'], as_index = False).agg({'Invoice': ['count'],
            'Quantity' : ['min', 'max', 'mean'],
            'spend' : ['min', 'max', 'mean'],
                })
        cid_gp_by.columns = ['Customer ID','count_Invoice_all','min_Q_all', 'max_Q_all', 'avg_Q_all', 'min_spend_all', 'max_spend_all', 'avg_spend_all' ]    
        cid_gp_by_c = data_mon_range[data_mon_range.c == 'C'].groupby(['Customer ID'], as_index = False).agg({
                'Invoice': ['count'],
                'Quantity' : ['min', 'max', 'mean'],
                'spend' : ['min', 'max', 'mean'],
                })
        cid_gp_by_c.columns = ['Customer ID','count_Invoice_c','min_Q_c', 'max_Q_c', 'avg_Q_c', 'min_spend_c', 'max_spend_c', 'avg_spend_c' ]
            
        cid_gp_by_p = data_mon_range[data_mon_range.c == 'p'].groupby(['Customer ID'], as_index = False).agg({
                'Invoice': ['count'],
                'Quantity' : ['min', 'max', 'mean'],
                'spend' : ['min', 'max', 'mean'],
                })
        cid_gp_by_p.columns = ['Customer ID','count_Invoice_p','min_Q_p', 'max_Q_p', 'avg_Q_p', 'min_spend_p', 'max_spend_p', 'avg_spend_p']

                    
        postage_purch =  data_mon_range.groupby(['Customer ID', 'Postage'], as_index = False)['spend'].agg(sum)

        postage_purch1 = pd.pivot_table(postage_purch, values = ['spend'] , index = ['Customer ID'],columns=['Postage'], aggfunc= sum, fill_value=0)
        postage_purch1.reset_index(inplace = True)
        postage_purch1.columns = ['Customer ID', 'Spend_Items','Spend_Postage']

        postage_purch1['Post_Spend_ratio'] = postage_purch1['Spend_Postage'] / postage_purch1['Spend_Items']
        postage_purch1.loc[postage_purch1.Post_Spend_ratio.isin(['inf', '-inf']),'Post_Spend_ratio'] = 999


        combined_df = pd.merge(cid_gp_by, cid_gp_by_c, how = 'outer', on = ['Customer ID'])
        combined_df = pd.merge(combined_df, cid_gp_by_p, how = 'outer', on = ['Customer ID'] )    
        combined_df = pd.merge(combined_df, postage_purch1, how = 'outer', on = ['Customer ID'])
        
        combined_df.fillna(0, inplace = True)

        colnames = list(combined_df.columns)
        colnames1 = [ x+'_'+str(mon) for x in colnames[1:]]
        colnames2 = ['Customer ID'] + colnames1
        combined_df.columns = colnames2
        
        cust_df = pd.merge(cust_df,combined_df, how = 'outer', on = ['Customer ID'])

    #Mergeing the label for this 18 month period
    cust_df = pd.merge(cust_df,cust_label, how = 'outer', on = ['Customer ID'])

    #Combining rows of different 18 month period
    cust_df_all_cols = cust_df_all_cols.append(cust_df)
    cust_df_all_cols.fillna(0, inplace = True)
    

num_cust = len(raw_data['Customer ID']. unique())
train_val = cust_df_all_cols.head( len(cust_df_all_cols) - num_cust)   
test_data = cust_df_all_cols.tail(num_cust)


# split train_val
X = train_val.iloc[:,:-1]
Y = train_val['next_month_purch']
train_X, val_X, train_Y, val_Y  = train_test_split(X,Y, test_size = .2, random_state = 42,shuffle = True, stratify = Y)

train_X = train_X.iloc[:,1:]
val_X = val_X.iloc[:,1:]
    
class_wt = (len(train_Y)-sum(train_Y)) / sum(train_Y)


model = XGBClassifier(n_estimators = 1000, max_depth  = 2, subsample  = .5, colsample_bytree  = 0.07, scale_pos_weight = class_wt, objective='binary:logistic' ,random_state  =42)
xgb = model.fit(train_X, train_Y)
y_val_pred = model.predict_proba(val_X)
prob_next_mon_purch = y_val_pred[:,1]
#predictions = [round(value) for value in prob_next_mon_purch]


#Calculate and report the best cutoff for Confusion matrix

cutoff = [x/100 for x in range(0,105,5)]
f1 = []
for c in cutoff:
    y_pred_bin = np.where(prob_next_mon_purch < c, 0,1)
    f1_sc = f1_score(val_Y, y_pred_bin)
    f1.append(f1_sc)
cutoff_idx = f1.index(max(f1))    
best_cutoff = cutoff[cutoff_idx]

# evaluate predictions for best cutoff
predictions = np.where(prob_next_mon_purch < best_cutoff, 0,1)

AUC = roc_auc_score(val_Y, prob_next_mon_purch)
print("AUC: %.2f%%" % (AUC * 100.0))

confusion_matrix(val_Y, predictions)
f1_score(val_Y, predictions)
    
# Prediction on latest data
prob_next_mon_purch = model.predict_proba(test_data.iloc[:,1:169])
prob_next_mon_purch = prob_next_mon_purch[:,1]
prediction = pd.DataFrame()
prediction['Customer ID'] = raw_data['Customer ID'].unique()
prediction['Prediction'] = prob_next_mon_purch

prediction.to_csv(r'D:\submission_XGB.csv')
    
    

    



    

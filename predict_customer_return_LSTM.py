import pandas as pd
import numpy as np
#from sklearn.model_selection import train_test_split
from matplotlib import pyplot
from datetime import timedelta
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import LSTM
from sklearn.metrics import accuracy_score, auc,roc_auc_score, precision_recall_fscore_support, confusion_matrix, f1_score




data = pd.read_csv(r'C:\Retail_CaseStudy\Data\Data.csv')


data = data[data['Customer ID'].notnull()]
# Cleaning the data
data['InvoiceDate_dt'] = pd.to_datetime(data.InvoiceDate, format='%Y-%m-%d %H:%M:%S') 
data['InvoiceDate_dt'].head()
#Shifiting time stamp by 9 days, because 
days = timedelta(9)
data['InvoiceDate_dt'] = data['InvoiceDate_dt'] - days

data['month_wise'] = [ (dt.year*100) + (dt.month ) for dt in data.InvoiceDate_dt]
data['day_wise'] = [ (dt.year*10000) + (dt.month *100) + (dt.day) for dt in data.InvoiceDate_dt]

data['spend'] = data.Price * data.Quantity


data['Postage'] = np.where(data.StockCode == 'POST', 'Postage','No_Postage')
data['UK'] = np.where(data.Country == 'United Kingdom', 1,0)

data['c'] = data.Invoice.str[0]
data.loc[data.c != 'C','c'] = 'p'
data_cancel = data[data.c == 'C']
data_purch = data[data.c == 'p']

# creating temporal features

# Num invoices/Customer/month
Num_Invoices1 = data.groupby(['Customer ID','month_wise'], as_index =False)['Invoice'].aggregate(pd.Series.nunique)
Num_Invoices1.rename(columns = {'Invoice':'Count_All_Invoice'}, inplace = True)

Num_Purch_Invoices1 = data_purch.groupby(['Customer ID','month_wise'], as_index =False)['Invoice'].aggregate(pd.Series.nunique)
Num_Purch_Invoices1.rename(columns = {'Invoice':'Count_Purch_Invoice'}, inplace = True)

Num_Cancel_Invoices1 = data_cancel.groupby(['Customer ID','month_wise'], as_index =False)['Invoice'].aggregate(pd.Series.nunique)
Num_Cancel_Invoices1.rename(columns = {'Invoice':'Count_Cancel_Invoice'}, inplace = True)


# avg Invoices in a day /Customer/month
Num_Daily_Invoices = data.groupby(['Customer ID','month_wise','day_wise'], as_index =False)['Invoice'].aggregate(pd.Series.nunique)
Num_Daily_Purch_Invoices = data_purch.groupby(['Customer ID','month_wise','day_wise'], as_index =False)['Invoice'].aggregate(pd.Series.nunique)
Num_Daily_Cancel_Invoices = data_cancel.groupby(['Customer ID','month_wise','day_wise'], as_index =False)['Invoice'].aggregate(pd.Series.nunique)

Num_Daily_Net_Purch_Invoices = pd.merge(Num_Daily_Purch_Invoices, Num_Daily_Cancel_Invoices, how = 'outer', on = ['Customer ID','month_wise','day_wise'], suffixes = ['_Purch','_Cancel'] )

Num_Daily_Net_Purch_Invoices.fillna(0, inplace = True)

Num_Daily_Net_Purch_Invoices['Invoice_Net_Purch'] = Num_Daily_Net_Purch_Invoices.Invoice_Purch - Num_Daily_Net_Purch_Invoices.Invoice_Cancel

Avg_Daily_Invoices = Num_Daily_Net_Purch_Invoices.groupby(['Customer ID','month_wise'], as_index =False).aggregate({'Invoice_Purch':np.mean,'Invoice_Cancel':np.mean,'Invoice_Net_Purch':np.mean})
Avg_Daily_Invoices.rename(columns = {'Invoice_Purch':'Dly_avg_Count_Purch_Invoice', 'Invoice_Cancel' :'Dly_avg_Count_Cancel_Invoice', 'Invoice_Net_Purch' : 'Dly_avg_Count_Net_Purch_Invoice'}, inplace = True)

# Total Spend/month
Tot_Purch_spend = data_purch.groupby(['Customer ID' , 'month_wise'] , as_index =False)['spend'].agg(sum)
Tot_Purch_spend.rename(columns = {'spend':'Tot_Purch_spend'}, inplace = True)

Tot_Cancel_spend =  data_cancel.groupby(['Customer ID' , 'month_wise'] , as_index =False)['spend'].agg(sum)
Tot_Cancel_spend.rename(columns = {'spend':'Tot_Cancel_spend'}, inplace = True)

Tot_Net_spend =  data.groupby(['Customer ID' , 'month_wise'] , as_index =False)['spend'].agg(sum)
Tot_Net_spend.rename(columns = {'spend':'Tot_Net_spend'}, inplace = True)


# Avg Spend/Invoice
Tot_Daily_Purch_spend = data_purch.groupby(['Customer ID','month_wise','day_wise'], as_index =False)['spend'].aggregate(sum)
Tot_Daily_Cancel_spend = data_cancel.groupby(['Customer ID','month_wise','day_wise'], as_index =False)['spend'].aggregate(sum)
Tot_Daily_Net_spend = pd.merge(Tot_Daily_Purch_spend, Tot_Daily_Cancel_spend, how = 'outer', on = ['Customer ID','month_wise','day_wise'], suffixes = ['_Purch','_Cancel'] )

Tot_Daily_Net_spend.fillna(0, inplace = True)

Tot_Daily_Net_spend['Net_spend'] = Tot_Daily_Net_spend.spend_Purch + Tot_Daily_Net_spend.spend_Cancel

Avg_Daily_Net_spend = Tot_Daily_Net_spend.groupby(['Customer ID','month_wise'], as_index =False).aggregate({'spend_Purch':np.mean,'spend_Cancel':np.mean,'Net_spend':np.mean})
Avg_Daily_Net_spend.rename(columns = {'spend_Purch':'Avg_Dly_Purch_Spend', 'spend_Cancel': 'Avg_Dly_Cancel_Spend','Net_spend' : 'Avg_Dly_Net_Spend'}, inplace = True)

# cancel/Purch ratio (month level)
cancel_purch_ratio = Tot_Daily_Net_spend.groupby(['Customer ID','month_wise'], as_index= False).agg({'spend_Purch':sum,'spend_Cancel':sum,'Net_spend':sum})

cancel_purch_ratio['c_p_ratio'] = np.abs(cancel_purch_ratio.spend_Cancel / cancel_purch_ratio.spend_Purch)

cancel_purch_ratio.loc[cancel_purch_ratio.c_p_ratio.isin(['inf']),'c_p_ratio'] = 0
cancel_purch_ratio.rename(columns = {'spend_Purch':'Tot_Purch_spend', 'spend_Cancel':'Tot_Cancel_spend', 'Net_spend':'Tot_Net_spend'}, inplace = True)


# postage/spend in Purchases
postage_purch =  data_purch.groupby(['Customer ID', 'month_wise', 'Postage'], as_index = False)['spend'].agg(sum)

postage_purch1 = pd.pivot_table(postage_purch, values = ['spend'] , index = ['Customer ID', 'month_wise'],columns=['Postage'], aggfunc= sum, fill_value=0)
postage_purch1.reset_index(inplace = True)
postage_purch1.columns = ['Customer ID', 'month_wise', 'Spend_Items','Spend_Postage']

postage_purch1['Post_Spend_ratio'] = postage_purch1['Spend_Postage'] / postage_purch1['Spend_Items']

postage_purch1.loc[postage_purch1.Post_Spend_ratio.isin(['inf']),'Post_Spend_ratio'] = 0


# postage/spend in cancellations
postage_cancel =  data_cancel.groupby(['Customer ID', 'month_wise', 'Postage'], as_index = False)['spend'].agg(sum)

postage_cancel1 = pd.pivot_table(postage_cancel, values = ['spend'] , index = ['Customer ID', 'month_wise'],columns=['Postage'], aggfunc= sum, fill_value=0)
postage_cancel1.reset_index(inplace = True)
postage_cancel1.columns = ['Customer ID', 'month_wise', 'Spend_Items','Spend_Postage']
postage_cancel1['Post_Spend_ratio'] = np.abs(postage_cancel1['Spend_Postage'] / postage_cancel1['Spend_Items'])

postage_cancel1.loc[postage_cancel1.Post_Spend_ratio.isin(['inf']),'Post_Spend_ratio'] = 0

# Number of items in each invoice
Tot_Invoice_Purch_items = data_purch.groupby(['Customer ID','month_wise','Invoice'], as_index =False)['Quantity'].aggregate(sum)
Tot_Invoice_Cancel_items = data_cancel.groupby(['Customer ID','month_wise','Invoice'], as_index =False)['Quantity'].aggregate(sum)
Tot_Invoice_Net_items = pd.merge(Tot_Invoice_Purch_items, Tot_Invoice_Cancel_items, how = 'outer', on = ['Customer ID','month_wise','Invoice'], suffixes = ['_Purch','_Cancel'] )

Tot_Invoice_Net_items.fillna(0, inplace = True)

Tot_Invoice_Net_items['Net_Quantity'] = Tot_Invoice_Net_items.Quantity_Purch + Tot_Invoice_Net_items.Quantity_Cancel

Avg_Invoice_Net_items = Tot_Invoice_Net_items.groupby(['Customer ID','month_wise'], as_index =False).aggregate({'Quantity_Purch':np.mean,'Quantity_Cancel':np.mean,'Net_Quantity':np.mean})
Avg_Invoice_Net_items.rename(columns = {'Quantity_Purch':'Avg_Dly_Purch_Quantity', 'Quantity_Cancel': 'Avg_Dly_Cancel_Quantity','Net_Quantity' : 'Avg_Dly_Net_Quantity'}, inplace = True)

Avg_Invoice_Net_items['Items_retention_ratio'] = Avg_Invoice_Net_items['Avg_Dly_Net_Quantity'] / Avg_Invoice_Net_items['Avg_Dly_Purch_Quantity']

Avg_Invoice_Net_items.loc[Avg_Invoice_Net_items.Items_retention_ratio.isin(['inf', '-inf']),'Items_retention_ratio'] = 0



#creating a DF that defines the structure of input data

cust = list(set(list(data['Customer ID'].unique())))
months = data['month_wise'].unique()
month_wise = []
for i in range(0, len(cust)):
    for j in months:
        month_wise.append(j)

cust1 = [c for c in cust for k in range(0, len(months))]
cust_month = pd.DataFrame({'Customer ID': cust1, 'month_wise':month_wise})

#calculating months since last purchase


cust_month['month_last_purch'] = 190001

for cust in tqdm(cust):
    purch_month_list = list(Num_Purch_Invoices1.loc[Num_Purch_Invoices1['Customer ID'] == cust,'month_wise'])
    
    purch_month_list.append(201111)
    for i in range(len(purch_month_list)-1):
        #print (i)
        start = purch_month_list[i]
        end = purch_month_list[i+1]
        #print("range = >=", start,"<",end)        
        
        cust_month.loc[(cust_month['Customer ID'] == cust) & (cust_month.month_wise >= start) & (cust_month.month_wise < end), 'month_last_purch'] = start

cust_month['month_wise1'] = [str(x)[0:4]+str('-') + str(x)[4:6]+str('-')+str('01') for x in cust_month.month_wise]
        
cust_month['month_last_purch1'] = [str(x)[0:4]+str('-') + str(x)[4:6]+str('-')+str('01') for x in cust_month.month_last_purch]

cust_month['month_wise1'] = pd.to_datetime(cust_month['month_wise1'])        
cust_month['month_last_purch1'] = pd.to_datetime(cust_month['month_last_purch1'])    

cust_month['months_since_last_purch'] = cust_month.month_wise1 - cust_month.month_last_purch1

cust_month['months_since_last_purch1'] = [x.days/30 for x in cust_month.months_since_last_purch]
cust_month['months_since_last_purch1'] = cust_month['months_since_last_purch1'].round(0)
cust_month.loc[cust_month.months_since_last_purch1> 100, 'months_since_last_purch1'] = 999
   
cust_month = cust_month[['Customer ID', 'month_wise', 'months_since_last_purch1']]

cust_month.columns = ['Customer ID', 'month_wise', 'months_since_last_purch']


#Merging the data
combined_df = pd.merge(Num_Invoices1, Num_Purch_Invoices1,how = 'outer', left_on = ['Customer ID', 'month_wise'] , right_on = ['Customer ID', 'month_wise'] )
combined_df = pd.merge(combined_df, Num_Cancel_Invoices1,how = 'outer', left_on = ['Customer ID', 'month_wise'] , right_on = ['Customer ID', 'month_wise'] )
combined_df = pd.merge(cust_month , combined_df, how = 'outer', left_on = ['Customer ID', 'month_wise'] , right_on = ['Customer ID', 'month_wise'] )

combined_df['Count_Net_Purch_Invoice'] = combined_df.Count_Purch_Invoice - combined_df.Count_Cancel_Invoice

#merging new feature set 2 (Daily average counts)
combined_df = pd.merge(combined_df, Avg_Daily_Invoices ,how = 'outer', left_on = ['Customer ID', 'month_wise'] , right_on = ['Customer ID', 'month_wise'])

#merging new feature set 3 (Spend Variables)
combined_df = pd.merge(combined_df, Avg_Daily_Net_spend ,how = 'outer', left_on = ['Customer ID', 'month_wise'] , right_on = ['Customer ID', 'month_wise'] )

#Merging feature set 4(Total spend and cancel : spend ratio)
combined_df = pd.merge(combined_df, cancel_purch_ratio ,how = 'outer', left_on = ['Customer ID', 'month_wise'] , right_on = ['Customer ID', 'month_wise'] )

#merging new feature set 5 (Postage Variables)
combined_df = pd.merge(combined_df, postage_purch1 ,how = 'outer', left_on = ['Customer ID', 'month_wise'] , right_on = ['Customer ID', 'month_wise'] )

#merging new feature set 6 (Postage on cancellations)
combined_df = pd.merge(combined_df, postage_cancel1 ,how = 'outer', left_on = ['Customer ID', 'month_wise'] , right_on = ['Customer ID', 'month_wise'], suffixes = ['_on_Purch','_on_Cancel'])

#merging new feature set 7 (Avg_Invoice_Net_items)
combined_df = pd.merge(combined_df, Avg_Invoice_Net_items ,how = 'outer', left_on = ['Customer ID', 'month_wise'] , right_on = ['Customer ID', 'month_wise'])

#Filling all NA with 0
combined_df.fillna(0, inplace =True)

#removing Data of 1st month (as it has only 9 days of Data)
min_month = combined_df.month_wise.unique()[0]
combined_df = combined_df[combined_df.month_wise != min_month]

#Standardization of the Values

scaler = StandardScaler()
#Excluding last 2 months of data from train data
month_filter = combined_df.month_wise.unique()[-2]

train_data_subset = combined_df.loc[combined_df.month_wise < month_filter]

train_data_subset = train_data_subset.iloc[:,2:]


#Fit only on the data to be used for training
scaler.fit(train_data_subset)

#Transform the entire dataset using parameters calculated on Train data
all_data_scaled = scaler.transform(combined_df.iloc[:,2:])


#LSTM Takes input in the form of Samples, timesteps, features
num_samples =  len(combined_df['Customer ID'].unique())
time_steps = len(combined_df['month_wise'].unique())
num_variables = combined_df.shape[1]-2

#Converting DF to array
#arr = combined_df.iloc[:,-num_variables:].values.reshape((num_samples, time_steps, num_variables))
arr = all_data_scaled.reshape((num_samples, time_steps, num_variables))



#Creating train_Y variable - Label
data_train = data[data.month_wise == 201109]
data_train_purch = data_train[data_train.c == 'p']

# all customer who shopped in the last month are considered as class 1 for testing
cust_list_train = purch_cust = list(data_train_purch['Customer ID'].unique())
label = [1 for k in range(0, len(purch_cust))]

all_cust = list(set(list(data['Customer ID'].unique())))

#remaining are considered as class 0
Remaining_cust = list(set(all_cust) - set(purch_cust))
label_0 = [0 for k in range(0, len(Remaining_cust))]

cust_list_train.extend(Remaining_cust)
label.extend(label_0)

cust_label_train = pd.DataFrame({'Customer ID' : cust_list_train, 'label':label})
cust_label_train.sort_values(by = 'Customer ID', axis = 0, ascending = True, inplace = True)


#Creating test_Y variable - Label
data_test = data[data.month_wise == 201110]
data_test_purch = data_test[data_test.c == 'p']

# all customer who shopped in the last month are considered as class 1 for testing
cust_list_test = purch_cust = list(data_test_purch['Customer ID'].unique())
label = [1 for k in range(0, len(purch_cust))]

all_cust = list(set(list(data['Customer ID'].unique())))

#remaining are considered as class 0
Remaining_cust = list(set(all_cust) - set(purch_cust))
label_0 = [0 for k in range(0, len(Remaining_cust))]

cust_list_test.extend(Remaining_cust)
label.extend(label_0)

cust_label_test = pd.DataFrame({'Customer ID' : cust_list_test, 'label':label})
cust_label_test.sort_values(by = 'Customer ID', axis = 0, ascending = True, inplace = True)

#Defining Train and test
#Training on data from month 1 to month 22, Predicting on month 23
#Testing on data from month 2 to month 23, predicting on month 24

train_X = arr[:,0:21,:]
train_Y = np.array(cust_label_train.label)
val_X = arr[:,1:22,:]
val_Y =  np.array(cust_label_test.label)

#Designing the network

# design network
model = Sequential()
model.add(LSTM(20, input_shape=(train_X.shape[1], train_X.shape[2])))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['AUC'])
# fit network
history = model.fit(train_X, train_Y, epochs=7, batch_size=72, validation_data=(val_X, val_Y), verbose=2, shuffle=True)

# plot history of AUC
pyplot.plot(history.history['auc'], label='train')
pyplot.plot(history.history['val_auc'], label='test')
pyplot.legend()
pyplot.show()


# plot history
pyplot.plot(history.history['loss'], label='train')
pyplot.plot(history.history['val_loss'], label='test')
pyplot.legend()
pyplot.show()



prob_next_mon_purch = model.predict(val_X)

val_Y_pred = prob_next_mon_purch.round()
accuracy = accuracy_score(val_Y, val_Y_pred)
print("Accuracy: %.2f%%" % (accuracy * 100.0))

AUC = roc_auc_score(val_Y, prob_next_mon_purch)
print("AUC: %.2f%%" % (AUC * 100.0))

confusion_matrix(val_Y, val_Y_pred)
precision_recall_fscore_support(val_Y, val_Y_pred)
    

cutoff = [x/100 for x in range(0,105,5)]
f1 = []
for c in cutoff:
    val_Y_pred = np.where(prob_next_mon_purch < c, 0,1)
    f1_sc = f1_score(val_Y, val_Y_pred)
    f1.append(f1_sc)
cutoff_idx = f1.index(max(f1))    
best_cutoff = cutoff[cutoff_idx]

# evaluate predictions for best cutoff
val_Y_pred = np.where(prob_next_mon_purch < best_cutoff, 0,1)

AUC = roc_auc_score(val_Y, prob_next_mon_purch) #AUC Remains same
print("AUC: %.2f%%" % (AUC * 100.0))

confusion_matrix(val_Y, val_Y_pred)
f1_score(val_Y, val_Y_pred)
 

# Latest_data for prediction
final_test_X = arr[:,2:23,:]
prob_next_mon_purch = model.predict(final_test_X)
   
prediction = pd.DataFrame()
prediction['Customer ID'] = data['Customer ID'].unique()
prediction['Prediction'] = prob_next_mon_purch

prediction.to_csv(r'C:\Retail_CaseStudy\submission.csv')
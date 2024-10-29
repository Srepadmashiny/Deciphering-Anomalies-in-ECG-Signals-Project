#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ {
# Name             :    signaks_Anomaly_Trainer_HB
# Authour          :    K. SrePadmashiny
# Reviewer         :    Signals Professor
# Date             :    29-Oct-2023
# Purpose          :    Signals Anomoly Detection Training
# Data             :    ECG Data
#                  :    
#                  :    
# Change History   :    
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Library s Begins ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
import pandas as pd 
import numpy as np
import pickle
import seaborn as sns
import matplotlib.pyplot as plt


from pyod.models.hbos import HBOS
from pyod.utils.data import generate_data
from pyod.utils.data import evaluate_print
from pyod.utils.example import visualize

from pyod.models.cblof import CBLOF


#Read Data
df=pd.read_csv("C:\\signals\\data\\\ecg.csv")

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
#                                        * 01 Variable Identification  *
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
df_numerics_only = df.select_dtypes(include=[np.number]) #Numeric Variable Identification
NumDataType = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
df_categorical_only = df.select_dtypes(exclude=NumDataType) #Categorical Variable Identification

print(df_numerics_only.columns)

print(df_categorical_only.columns)

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
#                                        * 02 Data Approach  *
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
print()
print("Data Set (Rows, Cols)")
print(". . . . . . . . . . . ")
print(df.shape)
print()

print("df_numerics_only (Rows, Cols)")
print(". . . . . . . . . . . . . . .")
print(df_numerics_only.shape)
print()

print("df_categorical_only (Rows, Cols)")
print(". . . . . . . . . . . . . . .")
print(df_categorical_only.shape)
print()

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
#                                        * 04 Bi-variate Analysis  *
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
def get_redundant_pairs(df):
    '''Get diagonal and lower triangular pairs of correlation matrix'''
    pairs_to_drop = set()
    cols = df.columns
    for i in range(0, df.shape[1]):
        for j in range(0, i+1):
            pairs_to_drop.add((cols[i], cols[j]))
    return pairs_to_drop

def get_top_abs_correlations(df, n=5):
    au_corr = df.corr().abs().unstack()
    labels_to_drop = get_redundant_pairs(df)
    au_corr = au_corr.drop(labels=labels_to_drop).sort_values(ascending=False)
    return au_corr[0:n]

#Continous and Continous
Corr_Num_Matix = df_numerics_only.corr().round(3)
print("~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~~ ~ ~ ~ ~ ")
print(Corr_Num_Matix)
print("~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~~ ~ ~ ~ ~ ")
print(". . . . . . . . . . . . . . . . . . Top Most Correlations . . . . . . . . . . . . . . . . . ")
print(get_top_abs_correlations(df_numerics_only, 150))
print()

# #Categorical and Target(Categorical)
# for column in list(df_categorical_only):
#     print(column)
#     grouped = dataset.groupby(['SFPA_color', column])
#     print(grouped.size())

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
#                                        * 03 Univariate Analysis  *
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
print(". . . . . . . . . . . . . . . . . . Central Tendancy Summary . . . . . . . . . . . . . . . . . ")
print(df_numerics_only.describe().transpose())
tmp= df_numerics_only.describe().transpose()
print()

print(". . . . . . . . . . . . . . . . . . Numerical  Missing Values . . . . . . . . . . . . . . . . . .")
print(len(df_numerics_only) - df_numerics_only.count())
print()

print(". . . . . . . . . . . . . . . . . . Categorical Missing Values . . . . . . . . . . . . . . . . . ")
print(len(df_categorical_only) - df_categorical_only.count())
print()


#  Numerical Variable Analysis`  *
for column in list(df_numerics_only):
    print(column)
    sns.violinplot(df_numerics_only[column],  linewidth=5, orient='h')
    #sns.distplot(df_numerics_only[column])
    plt.show(6)

#  Categorical Variable Analysis`  *
df_categorical_only.fillna("NaN") #correct Missing Values for Distribution view

print(". . . . . . . . . . . . . . . . . . Frequency Table . . . . . . . . . . . . . . . . . ")
print(df_categorical_only.apply(lambda x: x.value_counts()).T.stack())

# Distribution Plot
for column in list(df_categorical_only):
    sns.countplot(x=column, data=df_categorical_only)
    plt.show(10)



#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
#                                    *  Train / Test Split  *
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
train,test = df[1:4500], df[4501:4996]
print(train.shape)
print(test.shape)

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
#                                    *  Model Building *
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
SignalAnomolyModel = HBOS( contamination =.35)

SignalAnomolyModel.fit(train)
pred = SignalAnomolyModel.predict(test)



test['anomaly']=pred
outliers=test.loc[test['anomaly']==-1]
outlier_index=list(outliers.index)

#Find the number of anomalies and normal points here points classified -1 are anomalous
print(test['anomaly'].value_counts())

pickle.dump(SignalAnomolyModel,open("C:\\signals\\data\\\ecg.pickle", 'wb'))
print("Training Finished!")
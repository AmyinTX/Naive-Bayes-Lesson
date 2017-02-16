# -*- coding: utf-8 -*-
"""
Created on Mon Feb 13 09:05:09 2017

@author: abrown09
"""
#%% import packages
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from collections import Counter
from sklearn.naive_bayes import GaussianNB
from sklearn import metrics

#%% read in data
df = pd.read_csv('/Users/amybrown/Thinkful/Unit_4/Lesson_3/curric-data-001-data-sets/ideal-weight/ideal_weight.csv')

#%% clean data
df.columns = [c_name.replace("'", '') for c_name in df.columns.values.tolist()] # this worked! use this for previous lesson cleaning. removes quotes from column headers
df['sex'] = df['sex'].str.replace("'", '') # remove quotes from values in sex column

#%% plot actual v ideal weights
actual = df['actual']
ideal = df['ideal']

# may want to mess with binning, etc. 
plt.hist(actual, alpha=0.5, label='Actual', bins=20)
plt.hist(ideal,  alpha=0.5, label='Ideal', bins=20)
plt.legend(loc='upper right')
plt.show()

#%% plot diff var
diff = df['diff']
plt.hist(diff, alpha=0.5, bins=20)
plt.show()

#%% map sex to a categorical variable
df_dummy = pd.get_dummies(df['sex'])
df = pd.concat([df, df_dummy], axis=1) #note: only use either Female or Male in analysis

#%% are there more males or females in the dataset?
freq = print(Counter(df['sex']))
# there are 56 more females in the dataset than males

#%% split data into outcome and predictors
final_df = df[['actual', 'ideal', 'diff']]
class_df = df[['Male']]

#%% Naive Bayes Classifier ###

model = GaussianNB()
model.fit(final_df, class_df.values.ravel())
print(model)
expected = class_df.values.ravel()
predicted = model.predict(final_df)

print(metrics.classification_report(expected, predicted))
print(metrics.confusion_matrix(expected, predicted))

(expected==predicted).all() # checks whether the arrays are equivalent, which they are not
check = np.equal(expected, predicted)
mislabled = print(Counter(check)) # 14 points were mislabeled

print(model.predict([[145, 160, -15]])) # predicts male
print(model.predict([[160, 145, 15]])) # predicts female
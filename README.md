# Breast_Cancer

## Importing Library

import pandas as pd

import numpy as np

## Loading breast cancer dataset

df = pd.read_csv('breast-cancer.csv',header=None)

df.columns = ['Class',
'age',
'menopause',
'tumor-size',
'inv-nodes',
'node-caps',
'deg-malig',
'breast',
'breast-quad',
'irradiat']



#1.Class: no-recurrence-events, recurrence-events
#2. age: 10-19, 20-29, 30-39, 40-49, 50-59, 60-69, 70-79, 80-89, 90-99.
#3. menopause: lt40, ge40, premeno.
#4. tumor-size: 0-4, 5-9, 10-14, 15-19, 20-24, 25-29, 30-34, 35-39, 40-44, 45-49, 50-54, 55-59.
#5. inv-nodes: 0-2, 3-5, 6-8, 9-11, 12-14, 15-17, 18-20, 21-23, 24-26, 27-29, 30-32, 33-35, 36-39.
#6. node-caps: yes, no.
#7. deg-malig: 1, 2, 3.
#8. breast: left, right.
#9. breast-quad: left-up, left-low, right-up, right-low, central.
#10. irradiat: yes, no.

## Data View

df.head()

df.info()

## Data Organization

df['node-caps'].value_counts()

df['node-caps'].mask(df['node-caps']=='?',1, inplace=True)

df.loc[df['Class']=='no-recurrence-events', "Class"] = 0
df.loc[df['Class']=='recurrence-events', "Class"] = 1

df.loc[df['age']=='20-29','age'] = 0
df.loc[df['age']=='30-39','age'] = 1
df.loc[df['age']=='40-49','age'] = 2
df.loc[df['age']=='50-59','age'] = 3
df.loc[df['age']=='60-69','age'] = 4
df.loc[df['age']=='70-79','age'] = 5

df.loc[df['menopause']=='premeno','menopause'] = 0
df.loc[df['menopause']=='ge40','menopause'] = 1
df.loc[df['menopause']=='lt40','menopause'] = 2

df.loc[df['tumor-size']=='0-4','tumor-size'] = 0
df.loc[df['tumor-size']=='5-9','tumor-size'] = 1
df.loc[df['tumor-size']=='10-14','tumor-size'] = 2
df.loc[df['tumor-size']=='15-19','tumor-size'] = 3
df.loc[df['tumor-size']=='20-24','tumor-size'] = 4
df.loc[df['tumor-size']=='25-29','tumor-size'] = 5
df.loc[df['tumor-size']=='30-34','tumor-size'] = 6
df.loc[df['tumor-size']=='35-39','tumor-size'] = 7
df.loc[df['tumor-size']=='40-44','tumor-size'] = 8
df.loc[df['tumor-size']=='45-49','tumor-size'] = 9
df.loc[df['tumor-size']=='50-54','tumor-size'] = 10

df.loc[df['inv-nodes'] == '0-2','inv-nodes'] = 0
df.loc[df['inv-nodes'] == '3-5','inv-nodes'] = 1
df.loc[df['inv-nodes'] == '6-8','inv-nodes'] = 2
df.loc[df['inv-nodes'] == '9-11','inv-nodes'] = 3
df.loc[df['inv-nodes'] == '12-14','inv-nodes'] = 4
df.loc[df['inv-nodes'] == '15-17','inv-nodes'] = 5
df.loc[df['inv-nodes'] == '24-26','inv-nodes'] = 6

df.loc[df['node-caps']=='no','node-caps'] = 0
df.loc[df['node-caps']=='yes','node-caps'] = 1
df.loc[df['node-caps']=='?','node-caps'] = 2

df.loc[df['breast']=='left','breast'] = 0
df.loc[df['breast']=='right','breast'] = 1

df['breast-quad'].mask(df['breast-quad']=='?',0, inplace=True)
df.loc[df['breast-quad']=='left_low','breast-quad'] = 0
df.loc[df['breast-quad']=='left_up','breast-quad'] = 1
df.loc[df['breast-quad']=='right_up','breast-quad'] = 2
df.loc[df['breast-quad']=='right_low','breast-quad'] = 3
df.loc[df['breast-quad']=='central','breast-quad'] = 4

df.loc[df['irradiat']=='no','irradiat'] = 0
df.loc[df['irradiat']=='yes','irradiat'] = 1

df.head()

df['irradiat'].value_counts()

df.info()

## X and y differentiate

y = df.iloc[:,9]
y = y.astype('int64')
y

X = df.iloc[:,[0,2,6]]
X = X.astype('int64')
X

## Importing sklearn

from sklearn.model_selection import train_test_split

**Test Train Split**

X_train, X_test, Y_train, Y_test = train_test_split(X, y, random_state=0)

**Importing and Running DecisionTreeClassifier**

from sklearn import tree
from sklearn.tree import DecisionTreeClassifier

clf = tree.DecisionTreeClassifier(max_depth = 5, 
                             random_state = 0)

clf = clf.fit(X_train, Y_train)

tree.plot_tree(clf)

**Decision Tree Visalization with matplotlib**

import matplotlib.pyplot as plt

fn=['Class', 'menopause', 'deg-malig']
cn=['yes','no']
fig, axes = plt.subplots(nrows = 1,ncols = 1,figsize = (4,4), dpi=300)
tree.plot_tree(clf,
               feature_names = fn, 
               class_names=cn,
               filled = True
              )


clf.predict(X_test)

## IRIS DATA for Checking

#from sklearn.datasets import load_iris

#iris = load_iris()

#X, y = iris.data, iris.target

#clf = tree.DecisionTreeClassifier()

#clf = clf.fit(X, y)

#tree.plot_tree(clf)

#fn=['sepal length (cm)','sepal width (cm)','petal length (cm)','petal width (cm)']
#cn=['setosa', 'versicolor', 'virginica']
#fig, axes = plt.subplots(nrows = 1,ncols = 1,figsize = (4,4), dpi=300)
#tree.plot_tree(clf,
 #              feature_names = fn, 
 #              class_names=cn,
 #              filled = True
 #             )




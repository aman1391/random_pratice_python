# -*- coding: utf-8 -*-
"""
Created on Wed Jan 25 13:14:03 2017

@author: Admin
"""

 # -*- coding: utf-8 -*-
"""
Created on Wed Jan 25 12:20:05 2017

@author: Admin
"""


import os
import numpy as np
import pandas as pd
from sklearn import tree
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import normalize
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.tree import DecisionTreeClassifier
import matplotlib.pyplot as plt


##good to change the direcctory
os.chdir("C:\\Users\\Admin\\Desktop\\anchal's folder")

data_dir=os.getcwd()

X_train_path=os.path.join(data_dir,'train.csv')
X_test_path=os.path.join(data_dir,'test.csv')


###use pandas
X_train=pd.read_csv(X_train_path)

X_test=pd.read_csv(X_test_path)

X_train.head(5)

####joining the training and test data together to avoid confusion
X_All=pd.concat([X_train,X_test],axis=0)

###to check how our data exactly looks like
X_All.describe( )


##this is to check the behaviour of the data 
X_All[X_All.dtypes[(X_All.dtypes=="float64")|(X_All.dtypes=="int64")].index.values].hist(figsize=[30,30])

##the observation is that most of values lies in the 0th range , i.e. the variable name , which is quite odd
##the data is highly distored i.e. the data throws various similar values 
## now remove the columns which has a unique values  

###deleting the constant column
X_All = X_All.loc[:,X_All.apply(pd.Series.nunique) !=1]


###ploting of more variable and the variable with the 



y=X_All.dtypes[X_All.dtypes=="object"]

y=['protocol_type','service','flag','class']

######plotting the graphs
X_All.protocol_type.value_counts().plot(kind='bar')
X_All.service.value_counts().plot(kind='bar') ###this looks skewed 
X_All.flag.value_counts().plot(kind='bar') ###multiple levels SF has the level with high number of entries

##now see some of the missing values

X_All.isnull().sum()  ###this was quite observant in the first place when we plot the graphs


####now the modeling part
from sklearn.ensemble import RandomForestClassifier


lb= LabelEncoder()

for y_1 in y:
    X_All[y_1]=lb.fit_transform(X_All[y_1])
    

    
X_All.head()

#######building train and test
X_train_1=X_All[0:len(X_train)]
X_test_1=X_All[len(X_train)::]


target=X_train_1['class']


X_train_1=X_train_1.drop('class',axis=1)
X_test_1=X_test_1.drop('class',axis=1)


###randomForest Model

model=RandomForestClassifier(n_estimators=500,max_depth=10)
model=model.fit(X_train_1,target)
model.score(X_train_1,target)
out=model.predict(X_test_1)
out
#visualize the feature importances curve
importances = model.feature_importances_
std = np.std([tree.feature_importances_ for tree in model.estimators_],
             axis=0)
indices = np.argsort(importances)[::-1]

# Print the feature ranking
print("Feature ranking:")

for f in range(X_train_1.shape[1]):
    print("%d. feature %d (%f)" % (f + 1, indices[f], importances[indices[f]]))

# Plot the feature importances of the forest
plt.figure()
plt.title("Feature importances")
plt.bar(range(X_train_1.shape[1]), importances[indices],
       color="r", yerr=std[indices], align="center")
plt.xticks(range(X_train_1.shape[1]), indices)
plt.xlim([-1, X_train_1.shape[1]])
plt.show()
####decisiontree 
clf = DecisionTreeClassifier()
clf = clf.fit(X_train_1, target)
clf.score(X_train_1,target)


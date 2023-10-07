import pandas as pd 
import numpy as np
import seaborn as sns 
from sklearn.datasets import make_classification
from sklearn.preprocessing import MinMaxScaler,StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import *
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from joblib import *

#---------------create data-----------------#
X,y=make_classification(n_classes=2,n_samples=15000,n_features=5)

#-----------split data-------------#
x_train,x_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)

#---------------scale data---------------#
scale=MinMaxScaler()
x_train=scale.fit_transform(x_train)
x_test=scale.transform(x_test)

#--------------fit model-------------#
rf=RandomForestClassifier(max_depth=4)
rf.fit(x_train,y_train)

#---------------save model----------#
dump(rf,'rf_model.pkl')
#report train score
train_score=rf.score(x_train,y_train)*100

#report test score
test_score=rf.score(x_test,y_test)*100

# Write scores to a file
with open("metrics.txt", 'w') as outfile:
        outfile.write("Training variance explained: %2.1f%%\n" % train_score)
        outfile.write("Test variance explained: %2.1f%%\n" % test_score)

##########################################
##### PLOT FEATURE IMPORTANCE ############
##########################################
# Calculate feature importance in random forest
importances = rf.feature_importances_
labels = ['a','b','c','d','e']
feature_df = pd.DataFrame(list(zip(labels, importances)), columns = ["feature","importance"])
feature_df = feature_df.sort_values(by='importance', ascending=False,)

#image formatting
axis_fs = 18 #fontsize
title_fs = 22 #fontsize
sns.set(style="whitegrid")

ax = sns.barplot(x="importance", y="feature", data=feature_df)
ax.set_xlabel('Importance',fontsize = axis_fs) 
ax.set_ylabel('Feature', fontsize = axis_fs)#ylabel
ax.set_title('Random forest\nfeature importance', fontsize = title_fs)

plt.tight_layout()
plt.savefig("feature_importance.png",dpi=120) 
plt.close()


##########################################
############ PLOT RESIDUALS  #############
##########################################

y_pred = rf.predict(x_test) + np.random.normal(0,0.25,len(y_test))
y_jitter = y_test + np.random.normal(0,0.25,len(y_test))
res_df = pd.DataFrame(list(zip(y_jitter,y_pred)), columns = ["true","pred"])

ax = sns.scatterplot(x="true", y="pred",data=res_df)
ax.set_aspect('equal')
ax.set_xlabel('True wine quality',fontsize = axis_fs) 
ax.set_ylabel('Predicted wine quality', fontsize = axis_fs)#ylabel
ax.set_title('Residuals', fontsize = title_fs)

# Make it pretty- square aspect ratio
ax.plot([1, 10], [1, 10], 'black', linewidth=1)
plt.ylim((2.5,8.5))
plt.xlim((2.5,8.5))

plt.tight_layout()
plt.savefig("residuals.png",dpi=120) 


#----------load model--------#
from joblib import *
model=load('rf_model.pkl')

#---------get data for inference-----------#
from sklearn.datasets import make_classification
X,y=make_classification(n_classes=2,n_samples=15000,n_features=5)

y_pred=model.predict(X)
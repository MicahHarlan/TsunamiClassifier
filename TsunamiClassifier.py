import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split,GridSearchCV
from sklearn.linear_model import LinearRegression
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder
from sklearn.preprocessing import KBinsDiscretizer, Binarizer
from sklearn.preprocessing import LabelBinarizer, MultiLabelBinarizer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder,PolynomialFeatures
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
from sklearn.decomposition import PCA
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import cross_val_score
from sklearn.svm import LinearSVC,SVC
import numpy as np

#CLEANING THE DATA.
print("This program Compares a Logistic Regresson to a SVC on Tsunami Data")

np.set_printoptions(suppress=True)
standardizer = StandardScaler(with_mean=True, with_std=True)
le = LabelEncoder()
data2 = pd.read_csv("tsunami.csv")
disaster = pd.DataFrame()
data2 = data2.drop(0)
disaster['Year'] = data2['Year'] 
disaster['Validity'] = data2['Tsunami Event Validity']
data2['Latitude'] = pd.to_numeric(data2['Latitude'], errors='coerce').fillna(0)
disaster['Latitude'] = data2['Latitude']
data2['Longitude'] = pd.to_numeric(data2['Longitude'], errors='coerce').fillna(0)
disaster['Longitude'] = data2['Longitude']
data2['Total Deaths'] = pd.to_numeric(data2['Total Deaths'], errors='coerce').fillna(0)
disaster['Deaths'] = data2['Total Deaths']
data2['Maximum Water Height (m)'] = pd.to_numeric(data2['Maximum Water Height (m)'], errors='coerce').fillna(0)
disaster['Max Height'] = data2['Maximum Water Height (m)']
#data['Total Damage ($Mil)'] = pd.to_numeric(data['Total Damage ($Mil)'], errors='coerce').fillna(0)
#disaster['Cost'] = data['Total Damage ($Mil)']
data2['Total Houses Destroyed'] = pd.to_numeric(data2['Total Houses Destroyed'], errors='coerce').fillna(0)
disaster['Houses Destroyed'] = data2['Total Houses Destroyed']
disaster['Country'] = data2['Country']
disaster['Location'] = data2['Location Name']
disaster = disaster.reset_index()
disaster = disaster.dropna()
###############TSUNAMI############################

## NORMALIZING VALIDITY
norm = MinMaxScaler(feature_range=(0,1))
norm_valid = norm.fit_transform(disaster['Validity'].to_numpy().reshape(len(disaster['Validity']),1))
#NORMALIZING HOUSESDESTROYED and DEATHS
norm_houses = norm.fit_transform(disaster['Houses Destroyed'].to_numpy().reshape(len(disaster['Houses Destroyed']),1))
norm_deaths =  norm.fit_transform(disaster['Deaths'].to_numpy().reshape(len(disaster['Deaths']),1))
##STANDARDIZE WAVE HEIGHT
height_stand = disaster['Max Height'].to_numpy().reshape(len(disaster['Max Height']),1)
stand_wave = standardizer.fit_transform(height_stand)
#Years
norm_year = norm.fit_transform(disaster['Year'].to_numpy().reshape(len(disaster['Year']),1))
norm_height = norm.fit_transform(disaster['Max Height'].to_numpy().reshape(len(disaster['Max Height']),1))

stand_deaths =  disaster['Deaths'].to_numpy().reshape(len(disaster['Deaths']),1)
stand_d =  standardizer.fit_transform(stand_deaths)

stand_houses = disaster['Houses Destroyed'].to_numpy().reshape(len(disaster['Houses Destroyed']),1)
stand_house = standardizer.fit_transform(stand_houses)

lat = disaster['Latitude'].to_numpy().reshape(len(disaster['Validity']),1)
long = disaster['Longitude'].to_numpy().reshape(len(disaster['Longitude']),1)

#MOVE THIS DOWN

country = []
for i in disaster['Country']:
	if(i=="USA"):
		country.append(1)


	else:
		country.append(0)
		
disaster['Country'] = country




#LOGISTIC REGRESSION FOR COUNTRIES
lr = LogisticRegression(max_iter=99999)
cities = le.fit_transform(disaster['Location'])
country = le.fit_transform(disaster['Country'])
x = np.c_[norm_deaths,norm_height,cities,norm_valid,norm_houses] 
y = country 



############################################################
num_pcs = 4
pca = PCA(n_components=num_pcs)
pcs = pca.fit_transform(x)
p = make_pipeline(pca, lr)
p_scores = cross_val_score(p,x,y,cv=2)
print(f"mean score using logistic regression (pca{num_pcs}):  {p_scores.mean():.4f}")
############################################################
Xtrain, Xtest, ytrain, ytest = train_test_split(x,y,train_size=.2,shuffle=True)
p.fit(Xtrain,ytrain)
print("Tsunami Location: Logistic Regression Confusion Matrix")
print(f"{confusion_matrix(ytest, p.predict(Xtest))}")

##### SVM ######
x = np.c_[norm_deaths,norm_height,norm_valid,norm_houses] #NO CITY
num_pcs = 1
pca = PCA(n_components=num_pcs)
pcs = pca.fit_transform(x)
svm = SVC(kernel='poly',C=0.1,max_iter=9999999)
p = make_pipeline(pca,svm)
p_scores = cross_val_score(p,x,y,cv=5)
print(f"SVM's average cross validation: {p_scores.mean():.4f}")
Xtrain, Xtest, ytrain, ytest = train_test_split(x,y,train_size=.2,shuffle=True)
p.fit(Xtrain,ytrain)
print("Tsunami Location: SVC with PCA Confusion Matrix")
print(f"{confusion_matrix(ytest, p.predict(Xtest))}")







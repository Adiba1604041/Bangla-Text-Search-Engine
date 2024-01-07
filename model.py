import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import pickle
data = pd.read_csv( 'F:\eightdata.csv')

lab=LabelEncoder()
data['Bid'] = lab.fit_transform(data['Bid'])
data['DocID'] = lab.fit_transform(data['DocID'])

data['Keyword'] = lab.fit_transform(data['Keyword'])
#mapping = dict(zip(lab.classes_, range(1, len(lab.classes_)+1)))
mapping = dict(zip(lab.classes_, range(0, len(lab.classes_))))

x= data.iloc [:, : -1]
y= data['Bid']

X_train, X_test, Y_train,Y_test = train_test_split(x,y, test_size=0.2, random_state=40)

rf = RandomForestRegressor(n_estimators = 100, random_state = 0)
rf.fit(X_train, Y_train)
Y_pred = rf.predict(X_test)
a=rf.score(X_test, Y_test)

pickle.dump(rf,open('model.pkl','wb'))
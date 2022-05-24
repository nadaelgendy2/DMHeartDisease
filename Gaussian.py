import pandas as pd
from sklearn.naive_bayes import GaussianNB
import matplotlib.pyplot as plt
#from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report


HD=pd.read_csv(r"C:\Users\manar\Downloads\Compressed\Arabic tweets\heart.csv")
#HD.info()

#check for duplicates and removing them
#HD = HD[HD.duplicated()]
#print(HD.shape)
HD=HD.drop_duplicates()
#HD = HD[HD.duplicated()]
#print(HD.shape)

#check for null values
#print(HD.isnull().sum())

#check for outliers
#plt.boxplot(HD.drop('target', axis=1))
#plt.show()

#remove outliers
Q1 = HD.quantile(0.3)
Q3 = HD.quantile(0.7)
IQR = Q3-Q1
HD = HD[~((HD<(Q1-1.5*IQR))|(HD>(Q3+1.5*IQR))).any(axis=1)]

#dataframe of dataset
HD=pd.DataFrame(HD)
print(HD.shape)

#shuffling and sampling data by stratified sampling
#HD=HD.sample(frac=1)
#dsgroup=HD.groupby('age', group_keys=False)
#dssample=dsgroup.apply(lambda x: x.sample(frac=0.6))
#print(dssample.shape)

#train and test data
x=dssample.drop('target',axis=1)
y=dssample.target
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3, random_state=1)

#applaying naive bayes
GHD=GaussianNB()
GHD.fit(x_train, y_train)
y_prediction = GHD.predict(x_test)

#accuracy
print(classification_report(y_test,y_prediction))

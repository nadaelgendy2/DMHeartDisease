import pandas as pd
import seaborn as sb
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold

HD=pd.read_csv("/Users/nadahmed/Downloads/Semester 6/Data Mining/project/heart.csv")
#HD.info();
#check for duplicates before and after removing them
#var = HD[HD.duplicated()]
#print(var.shape)
HD=HD.drop_duplicates()
#var = HD[HD.duplicated()]
#print(var.shape)

#check for null values
#print(HD.isnull())

#check for outliers
#method1
#plt.boxplot(HD["fbs"])
#plt.show()

#method2
#print("Highest_value",HD['ca'].mean() + 3*HD['ca'].std())
#print("Lowest_value",HD['ca'].mean() - 3*HD['ca'].std())
#hd = HD[(HD['ca'] < -2.301701729570903) | (HD['ca'] > 3.7387878222861346)]
#print(hd)

#remove outliers 
Q1 = HD.quantile(0.25)
Q3 = HD.quantile(0.75)
IQR = Q3-Q1
data2 = HD[~((HD<(Q1-1.5*IQR))|(HD>(Q3+1.5*IQR))).any(axis=1)]
print(data2.shape)

#sampling data by stratified sampling
ds=pd.DataFrame(HD)
print(ds)
dsgroup=ds.groupby('age', group_keys=False)
dssample=dsgroup.apply(lambda x: x.sample(frac=0.6))
print(dssample)

                       





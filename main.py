import pandas as pd
import seaborn as sb
import matplotlib.pyplot as plt

HD=pd.read_csv("/Users/nadahmed/Downloads/Semester 6/Data Mining/project/heart.csv")
#HD.info();
#var = HD[HD.duplicated()]
#print(var.shape)
HD=HD.drop_duplicates()
#var = HD[HD.duplicated()]
#print(var.shape)


#plt.boxplot(HD["fbs"])
#plt.show()
Q1 = HD.quantile(0.25)
Q3 = HD.quantile(0.75)
IQR = Q3-Q1
data2 = HD[~((HD<(Q1-1.5*IQR))|(HD>(Q3+1.5*IQR))).any(axis=1)]
print(data2.shape)

#print("Highest allowed",HD['ca'].mean() + 3*HD['ca'].std())
#print("Lowest allowed",HD['ca'].mean() - 3*HD['ca'].std())

#new_df = HD[(HD['ca'] < -2.301701729570903) | (HD['ca'] > 3.7387878222861346)]
#print(new_df)





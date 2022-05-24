import pandas as pd
from sklearn.naive_bayes import GaussianNB
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2

HD=pd.read_csv("/Users/nadahmed/Downloads/Semester 6/Data Mining/project/heart.csv")
HD=HD.drop_duplicates()
#remove outliers
Q1 = HD.quantile(0.25)
Q3 = HD.quantile(0.75)
IQR = Q3-Q1
data2 = HD[~((HD<(Q1-1.5*IQR))|(HD>(Q3+1.5*IQR))).any(axis=1)]

Y= data2['target']
X = data2.drop(['target'], axis = 1)


#plt show gender against target
pd.crosstab(data2.sex,data2.target).plot(kind="bar",figsize=(5,5),color=['blue','red' ])
plt.xlabel("Sex (0 = female, 1= male)")
plt.xticks(rotation=0)
plt.legend(["Haven't Disease", "Have Disease"])
plt.ylabel('Frequency')
plt.show()
#plt show gender against target
plt.figure(figsize=(15, 15))
data2[data2["target"] == 0][data2.columns[7]].hist(bins=35, color='blue', label='Have Heart Disease = NO')
data2[data2["target"] == 1][data2.columns[7]].hist(bins=35, color='red', label='Have Heart Disease = YES')
plt.legend()
plt.xlabel("thalach")
#plt show gender against target
pd.crosstab(data2.oldpeak,data2.target).plot(kind="bar",figsize=(15,15),color=['blue','red' ])
plt.xlabel("oldpeak")
plt.xticks(rotation=0)
plt.legend(["Haven't Disease", "Have Disease"])
plt.ylabel('Frequency')
plt.show()
#plt show gender against target
pd.crosstab(data2.ca,data2.target).plot(kind="bar",figsize=(5,5),color=['blue','red' ])
plt.xlabel("ca")
plt.xticks(rotation=0)
plt.legend(["Haven't Disease", "Have Disease"])
plt.ylabel('Frequency')
plt.show()
#plt show age against target
pd.crosstab(data2.age,data2.target).plot(kind="bar",figsize=(10,10),color=['blue','red' ])
plt.xlabel('Age')
plt.xticks(rotation=0)
plt.legend(["Haven't Disease", "Have Disease"])
plt.ylabel('Frequency')
plt.show()

#Feature selection
Y = data2['target']
X = data2.drop(['target'], axis = 1)
#apply SelectKBest class to extract top 10 best features
bestfeatures = SelectKBest(score_func=chi2, k=13)
fit = bestfeatures.fit(X,Y)
dfscores = pd.DataFrame(fit.scores_)
dfcolumns = pd.DataFrame(X.columns)
#concat two dataframes for better visualization
featureScores = pd.concat([dfcolumns,dfscores],axis=1)
featureScores.columns = ['Specs','Score']

print(featureScores.nlargest(13,'Score'))
print('-------------------------------------------------------------------------------------------')

categorical_columns = []
for column in data2.columns:
    if len(data2[column].unique()) <= 10:
        categorical_columns.append(column)


#create dummies values
categorical_columns.remove('target')
data_set = pd.get_dummies(data2, columns=categorical_columns)

#using standardScaler for features which .get_dummies can't process in it 'continous cloumns.
st_scaler= StandardScaler()
scale_columns = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak']
data_set[scale_columns] = st_scaler.fit_transform(data_set[scale_columns])

#Train test split
y = data_set['target']
x = data_set.drop(['target'], axis = 1)
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.33, random_state = 0)



def naive_bayes():
    #applaying naive bayes
    GHD=GaussianNB()
    GHD.fit(x_train, y_train)
    y_prediction = GHD.predict(x_train)

    #accuracy(naive bayes)
    print('Naive Bayes Model: \n')
    print(f" Train Accuracy Score: {accuracy_score(y_train, y_prediction) * 100:.2f}%")
    print(classification_report(y_train,y_prediction))
    y_prediction = GHD.predict(x_test)
    print(f"Test Accuracy Score: {accuracy_score(y_test, y_prediction) * 100:.2f}%")
    print(classification_report(y_test,y_prediction))
    print('-------------------------------------------------------------------------------------------')


def knn():
    #applying knn
    knn = KNeighborsClassifier(n_neighbors=5)  # n_neighbors means k
    knn.fit(x_train, y_train)
    prediction = knn.predict(x_train)
    print('K-nearest Neighbour Model: \n')
    print(f" Train Accuracy Score: {accuracy_score(y_train,prediction) * 100:.2f}%")
    print(classification_report(y_train,prediction))
    prediction = knn.predict(x_test)
    print(f"Test Accuracy Score: {accuracy_score(y_test,prediction) * 100:.2f}%")
    print(classification_report(y_test, prediction))
    print('-------------------------------------------------------------------------------------------')



def Decisiontree_Classifier():
    # applying decision tree
    tree_classifer = DecisionTreeClassifier(random_state=0)
    tree_classifer.fit(x_train, y_train)
    pred = tree_classifer.predict(x_train)

    # accuracy(decision tree)
    print('Decision Tree Model: \n')
    print(f" Train Accuracy Score: {accuracy_score(y_train,pred) * 100:.2f}%")
    print(classification_report(y_train,pred))
    pred = tree_classifer.predict(x_test)
    print(f" Test Accuracy Score: {accuracy_score(y_test, pred) * 100:.2f}%")
    print(classification_report(y_test, pred))
    print('-------------------------------------------------------------------------------------------')

naive_bayes()
knn()
Decisiontree_Classifier()

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn import model_selection
from sklearn import linear_model

df = pd.read_csv("dataset.csv")
# print(df)

df.shape

df.columns


df.info()

df.describe()

df.groupby(['Hours'])['Scores'].mean()

# Exploring the dataset

plt.scatter(df['Hours'], df['Scores'], color='Blue',marker='o')
plt.title("Hours Vs Scores")
plt.xlabel("Hours studied")
plt.ylabel("Percentage Scoreed")
plt.show()

df.corr()

sns.lmplot(x="Hours",y="Scores", data=df)
plt.title("Plotting the regression line")
# sns.regplot(x="Hours", y="Scores", data=df)

# Dividing the data into attributes(inputs)
# and labels (outputs)

X = df.iloc[:, :-1].values
y = df.iloc[:, -1].values

# print(X)

# print(y)

# Splitting the dataset into the Training
# set and Test set


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 1/3, random_state = 0)

# Train the Simple Linear Regression
# model on the Training set

from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)

# Predict the Test set results

y_pred = regressor.predict(X_test)

# print(y_pred)

# Comparing Actual vs Predictes

df1 = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred}) 

print(df1)

# Visualising the Training set Results

# Plotting the training set
plt.scatter(X_train,y_train, color='red')
plt.plot(X_train,regressor.predict(X_train),color='blue')
plt.title('(Trainig set)')
plt.xlabel('Hours')
plt.ylabel('Scores')
plt.show()

# Visualising the Test set Results

plt.scatter(X_test, y_test, color = 'red')
plt.plot(X_train, regressor.predict(X_train), color = 'blue')
plt.title('(Testing set)')
plt.xlabel('Hours studied')
plt.ylabel('Percentage Scored')
plt.show()


# Checking the correlations
plt.figure(figsize=(10,5))
sns.heatmap(df.corr(),annot=True,cmap="YlOrBr",annot_kws={'fontsize':12})
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.show()


# Visualizing the differences between 
# actual Scores and predicted Scores

plt.scatter(y_test,y_pred,c='r')
plt.plot(y_test,y_pred,c='g')
plt.xlabel("Prices")
plt.ylabel("Predicted Score")
plt.title("Score vs Predicted Score")
plt.show()


# What will be predicted score if a student
# studies for 9.25 hrs/day?

# Prediction through our model

Hours = np.array([[9.25]])
predict=regressor.predict(Hours)
print("No of Hours = {}".format(Hours))
print("Predicted Score = {}".format(predict[0]))


#  Checking accuracy of our model

print("Train : ",regressor.score(X_train,y_train)*100)
print("Test : ",regressor.score(X_test,y_test)*100)


# Find the mean absolute error, r^2
# score error and Mean Squared Error

from sklearn import metrics  
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
print('Mean absolute error:', metrics.mean_absolute_error(y_test, regressor.predict(X_test))) 
print('r^2 score error:',r2_score(y_test, regressor.predict(X_test)))
print('Mean squared error: ',mean_squared_error(y_test, regressor.predict(X_test)))

# Mean absolute error: 4.691397441397446 which is
# quite accurate model for predicting the result
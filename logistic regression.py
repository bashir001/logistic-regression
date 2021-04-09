
		
# logistic-regression
#import the necessary modules/packages
import pandas as pd 
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
#read the datasets
df = pd.read_csv('[titanic (1).csv](https://github.com/bashir001/logistic-regression/files/6287173/titanic.1.csv)
')
data = pd.get_dummies(df, columns =['Siblings/Spouses', 'Pclass', 'Sex', 'Age', 'Fare'])

x = data.iloc[:, 1:]
y = data.iloc[:,0]
#spliting the data into training and testing
x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=0)
classifier = LogisticRegression()
classifier.fit(x_test, y_test)
predicted_y = classifier.predict(x_test)
print(predicted_y)
print(x_test.shape)

#to iterate through the predicted values and print the accuracy of the train and test data

for x in range(len(predicted_y)):
	if predicted_y[x] == 1:
		print(x, end="\t")
		print('accuracy: {:.2f}'. format(classifier.score(x_test, y_test)))
		print('accuracy: {:.2f}'. format(classifier.score(x_train, y_train)))
		print(df.shape)

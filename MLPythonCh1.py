from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
import numpy as np
iris = load_iris() # loads the iris database
knn = KNeighborsClassifier(n_neighbors=1) #KNN value to 1

#Splits 75% of the data into training data, and 25% of the data into testing data after randomly shuffling
X_train, X_test, y_train, y_test = train_test_split(iris['data'], iris['target'],
random_state=0)

knn.fit(X_train, y_train)

X_new = np.array([[6.4, 3.2, 5.3, 2.3]])

prediction = knn.predict(X_new)

print("Prediction is:")
if (prediction == 0):
    print("Setosa")
elif (prediction == 1):
    print("Versicolor")
elif (prediction == 2):
    print("Virginica")
else:
    print("Value " + prediction + " could not be classified")

print("With an accuracy of:")
print(knn.score(X_test, y_test))

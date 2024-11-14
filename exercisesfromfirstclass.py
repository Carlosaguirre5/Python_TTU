import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sklearn as sk
'''
#1 Pandas Exercise
#Create a Dataframe
data = pd.DataFrame({
    'Name:': ['Carlos', 'Mariela', 'Roscoe'],
    'Age:': [27,28,30],
    'City:': ['Barcelona','San jose','Madrid'],
})

#View data
print(data.head())
print(data.describe())


#2 Numpy Exercise
#Create an array from 1 to 10

array = np.arange(1, 11)

#Calculate mean
print('Mean:',np.mean(array))

#Reshape into 2x5
reshaped_array = array.reshape(2,5)
print("Reshaped array:\n", reshaped_array)

#3 Exercise Matplotlib
#Data

x = [1, 2, 3, 4, 5]
y = [i**2 for i in x]

#Plot 
plt.plot(x, y, marker='o')
plt.xlabel('X Values')
plt.ylabel('X Squared')
plt.title('Line Plot of Squares')
plt.show()

#4 Exercise Statistics/Outliers
#Data
numbers = [1, 2, 3, 4, 5, 100]

#Calculate mean and standard deviation
mean = np.mean(numbers)
std_dev = np.std(numbers)

#Find outliers

outliers = [x for x in numbers if abs(x-mean) > 2 * std_dev]
print('Outliers:', outliers)

#Supervised learning: Simple classification

#Height data (Features) and labels

from sklearn.neighbors import KNeighborsClassifier

heights = [[150], [160], [170], [180]]
labels = ["Short", "Short", "Tall", "Tall"]

#Compare len of heights and labels
#print(f'heights: {heights}, labels: {labels}')
#print(f'lenght of heights: {len(heights)}, lenght of labels: {len(labels)}')

# Train simple classifier

classifier = KNeighborsClassifier(n_neighbors=1)
classifier.fit(heights, labels)

# Classity new heigth

new_height = [[200]]
prediction = classifier.predict(new_height)
print("Prediction for 165cm:", prediction[0])


#Unsupervised Learning: Clustering with K-Means

from sklearn.cluster import KMeans

#Data (coordinates of points)

data = np.array([[1, 2], [2, 1], [3, 4], [5, 6], [6, 5],[8, 8]])

#Apply K-Means clustering with 2 clusters

kmeans = KMeans(n_clusters=2)
kmeans.fit(data)

#Print cluster centers and labels
print('Cluster centers:', kmeans.cluster_centers_)
print("Labels:", kmeans.labels_)



#NumPy Functions Pratice: Generating and Reshaping Arrays

#Generate array

array = np.linspace(0, 10, 5)

#Reshape into 5x1

reshaped_array = array.reshape(5, 1)
print('Array: \n', reshaped_array)

#GridSearch: Simple model tuning

from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier

# Dummy Data
X = [[1], [2], [3], [4], [5]]
y = [0, 0, 1, 1, 1]

#Compare len of heights and labels
#print(f'X: {X}, y: {y}')
#print(f'lenght of X: {len(X)}, lenght of y: {len(y)}')

# Define a Stratified K-Fold with fewer splits
cv = StratifiedKFold(n_splits=2)

# Define the model and parameter grid
model = KNeighborsClassifier()
param_grid = {'n_neighbors': [1, 2, 3,]}
grid_search = GridSearchCV(model, param_grid, cv=cv)

#Perform the grid search
grid_search.fit(X, y)

#Print the best parameter and best score
print('Best parameters found:', grid_search.best_params_)
print('Best cross-validation score:', grid_search.best_score_)

#Array Manipulation: Stacking and transposing

#Arrays
a = np.array([1, 2, 3])
b = np.array ([4, 5, 6])

#Horizontal Stacking

stacked = np.hstack((a, b))
print("Stacked array:", stacked)

#Transpose
stacked_2d = np.vstack((a,b))
transposed = stacked_2d.T
print('Tranposed array:\n', transposed)
'''
#Time Series Basics: Simple trend analysis

#Temperature Data
days = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
temperatures = [30, 32, 31, 33, 35, 36, 37]

#Plot 
plt.plot(days, temperatures, marker='o')
plt.xlabel('Day')
plt.ylabel('Temperature (C)')
plt.title('Temperature Over a Week')
plt.show()
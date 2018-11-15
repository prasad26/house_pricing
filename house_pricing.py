import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import mpl_toolkits
%matplotlib inline
from sklearn.linear_model import LinearRegression
from sklearn.cross_validation import train_test_split
from sklearn.ensemble import GradientBoostingRegressor

data = pd.read_csv("D:/ML/real-estate/kc_house_data.csv")

data.head()

data.describe()

data.isnull().sum()

data['bedrooms'].value_counts()
data['bedrooms'].value_counts().plot(kind = 'bar')
plt.title('number of bedrooms')
plt.xlabel('Bedrooms')
plt.ylabel('total_count')
sns.despine

plt.figure(figsize = (10, 10))
sns.jointplot(x = data.lat.values, y = data.long.values, size=10)
plt.xlabel('Latitude', fontsize = 12)
plt.ylabel('Longitude', fontsize = 12)
plt.show()
sns.despine

plt.scatter(data.price, data.sqft_living)
plt.title('price vs sqft_living')

plt.scatter(data.price, data.yr_built)
plt.title('price vs yr_built')

plt.scatter(data.price, data.bedrooms)
plt.title('price vs bedrooms')

plt.scatter(data.bedrooms, data.bathrooms)
plt.title('bedrooms vs bathrooms')
plt.xlabel('bedrooms')
plt.ylabel('bathrooms')
plt.show()
sns.despine

plt.scatter(data.bedrooms, data.price)
plt.title('bedrooms vs price')
plt.xlabel('bedrooms')
plt.ylabel('price')
plt.show()
sns.despine

plt.scatter(data.zipcode, data.price)
plt.title('price vs zipcode')

reg = LinearRegression()

labels = data['price']
conv_dates = [1 if values == 2014 else 0 for values in data.date]
data['date'] = conv_dates
train1 = data.drop(['id', 'price'], axis = 1)

x_train, x_test, y_train, y_test = train_test_split(train1, labels, test_size = 0.20, random_state = 2)

reg.fit(x_train, y_train)

reg.score(x_test, y_test)

clf = GradientBoostingRegressor(n_estimators = 400, max_depth = 5, min_samples_split = 2, learning_rate = 0.1, loss = 'ls')

clf.fit(x_train, y_train)

clf.score(x_test, y_test)
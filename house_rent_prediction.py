import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import r2_score


data = pd.read_csv('_All_Cities_Cleaned.csv')

le = LabelEncoder()
data['seller_type'] = le.fit_transform(data['seller_type'])
data['layout_type'] = le.fit_transform(data['layout_type'])
data['property_type'] = le.fit_transform(data['property_type'])
data['locality'] = le.fit_transform(data['locality'])
data['furnish_type'] = le.fit_transform(data['furnish_type'])
data['city'] = le.fit_transform(data['city'])

features = ['seller_type','bedroom','layout_type','property_type','locality','area','furnish_type','bathroom','city']
target = 'price'


X_train, X_test, y_train, y_test = train_test_split(data[features], data[target], test_size=0.2)


model = LinearRegression()
model.fit(X_train, y_train)


y_pred = model.predict(X_test)



mse = mean_squared_error(y_test, y_pred)
print('Mean Squared Error:', mse)


accuracy = r2_score(y_test, y_pred)
print('Accuracy:', accuracy)


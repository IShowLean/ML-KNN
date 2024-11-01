import pandas as pd
from sklearn.preprocessing import MinMaxScaler

data = pd.read_csv('vehicles_small.csv')

data = data.drop(columns=['id', 'url', 'region', 'region_url', 'VIN', 'model', 'image_url', 'description', 'county', 'state', 'lat', 'long', 'posting_date'])

data.rename(columns={
    'price': 'Price, €',
    'odometer': 'Mileage, km',
    'year': 'Year',
    'manufacturer': 'Company',
    'cylinders': 'Cylinders',
    'fuel': 'Fuel type',
    'drive': 'Drive type',
    'size': 'Size',
    'type': 'Body type',
    'paint_color': 'Body color',
    'transmission': 'Transmission',
}, inplace=True)

for column in data.select_dtypes(include='number').columns:
    data[column] = data[column].fillna(data[column].mean())

for column in data.select_dtypes(include='object').columns:
    data[column] = data[column].fillna(data[column].mode()[0])

target = 'Company'

data = pd.get_dummies(data, columns=data.select_dtypes(include='object').columns.difference([target]), dtype=int)

scaler = MinMaxScaler()
data[data.select_dtypes(include='number').columns] = scaler.fit_transform(data.select_dtypes(include='number'))

data.to_csv('vehicles_data.csv', index=False)

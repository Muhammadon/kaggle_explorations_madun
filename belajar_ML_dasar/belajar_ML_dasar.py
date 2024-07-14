# Belajar Machine Learning Dasar

# Dataset Melbourne Housing

# 1. importing pandas
    # pandas merupakan module untuk melakukan analisis data
import pandas as pd

# 2. memuat dataset sebagai pandas dataframe
file_path = './dataset/melb_data.csv'
housing_df = pd.read_csv(file_path)
# print(housing_df.head())

# 3. explorasi awal
    # menampilkan dimensi dataset
print(housing_df.shape)
print()

    # menampilkan daftar nama kolom
# print(housing_df.columns)
print()

    # ringkasan data (summary)
# print(housing_df.describe())
print()

    # kausus : menampilkan ukuran tanah terluas
# print(housing_df.describe().loc['max', 'Landsize'])
print()
    # alternatif ke dua
print(housing_df.describe()['Landsize']['max'])
print()

# 4. Machine Learning Model Dasar
    # pembersihan data (data cleaning)
# housing_df = housing_df.dropna() # akan menghapus data baris missing value / baris dengan data kosong
print(housing_df.shape)

    # memilih target prediski (prediction target) 
y = housing_df['Price']
# print(y)

    # memilih fitur (features selection) kita akan memilih kolom kolom yang akan menjadi parameter prediksi
features = ['Rooms', 'Bathroom', 'Landsize', 'Lattitude', 'Longtitude']
X = housing_df[features]
# print(X)
# print(X.describe())
print()

    # membangun model
        # membangun machine learning model dengan decision tree regressor
        # import decision tree regressor
from sklearn.tree import DecisionTreeRegressor

    # konvigurasi model
# housing_model = DecisionTreeRegressor(random_state=1)

    # training model
# housing_model.fit(X, y)

    # melakukan prediksi
# pred = housing_model.predict(X.head())
# print(pred)
print()
# print(y.head())
print()

    # evaluasi model(model evaluation)
        # melakukan validasi / evaluasi terhadap model untuk mengukur performa

        # importing evaluation metric(mean_absolute_error)
from sklearn.metrics import mean_absolute_error

# y_hat = housing_model.predict(X)
# print(mean_absolute_error(y, y_hat))

    # training dan testing dataset
from sklearn.model_selection import train_test_split

        # membagi dataset menjadi 2 bagian
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1)

        # konfigurasi dan training model
housing_model = DecisionTreeRegressor(random_state=1)
housing_model.fit(X_train, y_train)

        # evaluasi model
y_hat = housing_model.predict(X_test)
print(mean_absolute_error(y_test, y_hat))

    # optimasi model

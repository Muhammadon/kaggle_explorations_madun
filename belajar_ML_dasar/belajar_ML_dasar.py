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
def get_mae(max_leaf_nodes, X_train, X_test, y_train, y_test):
    model = DecisionTreeRegressor(max_leaf_nodes=max_leaf_nodes, random_state=0)
    model.fit(X_train, y_train)
    y_hat = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_hat)
    return mae

        # membandingkan mae dengan beberapa nilai max_leaf_nodes untuk menemukan jumlah leaf paling optimum
for max_leaf_nodes in [5, 50, 500, 5000]:
    leaf_mae = get_mae(max_leaf_nodes, X_train, X_test, y_train, y_test)
    print(f'Max leaf nodes: {max_leaf_nodes} \t Mean Absolute Error: {int(leaf_mae)}')
print()

# 5. Eksplorasi dengan random forest
    # membangun machine learning model dengan random forest regressor

    # importing RandomForestRegressor
from sklearn.ensemble import RandomForestRegressor

rf_model = RandomForestRegressor(n_estimators=100, random_state=1)
rf_model.fit(X_train, y_train)
y_hat = rf_model.predict(X_test)
print(f'Mean absolute error: {int(mean_absolute_error(y_test, y_hat))}')
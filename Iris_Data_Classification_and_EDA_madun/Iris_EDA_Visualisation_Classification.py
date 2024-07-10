# BAGIAN I
# Iris dataset : Simple Exploratory Data Analysis (EDA)

# Import Modules
import pandas as pd # olah dan analisis data

# Load dataset
iris_df = pd.read_csv('./dataset/iris/Iris.csv') # memuat file csv sebagai data frame
print(iris_df.head()) # Tampilkan 5 baris pertama
print()

# Drop column 'Id'
# iris_df = iris_df.drop(columns='Id')
iris_df.drop(columns='Id', inplace=True) # menghapus kolom bernama Id
print(iris_df.head()) # tampilkan 5 baris pertama
print()

# Identify the shape of the dataset
print(iris_df.shape) # bentuk/dimensi dataset (baris/kolom)
print()

# Get the list of columns
print(iris_df.columns) # menampilkan daaftar nama kolom
print()

# Identify data types for each columns
print(iris_df.dtypes) # menapilkan tipe data dari tiap kolom
print()

# Get bassic dataset information
print(iris_df.info()) # informasi dataset
print()

# Identify missing values
# iris_df.isnull().values.any()
print(iris_df.isna().values.any()) # mendeteksi keberadaan nilai kosong
print()

# Identify duplicate entries/rows
# iris_df[iris_df.duplicated(keep=false)] # tampilkan seluruh baris dengan duplikasi
print(iris_df[iris_df.duplicated()]) # tampilkan hanya baris duplikasi sekunder
print()
print(iris_df.duplicated().value_counts()) # hitung jumlah duplikasi data
print()

# Drop duplicate entries/rows
iris_df.drop_duplicates(inplace=True) # menghapus duplikasi data
print(iris_df.shape) # melihat kembali dimensi data
print()

# Describe the dataset
print(iris_df.describe()) # deskripsi statistik dari dataset
print()

# Correlation Matrix
numeric_df = iris_df.select_dtypes(include=[float, int]) # pilih hanya kolom numerik untuk perhitungan korelasi agar tidak eror
print(numeric_df.corr()) # korelasi antar kolom
print()

# =================================== SELESAI BAGIAN I ============================================================

# BAGIAN II
# Iris Dataset : Data Visualisation
import matplotlib.pyplot as plt # visualisasi data
import seaborn as sns # visualisasi data

# 1. Heatmap
sns.heatmap(data=numeric_df.corr()) # Membuat heatmap dari matriks korelasi
plt.show()
print()

# 2. Bar Plot
print(iris_df['Species'].value_counts()) # menghitung jumlah setiap spesies
iris_df['Species'].value_counts().plot.bar()
plt.tight_layout()
plt.show()
print()

# untuk melakukan bar plot, kita juga bisa memanfaatkan seborn
sns.countplot(data=iris_df, x='Species')
plt.tight_layout()
plt.show()

# 3. Pie Chart
iris_df['Species'].value_counts().plot.pie(autopct='%1.1f%%', labels=None, legend=True)
plt.tight_layout()
plt.show()

# 4. Line Plot
fig, ax = plt.subplots(nrows=2, ncols=2, figsize=(8, 8))

iris_df['SepalLengthCm'].plot.line(ax=ax[0][0])
ax[0][0].set_title('Sepal Length')

iris_df['SepalWidthCm'].plot.line(ax=ax[0][1])
ax[0][1].set_title('Sepal Width')

iris_df.PetalLengthCm.plot.line(ax=ax[1][0])
ax[1][0].set_title('Petal Length')

iris_df.PetalWidthCm.plot.line(ax=ax[1][1])
ax[1][1].set_title('Petal Width')
plt.show()

# bisa juga dilakukan secara keseluruhan dalam satu area
iris_df.plot()
plt.tight_layout()
plt.show()

# 5. Histogram
iris_df.hist(figsize=(6, 6), bins=10)
plt.tight_layout()
plt.show()

# 6. Boxplot
iris_df.boxplot()
plt.tight_layout()
plt.show()

# cara lain boxplot dengan mengelompokkan berdasarkan spesies
iris_df.boxplot(by='Species', figsize=(8, 8))
plt.tight_layout()
plt.show()

# 7. Scatter Plot
sns.scatterplot(x='SepalLengthCm', y='SepalWidthCm', data=iris_df, hue='Species')
plt.tight_layout()
plt.show()

# 8. Pair plot
sns.pairplot(iris_df, hue='Species', markers='+')
plt.tight_layout()
plt.show()

# 9. Violin Plot
sns.violinplot(data=iris_df, y='Species', x='SepalLengthCm', inner='quartile')
plt.tight_layout()
plt.show()

# =================================== SELESAI BAGIAN II ===========================================================

# BAGIAN III
# Iris Dataset : Classification Model, dengan pemanfaatan mechine learning

# 1. Import Modules
from sklearn.model_selection import train_test_split # pembagi dataset menjadi training dan testing set
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report # evaluasi performa model

# 2. Dataset : Feature & Class Label
X = iris_df.drop(columns='Species') # menempatkan features ke dalam variabel X dengan menghapus species, karna species adalah target
print(X.head()) # tampilkan 5 baris pertama
print()

y = iris_df['Species'] # menempatkan class label (target) ke dalam variabel y
print(y.head())
print()

# 3. split the dataset into a training set and a testing set
  # membagi dataset ke dalam training dan testing set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=10)

print('training dataset')
print(X_train.shape)
print(y_train.shape)
print('=================')
print('tetsting dataset')
print(X_test.shape)
print(y_test.shape)
print()

# 4. K Nearest Neighbors
from sklearn.neighbors import KNeighborsClassifier

k_range = list(range(1, 26))
scores = []
for k in k_range:
    model_knn = KNeighborsClassifier(n_neighbors=k) # konfigurasi algoritma
    model_knn.fit(X_train, y_train) # training model / classfier
    y_pred = model_knn.predict(X_test) # melakukan prediksi
    scores.append(accuracy_score(y_test, y_pred)) # evaluasi performa

  # visualisasi skor akurasi sebuah model, dengan 25 klai training
plt.plot(k_range, scores)
plt.xlabel('value of k for  KNN')
plt.ylabel('accuracy scor')
plt.title('accuracy scores for values of k of k_nearest_neighbors')
plt.tight_layout()
plt.show()
print()

  # train ulang dengan jumlah neighbors 3
model_knn = KNeighborsClassifier(n_neighbors=3) # konfigurasi algoritma
model_knn.fit(X_train, y_train) # training model / classifier
y_pred = model_knn.predict(X_test) # melakukan prediksi

  # accuracy score
print(accuracy_score(y_test, y_pred)) # evaluasi akurasi

  # evaluasi dengan confusio matrix
print(confusion_matrix(y_test, y_pred))

  # evaluasi classification report
print(classification_report(y_test, y_pred))

# 5. klasifikasi dengan Logistic Regression / alternatif lain knn_mode
from sklearn.linear_model import LogisticRegression

# model_logreg = LogisticRegression()
model_logreg = LogisticRegression(solver='lbfgs', multi_class='auto')
model_logreg.fit(X_train, y_train)
y_pred = model_logreg.predict(X_test)

# accuracy score
print(accuracy_score(y_test, y_pred))

# confusion matrix
print(confusion_matrix(y_test, y_pred))

# classification report
print(classification_report(y_test, y_pred))

# 6. klasifikasi dengan algoritma support vector classifier
from sklearn.svm import SVC

# model_svc = SVC()
model_svc = SVC(gamma='scale')
model_svc.fit(X_train, y_train)
y_pred = model_svc.predict(X_test)

# # 7. klasifikasi dengan decision tree classifier
from sklearn.tree import DecisionTreeClassifier

model_dt = DecisionTreeClassifier()
model_dt.fit(X_train, y_train)
y_pred = model_dt.predict(X_test)

# klasifikasi dengan algoritma random forest classifier
from sklearn.ensemble import RandomForestClassifier

# model_rf = RandomForestClassifier
model_rf = RandomForestClassifier(n_estimators=100)
model_rf.fit(X_test, y_test)
pred_rf = model_rf.predict(X_test)


# accuracy comparision for various models / membandingkan performa setiap model mechine learning dalam sklearn
models = [model_knn, model_logreg, model_svc, model_dt, model_rf]
accuracy_scores = []
for model in models:
    pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, pred)
    accuracy_scores.append(accuracy)
print(accuracy_scores)

# memvisualisasikannya
plt.bar(['KNN', 'LogReg', 'SVC', 'DT', 'RF'], accuracy_scores)
plt.ylim(0.90, 1.01)
plt.title('accuracy comparision for various models', fontsize=15, color='r')
plt.xlabel('Models', fontsize=18, color='g')
plt.ylabel('Accuracy Score', fontsize=18, color='g')
plt.tight_layout()
plt.show()
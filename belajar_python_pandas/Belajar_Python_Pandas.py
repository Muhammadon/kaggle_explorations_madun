# 1. import pandas
import pandas as pd

# 2. mengenal dataframe
    # membuat dataframe sederhana
data_pd = pd.DataFrame(data={'Minuman': ['Bandrek', 'Kopi Hitam'],
                   'harga': [4500, 3200]})
# print(data_pd)
print()

# 3. mengenal index pada dataframe
    # secara default index pada dataframe adalah integer yang akan di mulai dari 0
    # kita juga bisa juga secara explisit menentukan index tertentu
data_pd_2 = pd.DataFrame(data={'Budi': ['suka', 'suka'],
                               'wati': ['tidak suka', 'suka']},
                               index=['Mangga', 'jambu'])
# print(data_pd_2)
print()

# 4. mengenal series
    # atau bisa juga di sebut data 1 kolom, secara default indexnya juga integer dari 0
data_series = pd.Series(data=[100, 150, 200, 250, 300])
# print(data_series)
print()

    # kita juga bisa menentukan indexnya
data_series_2 = pd.Series(data=[3000, 3500, 4000], index=['Mangga', 'Pisang', 'Jambu'], name='Harga Buah')
# print(data_series_2)
print()

# 5. memuat file CSV ke dalam dataframe
wine_df = pd.read_csv('./dataset/winemag-data-130k-v2.csv')
# print(wine_df.head()) # akan menampilkan 5 data pertama, jika ingin lebih, masukkan parameter ke dalam fungsi 'head(10)'
print()

    # melihat dimensi dataframe dengan attribut shape
print(wine_df.shape)
print()

    # menggunakan salah satu kolom sebagai index, artinya kita tidak ingin index diberikan oleh pandasnya, kita akan mengambil index dari salah satu kolom data csv
wine_df = pd.read_csv('./dataset/winemag-data-130k-v2.csv', index_col=0)
# print(wine_df.head())
print()

    # melihat dimensi dataframe dengan attribut shape
print(wine_df.shape)
print()

# 6. menyimpan dataframe ke dalam file csv
minuman_df = pd.DataFrame(data={'Minuman': ['Bandrek', 'Kopi Hitam'],
                                'Harga': [4500, 3200]})
# print(minuman_df)
minuman_df.to_csv('./output/minuman.csv')
print()

    # menyimpan dataframe ke dalam file CSV tanpa menyertakan index
minuman_df.to_csv('./output/minuman.csv', index=False)
print()

# 7. melakukan akses pada kolom dataframe
# print(wine_df.head())
print()

    # terdapat dua cara untuk akses kolom pada dataframe
    # a. cara 1 : sebagai attribut
# print(wine_df.country)
print()

    # b. cara 2 : sebagai dictionary index
# print(wine_df['country'])
print()
# print(wine_df['country'][0])
print()

# 8. melakukan akses data pada dataframe
    # terdapat dua mekanisme untuk melakukan akses data pada dataframe : index-based selection dan label-based selection
    # kedua mekanisme akses ini mendahulukan baris(row) baru di ikuti kolom(column)

    # index-based selection
        # mekanisme ini dapat di lakukan dengan memanfaatkan method iloc
# print(wine_df.iloc[0]) # mengakses data baris index ke 0
print()

        # jika ingan mengakses spesifik data
# print(wine_df.iloc[2, 0]) # mengakses data baris index ke 2 dan kolom index ke 0
print()

        # slicing pada iloc
# print(wine_df.iloc[:, 0]) # mengakses semua baris dengan kolom index ke 0
print()
# print(wine_df.iloc[0, :7]) # mengakses baris index ke 0 dan 7 kolom pertama
print()
# print(wine_df.iloc[1:3, 0]) # mengakses baris dari index ke 1 - 3 (eksklusif) dan kolom index ke 0
print()

        # melewatkan list untuk melakukan akses pada data baris dan kolom tertentu dari dataframe
# print(wine_df.iloc[[0, 3, 8], 0]) # mengakses data baris index ke 0, 3, dan 8 dan kolom index ke 0
print()

        # menggunakan negative index untuk melakukan akses data pada baris dan kolom tertentu dari dataframe
# print(wine_df.iloc[-5:]) # mengakases data dari baris 5 index terakhir dari data
print()

    # label-based selection
        # mekanisme ini dapat di gunakan dengan method loc, sama seperti sebelumka, aksesnya mulai baris kemudian kom
        # dan kita di sini tidak mnyebutkan indeksnya, tapi langsung label/nama nya seperti 'country'
print(wine_df.loc[0, 'country']) # mengakses baris index ke 0 dan kolom country
print()

print(wine_df.loc[0:5, 'country']) # barisk indeks ke 0 sampe 4, kolom country
print()

print(wine_df.loc[:, ['taster_name', 'taster_twitter_handle', 'points']].head())
print()

    # Alternatif Lain
        # untuk melakukan akses data dapat di lakukan dengan model akses kolom-baris
print(wine_df[['country', 'province']][:].head()) # akses kolom country dan province, semua baris
print()

print(wine_df[['country', 'province']][:5]) # aksesk kolom country dan province, 5 baris pertama
print()

# 8. Mengganti Index
    # menjadikan nama kolom tertentu sebgai indeks, tidak menggunkan indeks yang di berikan program
print(wine_df.set_index('title').head()) # disini panda akan menghasilkan dataframe baru, artinya penggantian indeks tidak di lakukan dalama dataframe awal
# df = wine_df.set_index('title').head() # karena menghasilakan data baru bagusnya kita tampung ke dalam variabel baru
# print(wine_df.set_index('title', inplace=True).head()) # disini pandas akan melakukannya dalam dataframe yang sama

# wine_df.reset_index() # jika ingin mengembalikan de indeks default
print()

# 9. Seleksi data pada dataframe
    # menggunkan method isin
print(wine_df.loc[wine_df['country'].isin(['Italy', 'France'])].head())
print()
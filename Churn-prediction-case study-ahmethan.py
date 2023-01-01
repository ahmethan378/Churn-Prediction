import zipfile
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
import missingno as msno
from datetime import date
from sklearn.model_selection import train_test_split
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import MinMaxScaler, LabelEncoder, StandardScaler, RobustScaler
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
from lightgbm import LGBMClassifier


from collections import Counter
from sklearn.datasets import make_classification
#from imblearn.over_sampling import SMOTE
import warnings
from sklearn.metrics import confusion_matrix, accuracy_score, mean_squared_error, r2_score, roc_auc_score, roc_curve, classification_report
from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split, cross_val_score

warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter("ignore", category=ConvergenceWarning)
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter("ignore", category=ConvergenceWarning)


pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.float_format', lambda x: '%.3f' % x)
pd.set_option('display.width', 500)



churn = pd.read_excel("Case_Study.xlsx",sheet_name=0)
df=churn.copy()
df.head(3)

df.describe().T
df.info()
df.drop(df.columns[0],axis=1,inplace=True) # Gereksiz sütun silindi.

# Kolonların ismini küçük harfe çevirdim:

df.columns = [col.lower() for col in df.columns]


######################################
# Adım 1. Genel Resim
######################################

def check_df(dataframe):
    print("##################### Shape #####################")
    print(dataframe.shape)
    print("##################### Types #####################")
    print(dataframe.dtypes)
    print("##################### Head #####################")
    print(dataframe.head(3))
    print("##################### Tail #####################")
    print(dataframe.tail(3))
    print("##################### NA #####################")
    print(dataframe.isnull().sum())
    print("##################### Quantiles #####################")
    print(dataframe.quantile([0, 0.05, 0.50, 0.95, 0.99, 1]).T)

check_df(df)



##################################
# Adım 2: NUMERİK VE KATEGORİK DEĞİŞKENLERİN YAKALANMASI
##################################

def grab_col_names(dataframe, cat_th=10, car_th=20):
    """
    grab_col_names for given dataframe

    :param dataframe:
    :param cat_th:
    :param car_th:
    :return:
    """

    cat_cols = [col for col in dataframe.columns if dataframe[col].dtypes == "O"]

    num_but_cat = [col for col in dataframe.columns if dataframe[col].nunique() < cat_th and
                   dataframe[col].dtypes != "O"]

    cat_but_car = [col for col in dataframe.columns if dataframe[col].nunique() > car_th and
                   dataframe[col].dtypes == "O"]

    cat_cols = cat_cols + num_but_cat
    cat_cols = [col for col in cat_cols if col not in cat_but_car]

    num_cols = [col for col in dataframe.columns if dataframe[col].dtypes != "O"]
    num_cols = [col for col in num_cols if col not in num_but_cat]

    print(f"Observations: {dataframe.shape[0]}")
    print(f"Variables: {dataframe.shape[1]}")
    print(f'cat_cols: {len(cat_cols)}')
    print(f'num_cols: {len(num_cols)}')
    print(f'cat_but_car: {len(cat_but_car)}')
    print(f'num_but_cat: {len(num_but_cat)}')

    # cat_cols + num_cols + cat_but_car = değişken sayısı.
    # num_but_cat cat_cols'un içerisinde zaten.
    # dolayısıyla tüm şu 3 liste ile tüm değişkenler seçilmiş olacaktır: cat_cols + num_cols + cat_but_car
    # num_but_cat sadece raporlama için verilmiştir.

    return cat_cols, cat_but_car, num_cols, num_but_cat

cat_cols, cat_but_car, num_cols, num_but_cat = grab_col_names(df)
# 24 değişken içerisinden 13 tane kategorik, 5 tane nümerik değişken, 6 tane kardinal değişken tanımlandı.


######################################
# Adım 3: Kategorik Değişken Analizi (Analysis of Categorical Variables)
######################################

def cat_summary(dataframe, col_name, plot=False):
    print(pd.DataFrame({col_name: dataframe[col_name].value_counts(),
                        "Ratio": 100 * dataframe[col_name].value_counts() / len(dataframe)}))

    if plot:
        sns.countplot(x=dataframe[col_name], data=dataframe)
        plt.show()


for col in cat_cols:
    cat_summary(df, col)

#Veri setinde mevcut müşterinin ağırlığı yüksek gözüküyor ve bu model çıktısının bu yöne doğru eğilmesine neden olabilir, doğru model seçimi yapmalıyız.
#Hizmet bedeli değişkeni %92.47'si boş veri içeriyor bu alan sağlıklı bir çıktıya yönlendirmeyeceği için silinmesi gerekebilir.
#Yıl olarak yalnızca 2019 bulunuyor herhangi bir fikir sunmayacağı için bu sütun silinebilir.


######################################
# Adım 3: Sayısal Değişken Analizi (Analysis of Numerical Variables)
######################################

def num_summary(dataframe, numerical_col, plot=False):
    quantiles = [0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 0.95, 0.99]
    print(dataframe[numerical_col].describe(quantiles).T)

    if plot:
        dataframe[numerical_col].hist(bins=50)
        plt.xlabel(numerical_col)
        plt.title(numerical_col)
        plt.show()

    print("#####################################")


for col in num_cols:
    num_summary(df, col)

# Numerik sütunlar aykırı değerler içeriyorlar bunların düzenlemesini ilerki aşamalarda yapacağım.


######################################
# Adım 4: Hedef Değişken Analizi (Analysis of Target Variable)
# Kategorik değişkenlere göre hedef değişkenin ortalaması, hedef değişkene göre
# numerik değişkenlerin ortalaması)
######################################



def target_summary_with_cat(dataframe, target, categorical_col):
    print(pd.DataFrame({"TARGET_MEAN": dataframe.groupby(categorical_col)[target].mean()}), end="\n\n\n")

# Hedef Değişkenimiz olan Churn değişkeni kategorik olduğundan analiz yapamıyoruz bunu label encoder ile sayısal hale getirelim.

def label_encoder(dataframe, binary_col):
    labelencoder = LabelEncoder()
    dataframe[binary_col] = labelencoder.fit_transform(dataframe[binary_col])
    return dataframe

binary_cols = [col for col in df.columns if df[col].dtype not in [int, float]
               and df[col].nunique() == 2]

for col in binary_cols:
    df = label_encoder(df, col)


for col in cat_cols:
    target_summary_with_cat(df,"son müşteri statusu",col)

# Hedef değişken analizi için label_encoder işlemi yapmıştık şimdi tekrar eski haline getirelim.

cat_cols, cat_but_car, num_cols, num_but_cat = grab_col_names(df)



######################################
# Adım 5: Aykırı Değer Analizi
######################################
def outlier_thresholds(dataframe, variable, low_quantile=0.10, up_quantile=0.90):
    quantile_one = dataframe[variable].quantile(low_quantile)
    quantile_three = dataframe[variable].quantile(up_quantile)
    interquantile_range = quantile_three - quantile_one
    up_limit = quantile_three + 1.5 * interquantile_range
    low_limit = quantile_one - 1.5 * interquantile_range
    return low_limit, up_limit

def check_outlier(dataframe, col_name):
    low_limit, up_limit = outlier_thresholds(dataframe, col_name)
    if dataframe[(dataframe[col_name] > up_limit) | (dataframe[col_name] < low_limit)].any(axis=None):
        return True
    else:
        return False


for col in num_cols:
    if col != "son müşteri statusu":
      print(col, check_outlier(df, col))

# Ciro ve kart sayısı sütununda aykırı değerler bulunmaktadır.
# Bunun çözümü için low ve upper limitleri bularak aykırı bulduğumuz değerler ile değiştireceğim.

def replace_with_thresholds(dataframe, variable):
    low_limit, up_limit = outlier_thresholds(dataframe, variable)
    dataframe.loc[(dataframe[variable] < low_limit), variable] = low_limit
    dataframe.loc[(dataframe[variable] > up_limit), variable] = up_limit


for col in num_cols:
    if col != "son müşteri statusu":
        replace_with_thresholds(df,col)
# Aykırı değerler düzeltildi ve aşağıdan tekrar kontrol ediyorum.

for col in num_cols:
    if col != "son müşteri statusu":
      print(col, check_outlier(df, col))


######################################
# Eksik Değer Analizi
######################################


def missing_values_table(dataframe, na_name=False):
    na_columns = [col for col in dataframe.columns if dataframe[col].isnull().sum() > 0]

    n_miss = dataframe[na_columns].isnull().sum().sort_values(ascending=False)

    ratio = (dataframe[na_columns].isnull().sum() / dataframe.shape[0] * 100).sort_values(ascending=False)

    missing_df = pd.concat([n_miss, np.round(ratio, 2)], axis=1, keys=['n_miss', 'ratio'])

    print(missing_df, end="\n")

    if na_name:
        return na_columns

missing_values_table(df)
df.head(3)

df[[ "hizmet bedeli oranı", "hizmet bedeli beklenti tutarı"]] = df[
    ["hizmet bedeli oranı", "hizmet bedeli beklenti tutarı",]].replace('-', np.NaN)

df[[ "ciro","kart sayısı","kart başı ciro",
             "hizmet bedeli oranı","hizmet bedeli beklenti tutarı","sözleşme vadesi","sozlesmebitengunsayisi",
             "ziyaret_ort_sapma","sikayet_adedi_orani"]] = df[
    ["ciro","kart sayısı","kart başı ciro",
             "hizmet bedeli oranı","hizmet bedeli beklenti tutarı","sözleşme vadesi","sozlesmebitengunsayisi",
             "ziyaret_ort_sapma","sikayet_adedi_orani",]].replace('-', 0)
df.head(3)
df[[ "hizmet bedeli oranı"]] = df[
    ["hizmet bedeli oranı"]].replace('%', '')



# Bu fonsksiyon eksik değerlerin median veya mean ile doldurulmasını sağlar
def quick_missing_imp(data, num_method="median", cat_length=20, target="son müşteri statusu"):
    variables_with_na = [col for col in data.columns if
                         data[col].isnull().sum() > 0]  # Eksik değere sahip olan değişkenler listelenir

    temp_target = data[target]

    print("# BEFORE")
    print(data[variables_with_na].isnull().sum(), "\n\n")  # Uygulama öncesi değişkenlerin eksik değerlerinin sayısı

    # değişken object ve sınıf sayısı cat_lengthe eşit veya altındaysa boş değerleri mode ile doldur
    data = data.apply(lambda x: x.fillna(x.mode()[0]) if (x.dtype == "O" and len(x.unique()) <= cat_length) else x,
                      axis=0)

    # num_method mean ise tipi object olmayan değişkenlerin boş değerleri ortalama ile dolduruluyor
    if num_method == "mean":
        data = data.apply(lambda x: x.fillna(x.mean()) if x.dtype != "O" else x, axis=0)
    # num_method median ise tipi object olmayan değişkenlerin boş değerleri ortalama ile dolduruluyor
    elif num_method == "median":
        data = data.apply(lambda x: x.fillna(x.median()) if x.dtype != "O" else x, axis=0)

    data[target] = temp_target

    print("# AFTER \n Imputation method is 'MODE' for categorical variables!")
    print(" Imputation method is '" + num_method.upper() + "' for numeric variables! \n")
    print(data[variables_with_na].isnull().sum(), "\n\n")

    return data


df = quick_missing_imp(df, num_method="median", cat_length=17)

df.drop('son_6_aydaki_kart_sayisi_degisim',axis=1, inplace=True) # veri tipinde hata aldığım için silerek işlemlere devam ettim.
df.drop('hizmet bedeli oranı',axis=1, inplace=True) # veri tipinde hata aldığım için silerek işlemlere devam ettim.



##################
# Label Encoding & One-Hot Encoding
##################

cat_cols, cat_but_car, num_cols, num_but_cat = grab_col_names(df)


def label_encoder(dataframe, binary_col):
    labelencoder = LabelEncoder()
    dataframe[binary_col] = labelencoder.fit_transform(dataframe[binary_col])
    return dataframe

binary_cols = [col for col in df.columns if df[col].dtypes == "O"
               and len(df[col].unique()) == 2]

binary_cols

for col in binary_cols:
    label_encoder(df, col)


def one_hot_encoder(dataframe, categorical_cols, drop_first=False):
    dataframe = pd.get_dummies(dataframe, columns=categorical_cols, drop_first=drop_first)
    return dataframe

df = one_hot_encoder(df, ['hangi rakipten kazanıldı','müşteri alt sektör','segment','kartgrp','son sipariştarihi','ödeme tipi','bölge','i̇l'], drop_first=True)


df.head(3)

#Robust Scaler uygulayacağım değişkenleri seçtim.
df_num=df[["ciro","kart sayısı","kart başı ciro",
             "hizmet bedeli beklenti tutarı","sözleşme vadesi","sozlesmebitengunsayisi",
             "ziyaret_ort_sapma","sikayet_adedi_orani"]]

# Scaling işlemini uyguladığım veri setine x_transformed adını verdim.
col=df_num.columns
x_transformed=pd.DataFrame(RobustScaler().fit(df_num).transform(df_num), columns=col)
x_transformed.head(3)

##################################
# MODELLEME
##################################

y = df['son müşteri statusu']
X = df.drop(["son müşteri statusu", "müşteri kodu"], axis=1)

df.head(3)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=17)

#Random Forest uygulayalım.
from sklearn.ensemble import RandomForestClassifier

rf_model = RandomForestClassifier(random_state=46).fit(X_train, y_train)
y_pred = rf_model.predict(X_test)
accuracy_score(y_pred, y_test)

#Out[213]: 0.9957081545064378


# rf model kurulumu

# validasyon hatası, accuracy skoru, confusion matrix
cv_results = cross_val_score(rf_model, X_train, y_train, cv = 10, scoring= "accuracy")

print(cv_results.mean())
print(accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))


#Raporlama kısmı
cf_matrix = confusion_matrix(y_test, y_pred)
print(cf_matrix)
sns.heatmap(cf_matrix/np.sum(cf_matrix), annot=True,
            fmt='.2%', cmap='Blues');


#### True Negative --> %3.89 --> Churn olmayacağı tahmin edilmiş ve churn olmamış.
#### True Positive --> %95.68 --> Churn olacağı tahmin edilmiş ve churn olmuş.

#### False Positive --> %0.25 --> Churn olacağı tahmin edilmemiş ama churn olmamış.
#### False Negative --> %0.17 --> Churn olmayacağı tahmin edilmiş ama churn olmuş.
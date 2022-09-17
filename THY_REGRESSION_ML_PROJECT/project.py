# THY Regresyon Projesi
# İŞ PROBLEMİ:
# Türk Hava Yolları uçuşlarda bulunan sezonluk rezervasyonlu yolcu
# sayısını tahmin edebilmek için bir makine öğrenmesi regresyon
# modeli kurulmasını istemektedir.

# VERİ SETİ HİKAYESİ

#CARRIER Taşıyıcı firmanın ikili kodudur. 2 leg için de taşıyıcı firma aynıdır.
#AIRCRAFT_TYPE Uçak tipi bilgisidir.
#OND_SELL_CLASS En uzun legin satış sınıfıdır.
#LEG1_SELL_CLASS 1.leg in satış sınıfıdır.
#OND_CABIN_CLASS En uzun leg in kabin sınıfıdır.
#LEG1_CABIN_CLASS 1.leg in kabin sınıfıdır.
#HUB 1.leg uçuşundan 2.leg uçuşuna aktarma yapılan havalimanının IATA kodlarını içerir.
#DETUR_FACTOR Bir yolcunun uçmak istediği noktaya direkt uçmak yerine bağlantılı uçması durumunda yolunu ne kadar uzattığını ifade eden bir orandır.
#CONNECTION_TIME 1.leg uçuşundan 2.leg uçuşuna aktarma için beklenen süreyi dakika cinsinden ifade eder.
#PSGR_COUNT Toplam rezervasyonlu yolcu sayısıdır.
#LEG1_DEP_FULL 1. Leg’in kalkış tarih ve saatidir.
#LEG1_ARR_FULL 1. Leg’in varış tarih ve saatidir.
#LEG2_DEP_FULL 2. Leg’in kalkış tarih ve saatidir.
#LEG2_ARR_FULL 2. Leg’in varış tarih ve saatidir.
#LEG1_DURATION 1. Leg’in uçuş süresi.
#LEG2_DURATION 2. Leg’in uçuş süresi.
#FLIGHT_DURATION 1. Leg’in kalkış saatinden 2. Leg’in iniş saatine kadar geçen zaman
# Leg: 1 uçuşu temsil eder. 2 leg uçuş demek, 1 aktarmalı 2 adet uçuşu olan yolculuktur.
# Örnek: İstanbul - San Francisco uçuşunda Londra aktarması var ise; İstanbul - Londra 1. leg, Londra - San Francisco 2. legdir
########################################################################################################################

#İmport edilecekler.
import joblib
import numpy as np
import calendar
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier, AdaBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_validate, GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.model_selection import *
from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor
# !pip install catboost
# !pip install lightgbm
# !pip install xgboost

from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier

pd.set_option('display.max_columns', None)
pd.set_option('display.width', 500)
##############################################################################
#Yardımcı fonksiyonlar
def check_df(dataframe, head=5):
    print("##################### Shape #####################")
    print(dataframe.shape)
    print("##################### Types #####################")
    print(dataframe.dtypes)
    print("##################### Head #####################")
    print(dataframe.head(head))
    print("##################### Tail #####################")
    print(dataframe.tail(head))
    print("##################### NA #####################")
    print(dataframe.isnull().sum())
    print("##################### Quantiles #####################")
    print(dataframe.quantile([0, 0.05, 0.50, 0.95, 0.99, 1]).T)

def cat_summary(dataframe, col_name, plot=False):
    print(pd.DataFrame({col_name: dataframe[col_name].value_counts(),
                        "Ratio": 100 * dataframe[col_name].value_counts() / len(dataframe)}))
    print("##########################################")
    if plot:
        sns.countplot(x=dataframe[col_name], data=dataframe)
        plt.show(block=True)

def num_summary(dataframe, numerical_col, plot=False):
    quantiles = [0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 0.95, 0.99]
    print(dataframe[numerical_col].describe(quantiles).T)

    if plot:
        dataframe[numerical_col].hist(bins=20)
        plt.xlabel(numerical_col)
        plt.title(numerical_col)
        plt.show(block=True)

def target_summary_with_num(dataframe, target, numerical_col):
    print(dataframe.groupby(target).agg({numerical_col: "mean"}), end="\n\n\n")

def target_summary_with_cat(dataframe, target, categorical_col):
    print(pd.DataFrame({"TARGET_MEAN": dataframe.groupby(categorical_col)[target].mean()}), end="\n\n\n")

def correlation_matrix(df, cols):
    fig = plt.gcf()
    fig.set_size_inches(10, 8)
    plt.xticks(fontsize=10)
    plt.yticks(fontsize=10)
    fig = sns.heatmap(df[cols].corr(), annot=True, linewidths=0.5, annot_kws={'size': 12}, linecolor='w', cmap='RdBu')
    plt.show(block=True)

def grab_col_names(dataframe, cat_th=10, car_th=20):
    """
    Veri setindeki kategorik, numerik ve kategorik fakat kardinal değişkenlerin isimlerini verir.
    Not: Kategorik değişkenlerin içerisine numerik görünümlü kategorik değişkenler de dahildir.
    Parameters
    ------
        dataframe: dataframe
                Değişken isimleri alınmak istenilen dataframe
        cat_th: int, optional
                numerik fakat kategorik olan değişkenler için sınıf eşik değeri
        car_th: int, optinal
                kategorik fakat kardinal değişkenler için sınıf eşik değeri
    Returns
    ------
        cat_cols: list
                Kategorik değişken listesi
        num_cols: list
                Numerik değişken listesi
        cat_but_car: list
                Kategorik görünümlü kardinal değişken listesi
    Examples
    ------
        import seaborn as sns
        df = sns.load_dataset("iris")
        print(grab_col_names(df))
    Notes
    ------
        cat_cols + num_cols + cat_but_car = toplam değişken sayısı
        num_but_cat cat_cols'un içerisinde.
        Return olan 3 liste toplamı toplam değişken sayısına eşittir: cat_cols + num_cols + cat_but_car = değişken sayısı
    """

    # cat_cols, cat_but_car
    cat_cols = [col for col in dataframe.columns if dataframe[col].dtypes == "O"]
    num_but_cat = [col for col in dataframe.columns if dataframe[col].nunique() < cat_th and
                   dataframe[col].dtypes != "O"]
    cat_but_car = [col for col in dataframe.columns if dataframe[col].nunique() > car_th and
                   dataframe[col].dtypes == "O"]
    cat_cols = cat_cols + num_but_cat
    cat_cols = [col for col in cat_cols if col not in cat_but_car]

    # num_cols
    num_cols = [col for col in dataframe.columns if dataframe[col].dtypes != "O"]
    num_cols = [col for col in num_cols if col not in num_but_cat]

    # print(f"Observations: {dataframe.shape[0]}")
    # print(f"Variables: {dataframe.shape[1]}")
    # print(f'cat_cols: {len(cat_cols)}')
    # print(f'num_cols: {len(num_cols)}')
    # print(f'cat_but_car: {len(cat_but_car)}')
    # print(f'num_but_cat: {len(num_but_cat)}')
    return cat_cols, num_cols, cat_but_car

#Keşifci veri analizi

df = pd.read_csv("datasets/thyTrain.csv")
check_df(df)

# Değişken türlerinin ayrıştırılması
cat_cols, num_cols, cat_but_car = grab_col_names(df, cat_th=5, car_th=20)

# Target ile sayısal değişkenlerin incelemesi
for col in num_cols:
    target_summary_with_num(df, "PSGR_COUNT", col)

#ADIM 2: Target değişkeninin (PSGR_COUNT) standart sapma ve ortalama bilgilerini getiriniz.
df["PSGR_COUNT"].describe().T

#ADIM 3: Target değişkenini büyükten küçüğe sıralayıp veri setinin ilk 10 satırını getiriniz.
df["PSGR_COUNT"] = df["PSGR_COUNT"].sort_values()
df.head(10)

#ADIM 4: LEG1_DEP_FULL, LEG1_ARR_FULL, LEG2_DEP_FULL, LEG2_ARR_FULL değişkenlerinin min ve max değerlerini getiriniz.
LEG_LISTS = ["LEG1_DEP_FULL", "LEG1_ARR_FULL", "LEG2_DEP_FULL", "LEG2_ARR_FULL"]

for i in LEG_LISTS:
    print(i + " max= "+df[i].max() +" min= "+df[i].min())

#Adım 5: Her değişkendeki boş değer sayısını yazdırınız ve boş değer bulunan gözlemleri o değişkenin modu ile doldurunuz.
# degiskenlerdeki eksik deger sayisi
df.isnull().sum()
df["AIRCRAFT_TYPE"].fillna(df["AIRCRAFT_TYPE"].mode()[0], inplace=True)

#Adım 6: Aşağıda belirtilen değişkenleri oluşturunuz.
"""
LEG1_DEP_FULL değişkeninin ay bilgisi -> LEG1_DEP_MONTH
LEG1_DEP_FULL değişkeninin saat bilgisi -> LEG1_DEP_HOUR
LEG1_ARR_FULL değişkeninin ay bilgisi -> LEG1_ARR_MONTH
LEG1_ARR_FULL değişkeninin saat bilgisi -> LEG1_ARR_HOUR

LEG2_DEP_FULL değişkeninin ay bilgisi -> LEG2_DEP_MONTH
LEG2_DEP_FULL değişkeninin saat bilgisi -> LEG2_DEP_HOUR
LEG2_ARR_FULL değişkeninin ay bilgisi -> LEG2_ARR_MONTH
LEG2_ARR_FULL değişkeninin saat bilgisi ->LEG2_ARR_HOUR

LEG1_DEP_FULL değişkeninin haftanın günü bilgisi (Pazartesi, Salı,...) -> LEG1_DEP_DAY
LEG1_ARR_FULL değişkeninin haftanın günü bilgisi (Pazartesi, Salı,...) -> LEG1_ARR_DAY
LEG2_DEP_FULL değişkeninin haftanın günü bilgisi (Pazartesi, Salı,...) -> LEG2_DEP_DAY
LEG2_ARR_FULL değişkeninin haftanın günü bilgisi (Pazartesi, Salı,...) -> LEG2_ARR_DAY
LEG1_DURATION değişkenini dakika cinsine çeviriniz. (to_timedelta fonksiyonunu kullanarak zamana çevirip daha sonra dakikaya çevirme işlemini yapın.) (01:50 -> 110) -> LEG1_DURATION_MINUTES
LEG2_DURATION değişkenini dakika cinsine çeviriniz. (01:50 -> 110) -> LEG2_DURATION_MINUTES
FLIGHT_DURATION değişkenini dakika cinsine çeviriniz. (01:50 -> 110) -> FLIGHT_DURATION_MINUTES
LEG1_DURATION ile LEG2_DURATION değişkenlerini dakika cinsine çevirdikten sonra bu değişkenleri toplayınız. (LEG1_DURATION_MINUTES + LEG2_DURATION_MINUTES) (Aktarma için havalimanında bekleme süresi hariç uçuş süresi.) -> FLIGHT_DURATION_MINUTES_FLIGHTS
Aktarma dahil tüm uçuşun dakika cinsinden süresini aktarma hariç uçuş süresine böl. (FLIGHT_DURATION_MINUTES_FLIGHTS / FLIGHT_DURATION_MINUTES) -> CONNECTION_RATIO
LEG1_DURATION değişkeninin dakika cinsinden değerini tüm uçuşun dakika cinsinden değerine oranını bul. (LEG1_DURATION_MINUTES / FLIGHT_DURATION_MINUTES) -> LEG1_RATIO
LEG2_DURATION değişkeninin dakika cinsinden değerini tüm uçuşun dakika cinsinden değerine oranını bul. (LEG2_DURATION_MINUTES / FLIGHT_DURATION_MINUTES) -> LEG2_RATIO
(Üzerinde oynayacağımız değişkenleri datetime’a çevirmeyi unutmayınız.)
"""

df["LEG1_DEP_FULL"] = pd.to_datetime(df["LEG1_DEP_FULL"])
df["LEG1_ARR_FULL"] = pd.to_datetime(df["LEG1_ARR_FULL"])
df["LEG2_DEP_FULL"] = pd.to_datetime(df["LEG2_DEP_FULL"])
df["LEG2_ARR_FULL"] = pd.to_datetime(df["LEG2_ARR_FULL"])

df["LEG1_DEP_MONTH"] = df["LEG1_DEP_FULL"].dt.month
df["LEG1_DEP_HOUR"] = df["LEG1_DEP_FULL"].dt.hour

df["LEG1_ARR_MONTH"] = df["LEG1_ARR_FULL"].dt.month
df["LEG1_ARR_HOUR"] = df["LEG1_ARR_FULL"].dt.hour

df["LEG2_DEP_MONTH"] = df["LEG2_DEP_FULL"].dt.month
df["LEG2_DEP_HOUR"] = df["LEG2_DEP_FULL"].dt.hour

df["LEG2_ARR_MONTH"] = df["LEG2_ARR_FULL"].dt.month
df["LEG2_ARR_HOUR"] = df["LEG2_ARR_FULL"].dt.hour

df["LEG1_DEP_DAY"] = df["LEG1_DEP_FULL"].apply(lambda x: calendar.day_name[x.weekday()])
df["LEG1_ARR_DAY"] = df["LEG1_ARR_FULL"].apply(lambda x: calendar.day_name[x.weekday()])
df["LEG2_DEP_DAY"] = df["LEG2_DEP_FULL"].apply(lambda x: calendar.day_name[x.weekday()])
df["LEG2_ARR_DAY"] = df["LEG2_ARR_FULL"].apply(lambda x: calendar.day_name[x.weekday()])

df["LEG1_DURATION"] = pd.to_timedelta(df["LEG1_DURATION"])
df["LEG2_DURATION"] = pd.to_timedelta(df["LEG2_DURATION"])
df["FLIGHT_DURATION"] = pd.to_timedelta(df["FLIGHT_DURATION"])


df["LEG1_DURATION_MINUTES"] = df["LEG1_DURATION"].apply(lambda x: x.seconds / 60)
df["LEG2_DURATION_MINUTES"] = df["LEG2_DURATION"].apply(lambda x: x.seconds / 60)
df["FLIGHT_DURATION_MINUTES"] = df["FLIGHT_DURATION"].apply(lambda x: x.seconds / 60)

df["FLIGHT_DURATION_MINUTES_FLIGHTS"] = df["LEG1_DURATION_MINUTES"] + df["LEG2_DURATION_MINUTES"]

df["CONNECTION_RATIO"] = df["FLIGHT_DURATION_MINUTES_FLIGHTS"] / df["FLIGHT_DURATION_MINUTES"]

df["LEG1_RATIO"] = df["LEG1_DURATION_MINUTES"] / df["FLIGHT_DURATION_MINUTES"]
df["LEG2_RATIO"] = df["LEG2_DURATION_MINUTES"] / df["FLIGHT_DURATION_MINUTES"]

  #Adım 6.1: "LEG1_DEP_FULL", "LEG1_ARR_FULL", "LEG2_DEP_FULL", "LEG2_ARR_FULL", "LEG1_DURATION", "LEG2_DURATION", "FLIGHT_DURATION" değişkenlerini dataframe’den kaldırınız.
# Değişkenlerimizi ürettiğimiz için artık datetime tipli değişkenlerden kurtulabiliriz.

df.drop(["LEG1_DEP_FULL", "LEG1_ARR_FULL", "LEG2_DEP_FULL", "LEG2_ARR_FULL", "LEG1_DURATION", "LEG2_DURATION",
             "FLIGHT_DURATION"], axis=1, inplace=True)


#Adım 7: grab_col_names fonksiyonunu kullanarak değişkenleri sınıflandırınız.

cat_cols, num_cols, cat_but_car = grab_col_names(df, cat_th=5, car_th=20)

#Adım 7.1: num_cols değişkeninden PSGR_COUNT target değişkenini list comprehension kullanarak kaldırınız.

num_cols = [col for col in num_cols if "PSGR_COUNT" not in col]

#Adım 8: Outlier değerleri baskılayınız.

def outlier_thresholds(dataframe, col_name, q1=0.25, q3=0.75):
    quartile1 = dataframe[col_name].quantile(q1)
    quartile3 = dataframe[col_name].quantile(q3)
    interquantile_range = quartile3 - quartile1
    up_limit = quartile3 + 1.5 * interquantile_range
    low_limit = quartile1 - 1.5 * interquantile_range
    return low_limit, up_limit

def replace_with_thresholds(dataframe, variable, q1=0.25 , q3=0.75):
    low_limit, up_limit = outlier_thresholds(dataframe, variable, q1, q3)
    dataframe.loc[(dataframe[variable] < low_limit), variable] = low_limit
    dataframe.loc[(dataframe[variable] > up_limit), variable] = up_limit


def check_outlier(dataframe, col_name, q1=0.25, q3=0.75):
    low_limit, up_limit = outlier_thresholds(dataframe, col_name, q1, q3)
    if dataframe[(dataframe[col_name] > up_limit) | (dataframe[col_name] < low_limit)].any(axis=None):
        return True
    else:
        return False


for col in num_cols:
    replace_with_thresholds(df, col)

#Adım 8.1: Rare sınıfları tek bir sınıf haline getiriniz. (rare_perc = 0.05)


def rare_encoder(dataframe, rare_perc):
    temp_df = dataframe.copy()

    rare_columns = [col for col in temp_df.columns if temp_df[col].dtypes == 'O'
                    and (temp_df[col].value_counts() / len(temp_df) < rare_perc).any(axis=None)]

    for var in rare_columns:
        tmp = temp_df[var].value_counts() / len(temp_df)
        rare_labels = tmp[tmp < rare_perc].index
        temp_df[var] = np.where(temp_df[var].isin(rare_labels), 'Rare', temp_df[var])

    return temp_df



df = rare_encoder(df, 0.05)

#Adım 8.2: Kategorik değişkenlere One Hot encoding uygulayınız.


def one_hot_encoder(dataframe, categorical_cols, drop_first=False):
    dataframe = pd.get_dummies(dataframe, columns=categorical_cols, drop_first=drop_first)
    return dataframe


ohe_cols = [col for col in df.columns if (10 >= df[col].nunique() > 2) & (col != "PSGR_COUNT")]
df = one_hot_encoder(df, ohe_cols)

#Adım 8.3: Standard Scaler yapın.


scaler = StandardScaler()
df[num_cols] = scaler.fit_transform(df[num_cols])


#Adım 9: Cross Validation kullanarak makine öğrenmesi modellerini eğitin ve rmse değerlerini getirin.


y = df["PSGR_COUNT"]
X = df.drop(["PSGR_COUNT"], axis=1)

# MODEL

models = [("LR", LinearRegression()),
          ("Ridge", Ridge()),
          ("Lasso", Lasso()),
          ("ElasticNet", ElasticNet()),
          ("KNN", KNeighborsRegressor()),
          ("CART", DecisionTreeRegressor()),
          ("RF", RandomForestRegressor()),
          ("SVR", SVR()),
          ("GBM", GradientBoostingRegressor()),
          ("XGBoost", XGBRegressor(objective="reg:squarederror")),
          ("LightGBM", LGBMRegressor()),
          ("CatBoost", CatBoostRegressor(verbose=False))]

for name, regressor in models:
    rmse = np.mean(np.sqrt(-cross_val_score(regressor, X, y, cv=10, scoring="neg_mean_squared_error")))
    print(f"RMSE {name} :", rmse)

#Adım 9.1: Belirli algoritmalar üzerinde GridSearchCV uygulayın ve rmse değerlerini getirin.


rf_params = {"max_depth": [5, 8, None],
                 "max_features": [5, 7, "auto"],
                 "min_samples_split": [8, 20],  # 20nin üstlerini dene
                 "n_estimators": [200]}

xgboost_params = {"learning_rate": [0.1],
                      "max_depth": [5],
                      "n_estimators": [100],
                      "colsample_bytree": [0.5]}

lightgbm_params = {"learning_rate": [0.01, 0.1],
                       "n_estimators": [300, 500],
                       "colsample_bytree": [0.7, 1]}

regressors = [#("RF", RandomForestRegressor(), rf_params),
              #("XGBoost", XGBRegressor(objective="reg:squarederror"), xgboost_params),
              ("LightGBM", LGBMRegressor(), lightgbm_params)]

best_models = {}

for name, regressor, params in regressors:
    print(f"########## {name} ##########")

    gs_best = GridSearchCV(regressor, params, cv=3, n_jobs=-1, verbose=False).fit(X, y)

    final_model = regressor.set_params(**gs_best.best_params_)
    rmse = np.mean(np.sqrt(-cross_val_score(final_model, X, y, cv=10, scoring="neg_mean_squared_error")))
    print(f"RMSE (After): {round(rmse, 4)} ({name}) ")

    print(f"{name} best params: {gs_best.best_params_}", end="\n\n")

    best_models[name] = final_model

#Adım 10: Feature Importance tablosunu yazdırınız.


# FEATURE IMPORTANCE

def plot_importance(model, features, num=len(X), save=False):

    feature_imp = pd.DataFrame({"Value": model.feature_importances_, "Feature": features.columns})
    plt.figure(figsize=(10, 10))
    sns.set(font_scale=1)
    sns.barplot(x="Value", y="Feature", data=feature_imp.sort_values(by="Value",
                                                                         ascending=False)[0:num])
    plt.title("Features")
    plt.tight_layout()
    plt.show()
    if save:
        plt.savefig("importances.png")

model = LGBMRegressor()
model.fit(X, y)

plot_importance(model, X)


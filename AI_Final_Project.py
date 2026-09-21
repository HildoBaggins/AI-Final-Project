# AI Final Project
# Hilliard Domangue
# Extreme weather preditor

# I ran this code on VS code

import numpy as np
import matplotlib.pyplot as plt
import glob
import pandas as pd
import sklearn
import sklearn.cluster
import sklearn.metrics
import sklearn.preprocessing


def plot(data, kmeans, f1, f2, size=200, title="KMeans Clustering"):
    plt.figure(figsize=(8, 6))
    plt.scatter(
        data[f1], data[f2], c=data["Cluster"], cmap="viridis", s=50, label="Data Points"
    )
    plt.scatter(
        kmeans.cluster_centers_[:, 0],
        kmeans.cluster_centers_[:, 1],
        s=size,
        c="red",
        marker="X",
        label="Centers",
    )
    plt.title(title)
    plt.xlabel(f1)
    plt.ylabel(f2)
    plt.legend()
    plt.show()


year = 0
num_of_files = 10
encode = sklearn.preprocessing.OneHotEncoder(dtype=float)
scale = sklearn.preprocessing.StandardScaler

# data files for different years but the same month(September)
datafiles = np.empty(num_of_files, dtype=object)
datafiles[0] = glob.glob(
    "/Users/hillarddomangue/Documents/AI/Final_Project/Data/201509/*.csv"
)
datafiles[1] = glob.glob(
    "/Users/hillarddomangue/Documents/AI/Final_Project/Data/201609/*.csv"
)
datafiles[2] = glob.glob(
    "/Users/hillarddomangue/Documents/AI/Final_Project/Data/201709/*.csv"
)
datafiles[3] = glob.glob(
    "/Users/hillarddomangue/Documents/AI/Final_Project/Data/201809/*.csv"
)
datafiles[4] = glob.glob(
    "/Users/hillarddomangue/Documents/AI/Final_Project/Data/201909/*.csv"
)
datafiles[5] = glob.glob(
    "/Users/hillarddomangue/Documents/AI/Final_Project/Data/202009/*.csv"
)
datafiles[6] = glob.glob(
    "/Users/hillarddomangue/Documents/AI/Final_Project/Data/202109/*.csv"
)
datafiles[7] = glob.glob(
    "/Users/hillarddomangue/Documents/AI/Final_Project/Data/202209/*.csv"
)
datafiles[8] = glob.glob(
    "/Users/hillarddomangue/Documents/AI/Final_Project/Data/202309/*.csv"
)
datafiles[9] = glob.glob(
    "/Users/hillarddomangue/Documents/AI/Final_Project/Data/202409/*.csv"
)

# This step takes a while (over 400 million dataframes)
# The data was greatly reduced because of time and hardware constraints
# From dataset at https://www.ncei.noaa.gov/data/global-marine/archive/

df1 = pd.read_csv(
    datafiles[0][0],
    na_values="?",
)
df2 = pd
for i in range(num_of_files):
    for j in datafiles[i]:

        if year % 5 == 0:
            print("year: 20{}".format(15 + i))
        year += 1

        df2 = pd.read_csv(j, na_values="?")
        df1 = pd.concat([df1, df2])
        print("size of data: {}".format(df1.size))

# Preprocessing
df1 = df1.fillna(0)

df1 = df1.drop("TRIM_FLAG", axis=1)
df1 = df1.drop("NCDC_QC_FLAGS", axis=1)
df1 = df1.drop("HI_CLD_TYPE", axis=1)
df1 = df1.drop("DUR_OF_PER", axis=1)
df1 = df1.drop("VV_IND", axis=1)
df1 = df1.drop("WAVE_DIR", axis=1)
df1 = df1.drop("ICE_ACCR_ON_SHIP", axis=1)
df1 = df1.drop("DPT_IND", axis=1)
df1 = df1.drop("AMT_PRECIP", axis=1)
df1 = df1.drop("ELEVATION", axis=1)
df1 = df1.drop("NAME", axis=1)
df1 = df1.drop("CLD_HGT", axis=1)
df1 = df1.drop("ICE_SIT_TREND", axis=1)
df1 = df1.drop("MID_CLD_TYPE", axis=1)
df1 = df1.drop("LOW_CLD_TYPE", axis=1)
df1 = df1.drop("RATE_OF_I", axis=1)
df1 = df1.drop("STAGE_OF_DEVELP", axis=1)
df1 = df1.drop("TRUE_BEARING_ICE_EDGE", axis=1)
df1 = df1.drop("ICE_OF_LAND_ORIGIN", axis=1)
df1 = df1.drop("STATION", axis=1)
df1 = df1.drop("DATE", axis=1)

# KMeans with a couple of variables that I thought might be important
kmeans = sklearn.cluster.KMeans(n_clusters=2, random_state=0)
kmeans.fit(df1)
df1["Cluster"] = kmeans.fit_predict(df1)

plot(df1, kmeans, "AIR_TEMP", "LONGITUDE", 200, "KMeans Clustering 1")
plot(df1, kmeans, "WIND_SPEED", "SEA_SURF_TEMP", 200, "KMeans Clustering 2")

# PCA with KMeans
pca = sklearn.decomposition.PCA(n_components=2)
principal_components = pca.fit_transform(df1)

pca_df = pd.DataFrame(data=principal_components, columns=["PC1", "PC2"])
kmeans = sklearn.cluster.KMeans(n_clusters=2, random_state=0)
kmeans.fit(pca_df)
pca_df["Cluster"] = kmeans.fit_predict(pca_df)
plot(pca_df, kmeans, "PC1", "PC2", 200, "KMeans Clustering w/ PCA")

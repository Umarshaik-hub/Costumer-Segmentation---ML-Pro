# Customer Segmentation using K-Means
# CSV columns expected: Customer_ID, Age, Gender, Annual_Income, Spending_Score, Purchase_Frequency

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

df=pd.read_csv("customers.csv").dropna().copy()
features=["Annual_Income","Spending_Score","Age","Purchase_Frequency"]
X=df[features]

scaler=StandardScaler()
X_scaled=scaler.fit_transform(X)

# Elbow + silhouette
inertias=[]; silhouettes=[]
ks=range(2,11)
for k in ks:
    km=KMeans(n_clusters=k,random_state=42,n_init=10)
    labels=km.fit_predict(X_scaled)
    inertias.append(km.inertia_)
    silhouettes.append(silhouette_score(X_scaled,labels))

plt.plot(list(ks),inertias,marker="o")
plt.xlabel("K");plt.ylabel("WCSS");plt.title("Elbow Method");plt.show()

plt.plot(list(ks),silhouettes,marker="o")
plt.xlabel("K");plt.ylabel("Silhouette Score");plt.title("Silhouette Analysis");plt.show()

# Set K after inspecting the plots
K=4
kmeans=KMeans(n_clusters=K,random_state=42,n_init=10)
df["Cluster"]=kmeans.fit_predict(X_scaled)

print(df.groupby("Cluster")[features].mean())
sns.scatterplot(data=df,x="Annual_Income",y="Spending_Score",hue="Cluster",palette="tab10")
plt.title("Customer Segments");plt.show()

df.to_csv("customer_segments.csv",index=False)

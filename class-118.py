import pandas as pd
df = pd.read_csv("petals_sepal.csv")
print(df.head())
import plotly.express as px
fig=px.scatter(df,x="battale_sixe",y="sappale_size")
fig.show()
from sklearn.cluster import KMeans
x=df.iLock[:,[0,1]].values
wcss=[]
for i in range(1,11):
    kmeans=KMeans(n_clusters=i,init="k-means++",random_states=42)
    kmeans.fit(x)
    wcss.append(kmeans.inertia)

#import pickle
import numpy as np
#fp=open("result.pkl","rb")
#obj=pickle.load(fp)
import joblib
obj=joblib.load("result.jbl")
vec=[e[0] for e in obj if e.shape[1]>0]
#print(len(obj))
for e in obj:
    if(len(e.shape)!=3):
        print(e.shape)
out=np.concatenate(vec,0)
print(out.shape)

#print(obj[0][0].shape)

X=out
idx=np.arange(X.shape[0])
np.random.shuffle(idx)

import umap
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

idx=idx[:1000]
print(X.shape)
umap = umap.UMAP(n_components=2, random_state=0)
X_umap = umap.fit_transform(X[idx])
plt.figure(figsize=(32, 32))
plt.scatter(X_umap[:, 0], X_umap[:, 1], cmap='jet', alpha=0.5, s=3 )
plt.title("UMAP")
plt.savefig("umap.png")

pca = PCA(n_components=2, random_state=0)
X_pca = pca.fit_transform(X[idx])
plt.figure(figsize=(32, 32))
plt.scatter(X_pca[:, 0], X_pca[:, 1], cmap='jet', alpha=0.5, s=3 )
plt.title("PCA")
plt.savefig("pca.png")


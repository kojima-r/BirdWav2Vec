import joblib
import numpy as np
"""
obj=joblib.load("result.jbl")
out=[]
for e in obj:
    if e.shape[1]>0:
        v=np.mean(e[0],0)
        print(v.shape)
        out.append(v)
    else:
        out.append(None)

joblib.dump(out,"result_vec.jbl")
"""
out=joblib.load("result_vec.jbl")
labels=joblib.load("labels.jbl")

X=np.array([e for e in out if e is not None])
idx=np.arange(len(X))
import umap
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

plt.rcParams["font.size"] = 8

Y_label=np.array([l["label"] for l in labels])
Y_type =np.array([l["type"]=="br" for l in labels])
Y_s =[l["description"] for l in labels]
print("X:",X.shape)
print("Y:",Y_label.shape)

umap = umap.UMAP(n_components=2, random_state=0)
X_umap = umap.fit_transform(X[idx])
pca = PCA(n_components=2, random_state=0)
X_pca = pca.fit_transform(X[idx])

plt.figure(figsize=(64, 64))
plt.scatter(X_umap[:, 0], X_umap[:, 1], c=Y_label ,cmap='jet', alpha=0.5, s=3 )
for i, s in enumerate(Y_s):
    plt.text(X_umap[i,0], X_umap[i,1],s)
plt.title("UMAP")
plt.savefig("umap_v_l_txt.png")
print("[SAVE]","umap_v_l_txt.png")
plt.clf()

plt.figure(figsize=(32, 32))
plt.scatter(X_umap[:, 0], X_umap[:, 1], c=Y_label ,cmap='jet', alpha=0.5, s=3 )
plt.title("UMAP")
plt.savefig("umap_v_l.png")
print("[SAVE]","umap_v_l.png")
plt.clf()


plt.figure(figsize=(32, 32))
plt.scatter(X_umap[:, 0], X_umap[:, 1], c=Y_type ,cmap='jet', alpha=0.5, s=3 )
plt.title("UMAP")
plt.savefig("umap_v_t.png")
print("[SAVE]","umap_v_t.png")
plt.clf()

plt.figure(figsize=(64, 64))
plt.scatter(X_pca[:, 0], X_pca[:, 1],  c=Y_label ,cmap='jet', alpha=0.5, s=3 )
for i, s in enumerate(Y_s):
    plt.text(X_pca[i,0], X_pca[i,1],s)
plt.title("PCA")
plt.savefig("pca_v_l_txt.png")
print("[SAVE]","pca_v_l_txt.png")

plt.figure(figsize=(32, 32))
plt.scatter(X_pca[:, 0], X_pca[:, 1],  c=Y_label ,cmap='jet', alpha=0.5, s=3 )
plt.title("PCA")
plt.savefig("pca_v_l.png")
print("[SAVE]","pca_v_l.png")

plt.figure(figsize=(32, 32))
plt.scatter(X_pca[:, 0], X_pca[:, 1],  c=Y_type ,cmap='jet', alpha=0.5, s=3 )
plt.title("PCA")
plt.savefig("pca_v_t.png")
print("[SAVE]","pca_v_t.png")

#plt.figure(figsize=(32, 32))
#plt.scatter(X_pca[:, 0], X_pca[:, 1],  c=Y_label ,cmap='jet', alpha=0.5, s=3 )


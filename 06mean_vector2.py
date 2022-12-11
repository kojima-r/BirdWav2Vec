import joblib
import numpy as np

out=joblib.load("result_vec.jbl")
labels=joblib.load("labels.jbl")

X=np.array([e for e in out if e is not None])
import umap
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

plt.rcParams["font.size"] = 8

data={}
data_info={}
for i,l in enumerate(labels):
    vec=X[i]
    s=l["description"]
    if s not in data:
        data[s]=[]
        data_info[s]=l
    data[s].append(vec)
data_v=[]
labels=[]
label_mapping={}
for k,v in data.items():
    vv=np.mean(v,axis=0)
    data_v.append(vv)
    data_info[k]["vector"]=vv.tolist()
    key=(data_info[k]["type"],data_info[k]["label"])
    if key not in label_mapping:
        label_mapping[key]=len(label_mapping)
    new_label=label_mapping[key]
    data_info[k]["label"]=new_label
    labels.append(data_info[k])


import joblib
import json

joblib.dump(labels, 'result_mean_vec.jbl', compress=3)
path='result_mean_vec.json'
fp = open(path, "w")
json.dump(labels, fp)

X=np.array(data_v)
Y_label=np.array([l["label"] for l in labels])
Y_type =np.array([l["type"]=="br" for l in labels])
Y_s =[l["description"] for l in labels]
print("X:",X.shape)
print("Y:",Y_label.shape)

idx=np.arange(len(X))
umap = umap.UMAP(n_components=2, random_state=0)
X_umap = umap.fit_transform(X[idx])
pca = PCA(n_components=2, random_state=0)
X_pca = pca.fit_transform(X[idx])

plt.figure(figsize=(64, 64))
plt.scatter(X_umap[:, 0], X_umap[:, 1], c=Y_label ,cmap='jet', alpha=0.5, s=3 )
for i, s in enumerate(Y_s):
    plt.text(X_umap[i,0], X_umap[i,1],s)
plt.title("UMAP")
plt.savefig("umap_mv_l_txt.png")
print("[SAVE]","umap_mv_l_txt.png")
plt.clf()

plt.figure(figsize=(32, 32))
plt.scatter(X_umap[:, 0], X_umap[:, 1], c=Y_label ,cmap='jet', alpha=0.5, s=3 )
plt.title("UMAP")
plt.savefig("umap_mv_l.png")
print("[SAVE]","umap_mv_l.png")
plt.clf()


plt.figure(figsize=(32, 32))
plt.scatter(X_umap[:, 0], X_umap[:, 1], c=Y_type ,cmap='jet', alpha=0.5, s=3 )
plt.title("UMAP")
plt.savefig("umap_mv_t.png")
print("[SAVE]","umap_mv_t.png")
plt.clf()

plt.figure(figsize=(64, 64))
plt.scatter(X_pca[:, 0], X_pca[:, 1],  c=Y_label ,cmap='jet', alpha=0.5, s=3 )
for i, s in enumerate(Y_s):
    plt.text(X_pca[i,0], X_pca[i,1],s)
plt.title("PCA")
plt.savefig("pca_mv_l_txt.png")
print("[SAVE]","pca_mv_l_txt.png")

plt.figure(figsize=(32, 32))
plt.scatter(X_pca[:, 0], X_pca[:, 1],  c=Y_label ,cmap='jet', alpha=0.5, s=3 )
plt.title("PCA")
plt.savefig("pca_mv_l.png")
print("[SAVE]","pca_mv_l.png")

plt.figure(figsize=(32, 32))
plt.scatter(X_pca[:, 0], X_pca[:, 1],  c=Y_type ,cmap='jet', alpha=0.5, s=3 )
plt.title("PCA")
plt.savefig("pca_mv_t.png")
print("[SAVE]","pca_mv_t.png")

#plt.figure(figsize=(32, 32))
#plt.scatter(X_pca[:, 0], X_pca[:, 1],  c=Y_label ,cmap='jet', alpha=0.5, s=3 )


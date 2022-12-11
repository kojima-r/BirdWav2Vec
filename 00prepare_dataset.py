import datasets
import numpy as np
from datasets import load_dataset
name1="kojima-r/birdbr"
name2="kojima-r/birdjpbook"
dataset1 = load_dataset(name1, split='train')
dataset2 = load_dataset(name2, split='train')
print("br>",len(dataset1))
print("jpbook>",len(dataset2))
new_c = np.array(["br"] * len(dataset1))
dataset1 = dataset1.add_column("type", new_c)
new_c = np.array(["jpbook"] * len(dataset2))
dataset2 = dataset2.add_column("type", new_c)
print(dataset1[0])
print(dataset2[0])
ds=datasets.concatenate_datasets([dataset1,dataset2])
print(ds[0])
print("jp_all>",len(ds))
ds.push_to_hub("kojima-lab/bird-jp-all")
#

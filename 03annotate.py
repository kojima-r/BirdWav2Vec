from datasets import DatasetDict, concatenate_datasets, load_dataset
from torch.utils.data.dataloader import DataLoader
from transformers import (
    AdamW,
    SchedulerType,
    Wav2Vec2Config,
    Wav2Vec2FeatureExtractor,
    Wav2Vec2ForPreTraining)

import numpy as np

import datasets
from datasets import load_dataset
def prepare_dataset(batch):
    audio = batch["audio"]
    if len(audio["array"])>=44100*0.1:
        return True
    return False
name="kojima-lab/bird-jp-all"
dataset = load_dataset(name, split='train')
dataset = dataset.filter(prepare_dataset)

dataset = dataset.cast_column(
    "audio", datasets.features.Audio(sampling_rate=16000)
    )

print(len(dataset))
print(dataset[0])

idx=np.arange(len(dataset))
labels=[]
for i in idx:
    i=int(i)
    obj=dataset[i]
    ll={
        "label":obj['label'],
        "description": obj['description'],
        "type":obj['type'],
        }
    labels.append(ll)
    #print(">>",i)
    #sound_data = dataset[i]['audio']['array']
import joblib
joblib.dump(labels, 'labels.jbl', compress=3)
#joblib.dump(result_q_list, 'result_q.jbl', compress=3)


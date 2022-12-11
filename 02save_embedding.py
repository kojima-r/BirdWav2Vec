from datasets import DatasetDict, concatenate_datasets, load_dataset
from torch.utils.data.dataloader import DataLoader
from transformers import (
    AdamW,
    SchedulerType,
    Wav2Vec2Config,
    Wav2Vec2FeatureExtractor,
    Wav2Vec2ForPreTraining)

import numpy as np

processor = Wav2Vec2FeatureExtractor.from_pretrained("patrickvonplaten/wav2vec2-base-v2")
#feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained("./wav2vec2-birddb")
#config = Wav2Vec2Config.from_pretrained("wav2vec2-birddb")
#model = Wav2Vec2ForPreTraining(config)
model = Wav2Vec2ForPreTraining.from_pretrained("./wav2vec2-bird-jp-all")
#print(config)
print(model)

import datasets
from datasets import load_dataset
def prepare_dataset(batch):
    audio = batch["audio"]
    if len(audio["array"])>=44100*0.1:
        return True
    return False
name="kojima-r/bird-jp-all"
dataset = load_dataset(name, split='train')
dataset = dataset.filter(prepare_dataset)

dataset = dataset.cast_column(
    "audio", datasets.features.Audio(sampling_rate=processor.sampling_rate)
    )

print(len(dataset))
print(dataset[0])

idx=np.arange(len(dataset))
#np.random.shuffle(idx)
#sound_data = [dataset[i]['audio']['array'] for i in range(3)]
result_list=[]
result_q_list=[]
for i in idx:
    i=int(i)
    print(">>",i)
    print(dataset[i])
    sound_data = dataset[i]['audio']['array']

    input_values = processor(sound_data, sampling_rate=16000, return_tensors="pt").input_values
    result=model(input_values)
    #logits = result.logits#
    #states=result.last_hidden_state
    #feat=result.extract_features

    #print(states.shape)
    #print(result)
    st   = result.projected_states.detach().numpy()
    st_q = result.projected_quantized_states.detach().numpy()

    result_list.append(st)
    result_q_list.append(st_q)
    #print(result.projected_states.detach().numpy().shape)
    #print(">>",result.projected_quantized_states.detach().numpy().shape)
import joblib
joblib.dump(result_list, 'result.jbl', compress=3)
joblib.dump(result_q_list, 'result_q.jbl', compress=3)
#import pickle
#with open("result.pkl", mode="wb") as fp:
#    pickle.dump(result_list, fp)


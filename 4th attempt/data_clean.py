import os
import pickle
from collections import Counter
import pandas as pd
from random import shuffle

def load_data(directory):
    all_data=[]
    for file in os.listdir(directory):
        if file.endswith('.pkl'):
            with open(os.path.join(directory,file),'rb') as f:
                data=pickle.load(f)
                shuffle(data)
                all_data.extend(data)
    return all_data

def clean_and_undersample(data):
    allowed_inputs={(1,0,0,0),(0,0,1,0),(0,0,0,1)}

    filtered_data=[]
    for image,inputs in data:
        print(image)
        print(inputs)
        if tuple(inputs) in {(0,0,1,0),(1,0,1,0)}:
            inputs=[0,0,1,0]
        elif tuple(inputs) in {(0,0,0,1),(1,0,0,1)}:
            inputs=[0,0,0,1]
        if tuple(inputs) in allowed_inputs:
            filtered_data.append((image,inputs))

    input_counts=Counter([tuple(inputs) for _,inputs in filtered_data])
    min_count=min(input_counts.values())
    undersampled_data=[]
    input_samples={inputs:0 for inputs in allowed_inputs}
    for image,inputs in filtered_data:
        if input_samples[tuple(inputs)]<min_count:
            undersampled_data.append((image,inputs))
            input_samples[tuple(inputs)]+=1
    return undersampled_data

data_directory='./data'
data=load_data(data_directory)
balanced_data=clean_and_undersample(data)
df=pd.DataFrame(balanced_data)
df.to_pickle('cleaned_data.pkl')
print(df.head())
print(Counter(df[1].apply(str)))
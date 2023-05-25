import numpy as np
import pandas
import torch
import torchaudio
import soundfile as sf
import timeit

# Takes the directory with the data and loads it
def load_data(directory):
    df = pandas.read_csv(directory+'/train_metadata.csv')
    audio_ogg = []
    for file in df['filename']:
         audio_ogg.append(sf.read(directory+"/train_audio/"+file))


    audio_ogg = np.asarray(audio_ogg)
    df['audio1'] = audio_ogg[:,0]
    df['audio2'] = audio_ogg[:,1]
    df['audio3'] = audio_ogg[:,2]
    chosen_coloumns = ['latitude', 'longitude', 'common_name', 'rating', 'time', 'audio']
    return df[chosen_coloumns]

path = r"C:\Users\zhakk\OneDrive\Skrivebord\Uni\Kandidat\AML-Final\BirdCLEFData"

# Takes the directory with the data and loads it
def load_audiofile(directory):
    df = pandas.read_csv(directory+'/train_metadata.csv')
    audio_ogg = []
    for file in df['filename']:
         audio_ogg.append(sf.read(directory+"/train_audio/"+file))


    audio_ogg = np.asarray(audio_ogg)
    df['audio1'] = audio_ogg[:,0]
    df['audio2'] = audio_ogg[:,1]
    df['audio3'] = audio_ogg[:,2]
    chosen_coloumns = ['latitude', 'longitude', 'common_name', 'rating', 'time', 'audio']
    return df[chosen_coloumns]

path = r"C:\Users\zhakk\OneDrive\Skrivebord\Uni\Kandidat\AML-Final\BirdCLEFData"

df = load_data(path)
print(df)

df.to_csv(path)
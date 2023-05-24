import pandas
import torch
import torchaudio
import soundfile as sf

# Takes the directory with the data and loads it
def load_data(directory):
    df = pandas.read_csv(directory+'/train_metadata.csv')
    # audio_ogg = []
    #for file in df['filename']:
    #     audio_ogg.append(sf.read(directory+"/"+file))
    df['audio'] = audio_ogg[0,:]
    df['audio'] = audio_ogg[1,:]
    df['audio'] = audio_ogg[2,:]
    chosen_coloumns = ['latitude', 'longitude', 'common_name', 'rating', 'time', 'audio']
    return df[chosen_coloumns]

df = load_data(r'C:\Users\birk\OneDrive - University of Copenhagen\Documents\KU tid\AppML\APPML-BirdCLEF\data')
print(df)
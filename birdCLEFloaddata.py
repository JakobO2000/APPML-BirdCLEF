import pandas
import torch
import torchaudio
import soundfile as sf
print("Hello world")
print(torch.__version__)
print(torchaudio.__version__)
# Takes the directory with the data and loads it
def load_data(directory):
    df = pandas.DataFrame(directory+"/train_metadata.csv")
    audio_ogg = []
    for file in df['filename']:
        audio_ogg.append(sf.read(directory+"/"+file))
    df['audio'] = audio_ogg
    chosen_coloumns = ['latitude', 'longitude', 'common_name', 'rating', 'time', 'audio']
    return df[chosen_coloumns]
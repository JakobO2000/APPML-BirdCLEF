import numpy as np
import pandas
import torch
import torchaudio
import soundfile as sf
import timeit
import librosa

# Takes the directory with the data and returns pandas with metadata
def load_metadata(directory):
    df = pandas.read_csv(directory+'/train_metadata.csv')
    df['filename'] = directory+"/train_audio/"+df['filename']
    chosen_coloumns = ['latitude', 'longitude', 'common_name', 'rating', 'filename']
    return df[chosen_coloumns]


# Takes filepath from metadata dataframe and returns audio file
def load_audiofile(filepath):
    audio, sr = librosa.load(filepath)
    return audio, sr


# Converts ogg audio to waveform and spectrogram. Exact values for melspectrogram function might need to be changed values currently chosen from https://www.kaggle.com/code/awsaf49/birdclef23-pretraining-is-all-you-need-train
# audio -- Can be filepath from metadata dataframe or numpy array with ogg data
def get_melspectrogram(audio, sr=22050):
    if type(audio) is str:
        audio, sr = load_audiofile(audio)

    melspectrogram = librosa.feature.melspectrogram(y=audio, 
                                    sr=sr, 
                                    n_mels=128,
                                    n_fft=2028,
                                    hop_length=512, #base value from function in notebook it is calculated as duration_of_audio*sr//(384-1)
                                    fmax=11000,
                                    fmin=20,
                                    )
    melspectrogram = librosa.power_to_db(melspectrogram, ref=1.0)
    return melspectrogram

#Calculates Short Time Fourier Transformation of an audio file
# audio -- Can be filepath from metadata dataframe or numpy array with ogg data
def get_STFT(audio, sr=22050):
    if type(audio) is str:
        audio, sr = load_audiofile(audio)
    stft_audio = librosa.stft(audio, n_fft=2028, hop_length=512)
    return stft_audio


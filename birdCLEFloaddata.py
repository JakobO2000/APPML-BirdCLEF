import numpy as np
import pandas
import torch
import torchaudio
import soundfile as sf
from scipy.signal import stft, lfilter

# Takes the directory with the data and returns pandas with metadata
def load_metadata(directory, trim=False):
    if trim:
        df = pandas.read_csv(directory+'/train_metadata_trim.csv')
    else:
        df = pandas.read_csv(directory+'/train_metadata.csv')
    df['filename'] = directory+"/train_audio/"+df['filename']
    chosen_coloumns = ['latitude', 'longitude', 'common_name', 'rating', 'filename']
    return df[chosen_coloumns]


# Takes filepath from metadata dataframe and returns audio file
def load_audiofile(filepath):
    audio, sr = sf.read(filepath)
    if len(audio.shape) > 1:
        audio = np.mean(audio, axis=1)
    return audio.astype(np.float32), sr

def pcen(E, alpha=0.98, delta=2, r=0.5, s=0.025, eps=1e-6):   
    M = lfilter([s], [1, s - 1], E)
    smooth = (eps + M)**(-alpha)
    return (E * smooth + delta)**r - delta**r
# Converts ogg audio to waveform and spectrogram. Exact values for melspectrogram function might need to be changed values currently chosen from https://www.kaggle.com/code/awsaf49/birdclef23-pretraining-is-all-you-need-train
# audio -- Can be filepath from metadata dataframe or numpy array with ogg data
def get_melspectrogram(audio, sr=32000, n_mels=128, n_fft=2028, hop_length=512, fmax=16000, fmin=20,power=2.0,top_db=100):
    if type(audio) is str:
        audio, sr = load_audiofile(audio)
    waveform = torch.from_numpy(audio)
    transform = torchaudio.transforms.MelSpectrogram( 
                                    sample_rate=sr, 
                                    n_mels=n_mels,
                                    n_fft=n_fft,
                                    hop_length=hop_length, #base value from function in notebook it is calculated as duration_of_audio*sr//(384-1)
                                    f_max=fmax,
                                    f_min=fmin,
                                    power=2.0
                                    )
    melspectrogram = transform(waveform)

    melspectrogram = torchaudio.transforms.AmplitudeToDB()(melspectrogram)

    melspectrogram = torch.nn.functional.normalize(melspectrogram, p=2, dim=0)
    melspectrogram = (melspectrogram * 255)

    return melspectrogram

#Calculates Short Time Fourier Transformation of an audio file
# audio -- Can be filepath from metadata dataframe or numpy array with ogg data
def get_STFT(audio, sr=32000, n_fft=2028, nperseg=512):
    if type(audio) is str:
        audio, sr = load_audiofile(audio)
    stft_audio = stft(audio, nfft=n_fft, nperseg=nperseg)
    return stft_audio


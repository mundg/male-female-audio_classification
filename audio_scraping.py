
import numpy as np
import pandas as pd
import scipy.stats
import librosa

import os
import requests
from bs4 import BeautifulSoup

import tempfile
import io 
import tarfile


def get_readme_info(tar_object):
    # Default values 
    metadata_dict = {}
    user_name = 'no_user_name' 
    gender = 'no_gender'
    age_range = 'no_age_range'
    language = 'no_language'
    dialect = 'no_dialect'

    readme_members = [file for file in tar_object.getmembers() if "README" in file.name]
    if not readme_members:
        return {'user_name': user_name, 'gender': gender, 'age_range': age_range, 'language': language, 'dialect': dialect}

    readme_file = tar_object.extractfile(readme_members[0])
    if readme_file is None:
        return {'user_name': user_name, 'gender': gender, 'age_range': age_range, 'language': language, 'dialect': dialect}

    try:
        readme_txt = readme_file.read().decode('utf-8', errors='ignore').lower()
        for line in readme_txt.split("\n"):
            if "user name" in line:
                user_name = line.split(":")[1].strip()

            if "gender" in line:
                gender = line.split(":")[1].strip()

            if "age range" in line:
                age_range = line.split(":")[1].strip()

            if "language" in line: 
                language = line.split(":")[1].strip()

            if "dialect" in line: 
                dialect = line.split(":")[1].strip()
    except Exception as e:
        print(f"Error reading README file: {e}")

    # Store values
    metadata_dict['user_name'] = user_name
    metadata_dict['gender'] = gender
    metadata_dict['age_range'] = age_range
    metadata_dict['language'] = language
    metadata_dict['dialect'] = dialect

    return metadata_dict

def load_base_features_weighted(temp_file_path):    
    y, sr = librosa.load(temp_file_path, sr = None)
    features_dict = {}

    # Spectogram
    stft = np.abs(librosa.stft(y))
    # Frequency Bins
    frequencies = librosa.fft_frequencies(sr=sr)
    frequencies_khz = frequencies/1000
    # Magnitude Spectrum
    magnitude_spectrum = np.mean(stft, axis=1)

    # Calculate Mean Frequency
    weighted_frequencies = frequencies_khz * magnitude_spectrum
    mean_freq = np.sum(weighted_frequencies) / np.sum(magnitude_spectrum)
    features_dict['mean_freq_khz'] = mean_freq

    # Peak Frequency (Khz)
    peak_freq = frequencies_khz[np.argmax(magnitude_spectrum)]
    features_dict['peak_freq_khz'] = peak_freq

    # Frequency Distribution (Khz)
    weighted_frequencies_dist = np.repeat(frequencies_khz, np.round(magnitude_spectrum * 100).astype(int))
    human_speech_filter =  np.logical_and(weighted_frequencies_dist >= 0.085, weighted_frequencies_dist <= 0.255)
    freq_dist = weighted_frequencies_dist[human_speech_filter]

    features_dict['std_freq_khz'] =  np.std(freq_dist)
    features_dict['median_freq_khz'] =  np.median(freq_dist)
    features_dict['q1_freq_khz'] =  np.percentile(freq_dist, 25)
    features_dict['q3_freq_khz'] =  np.percentile(freq_dist, 75)
    features_dict['iqr_freq_khz'] =  np.percentile(freq_dist, 75) - np.percentile(freq_dist, 25)
    features_dict['skewness'] =  scipy.stats.skew(freq_dist)
    features_dict['kurtosis'] =  scipy.stats.kurtosis(freq_dist)
    features_dict['mode_freq_khz'] =  scipy.stats.mode(freq_dist)[0]

    return features_dict


source_url = 'https://repository.voxforge1.org/downloads/SpeechCorpus/Trunk/Audio/Main/16kHz_16bit/'
audio_url = requests.get(source_url)
content = BeautifulSoup(audio_url.content, 'html.parser')
audio_files_download_link = [f"{source_url}{a['href']}" for a in content.find_all('a', href=True) if a['href'].endswith('.tgz')]

results = []

for tgz_file_url in audio_files_download_link:
    print(tgz_file_url)
    response_file_sample = requests.get(tgz_file_url)
    tgz_content = io.BytesIO(response_file_sample.content)
    with tarfile.open(fileobj=tgz_content, mode='r:gz') as tar:
        
            # Get Metadata
            metadata_dict = get_readme_info(tar_object = tar)

            wav_members = [m for m in tar.getmembers() if m.name.endswith('.wav')]

            samples_processed = 0 
            max_samples = 3 

            for wav in wav_members:
                # Get atleast 3 samples
                if samples_processed >= max_samples:
                    break
                samples_processed += 1 
                

                wav_file = tar.extractfile(wav)
                
                if wav_file:
                    file_name = os.path.basename(wav.name)
                    print(f"processing.. {file_name}")
                    wav_data = wav_file.read()

                    with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_file:
                        temp_path = temp_file.name
                        temp_file.write(wav_data)

                    # Load Audio 
                    try: 
                        features_dict = load_base_features_weighted(temp_path)
                        all_features_dict = {**metadata_dict, **features_dict}
                        results.append(all_features_dict)
                    except Exception as e:
                        print(f"Error processing {file_name}: {str(e)}")

audio_data = pd.DataFrame(results)
audio_data.to_csv('data/audio_features_df.csv', index=False)
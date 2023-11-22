
from sklearn.model_selection import train_test_split, StratifiedKFold
import pandas as pd
import glob
import os
from utils.vad_tools import vad
from sklearn.model_selection import train_test_split, StratifiedKFold


'''RAVDESS - loader'''

def extract_speech_info(file_path, features):
    file_name = os.path.splitext(os.path.basename(file_path))[0]
    speech_info = dict(zip(features, file_name.split("-")))
    speech_info['audio_files'] = file_path
    return speech_info

def get_metadata(dataset_dir):
    speech_list = glob.glob(os.path.join(dataset_dir, '*/*.wav'))
    features = ['modality', 'vocal_channel', 'emotion', 'emotion_intensity', 'statement', 'repetition', 'actor']
    
    meta_info = pd.DataFrame([extract_speech_info(f, features) for f in speech_list])
    return meta_info

def split_metadata(meta_data, use_kfold=False, num_splits=5, test_ratio=0.2, apply_stratification=True, target_column=None, show_logs=True, random_seed=None):
    if use_kfold:
        if show_logs:
            print("Using Stratified K-Fold Cross Validation...")
            total_samples = len(meta_data)
            print('Total samples:', total_samples)
        
        stratified_kfold = StratifiedKFold(n_splits=num_splits, random_state=random_seed, shuffle=True)
        return stratified_kfold.split(meta_data.drop(columns=[target_column]), meta_data[target_column])
    else:
        if show_logs:
            print("Using Train-Test Split...")
        
        train_set, test_set = train_test_split(meta_data, stratify=meta_data[target_column] if apply_stratification else None, shuffle=True, test_size=test_ratio, random_state=random_seed)
        
        if show_logs:
            print(f'Test size ratio: {test_ratio}')
            print('Train set size:', len(train_set))
            print('Test set size:', len(test_set))
        
        return train_set, test_set
    
'''Emo-DB - loader'''


if __name__ == "__main__":
    meta = get_metadata('./dataset/ravdess/')
    print(meta)
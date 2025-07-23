import os
import numpy as np
import torch
import torchaudio
from torch import Tensor
from torch.utils.data import Dataset
import librosa
import random
from datautils.dataio import pad, load_audio
import torch
import soundfile as sf
import shlex


#########################
# Set up logging
#########################
import logging
from logger import setup_logging
setup_logging()
logger = logging.getLogger(__name__)

# def gen_list(protocol_file, is_train=False, is_dev=False, is_eval=False):
#     label_mapping = {
#         "clean": 0,
#         "Environment noise": 1,
#         "Nature noise": 2,
#         "BGM": 3,
#         "Overlapping Speech": 4,
#         "White noise": 5,
#         "Pink noise": 6,
#         "pitch_shift": 7,
#         "Time stretch": 8,
#         "Auto-Tune": 9,
#         "Reverberation": 10
#     }

#     d_meta = {}
#     file_list = []

#     with open(protocol_file, "r") as f:
#         for line in f:
#             utt, subset, label = line.strip().split("\t")
#             if (is_train and subset.lower() == "train") or \
#                (is_dev and subset.lower() == "dev") or \
#                (is_eval and subset.lower() == "eval"):
#                 file_list.append(utt)  # 파일 경로 그대로 추가
#                 d_meta[utt] = label_mapping.get(label, -1)  # 라벨 매핑
    
#     return d_meta, file_list

def gen_list(protocol_file, is_train=False, is_dev=False, is_eval=False):
    
    label_mapping = {
        "clean": 0,
        "environment_noise": 1,
        "background_noise": 1,
        "background_music": 2,
        "overlapping_speech": 3,
        "white_noise": 4,
        "pink_noise": 5,
        "pitch_shift": 6,
        "time_stretch": 7,
        "auto_tune": 8,
        "reverberation": 9
    }

    d_meta = {}
    file_list = []
    if (is_train):
        with open(protocol_file, "r") as f:
            for line in f:
                utt, subset, label = line.strip().split("\t")
                if subset.lower() == "train":  # 훈련 데이터만 필터링
                    file_list.append(utt)  # 파일 경로 그대로 추가
                    d_meta[utt] = label_mapping.get(label, -1)  # 라벨 매핑
        return d_meta, file_list
    
    if (is_dev):
        with open(protocol_file, "r") as f:
            for line in f:
                utt, subset, label = line.strip().split("\t")
                if subset.lower() == "dev":  # 훈련 데이터만 필터링
                    file_list.append(utt)  # 파일 경로 그대로 추가
                    d_meta[utt] = label_mapping.get(label, -1)  # 라벨 매핑
        return d_meta, file_list
    # 파일 경로만 한줄에 하나씩 있는 경우에 사용용
    # if (is_eval):
    #     with open(protocol_file, "r") as f:
    #         for line in f:
    #             utt = line.strip()
    #             file_list.append(utt)  # 파일 경로만 추가
    #     return file_list
    if (is_eval):
        with open(protocol_file, "r") as f:
            for line in f:
                utt = line.strip().split("\t")[0]
                file_list.append(utt)
        return file_list
        
class Dataset_base(Dataset):
    def __init__(self, args, list_IDs, labels, **kwargs):
        """
        Args:
        - args: Additional arguments (not actively used here).
        - list_IDs: List of file paths (absolute paths from protocol file).
        - labels: Dictionary mapping file paths to their labels.
        - kwargs: Other optional parameters (e.g., augmentation settings).
        """
        self.list_IDs = list_IDs  # 파일 경로 리스트 (절대 경로)
        self.labels = labels      # 파일 라벨 매핑
        self.args = args

        # 공통 속성 설정
        self.trim_length = kwargs.get('trim_length', 64000)
        self.wav_samp_rate = kwargs.get('wav_samp_rate', 16000)
        self.augmentation_methods = kwargs.get('augmentation_methods', [])
        self.eval_augment = kwargs.get('eval_augment', None)
        self.repeat_pad = kwargs.get('repeat_pad', True)
        self.is_train = kwargs.get('is_train', False)
        
    def __len__(self):
        """Return the total number of samples."""
        return len(self.list_IDs)

def fix_length(waveform, target_len=64000, random_start=True):
    """waveform: torch.Tensor [1, N] or [N]"""
    if waveform.ndim == 2:  # [1, N]
        waveform = waveform.squeeze(0)
    cur_len = waveform.shape[-1]
    if cur_len > target_len:
        if random_start:
            start = np.random.randint(0, cur_len - target_len)
        else:
            start = 0
        waveform = waveform[start : start + target_len]
    elif cur_len < target_len:
        repeat = (target_len // cur_len) + 1
        waveform = waveform.repeat(repeat)[:target_len]
    return waveform

class Dataset_for(Dataset_base):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.target_len = 80000  # 5초
        self.sr = 16000
        self.n_fft = 2048
        self.hop_length = 512
        self.n_mfcc = 128  # 또는 128 등 실험값에 따라
        self.n_mels = 128
        self.window = 'hamming'

    def __getitem__(self, idx):
        utt_id = self.list_IDs[idx]
        # 캐시 디렉토리 설정 (wav 파일과 동일한 위치에 'mfcc_cache' 폴더)
        cache_dir = os.path.join(os.path.dirname(utt_id), 'mfcc_cache')
        os.makedirs(cache_dir, exist_ok=True)
        # 캐시 파일명: 원본 wav 파일명 + .mfcc.npy
        base = os.path.splitext(os.path.basename(utt_id))[0]
        cache_path = os.path.join(cache_dir, base + '.mfcc.npy')

        if os.path.exists(cache_path):
            mfcc = np.load(cache_path)
        else:
            wav, fs = librosa.load(utt_id, sr=self.sr)
            wav = fix_length(wav, self.target_len, random_start=self.is_train)
            mfcc = librosa.feature.mfcc(
                y=wav,
                sr=self.sr,
                n_mfcc=self.n_mfcc,
                n_fft=self.n_fft,
                hop_length=self.hop_length,
                n_mels=self.n_mels,
                window=self.window,
            )
            mfcc = mfcc / (np.linalg.norm(mfcc) + 1e-8)
            np.save(cache_path, mfcc)
        mfcc = torch.from_numpy(mfcc).float()
        mfcc = mfcc.unsqueeze(0)
        # print("mfcc shape:", mfcc.shape)
        target = self.labels[utt_id]
        return mfcc, target
    
class Dataset_for_dev(Dataset_base):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.target_len = 80000
        self.sr = 16000
        self.n_fft = 2048
        self.hop_length = 512
        self.n_mfcc = 128
        self.n_mels = 128
        self.window = 'hamming'

    def __getitem__(self, idx):
        utt_id = self.list_IDs[idx]
        # 캐시 디렉토리 설정 (wav 파일과 동일한 위치에 'mfcc_cache' 폴더)
        cache_dir = os.path.join(os.path.dirname(utt_id), 'mfcc_cache')
        os.makedirs(cache_dir, exist_ok=True)
        # 캐시 파일명: 원본 wav 파일명 + .mfcc.npy
        base = os.path.splitext(os.path.basename(utt_id))[0]
        cache_path = os.path.join(cache_dir, base + '.mfcc.npy')

        if os.path.exists(cache_path):
            mfcc = np.load(cache_path)
        else:
            wav, fs = librosa.load(utt_id, sr=self.sr)
            wav = fix_length(wav, self.target_len, random_start=self.is_train)
            mfcc = librosa.feature.mfcc(
                y=wav,
                sr=self.sr,
                n_mfcc=self.n_mfcc,
                n_fft=self.n_fft,
                hop_length=self.hop_length,
                n_mels=self.n_mels,
                window=self.window,
            )
            mfcc = mfcc / (np.linalg.norm(mfcc) + 1e-8)
            np.save(cache_path, mfcc)
        mfcc = torch.from_numpy(mfcc).float()
        target = self.labels[utt_id]
        mfcc = mfcc.unsqueeze(0)
        return mfcc, target

class Dataset_for_eval(Dataset_base):
    def __init__(self, args, list_IDs, labels, enable_chunking=False):
        super().__init__(args, list_IDs, labels)
        self.enable_chunking = enable_chunking
        self.target_len = 80000
        self.sr = 16000
        self.n_fft = 2048
        self.hop_length = 512
        self.n_mfcc = 128
        self.n_mels = 128
        self.window = 'hamming'

    def __getitem__(self, idx):
        utt_id = self.list_IDs[idx]
        wav, sr = librosa.load(utt_id, sr=self.sr)
        if not self.enable_chunking:
            wav = fix_length(wav, self.target_len, random_start=False)
        mfcc = librosa.feature.mfcc(
            y=wav,
            sr=self.sr,
            n_mfcc=self.n_mfcc,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            n_mels=self.n_mels,
            window=self.window,
        )
        mfcc = mfcc / (np.linalg.norm(mfcc) + 1e-8)
        mfcc = torch.from_numpy(mfcc).float()
        mfcc = mfcc.unsqueeze(0)
        if self.labels is not None:
            target = self.labels[utt_id]
            return mfcc, target
        else:
            return mfcc, utt_id
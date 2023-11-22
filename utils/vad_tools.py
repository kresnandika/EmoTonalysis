import numpy as np
import librosa
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d

def normalize_array(arr):
    return (arr - np.min(arr)) / (np.max(arr) - np.min(arr))

def compute_gradient(arr, sigma=10):
    smoothed_arr = gaussian_filter1d(arr, sigma=sigma)
    return np.gradient(smoothed_arr)

def find_vad_points(time_mag, n_spare=3):
    sorted_time_mag = np.sort(time_mag)
    sorted_time_mag_smoothed_grad = compute_gradient(sorted_time_mag)
    threshold = np.max(sorted_time_mag_smoothed_grad)
    va_indices = np.where(sorted_time_mag_smoothed_grad > threshold)[0]
    if len(va_indices) > 0:
        va_point = [max(0, va_indices[0] - n_spare), min(len(time_mag) - 1, va_indices[-1] + n_spare)]
        return va_point
    else:
        return [0, 0]

def vad(spec, plot=False, n_spare=3):
    time_mag = normalize_array(librosa.amplitude_to_db(spec)).sum(axis=0)
    va_point = find_vad_points(time_mag, n_spare)
    if plot:
        plt.plot(time_mag)
        vad_line = np.zeros_like(time_mag)
        vad_line[va_point[0]: va_point[1] + 1] = np.max(time_mag)
        plt.plot(vad_line)
    return va_point

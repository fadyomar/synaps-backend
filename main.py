from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List
import numpy as np
from scipy.signal import butter, filtfilt, iirnotch

app = FastAPI(title="Synaps Backend API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

class FilterRequest(BaseModel):
    signal: List[float]
    sample_rate: float
    bandpass_low: float
    bandpass_high: float
    notch_enabled: bool
    car_enabled: bool
    channels: List[List[float]]
    channel_names: List[str]

class FilterResponse(BaseModel):
    filtered_channels: dict
    fft_freqs: List[float]
    fft_magnitudes: List[float]
    band_powers: dict

def butter_bandpass(lowcut, highcut, fs, order=4):
    nyq = fs / 2
    low = lowcut / nyq
    high = highcut / nyq
    low = max(0.001, min(low, 0.999))
    high = max(0.001, min(high, 0.999))
    if low >= high:
        high = min(low + 0.1, 0.999)
    b, a = butter(order, [low, high], btype='band')
    return b, a

def apply_bandpass(signal, lowcut, highcut, fs):
    b, a = butter_bandpass(lowcut, highcut, fs)
    return filtfilt(b, a, signal).tolist()

def apply_notch(signal, fs, freq=50.0, Q=30.0):
    b, a = iirnotch(freq, Q, fs)
    return filtfilt(b, a, signal).tolist()

def apply_car(channels_data):
    arr = np.array(channels_data)
    mean = np.mean(arr, axis=0)
    return (arr - mean).tolist()

def compute_fft(signal, sr):
    N = len(signal)
    freqs = np.fft.rfftfreq(N, d=1/sr)
    magnitudes = np.abs(np.fft.rfft(signal)) / N
    mask = freqs <= 50
    return freqs[mask].tolist(), magnitudes[mask].tolist()

def compute_band_power(freqs, magnitudes, low, high):
    freqs = np.array(freqs)
    magnitudes = np.array(magnitudes)
    mask = (freqs >= low) & (freqs <= high)
    band = magnitudes[mask]
    return float(np.mean(band)) if len(band) > 0 else 0.0

@app.get("/")
def root():
    return {"status": "Synaps Backend is running", "version": "0.1.0"}

@app.post("/filter", response_model=FilterResponse)
def filter_signal(req: FilterRequest):
    sr = req.sample_rate
    channels_data = req.channels
    channel_names = req.channel_names

    # Apply CAR first
    if req.car_enabled and len(channels_data) > 1:
        channels_data = apply_car(channels_data)

    filtered = {}
    for i, (ch_signal, ch_name) in enumerate(zip(channels_data, channel_names)):
        sig = list(ch_signal)

        # Bandpass
        try:
            sig = apply_bandpass(sig, req.bandpass_low, req.bandpass_high, sr)
        except Exception:
            pass

        # Notch
        if req.notch_enabled:
            try:
                sig = apply_notch(sig, sr)
            except Exception:
                pass

        filtered[ch_name] = sig

    # FFT on first channel
    first_ch = list(filtered.values())[0]
    freqs, magnitudes = compute_fft(first_ch, sr)

    band_powers = {
        "Delta": compute_band_power(freqs, magnitudes, 0, 4),
        "Theta": compute_band_power(freqs, magnitudes, 4, 8),
        "Alpha": compute_band_power(freqs, magnitudes, 8, 13),
        "Beta":  compute_band_power(freqs, magnitudes, 13, 30),
        "Gamma": compute_band_power(freqs, magnitudes, 30, 50),
    }

    return FilterResponse(
        filtered_channels=filtered,
        fft_freqs=freqs,
        fft_magnitudes=magnitudes,
        band_powers=band_powers,
    )

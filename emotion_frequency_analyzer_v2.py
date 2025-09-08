#!/usr/bin/env python3
"""
EmotionFrequencyAnalyzerV2 (FFT-based bandpower, optional multiprocessing)
"""

from typing import Dict, Tuple, Union, List, Optional
import numpy as np
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

EPS = 1e-12


class Kalman1D:
    def __init__(self, q: float = 1e-5, r: float = 1e-2, initial_x: float = 0.0, initial_p: float = 1.0):
        self.q = q
        self.r = r
        self.x = initial_x
        self.p = initial_p

    def filter(self, measurements: np.ndarray) -> np.ndarray:
        x = self.x
        p = self.p
        out = np.empty_like(measurements)
        for i, z in enumerate(measurements):
            p = p + self.q
            k = p / (p + self.r)
            x = x + k * (z - x)
            p = (1 - k) * p
            out[i] = x
        self.x = x
        self.p = p
        return out


def bandpower_via_rfft(x: np.ndarray, fs: float, bands: Dict[str, Tuple[float, float]], nfft: Optional[int] = None) -> Dict[str, float]:
    n = x.shape[0]
    if nfft is None:
        nfft = n
    win = np.hanning(n)
    xw = x * win
    spec = np.fft.rfft(xw, n=nfft)
    psd = (np.abs(spec) ** 2) / (np.sum(win ** 2) + EPS)
    freqs = np.fft.rfftfreq(nfft, d=1.0 / fs)
    band_powers: Dict[str, float] = {}
    for name, (low, high) in bands.items():
        idx = np.where((freqs >= low) & (freqs <= high))[0]
        band_powers[name] = float(np.sum(psd[idx]) + EPS) if idx.size > 0 else 0.0
    return band_powers


def _process_window_worker(args):
    seg, fs, bands = args
    if seg.ndim == 2:
        avg = np.mean(seg, axis=0)
    else:
        avg = seg
    return bandpower_via_rfft(avg, fs, bands, nfft=None)


class EmotionFrequencyAnalyzerV2:
    def __init__(
        self,
        sampling_rate: int = 512,
        window_size: int = 2048,
        hop_size: int = 1024,
        snr_threshold_db: float = 20.0,
        kalman_q: float = 1e-5,
        kalman_r: float = 1e-2,
        use_multiprocessing: bool = False,
        max_workers: int = 4,
        min_windows_for_mp: int = 4,
    ) -> None:
        self.fs = sampling_rate
        self.window_size = window_size
        self.hop_size = hop_size
        self.snr_threshold_db = snr_threshold_db
        self.kalman_q = kalman_q
        self.kalman_r = kalman_r
        self.use_multiprocessing = use_multiprocessing
        self.max_workers = max_workers
        self.min_windows_for_mp = min_windows_for_mp

        self.frequency_bands = {
            'delta': (0.5, 4.0),
            'theta': (4.0, 8.0),
            'alpha': (8.0, 13.0),
            'beta': (13.0, 30.0),
            'gamma': (30.0, 100.0),
        }
        self.emotion_coeffs = {
            'joy': {'alpha': 0.35, 'beta': 0.25},
            'sadness': {'alpha': -0.25, 'theta': 0.20},
            'anger': {'beta': 0.40, 'gamma': 0.25},
        }

    def _ensure_numpy(self, eeg_input: Union[np.ndarray, Dict[str, np.ndarray]]):
        if isinstance(eeg_input, dict):
            channels = list(eeg_input.keys())
            arrs = [np.asarray(eeg_input[ch], dtype=np.float64) for ch in channels]
            n_samples = max(a.shape[0] for a in arrs)
            data = np.zeros((len(arrs), n_samples), dtype=np.float64)
            for i, a in enumerate(arrs):
                data[i, :a.shape[0]] = a
            return data, channels
        elif isinstance(eeg_input, np.ndarray):
            if eeg_input.ndim == 1:
                return eeg_input[np.newaxis, :], ['ch0']
            elif eeg_input.ndim == 2:
                channels = [f'ch{i}' for i in range(eeg_input.shape[0])]
                return eeg_input.astype(np.float64), channels
            else:
                raise ValueError("eeg_input must be 1D or 2D array")
        else:
            raise TypeError("eeg_input must be numpy.ndarray or dict")

    def _apply_kalman(self, data: np.ndarray) -> np.ndarray:
        filtered = np.empty_like(data)
        for i in range(data.shape[0]):
            kf = Kalman1D(q=self.kalman_q, r=self.kalman_r, initial_x=float(np.median(data[i, :])))
            filtered[i, :] = kf.filter(data[i, :])
        return filtered

    def _calc_snr_db(self, signal: np.ndarray, noise_estimate: np.ndarray) -> float:
        ps = np.mean(signal ** 2) + EPS
        pn = np.mean(noise_estimate ** 2) + EPS
        return float(10.0 * np.log10(ps / pn))

    def analyze_eeg_to_emotion(self, eeg_input: Union[np.ndarray, Dict[str, np.ndarray]], sampling_rate: int = None) -> Dict:
        t0 = time.time()
        if sampling_rate is not None and sampling_rate != self.fs:
            raise ValueError("sampling_rate mismatch")

        data, channel_names = self._ensure_numpy(eeg_input)
        n_ch, n_samples = data.shape

        noise_est = np.mean(np.diff(data, axis=1) ** 2) if n_samples > 1 else 1e-6
        snr_db = self._calc_snr_db(data, np.sqrt(noise_est) * np.ones_like(data))
        signal_quality = 'good' if snr_db > self.snr_threshold_db else ('poor' if snr_db > (self.snr_threshold_db - 6) else 'reject')

        try:
            filtered = self._apply_kalman(data)
        except Exception:
            filtered = data.copy()

        starts = []
        start = 0
        while start + self.window_size <= n_samples:
            starts.append(start)
            start += self.hop_size
        if not starts:
            starts = [0]

        band_accum = {b: 0.0 for b in self.frequency_bands.keys()}
        window_count = 0

        if self.use_multiprocessing and len(starts) >= self.min_windows_for_mp:
            tasks = []
            for s in starts:
                seg = filtered[:, s:s + self.window_size]
                tasks.append((seg, self.fs, self.frequency_bands))
            with ProcessPoolExecutor(max_workers=self.max_workers) as exe:
                futures = [exe.submit(_process_window_worker, t) for t in tasks]
                for fut in as_completed(futures):
                    try:
                        band_feats = fut.result()
                    except Exception:
                        continue
                    for k, v in band_feats.items():
                        band_accum[k] += v
                    window_count += 1
        else:
            for s in starts:
                seg = filtered[:, s:s + self.window_size]
                avg = np.mean(seg, axis=0)
                band_feats = bandpower_via_rfft(avg, self.fs, self.frequency_bands)
                for k, v in band_feats.items():
                    band_accum[k] += v
                window_count += 1

        band_powers = {k: (v / max(1, window_count)) for k, v in band_accum.items()}

        total = sum(band_powers.values()) + EPS
        norm = {k: v / total for k, v in band_powers.items()}
        raw_scores = {}
        for emotion, coeffs in self.emotion_coeffs.items():
            s = 0.0
            for band, coeff in coeffs.items():
                s += coeff * norm.get(band, 0.0)
            raw_scores[emotion] = s
        raw_scores['neutral'] = 0.0
        relu_scores = {k: max(0.0, v) for k, v in raw_scores.items()}
        vals = np.array(list(relu_scores.values()), dtype=np.float64)
        if vals.sum() <= EPS:
            soft = np.ones_like(vals) / len(vals)
        else:
            exps = np.exp(vals - np.max(vals))
            soft = exps / (exps.sum() + EPS)
        emotion_vector = {k: float(soft[i]) for i, k in enumerate(relu_scores.keys())}
        primary = max(emotion_vector.items(), key=lambda kv: kv[1])[0]
        primary_score = emotion_vector.get(primary, 0.0)
        snr_factor = min(1.0, max(0.0, (snr_db - self.snr_threshold_db + 6.0) / 12.0))
        confidence = float(primary_score * 0.7 + snr_factor * 0.3)
        if signal_quality == 'reject':
            confidence = min(confidence, 0.25)

        metadata = {
            'channels': channel_names,
            'n_channels': n_ch,
            'n_samples': n_samples,
            'window_size': self.window_size,
            'hop_size': self.hop_size,
            'processing_time_ms': (time.time() - t0) * 1000.0,
        }

        return {
            'emotion_vector': emotion_vector,
            'primary_emotion': primary,
            'band_powers': band_powers,
            'snr_db': float(snr_db),
            'signal_quality': signal_quality,
            'confidence': float(confidence),
            'metadata': metadata,
        }


__all__ = ["EmotionFrequencyAnalyzerV2"]


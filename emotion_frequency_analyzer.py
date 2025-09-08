#!/usr/bin/env python3
"""
emotion_frequency_analyzer.py
- EEG band-power extraction via Goertzel, sliding windows, Kalman denoising, SNR checks
- Maps band features to emotion scores (joy, sadness, anger, neutral)
"""

from typing import Dict, Tuple, Union, List
import numpy as np
import time

EPS = 1e-12


class Kalman1D:
    """Simple 1D Kalman filter for denoising each channel (scalar process)."""

    def __init__(self, q: float = 1e-5, r: float = 1e-2, initial_x: float = 0.0, initial_p: float = 1.0) -> None:
        self.q = q
        self.r = r
        self.x = initial_x
        self.p = initial_p

    def filter(self, measurements: np.ndarray) -> np.ndarray:
        x = self.x
        p = self.p
        out = np.empty_like(measurements)
        for _, z in enumerate(measurements):
            p = p + self.q
            k = p / (p + self.r)
            x = x + k * (z - x)
            p = (1 - k) * p
            out[_] = x
        self.x = x
        self.p = p
        return out


def goertzel_power_block(x: np.ndarray, freqs: np.ndarray, fs: float) -> np.ndarray:
    """Vectorized Goertzel for a 1D signal x and multiple target frequencies."""
    n = x.shape[0]
    k = np.round((n * freqs) / fs).astype(int)
    omega = (2.0 * np.pi * k) / n
    cos_omega = np.cos(omega)
    coeff = 2.0 * cos_omega
    powers = np.zeros_like(freqs, dtype=np.float64)
    for i, c in enumerate(coeff):
        s_prev = 0.0
        s_prev2 = 0.0
        for sample in x:
            s = sample + c * s_prev - s_prev2
            s_prev2 = s_prev
            s_prev = s
        real = s_prev - s_prev2 * cos_omega[i]
        imag = s_prev2 * np.sin(omega[i])
        powers[i] = real * real + imag * imag
    return powers


class EmotionFrequencyAnalyzer:
    def __init__(
        self,
        sampling_rate: int = 512,
        window_size: int = 2048,
        hop_size: int = 1024,
        snr_threshold_db: float = 20.0,
        kalman_q: float = 1e-5,
        kalman_r: float = 1e-2,
    ) -> None:
        self.fs = sampling_rate
        self.window_size = window_size
        self.hop_size = hop_size
        self.snr_threshold_db = snr_threshold_db
        self.frequency_bands: Dict[str, Tuple[float, float]] = {
            'delta': (0.5, 4.0),
            'theta': (4.0, 8.0),
            'alpha': (8.0, 13.0),
            'beta': (13.0, 30.0),
            'gamma': (30.0, 100.0),
        }
        self.emotion_coeffs: Dict[str, Dict[str, float]] = {
            'joy': {'alpha': 0.35, 'beta': 0.25},
            'sadness': {'alpha': -0.25, 'theta': 0.20},
            'anger': {'beta': 0.40, 'gamma': 0.25},
        }
        self.kalman_q = kalman_q
        self.kalman_r = kalman_r

    def _ensure_numpy(self, eeg_input: Union[np.ndarray, Dict[str, np.ndarray]]) -> Tuple[np.ndarray, List[str]]:
        if isinstance(eeg_input, dict):
            channels = list(eeg_input.keys())
            arrs = [np.asarray(eeg_input[ch], dtype=np.float64) for ch in channels]
            n_samples = max(a.shape[0] for a in arrs)
            data = np.zeros((len(arrs), n_samples), dtype=np.float64)
            for i, a in enumerate(arrs):
                data[i, : a.shape[0]] = a
            return data, channels
        elif isinstance(eeg_input, np.ndarray):
            if eeg_input.ndim == 1:
                return eeg_input[np.newaxis, :], ['ch0']
            elif eeg_input.ndim == 2:
                channels = [f'ch{i}' for i in range(eeg_input.shape[0])]
                return eeg_input.astype(np.float64), channels
            else:
                raise ValueError('eeg_input numpy array must be 1D or 2D (channels x samples).')
        else:
            raise TypeError('eeg_input must be numpy.ndarray or dict')

    def _apply_kalman(self, data: np.ndarray) -> np.ndarray:
        filtered = np.empty_like(data)
        for i in range(data.shape[0]):
            kf = Kalman1D(q=self.kalman_q, r=self.kalman_r, initial_x=float(np.median(data[i, :])))
            filtered[i, :] = kf.filter(data[i, :])
        return filtered

    def _calc_snr_db(self, signal: np.ndarray, noise_estimate: np.ndarray) -> float:
        ps = np.mean(signal ** 2) + EPS
        pn = np.mean(noise_estimate ** 2) + EPS
        snr = 10.0 * np.log10(ps / pn)
        return float(snr)

    def _band_power_goertzel(self, segment: np.ndarray, band: Tuple[float, float], fs: float, n_freqs: int = 5) -> float:
        low, high = band
        if high <= low:
            return 0.0
        freqs = np.linspace(max(0.1, low), min(fs / 2 - 0.1, high), n_freqs)
        powers = goertzel_power_block(segment, freqs, fs)
        return float(np.mean(powers) + EPS)

    def _extract_band_features(self, data_segment: np.ndarray) -> Dict[str, float]:
        if data_segment.ndim == 2:
            avg = np.mean(data_segment, axis=0)
        else:
            avg = data_segment
        band_powers: Dict[str, float] = {}
        for band_name, (low, high) in self.frequency_bands.items():
            band_powers[band_name] = self._band_power_goertzel(avg, (low, high), self.fs, n_freqs=5)
        return band_powers

    def _map_bands_to_emotions(self, band_features: Dict[str, float]) -> Tuple[Dict[str, float], str]:
        total = sum(band_features.values()) + EPS
        norm = {k: v / total for k, v in band_features.items()}
        raw_scores: Dict[str, float] = {}
        for emotion, coeffs in self.emotion_coeffs.items():
            s = 0.0
            for band, coeff in coeffs.items():
                band_val = norm.get(band, 0.0)
                s += coeff * band_val
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
        return emotion_vector, primary

    def analyze_eeg_to_emotion(self, eeg_input: Union[np.ndarray, Dict[str, np.ndarray]], sampling_rate: int = None) -> Dict:
        t0 = time.time()
        if sampling_rate is not None and sampling_rate != self.fs:
            raise ValueError(f'sampling_rate mismatch: analyzer fs={self.fs}, provided={sampling_rate}')

        data, channel_names = self._ensure_numpy(eeg_input)
        n_ch, n_samples = data.shape

        _ = np.mean(data ** 2)
        noise_est = np.mean(np.diff(data, axis=1) ** 2) if n_samples > 1 else 1e-6
        snr_db = self._calc_snr_db(data, np.sqrt(noise_est) * np.ones_like(data))
        signal_quality = 'good' if snr_db > self.snr_threshold_db else ('poor' if snr_db > (self.snr_threshold_db - 6) else 'reject')

        try:
            filtered = self._apply_kalman(data)
        except Exception:
            filtered = data.copy()

        band_accum = {b: 0.0 for b in self.frequency_bands.keys()}
        window_count = 0
        start = 0
        while start + self.window_size <= n_samples:
            seg = filtered[:, start : start + self.window_size]
            band_feats = self._extract_band_features(seg)
            for k, v in band_feats.items():
                band_accum[k] += v
            window_count += 1
            start += self.hop_size
        if window_count == 0:
            band_feats = self._extract_band_features(filtered)
            for k, v in band_feats.items():
                band_accum[k] += v
            window_count = 1

        band_powers = {k: (v / window_count) for k, v in band_accum.items()}
        emotion_vector, primary = self._map_bands_to_emotions(band_powers)

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


if __name__ == '__main__':
    fs = 512
    t = np.arange(0, 5.0, 1.0 / fs)
    sig = 0.8 * np.sin(2 * np.pi * 10 * t) + 0.5 * np.sin(2 * np.pi * 20 * t) + 0.05 * np.random.randn(t.size)
    eeg = np.stack([sig, sig * 0.9])
    analyzer = EmotionFrequencyAnalyzer(sampling_rate=fs)
    res = analyzer.analyze_eeg_to_emotion(eeg)
    print(res)


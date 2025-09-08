import numpy as np
import pytest

from emotion_frequency_analyzer import EmotionFrequencyAnalyzer


def synth_signal(fs: int, duration: float, components: list[tuple[float, float]]) -> np.ndarray:
    t = np.arange(0, duration, 1.0 / fs)
    sig = np.zeros_like(t)
    for f, a in components:
        sig += a * np.sin(2 * np.pi * f * t)
    sig += 0.01 * np.random.randn(t.size)
    return sig


@pytest.fixture(scope="module")
def analyzer() -> EmotionFrequencyAnalyzer:
    return EmotionFrequencyAnalyzer(sampling_rate=512, window_size=2048, hop_size=1024)


def test_joy_detection(analyzer: EmotionFrequencyAnalyzer) -> None:
    fs = 512
    sig = synth_signal(fs, duration=6.0, components=[(10.0, 1.0), (20.0, 0.6)])
    eeg = np.stack([sig, 0.9 * sig])
    res = analyzer.analyze_eeg_to_emotion(eeg)
    assert res['snr_db'] > 5.0
    assert res['primary_emotion'] == 'joy'
    assert res['confidence'] > 0.2


def test_sadness_detection(analyzer: EmotionFrequencyAnalyzer) -> None:
    fs = 512
    sig = synth_signal(fs, duration=6.0, components=[(6.0, 1.0), (10.0, 0.2)])
    eeg = np.stack([sig, 1.05 * sig])
    res = analyzer.analyze_eeg_to_emotion(eeg)
    assert res['primary_emotion'] == 'sadness'


def test_anger_detection(analyzer: EmotionFrequencyAnalyzer) -> None:
    fs = 512
    sig = synth_signal(fs, duration=6.0, components=[(20.0, 1.0), (40.0, 0.8)])
    eeg = np.stack([sig, 0.8 * sig])
    res = analyzer.analyze_eeg_to_emotion(eeg)
    assert res['primary_emotion'] == 'anger'


def test_low_snr_rejection(analyzer: EmotionFrequencyAnalyzer) -> None:
    fs = 512
    t = np.arange(0, 3.0, 1.0 / fs)
    noise = 0.5 * np.random.randn(t.size)
    eeg = np.stack([noise, noise])
    res = analyzer.analyze_eeg_to_emotion(eeg)
    assert res['signal_quality'] in ('reject', 'poor')
    assert res['confidence'] <= 0.5


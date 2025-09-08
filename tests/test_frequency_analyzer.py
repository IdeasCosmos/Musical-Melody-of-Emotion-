#!/usr/bin/env python3
import numpy as np
import unittest

from emotion_frequency_analyzer import EmotionFrequencyAnalyzer


def synth_signal(fs: int, duration: float, components):
    t = np.arange(0, duration, 1.0 / fs)
    sig = np.zeros_like(t)
    for f, a in components:
        sig += a * np.sin(2 * np.pi * f * t)
    sig += 0.01 * np.random.randn(t.size)
    return sig


class TestEmotionFrequencyAnalyzer(unittest.TestCase):
    def setUp(self) -> None:
        self.analyzer = EmotionFrequencyAnalyzer(sampling_rate=512, window_size=2048, hop_size=1024)

    def test_joy_detection(self):
        fs = 512
        sig = synth_signal(fs, duration=6.0, components=[(10.0, 1.0), (20.0, 0.6)])
        eeg = np.stack([sig, 0.9 * sig])
        res = self.analyzer.analyze_eeg_to_emotion(eeg)
        self.assertIn('joy', res['emotion_vector'])
        self.assertEqual(res['primary_emotion'], 'joy')
        self.assertGreater(res['confidence'], 0.2)

    def test_sadness_detection(self):
        fs = 512
        sig = synth_signal(fs, duration=6.0, components=[(6.0, 1.0), (10.0, 0.2)])
        eeg = np.stack([sig, 1.05 * sig])
        res = self.analyzer.analyze_eeg_to_emotion(eeg)
        self.assertEqual(res['primary_emotion'], 'sadness')

    def test_anger_detection(self):
        fs = 512
        sig = synth_signal(fs, duration=6.0, components=[(20.0, 1.0), (40.0, 0.8)])
        eeg = np.stack([sig, 0.8 * sig])
        res = self.analyzer.analyze_eeg_to_emotion(eeg)
        self.assertEqual(res['primary_emotion'], 'anger')

    def test_low_snr_rejection(self):
        fs = 512
        t = np.arange(0, 3.0, 1.0 / fs)
        noise = 0.5 * np.random.randn(t.size)
        eeg = np.stack([noise, 1.0 * noise])
        res = self.analyzer.analyze_eeg_to_emotion(eeg)
        self.assertIn(res['signal_quality'], ('reject', 'poor'))
        self.assertLessEqual(res['confidence'], 0.5)


if __name__ == '__main__':
    unittest.main()


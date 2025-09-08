#!/usr/bin/env python3
import unittest
import numpy as np

from integrated_emotion_engine import IntegratedEmotionEngine


class TestIntegratedEmotionEngine(unittest.TestCase):
    def setUp(self) -> None:
        self.engine = IntegratedEmotionEngine()

    def test_fusion_with_sarcasm(self):
        fs = 512
        t = np.arange(0, 5.0, 1.0 / fs)
        sig = 0.8 * np.sin(2 * np.pi * 10 * t) + 0.5 * np.sin(2 * np.pi * 20 * t)
        eeg = np.stack([sig, 0.9 * sig])

        text = "이게 재미있네요. 참 잘했어."
        res = self.engine.analyze(text, eeg, sampling_rate=fs)
        self.assertIn('fused_emotion_vector', res)
        self.assertIn('confidence', res)
        self.assertIn('primary_emotion', res)


if __name__ == '__main__':
    unittest.main()


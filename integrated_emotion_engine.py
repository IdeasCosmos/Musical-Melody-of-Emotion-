#!/usr/bin/env python3
"""
IntegratedEmotionEngine
- Orchestrates EEG-based emotion analysis and Korean text emotion processing.
"""

from typing import Dict, Union, Optional

from emotion_frequency_analyzer import EmotionFrequencyAnalyzer
from emotion_frequency_analyzer_v2 import EmotionFrequencyAnalyzerV2
from korean_emotion_processor import KoreanEmotionProcessor


class IntegratedEmotionEngine:
    def __init__(
        self,
        use_v2_analyzer: bool = False,
        sampling_rate: int = 512,
        window_size: int = 2048,
        hop_size: int = 1024,
    ) -> None:
        self.text_processor = KoreanEmotionProcessor()
        if use_v2_analyzer:
            self.eeg_analyzer = EmotionFrequencyAnalyzerV2(
                sampling_rate=sampling_rate, window_size=window_size, hop_size=hop_size
            )
        else:
            self.eeg_analyzer = EmotionFrequencyAnalyzer(
                sampling_rate=sampling_rate, window_size=window_size, hop_size=hop_size
            )

    def analyze(
        self,
        eeg_input: Union[Dict[str, "np.ndarray"], "np.ndarray"],
        text: Optional[str] = None,
        sampling_rate: Optional[int] = None,
    ) -> Dict:
        """
        Run EEG analyzer and optional Korean text processor, return combined report.
        """
        eeg_result = self.eeg_analyzer.analyze_eeg_to_emotion(eeg_input, sampling_rate=sampling_rate)
        text_result = self.text_processor.text_to_emotion_music(text or "") if text is not None else None

        combined = {
            'eeg': eeg_result,
            'text': text_result,
        }

        if text_result is not None:
            combined['summary'] = {
                'primary_eeg_emotion': eeg_result.get('primary_emotion'),
                'primary_text_emotion': text_result.get('primary_emotion'),
                'music_chord': text_result.get('music_chord'),
                'confidence': eeg_result.get('confidence'),
            }

        return combined


__all__ = [
    'IntegratedEmotionEngine',
]
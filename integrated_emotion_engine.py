#!/usr/bin/env python3
"""
integrated_emotion_engine.py
- IntegratedEmotionEngine: EEG 분석기와 한국어 텍스트 감정 처리기를 통합.
"""

from typing import Any, Dict, Optional, Union
import numpy as np

from emotion_frequency_analyzer import EmotionFrequencyAnalyzer
from korean_emotion_processor import KoreanEmotionProcessor


class IntegratedEmotionEngine:
    """EEG 기반 감정 추정과 한국어 텍스트 감정 신호를 함께 제공하는 엔진."""

    def __init__(
        self,
        eeg_analyzer: Optional[EmotionFrequencyAnalyzer] = None,
        text_processor: Optional[KoreanEmotionProcessor] = None,
    ) -> None:
        self.eeg_analyzer = eeg_analyzer or EmotionFrequencyAnalyzer()
        self.text_processor = text_processor or KoreanEmotionProcessor()

    def analyze_eeg(
        self,
        eeg_input: Union[np.ndarray, Dict[str, np.ndarray]],
        sampling_rate: Optional[int] = None,
    ) -> Dict[str, Any]:
        return self.eeg_analyzer.analyze_eeg_to_emotion(eeg_input, sampling_rate=sampling_rate)

    def analyze_text(self, text: str) -> Dict[str, Any]:
        return self.text_processor.text_to_emotion_music(text)

    def analyze(
        self,
        eeg_input: Optional[Union[np.ndarray, Dict[str, np.ndarray]]] = None,
        sampling_rate: Optional[int] = None,
        text: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        통합 분석:
        - eeg_input 제공 시 EEG 감정 결과 반환
        - text 제공 시 한국어 텍스트 감정 결과 반환
        - 둘 다 제공 시 간단한 결합 신뢰도(fused_confidence)를 추가
        """
        result: Dict[str, Any] = {}
        if eeg_input is not None:
            result['eeg'] = self.analyze_eeg(eeg_input, sampling_rate)
        if text is not None:
            result['text'] = self.analyze_text(text)

        if 'eeg' in result and 'text' in result:
            eeg_conf = float(result['eeg'].get('confidence', 0.0))
            text_intensity = float(abs(result['text'].get('intensity', 0.0)))
            fused_confidence = max(0.0, min(1.0, 0.5 * eeg_conf + 0.5 * min(1.0, text_intensity)))
            result['fused_confidence'] = fused_confidence

        return result


__all__ = ['IntegratedEmotionEngine']

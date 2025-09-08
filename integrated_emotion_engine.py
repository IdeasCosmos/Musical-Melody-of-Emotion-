#!/usr/bin/env python3
"""
integrated_emotion_engine.py
- 통합 엔진: 한국어 텍스트 감정 처리와 EEG 기반 감정 추정을 결합.
"""

from typing import Dict, Any, Optional, Union

import numpy as np

from korean_emotion_processor import KoreanEmotionProcessor
from emotion_frequency_analyzer import EmotionFrequencyAnalyzer


class IntegratedEmotionEngine:
    """
    텍스트와 EEG를 함께 받아 최종 감정 벡터와 메타데이터를 제공합니다.
    간단한 late-fusion 규칙:
      - 텍스트 감정이 '반어법'이면 EEG 결과 신뢰도를 10% 감소
      - 텍스트 주요 감정이 joy/anger/sadness와 매칭될 경우, 해당 EEG 점수에 10% 가중
    """

    def __init__(
        self,
        eeg_sampling_rate: int = 512,
        eeg_window_size: int = 2048,
        eeg_hop_size: int = 1024,
    ) -> None:
        self.text_proc = KoreanEmotionProcessor()
        self.eeg_analyzer = EmotionFrequencyAnalyzer(
            sampling_rate=eeg_sampling_rate,
            window_size=eeg_window_size,
            hop_size=eeg_hop_size,
        )

        # 텍스트 감정 라벨을 EEG 라벨로 맵핑 (간단 매핑)
        self.text_to_eeg_label: Dict[str, str] = {
            '놀람/깨달음': 'joy',
            '발견/인정': 'joy',
            '정당화/단호': 'anger',
            '공유지식': 'neutral',
            '반어법': 'neutral',
            '중립': 'neutral',
        }

    def analyze(
        self,
        text: str,
        eeg_input: Union[np.ndarray, Dict[str, np.ndarray]],
        sampling_rate: Optional[int] = None,
    ) -> Dict[str, Any]:
        text_res = self.text_proc.text_to_emotion_music(text)
        eeg_res = self.eeg_analyzer.analyze_eeg_to_emotion(eeg_input, sampling_rate=sampling_rate)

        fused = dict(eeg_res['emotion_vector'])

        # 텍스트 주요 감정
        text_primary = text_res.get('primary_emotion', '중립')
        eeg_label = self.text_to_eeg_label.get(text_primary, 'neutral')

        # 가중 조정
        if eeg_label in fused:
            fused[eeg_label] = float(min(1.0, fused[eeg_label] * 1.10))

        # 반어법 처리: EEG confidence 살짝 감소
        confidence = float(eeg_res.get('confidence', 0.0))
        if text_res.get('is_sarcasm', False):
            confidence = max(0.0, confidence * 0.9)

        # 정규화 재적용
        total = sum(fused.values()) or 1.0
        fused = {k: float(v / total) for k, v in fused.items()}
        primary = max(fused.items(), key=lambda kv: kv[1])[0]

        return {
            'text_result': text_res,
            'eeg_result': eeg_res,
            'fused_emotion_vector': fused,
            'primary_emotion': primary,
            'confidence': confidence,
        }


__all__ = ["IntegratedEmotionEngine"]
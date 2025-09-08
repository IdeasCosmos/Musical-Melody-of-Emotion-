#!/usr/bin/env python3
"""
korean_emotion_processor.py
- KoreanEmotionProcessor: 한국어 형태소 분석, 어미별 감정 점수 계산, 반어법 감지.
- text_to_emotion_music: 텍스트 → 형태소 추출 → 어미 매핑 → 감정 강도 → 음악 코드 생성.
- Konlpy 시도 후 실패 시 간단 정규식 기반 fallback 사용.
"""

import time
import sys
import re
from typing import Dict, List

# Konlpy 시도 (환경에 없을 수 있음)
try:
    from konlpy.tag import Komoran
    KONLPY_AVAILABLE = True
except Exception:
    KONLPY_AVAILABLE = False


class KoreanEmotionProcessor:
    """
    한국어 감정 처리기: 형태소 분석, 어미 감정 매핑, 반어법 감지.
    """

    def __init__(self):
        # 제공된 한국어 데이터
        self.ending_emotions = {
            '네요': {'emotion': '놀람/깨달음', 'intensity': '중'},
            '군요': {'emotion': '발견/인정', 'intensity': '중'},
            '거든요': {'emotion': '정당화/단호', 'intensity': '중'},
            '잖아요': {'emotion': '공유지식', 'intensity': '중'},
        }
        # 반어법 키워드
        self.sarcasm_keywords = ["참", "그래 잘했어"]
        # 강도 맵핑 (숫자화)
        self.intensity_map = {'중': 0.5, '강': 0.8, '약': 0.3}
        # 음악 코드 매핑 (감정 → 코드, 임시 예시)
        self.emotion_to_chord = {
            '놀람/깨달음': 'C Major',
            '발견/인정': 'G Major',
            '정당화/단호': 'D Minor',
            '공유지식': 'F Major',
            '반어법': 'E7',  # 긴장/반전 코드
        }
        if KONLPY_AVAILABLE:
            self.komoran = Komoran()

    def extract_morphemes(self, text: str) -> List[str]:
        """형태소 추출: Konlpy 사용 또는 fallback 정규식.

        Fallback에서는 문장 내 어디서든 어미를 탐지합니다(문장부호와 무관).
        """
        if KONLPY_AVAILABLE:
            return [morph for morph, tag in self.komoran.pos(text) if tag in ('EF', 'EC')]
        # Fallback: 간단 어미 매칭 (문장 끝 고정 아님)
        return re.findall(r'(네요|군요|거든요|잖아요)', text)

    def detect_sarcasm(self, text: str) -> bool:
        """반어법 감지: 키워드 매칭."""
        return any(keyword in text for keyword in self.sarcasm_keywords)

    def calculate_emotion_scores(self, morphemes: List[str]) -> Dict[str, float]:
        """어미별 감정 점수 계산."""
        scores: Dict[str, float] = {}
        for morph in morphemes:
            if morph in self.ending_emotions:
                info = self.ending_emotions[morph]
                emotion = info['emotion']
                intensity = self.intensity_map.get(info['intensity'], 0.5)
                scores[emotion] = intensity
        return scores

    def text_to_emotion_music(self, text: str) -> Dict:
        """텍스트 → 감정-음악 변환."""
        start_time = time.time()
        morphemes = self.extract_morphemes(text)
        is_sarcasm = self.detect_sarcasm(text)
        scores = self.calculate_emotion_scores(morphemes)

        # 감정 강도 계산 (평균)
        if scores:
            total_intensity = sum(scores.values()) / max(1, len(scores))
            primary_emotion = max(scores, key=scores.get)
        else:
            total_intensity = 0.0
            primary_emotion = '중립'

        # 반어법 적용 (감정 반전)
        if is_sarcasm:
            primary_emotion = '반어법'
            total_intensity *= -1

        # 음악 코드 생성
        chord = self.emotion_to_chord.get(primary_emotion, 'C Minor')

        processing_time_ms = (time.time() - start_time) * 1000.0
        memory_usage_bytes = (
            sys.getsizeof(text)
            + sys.getsizeof(morphemes)
            + sys.getsizeof(scores)
        )

        return {
            'morphemes': morphemes,
            'emotions': scores,
            'primary_emotion': primary_emotion,
            'intensity': total_intensity,
            'music_chord': chord,
            'is_sarcasm': is_sarcasm,
            'processing_time_ms': processing_time_ms,
            'memory_usage_bytes': memory_usage_bytes,
        }


__all__ = [
    'KoreanEmotionProcessor',
    'KONLPY_AVAILABLE',
]

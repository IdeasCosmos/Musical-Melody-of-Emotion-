#!/usr/bin/env python3
"""
Korean Emotion Processor
- Provides lightweight Korean morpheme ending extraction and sarcasm detection.
- Falls back to simple string/regex matching when konlpy is unavailable.
"""

import time
import sys
import re
from typing import Dict, List

# Try konlpy (optional)
try:
    from konlpy.tag import Komoran  # type: ignore
    KONLPY_AVAILABLE = True
except Exception:
    KONLPY_AVAILABLE = False


class KoreanEmotionProcessor:
    """한국어 감정 처리기: 형태소(어미) 기반 감정 매핑 + 반어법 감지."""

    def __init__(self):
        # 어미 → 감정/강도 제공 데이터
        self.ending_emotions: Dict[str, Dict[str, str]] = {
            '네요': {'emotion': '놀람/깨달음', 'intensity': '중'},
            '군요': {'emotion': '발견/인정', 'intensity': '중'},
            '거든요': {'emotion': '정당화/단호', 'intensity': '중'},
            '잖아요': {'emotion': '공유지식', 'intensity': '중'},
        }

        self.sarcasm_keywords: List[str] = ["참", "그래 잘했어"]
        self.intensity_map: Dict[str, float] = {'강': 0.8, '중': 0.5, '약': 0.3}
        self.emotion_to_chord: Dict[str, str] = {
            '놀람/깨달음': 'C Major',
            '발견/인정': 'G Major',
            '정당화/단호': 'D Minor',
            '공유지식': 'F Major',
            '반어법': 'E7',
        }

        if KONLPY_AVAILABLE:
            self.komoran = Komoran()

    def extract_morphemes(self, text: str) -> List[str]:
        """어미 형태소 추출 (konlpy 사용 가능 시 POS 태깅, 아니면 정규식)."""
        if KONLPY_AVAILABLE:
            return [morph for morph, tag in self.komoran.pos(text) if tag in ('EF', 'EC')]
        # fallback: 문장 내부 어디에나 등장하는 지정 어미를 수집
        return re.findall(r'(네요|군요|거든요|잖아요)', text)

    def detect_sarcasm(self, text: str) -> bool:
        """간단 키워드 기반 반어법 감지."""
        return any(keyword in text for keyword in self.sarcasm_keywords)

    def calculate_emotion_scores(self, morphemes: List[str]) -> Dict[str, float]:
        """어미별 감정 점수 산출."""
        scores: Dict[str, float] = {}
        for morph in morphemes:
            info = self.ending_emotions.get(morph)
            if not info:
                continue
            emotion = info['emotion']
            intensity = self.intensity_map.get(info.get('intensity', '중'), 0.5)
            scores[emotion] = max(scores.get(emotion, 0.0), intensity)
        return scores

    def text_to_emotion_music(self, text: str) -> Dict:
        """
        텍스트 → 감정-음악 매핑 결과 반환.
        returns: dict with morphemes, emotions, primary_emotion, intensity, music_chord, is_sarcasm, processing_time_ms, memory_usage_bytes
        """
        t0 = time.time()
        morphemes = self.extract_morphemes(text)
        is_sarcasm = self.detect_sarcasm(text)
        scores = self.calculate_emotion_scores(morphemes)

        avg_intensity = (sum(scores.values()) / len(scores)) if scores else 0.0
        primary_emotion = max(scores, key=scores.get) if scores else '중립'
        if is_sarcasm:
            primary_emotion = '반어법'

        chord = self.emotion_to_chord.get(primary_emotion, 'C Minor')

        result = {
            'morphemes': morphemes,
            'emotions': scores,
            'primary_emotion': primary_emotion,
            'intensity': avg_intensity,
            'music_chord': chord,
            'is_sarcasm': is_sarcasm,
            'processing_time_ms': (time.time() - t0) * 1000.0,
            'memory_usage_bytes': (
                sys.getsizeof(text)
                + sys.getsizeof(morphemes)
                + sys.getsizeof(scores)
            ),
        }
        return result


__all__ = [
    'KoreanEmotionProcessor',
    'KONLPY_AVAILABLE',
]
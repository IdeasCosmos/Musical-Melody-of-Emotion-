#!/usr/bin/env python3
"""
korean_emotion_processor.py
- KoreanEmotionProcessor: 한국어 형태소 분석(간단화), 어미별 감정 점수 계산, 반어법 감지.
- text_to_emotion_music: 텍스트 → 형태소 추출 → 어미 매핑 → 감정 강도 → 음악 코드 생성.
- Konlpy 대체: 환경 제한으로 단순 문자열 매칭 사용 (konlpy import 시도 후 fallback).
"""

import time
import sys
import re
from typing import Dict, List

# Konlpy 시도 (환경에 없을 수 있음)
try:
    from konlpy.tag import Komoran
    KONLPY_AVAILABLE = True
except ImportError:
    KONLPY_AVAILABLE = False
    # 환경에 없을 수 있으므로 경고만 출력
    print("Konlpy not available, using simple string matching for morpheme analysis.")


class KoreanEmotionProcessor:
    """
    한국어 감정 처리기: 형태소 분석, 어미 감정 매핑, 반어법 감지.
    """

    def __init__(self) -> None:
        # 제공된 한국어 데이터
        self.ending_emotions: Dict[str, Dict[str, str]] = {
            '네요': {'emotion': '놀람/깨달음', 'intensity': '중'},
            '군요': {'emotion': '발견/인정', 'intensity': '중'},
            '거든요': {'emotion': '정당화/단호', 'intensity': '중'},
            '잖아요': {'emotion': '공유지식', 'intensity': '중'},
        }

        # 반어법 키워드 (간단한 휴리스틱)
        self.sarcasm_keywords: List[str] = [
            "참",
            "그래 잘했어",
        ]

        # 강도 맵핑 (숫자화)
        self.intensity_map: Dict[str, float] = {'중': 0.5, '강': 0.8, '약': 0.3}

        # 음악 코드 매핑 (감정 → 코드, 임시 예시)
        self.emotion_to_chord: Dict[str, str] = {
            '놀람/깨달음': 'C Major',
            '발견/인정': 'G Major',
            '정당화/단호': 'D Minor',
            '공유지식': 'F Major',
            '반어법': 'E7',  # 긴장/반전 코드
            '중립': 'C Minor',
        }

        if KONLPY_AVAILABLE:
            self.komoran = Komoran()
        else:
            self.komoran = None

    def extract_morphemes(self, text: str) -> List[str]:
        """형태소 추출: Konlpy 사용 또는 fallback 정규식 매칭.

        - Konlpy 사용 시 어미(EF, EC)만 수집
        - Fallback: 문장 말미의 대표 어미를 정규식으로 추출
        """
        if KONLPY_AVAILABLE and self.komoran is not None:
            return [morph for morph, tag in self.komoran.pos(text) if tag in ['EF', 'EC']]
        # 간단 패턴: 문장 끝 어미 탐지 (여러 문장 지원)
        endings: List[str] = []
        for sentence in re.split(r"[\.!?\n]+", text):
            sentence = sentence.strip()
            if not sentence:
                continue
            match = re.search(r"(네요|군요|거든요|잖아요)$", sentence)
            if match:
                endings.append(match.group(1))
        return endings

    def detect_sarcasm(self, text: str) -> bool:
        """반어법 감지: 키워드 매칭 (대소문자 그대로 매칭)."""
        for kw in self.sarcasm_keywords:
            if kw in text:
                return True
        return False

    def calculate_emotion_scores(self, morphemes: List[str]) -> Dict[str, float]:
        """어미별 감정 점수 계산."""
        scores: Dict[str, float] = {}
        for morph in morphemes:
            if morph in self.ending_emotions:
                emotion_label = self.ending_emotions[morph]['emotion']
                intensity_label = self.ending_emotions[morph]['intensity']
                intensity_value = self.intensity_map.get(intensity_label, 0.5)
                # 여러 번 나타나면 누적의 최대값을 사용
                prev = scores.get(emotion_label, 0.0)
                scores[emotion_label] = max(prev, intensity_value)
        return scores

    def text_to_emotion_music(self, text: str) -> Dict:
        """텍스트 → 감정-음악 변환."""
        start_time = time.time()
        morphemes = self.extract_morphemes(text)
        is_sarcasm = self.detect_sarcasm(text)
        scores = self.calculate_emotion_scores(morphemes)

        # 감정 강도 계산 (평균)
        total_intensity = sum(scores.values()) / max(1, len(scores)) if scores else 0.0
        primary_emotion = max(scores, key=scores.get) if scores else '중립'

        # 반어법 적용 (감정 반전)
        if is_sarcasm:
            primary_emotion = '반어법'
            total_intensity *= -1  # 부정으로 flip

        # 음악 코드 생성
        chord = self.emotion_to_chord.get(primary_emotion, 'C Minor')

        processing_time = (time.time() - start_time) * 1000.0  # ms
        memory_usage = (
            sys.getsizeof(text)
            + sys.getsizeof(morphemes)
            + sys.getsizeof(scores)
        )  # approximate bytes

        return {
            'morphemes': morphemes,
            'emotions': scores,
            'primary_emotion': primary_emotion,
            'intensity': total_intensity,
            'music_chord': chord,
            'is_sarcasm': is_sarcasm,
            'processing_time_ms': processing_time,
            'memory_usage_bytes': memory_usage,
        }


__all__ = ["KoreanEmotionProcessor"]


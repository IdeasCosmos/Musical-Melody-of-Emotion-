#!/usr/bin/env python3
import unittest
from korean_emotion_processor import KoreanEmotionProcessor


class TestKoreanEmotionProcessor(unittest.TestCase):
    def setUp(self) -> None:
        self.processor = KoreanEmotionProcessor()

    def test_extract_morphemes(self):
        text = "이게 재미있네요. 정말 그렇군요!"
        morphemes = self.processor.extract_morphemes(text)
        self.assertIn('네요', morphemes)
        self.assertIn('군요', morphemes)

    def test_detect_sarcasm(self):
        text1 = "참 잘했어."
        text2 = "좋은 아침입니다."
        self.assertTrue(self.processor.detect_sarcasm(text1))
        self.assertFalse(self.processor.detect_sarcasm(text2))

    def test_calculate_emotion_scores(self):
        morphemes = ['네요', '군요']
        scores = self.processor.calculate_emotion_scores(morphemes)
        self.assertIn('놀람/깨달음', scores)
        self.assertEqual(scores['놀람/깨달음'], 0.5)

    def test_text_to_emotion_music(self):
        text = "이게 재미있네요. 참 잘했어."
        result = self.processor.text_to_emotion_music(text)
        self.assertIn('emotions', result)
        self.assertTrue(result['is_sarcasm'])
        self.assertIn(result['primary_emotion'], ['반어법', '놀람/깨달음', '발견/인정', '정당화/단호', '공유지식', '중립'])


if __name__ == '__main__':
    unittest.main()


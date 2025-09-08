import unittest

from korean_emotion_processor import KoreanEmotionProcessor


class TestKoreanEmotionProcessor(unittest.TestCase):
    def setUp(self) -> None:
        self.processor = KoreanEmotionProcessor()

    def test_extract_morphemes(self) -> None:
        text = "이게 재미있네요. 정말 그렇군요"
        morphemes = self.processor.extract_morphemes(text)
        self.assertIn('네요', morphemes)
        self.assertIn('군요', morphemes)

    def test_detect_sarcasm(self) -> None:
        self.assertTrue(self.processor.detect_sarcasm("참 잘했어."))
        self.assertFalse(self.processor.detect_sarcasm("좋은 아침입니다."))

    def test_calculate_emotion_scores(self) -> None:
        scores = self.processor.calculate_emotion_scores(['네요', '군요'])
        self.assertIn('놀람/깨달음', scores)
        self.assertGreaterEqual(scores['놀람/깨달음'], 0.5)

    def test_text_to_emotion_music(self) -> None:
        result = self.processor.text_to_emotion_music("이게 재미있네요. 참 잘했어.")
        self.assertIn('emotions', result)
        self.assertTrue(result['is_sarcasm'])
        self.assertLess(result['processing_time_ms'], 500.0)


if __name__ == '__main__':
    unittest.main()


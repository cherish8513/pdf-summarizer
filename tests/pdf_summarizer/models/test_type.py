from unittest import TestCase

from pdf_summarizer.models.type import Type


class TestType(TestCase):
    def test_type(self):
        text = Type.TEXT
        self.assertEqual(text.value, "text")

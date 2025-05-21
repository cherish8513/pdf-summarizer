import unittest

from pdf_summarizer.models.type import Type


class MyTestCase(unittest.TestCase):
    def test_type(self):
        text = Type.TEXT
        self.assertEqual(text.value, "text")


if __name__ == '__main__':
    unittest.main()

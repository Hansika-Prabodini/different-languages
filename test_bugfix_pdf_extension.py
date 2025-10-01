import unittest
from unittest.mock import Mock, patch
import sys

# Ensure app module is importable
sys.path.append('.')
import app


class TestPdfToWordCaseInsensitiveExtension(unittest.TestCase):
    @patch('app.Converter')
    def test_uppercase_pdf_extension(self, mock_converter_class):
        mock_converter = Mock()
        mock_converter_class.return_value = mock_converter

        mock_file = Mock()
        mock_file.name = 'Report.PDF'

        result = app.pdf_to_word(mock_file)

        expected = 'Report.docx'
        self.assertEqual(result, expected)
        mock_converter_class.assert_called_once_with('Report.PDF')
        mock_converter.convert.assert_called_once_with(expected, multi_processing=True, start=0, end=None)
        mock_converter.close.assert_called_once()

    @patch('app.Converter')
    def test_mixedcase_pdf_extension_and_string_input(self, mock_converter_class):
        mock_converter = Mock()
        mock_converter_class.return_value = mock_converter

        # Pass a string path instead of a file-like object
        input_path = 'dir/thing.PdF'
        result = app.pdf_to_word(input_path)

        expected = 'dir/thing.docx'
        self.assertEqual(result, expected)
        mock_converter_class.assert_called_once_with(input_path)
        mock_converter.convert.assert_called_once_with(expected, multi_processing=True, start=0, end=None)
        mock_converter.close.assert_called_once()


if __name__ == '__main__':
    unittest.main(verbosity=2)

import unittest
from unittest.mock import Mock, patch
import sys

# Ensure module import
sys.path.append('.')
import app

class TestPdfToWordBugfix(unittest.TestCase):
    @patch('app.Converter')
    def test_uppercase_pdf_extension_converts_correctly(self, mock_converter_class):
        # Given a file with uppercase .PDF extension
        mock_file = Mock()
        mock_file.name = 'REPORT.PDF'

        mock_converter = Mock()
        mock_converter_class.return_value = mock_converter

        # When converting
        result = app.pdf_to_word(mock_file)

        # Then the output should use .docx and not leave .PDF
        self.assertEqual(result, 'REPORT.docx')
        mock_converter.convert.assert_called_once_with('REPORT.docx', multi_processing=True, start=0, end=None)
        mock_converter.close.assert_called_once()

if __name__ == '__main__':
    unittest.main()

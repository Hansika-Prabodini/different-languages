import unittest
from unittest.mock import Mock, patch
import sys

# Ensure we can import the module under test
sys.path.append('.')
import app


class TestPdfToWordCaseInsensitive(unittest.TestCase):
    @patch('app.Converter')
    def test_mixed_case_extension_converts_correctly(self, mock_converter_class):
        # Arrange
        mock_converter = Mock()
        mock_converter_class.return_value = mock_converter

        mock_file = Mock()
        mock_file.name = 'My.Report.PdF'

        # Act
        result = app.pdf_to_word(mock_file)

        # Assert: filename uses .docx and preserves base name
        self.assertEqual(result, 'My.Report.docx')
        mock_converter_class.assert_called_once_with('My.Report.PdF')
        mock_converter.convert.assert_called_once()
        mock_converter.close.assert_called_once()


if __name__ == '__main__':
    unittest.main()

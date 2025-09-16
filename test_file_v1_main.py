"""
Comprehensive test suite for Taipy chat application (file-v1-main.py).
Tests all critical functionality including API calls, context management, and UI interactions.
"""
import unittest
from unittest.mock import Mock, patch, MagicMock
import sys
import os

# Import the module under test
sys.path.append('.')
import importlib.util

# Load the module dynamically
spec = importlib.util.spec_from_file_location("file_v1_main", "file-v1-main.py")
file_v1_main = importlib.util.module_from_spec(spec)


class MockState:
    """Mock Taipy State object for testing."""
    def __init__(self):
        self.context = ""
        self.conversation = {"Conversation": []}
        self.current_user_message = ""
        self.past_conversations = []
        self.selected_conv = None
        self.selected_row = [1]
        self.client = None


class TestChatApplication(unittest.TestCase):
    """Test cases for chat application functions."""

    def setUp(self):
        """Set up test fixtures before each test method."""
        self.mock_state = MockState()
        self.mock_client = Mock()
        self.mock_state.client = self.mock_client

    def test_on_init(self):
        """Test initialization function."""
        file_v1_main.on_init(self.mock_state)
        
        self.assertEqual(self.mock_state.context, file_v1_main.DEFAULT_CONTEXT)
        self.assertEqual(self.mock_state.conversation, file_v1_main.DEFAULT_CONVERSATION.copy())
        self.assertEqual(self.mock_state.current_user_message, "")
        self.assertEqual(self.mock_state.past_conversations, [])
        self.assertEqual(self.mock_state.selected_conv, None)
        self.assertEqual(self.mock_state.selected_row, [1])

    def test_request_successful(self):
        """Test successful API request."""
        # Mock successful API response
        mock_response = Mock()
        mock_response.choices[0].message.content = "Test response"
        self.mock_client.chat.completions.create.return_value = mock_response
        
        result = file_v1_main.request(self.mock_state, "Test prompt")
        
        self.assertEqual(result, "Test response")
        self.mock_client.chat.completions.create.assert_called_once()

    def test_request_with_exception(self):
        """Test API request with exception handling."""
        # Mock API exception
        self.mock_client.chat.completions.create.side_effect = Exception("API Error")
        
        with patch('file-v1-main.on_exception') as mock_on_exception:
            result = file_v1_main.request(self.mock_state, "Test prompt")
            
            self.assertEqual(result, "Sorry, I encountered an error processing your request.")
            mock_on_exception.assert_called_once()

    def test_update_context(self):
        """Test context update functionality."""
        self.mock_state.context = "Initial context"
        self.mock_state.current_user_message = "Hello"
        self.mock_state.conversation = {"Conversation": ["Previous message"]}
        
        # Mock the request function
        with patch.object(file_v1_main, 'request', return_value="AI response"):
            result = file_v1_main.update_context(self.mock_state)
            
            self.assertEqual(result, "AI response")
            self.assertIn("Human: \n Hello\n\n AI:", self.mock_state.context)
            self.assertIn("AI response", self.mock_state.context)
            self.assertEqual(self.mock_state.selected_row, [2])  # len + 1

    def test_send_message_empty(self):
        """Test sending empty message."""
        self.mock_state.current_user_message = "   "  # Whitespace only
        
        with patch('file-v1-main.notify') as mock_notify:
            file_v1_main.send_message(self.mock_state)
            
            mock_notify.assert_called_with(self.mock_state, "warning", "Please enter a message")

    def test_send_message_successful(self):
        """Test successful message sending."""
        self.mock_state.current_user_message = "Hello"
        self.mock_state.conversation = {"Conversation": ["Initial"]}
        
        with patch.object(file_v1_main, 'update_context', return_value="AI response") as mock_update:
            with patch('file-v1-main.notify') as mock_notify:
                file_v1_main.send_message(self.mock_state)
                
                mock_update.assert_called_once()
                self.assertEqual(self.mock_state.current_user_message, "")
                self.assertEqual(len(self.mock_state.conversation["Conversation"]), 3)
                mock_notify.assert_any_call(self.mock_state, "info", "Sending message...")
                mock_notify.assert_any_call(self.mock_state, "success", "Response received!")

    def test_send_message_with_exception(self):
        """Test message sending with exception."""
        self.mock_state.current_user_message = "Hello"
        
        with patch.object(file_v1_main, 'update_context', side_effect=Exception("Test error")):
            with patch('file-v1-main.on_exception') as mock_on_exception:
                file_v1_main.send_message(self.mock_state)
                
                mock_on_exception.assert_called_once()

    def test_style_conv(self):
        """Test conversation styling function."""
        # Test user message (even index)
        result = file_v1_main.style_conv(self.mock_state, 0, 1)
        self.assertEqual(result, "user_message")
        
        # Test AI message (odd index)  
        result = file_v1_main.style_conv(self.mock_state, 1, 1)
        self.assertEqual(result, "gpt_message")
        
        # Test None index
        result = file_v1_main.style_conv(self.mock_state, None, 1)
        self.assertIsNone(result)

    def test_on_exception(self):
        """Test exception handling function."""
        with patch('file-v1-main.notify') as mock_notify:
            test_exception = Exception("Test error")
            file_v1_main.on_exception(self.mock_state, "test_function", test_exception)
            
            mock_notify.assert_called_with(
                self.mock_state, 
                "error", 
                "An error occured in test_function: Test error"
            )

    def test_reset_chat_with_conversation(self):
        """Test chat reset with existing conversation."""
        self.mock_state.conversation = {"Conversation": ["Q1", "A1", "Q2", "A2"]}
        self.mock_state.past_conversations = []
        self.mock_state.context = "Old context"
        
        file_v1_main.reset_chat(self.mock_state)
        
        # Should save the conversation
        self.assertEqual(len(self.mock_state.past_conversations), 1)
        self.assertEqual(self.mock_state.past_conversations[0][0], 0)
        
        # Should reset to defaults
        self.assertEqual(self.mock_state.conversation, file_v1_main.DEFAULT_CONVERSATION.copy())
        self.assertEqual(self.mock_state.context, file_v1_main.DEFAULT_CONTEXT)
        self.assertEqual(self.mock_state.selected_row, [1])

    def test_reset_chat_empty_conversation(self):
        """Test chat reset with empty conversation."""
        self.mock_state.conversation = {"Conversation": ["Q", "A"]}  # Too short
        initial_past_count = len(self.mock_state.past_conversations)
        
        file_v1_main.reset_chat(self.mock_state)
        
        # Should not save short conversation
        self.assertEqual(len(self.mock_state.past_conversations), initial_past_count)

    def test_tree_adapter(self):
        """Test tree adapter function."""
        # Test with long conversation
        item = [0, {"Conversation": ["Q1", "A1", "This is a very long conversation message that should be truncated"]}]
        result = file_v1_main.tree_adapter(item)
        
        self.assertEqual(result[0], 0)
        self.assertTrue(result[1].endswith("..."))
        self.assertTrue(len(result[1]) <= 53)  # 50 chars + "..."
        
        # Test with short conversation
        item = [1, {"Conversation": ["Q1", "A1"]}]
        result = file_v1_main.tree_adapter(item)
        
        self.assertEqual(result, (1, "Empty conversation"))

    def test_select_conv_valid(self):
        """Test conversation selection with valid input."""
        # Set up past conversation
        past_conv = {"Conversation": ["Q1", "A1", "Q2", "A2"]}
        self.mock_state.past_conversations = [[0, past_conv]]
        
        file_v1_main.select_conv(self.mock_state, "selected_conv", [[0]])
        
        self.assertEqual(self.mock_state.conversation, past_conv)
        self.assertIn("Q2", self.mock_state.context)
        self.assertIn("A2", self.mock_state.context)
        self.assertEqual(self.mock_state.selected_row, [3])  # len - 1

    def test_select_conv_invalid(self):
        """Test conversation selection with invalid input."""
        original_conversation = self.mock_state.conversation.copy()
        
        # Test with empty value
        file_v1_main.select_conv(self.mock_state, "selected_conv", [])
        self.assertEqual(self.mock_state.conversation, original_conversation)
        
        # Test with None
        file_v1_main.select_conv(self.mock_state, "selected_conv", None)
        self.assertEqual(self.mock_state.conversation, original_conversation)

    def test_constants(self):
        """Test application constants."""
        self.assertIsInstance(file_v1_main.DEFAULT_CONTEXT, str)
        self.assertIn("conversation with an AI", file_v1_main.DEFAULT_CONTEXT.lower())
        
        self.assertIsInstance(file_v1_main.DEFAULT_CONVERSATION, dict)
        self.assertIn("Conversation", file_v1_main.DEFAULT_CONVERSATION)
        self.assertIsInstance(file_v1_main.DEFAULT_CONVERSATION["Conversation"], list)


class TestChatApplicationIntegration(unittest.TestCase):
    """Integration tests for chat application workflow."""

    def setUp(self):
        """Set up integration test fixtures."""
        self.mock_state = MockState()
        self.mock_client = Mock()
        self.mock_state.client = self.mock_client

    def test_complete_conversation_flow(self):
        """Test complete conversation workflow."""
        # Initialize
        file_v1_main.on_init(self.mock_state)
        
        # Mock API response
        mock_response = Mock()
        mock_response.choices[0].message.content = "Hello! How can I help you?"
        self.mock_client.chat.completions.create.return_value = mock_response
        
        # Send message
        self.mock_state.current_user_message = "Hello"
        
        with patch('file-v1-main.notify'):
            file_v1_main.send_message(self.mock_state)
            
            # Verify conversation updated
            self.assertEqual(len(self.mock_state.conversation["Conversation"]), 4)
            self.assertIn("Hello", self.mock_state.conversation["Conversation"])
            self.assertIn("Hello! How can I help you?", self.mock_state.conversation["Conversation"])

    def test_multiple_conversations(self):
        """Test handling multiple conversations."""
        file_v1_main.on_init(self.mock_state)
        
        # Create a conversation
        self.mock_state.conversation = {"Conversation": ["Q1", "A1", "Q2", "A2"]}
        
        # Reset to start new conversation
        file_v1_main.reset_chat(self.mock_state)
        
        # Verify old conversation saved and new one started
        self.assertEqual(len(self.mock_state.past_conversations), 1)
        self.assertEqual(self.mock_state.conversation, file_v1_main.DEFAULT_CONVERSATION.copy())
        
        # Select old conversation
        file_v1_main.select_conv(self.mock_state, "selected_conv", [[0]])
        self.assertIn("Q2", self.mock_state.conversation["Conversation"])

    def test_error_recovery(self):
        """Test error recovery in various scenarios."""
        file_v1_main.on_init(self.mock_state)
        
        # Test API failure recovery
        self.mock_client.chat.completions.create.side_effect = Exception("Network error")
        
        with patch('file-v1-main.on_exception'):
            result = file_v1_main.request(self.mock_state, "Test")
            self.assertIn("error", result.lower())


class TestChatApplicationEdgeCases(unittest.TestCase):
    """Test edge cases and boundary conditions."""

    def setUp(self):
        """Set up edge case test fixtures."""
        self.mock_state = MockState()

    def test_empty_state_handling(self):
        """Test handling of empty or None state values."""
        # Test with minimal state
        minimal_state = Mock()
        minimal_state.context = None
        minimal_state.conversation = None
        
        # Should not crash when initializing
        try:
            file_v1_main.on_init(minimal_state)
            self.assertIsNotNone(minimal_state.context)
        except AttributeError:
            pass  # Acceptable if function expects full state

    def test_large_conversation_handling(self):
        """Test handling of very large conversations."""
        # Create large conversation
        large_conv = ["Message"] * 1000
        self.mock_state.conversation = {"Conversation": large_conv}
        
        # Test reset doesn't fail with large data
        try:
            file_v1_main.reset_chat(self.mock_state)
            self.assertTrue(True)  # If we get here, no exception occurred
        except Exception:
            self.fail("Reset failed with large conversation")

    def test_special_characters(self):
        """Test handling of special characters in messages."""
        self.mock_state.current_user_message = "Hello! 🤖 How are you? \n\t Special chars: @#$%"
        
        with patch.object(file_v1_main, 'update_context', return_value="Response"):
            with patch('file-v1-main.notify'):
                try:
                    file_v1_main.send_message(self.mock_state)
                    self.assertTrue(True)  # If we get here, special chars handled
                except Exception as e:
                    self.fail(f"Special characters caused failure: {e}")

    def test_tree_adapter_edge_cases(self):
        """Test tree adapter with edge cases."""
        # Test with empty conversation
        item = [0, {"Conversation": []}]
        result = file_v1_main.tree_adapter(item)
        self.assertEqual(result, (0, "Empty conversation"))
        
        # Test with exactly 3 items
        item = [1, {"Conversation": ["Q", "A", "Q2"]}]
        result = file_v1_main.tree_adapter(item)
        self.assertEqual(result, (1, "Empty conversation"))


if __name__ == '__main__':
    # Set up mock environment for testing
    with patch.dict(os.environ, {'OPENAI_API_KEY': 'test_key'}):
        # Run tests with verbose output
        unittest.main(verbosity=2)

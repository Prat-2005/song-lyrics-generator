import unittest
import sys
from pathlib import Path
from unittest.mock import patch

# Ensure src is in the python path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'src'))

# Patch retrieval globally so importing agent doesn't load index
with patch('retrieval.LyricsRetriever', autospec=True):
    from agent import parse_tool_arguments, normalize_tool_calls, execute_tool_call

class TestAgent(unittest.TestCase):
    def test_parse_tool_arguments(self):
        """Test parsing of tool arguments from different formats"""
        # Already a dict
        self.assertEqual(parse_tool_arguments({"word": "test"}), {"word": "test"})
        # Valid JSON string
        self.assertEqual(parse_tool_arguments('{"word": "test"}'), {"word": "test"})
        # Invalid JSON string
        self.assertEqual(parse_tool_arguments('{word: test}'), {})
        # Not a dict or string
        self.assertEqual(parse_tool_arguments(123), {})

    def test_normalize_tool_calls(self):
        """Test normalization of tool calls from text response"""
        # Already has tool_calls
        resp = {"tool_calls": [{"function": {"name": "test"}}]}
        self.assertEqual(normalize_tool_calls(resp.copy()), resp)
        
        # Valid JSON tool call in content
        resp2 = {"content": '{"name": "get_rhymes", "arguments": "{\\"word\\": \\"test\\"}"}'}
        normalized = normalize_tool_calls(resp2)
        self.assertIn("tool_calls", normalized)
        self.assertEqual(normalized["tool_calls"][0]["function"]["name"], "get_rhymes")
        
        # Plain text content
        resp3 = {"content": "Here is a lyric"}
        normalized3 = normalize_tool_calls(resp3.copy())
        self.assertNotIn("tool_calls", normalized3)

    @patch('agent.TOOL_FUNCTIONS')
    def test_execute_tool_call(self, mock_tools):
        """Test executing tool calls"""
        mock_tools.__contains__.return_value = True
        mock_tools.__getitem__.return_value = lambda word: ["test1", "test2"]
        
        # Successful call
        result = execute_tool_call("get_rhymes", {"word": "test"})
        self.assertEqual(result, ["test1", "test2"])
        
        # Unknown tool
        mock_tools.__contains__.return_value = False
        result_unknown = execute_tool_call("unknown_tool", {})
        self.assertTrue(isinstance(result_unknown, str))
        self.assertIn("Error: unknown tool", result_unknown)
        
        # Bad args
        mock_tools.__contains__.return_value = True
        result_bad_args = execute_tool_call("get_rhymes", "not a dict")
        self.assertTrue(isinstance(result_bad_args, str))
        self.assertIn("args must be a dictionary", result_bad_args)

if __name__ == '__main__':
    unittest.main()

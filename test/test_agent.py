import pytest
from unittest.mock import patch, MagicMock
from src.agent import execute_tool_call, resolve_response_with_tools, generate_lyrics_v2, sanitize_lyrics_output

def test_sanitize_lyrics_output():
    raw_text = '<tools></tools>\n{"name": "get_similar_lines", "arguments": "{\\"query\\": \\"test\\"}"}\n```json\n{"name": "check_sentiment"}\n```\nHere are the real lyrics line 1\nHere are the real lyrics line 2'
    cleaned = sanitize_lyrics_output(raw_text)
    assert "Here are the real lyrics line 1" in cleaned
    assert "get_similar_lines" not in cleaned
    assert "<tools>" not in cleaned

def test_execute_tool_call_success():
    # Mock a simple tool function
    mock_func = MagicMock(return_value="Tool Result")
    with patch('src.agent.TOOL_FUNCTIONS', {"my_tool": mock_func}):
        result = execute_tool_call("my_tool", {"arg1": "val1"})
        assert result == "Tool Result"
        mock_func.assert_called_with(arg1="val1")

def test_execute_tool_call_unknown():
    with patch('src.agent.TOOL_FUNCTIONS', {}):
        result = execute_tool_call("unknown", {})
        assert "Error: unknown tool" in result

def test_execute_tool_call_bad_args():
    mock_func = MagicMock()
    with patch('src.agent.TOOL_FUNCTIONS', {"my_tool": mock_func}):
        result = execute_tool_call("my_tool", "not a dict")
        assert "Error: args must be a dictionary" in result

@patch('src.agent.call_local_llm')
def test_resolve_response_with_tools_no_tools(mock_llm):
    mock_llm.return_value = {"role": "assistant", "content": "Plain text response"}

    messages = [{"role": "user", "content": "Hello"}]
    tool_logs = []
    response = resolve_response_with_tools(messages, 1.0, tool_logs)

    assert response["content"] == "Plain text response"
    assert len(tool_logs) == 0

@patch('src.agent.call_local_llm')
@patch('src.agent.execute_tool_call')
def test_resolve_response_with_tools_with_tool(mock_execute, mock_llm):
    # 1st call: LLM wants to call a tool
    # 2nd call: LLM finishes after seeing tool result
    mock_llm.side_effect = [
        {"role": "assistant", "tool_calls": [{"function": {"name": "get_rhymes", "arguments": {"word": "blue"}}}]},
        {"role": "assistant", "content": "Rhymes with blue are true and new."}
    ]
    mock_execute.return_value = ["true", "new"]

    messages = [{"role": "user", "content": "What rhymes with blue?"}]
    tool_logs = []
    response = resolve_response_with_tools(messages, 1.0, tool_logs)

    assert response["content"] == "Rhymes with blue are true and new."
    assert len(tool_logs) == 1
    assert tool_logs[0]["tool"] == "get_rhymes"

@patch('src.agent.resolve_response_with_tools')
@patch('src.agent.call_local_llm')
@patch('src.agent.get_similar_lines')
def test_generate_lyrics_v2_flow(mock_similar, mock_llm, mock_resolve):
    mock_similar.return_value = ["Line 1", "Line 2"]
    mock_resolve.return_value = {"role": "assistant", "content": "Generated lyrics"}
    # Mock critique to say FINAL immediately
    mock_llm.return_value = {"role": "assistant", "content": "FINAL"}

    lyrics, logs = generate_lyrics_v2("Theme", "Artist")

    assert lyrics == "Generated lyrics"
    mock_similar.assert_called()
    mock_resolve.assert_called()

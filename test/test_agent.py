from unittest.mock import patch

from src.agent import generate_lyrics, _strip_formatting, _passes_targets, TARGETS


def test_strip_formatting_removes_code_fence():
    raw = "```\nHere are the lyrics\nline two\n```"
    assert _strip_formatting(raw) == "Here are the lyrics\nline two"


def test_strip_formatting_handles_empty():
    assert _strip_formatting("") == ""
    assert _strip_formatting(None) == ""


def test_passes_targets_true_when_all_met():
    metrics = {k: v + 0.1 for k, v in TARGETS.items()}
    assert _passes_targets(metrics)


def test_passes_targets_false_when_one_missed():
    metrics = {k: v + 0.1 for k, v in TARGETS.items()}
    metrics["rhyme_density"] = 0.0
    assert not _passes_targets(metrics)


@patch("src.agent.get_similar_lines")
@patch("src.agent.call_llm")
@patch("src.agent._metrics_for")
def test_generate_lyrics_no_revision_when_targets_met(mock_metrics, mock_llm, mock_similar):
    mock_similar.return_value = ["Line 1", "Line 2"]
    mock_llm.return_value = ("Generated lyrics", None)
    mock_metrics.return_value = {k: v + 0.1 for k, v in TARGETS.items()}

    lyrics, log, metrics = generate_lyrics("heartbreak", "Drake")

    assert lyrics == "Generated lyrics"
    # Only the draft call — no revision call — since targets were already met.
    assert mock_llm.call_count == 1
    steps = [entry["step"] for entry in log]
    assert "Revision" in steps
    assert any("Not needed" in entry["info"] for entry in log if entry["step"] == "Revision")


@patch("src.agent.get_similar_lines")
@patch("src.agent.call_llm")
@patch("src.agent._metrics_for")
def test_generate_lyrics_revises_when_targets_missed(mock_metrics, mock_llm, mock_similar):
    mock_similar.return_value = ["Line 1"]
    mock_llm.side_effect = [
        ("First draft", None),
        ("Revised draft", None),
    ]
    # First call under target, second call meets target.
    mock_metrics.side_effect = [
        {k: 0.0 for k in TARGETS},
        {k: v + 0.1 for k, v in TARGETS.items()},
    ]

    lyrics, log, metrics = generate_lyrics("heartbreak", "Drake")

    assert lyrics == "Revised draft"
    assert mock_llm.call_count == 2


@patch("src.agent.get_similar_lines")
@patch("src.agent.call_llm")
def test_generate_lyrics_returns_error_string_on_failure(mock_llm, mock_similar):
    mock_similar.return_value = []
    mock_llm.return_value = (None, "Local model unreachable and no fallback configured")

    lyrics, log, metrics = generate_lyrics("heartbreak")

    assert lyrics.startswith("Error:")
    assert metrics == {}

"""
Lyrics generation.

Design notes (read this before you "improve" it again):

- There is NO agent loop and NO tool-calling here on purpose. Retrieval
  (get_similar_lines) and rhyme lookups are things WE decide to do in Python,
  not things the model needs to be given as a "tool" and asked to call. That
  removes an entire class of bugs: malformed tool-call JSON, models leaking
  <tools> tags into the output, multi-round tool negotiation, etc.

- There is at most ONE revision pass, and it only happens if the metrics
  (rhyme density / syllable consistency / originality) actually fall short
  of target. We check that in code. We never ask the model to grade its own
  work and decide when to stop — that's slow and unreliable.

- Local model first, cloud fallback second. If both fail, we return a clear
  error string instead of silently returning None / crashing.
"""

import json
import re
import time

import requests
from openai import OpenAI

from src.config import (
    LOCAL_URL, LOCAL_MODEL, LOCAL_PROVIDER, MODEL_TIMEOUT,
    FALLBACK_PROVIDER, FALLBACK_MODEL, FALLBACK_API_KEY, FALLBACK_BASE_URL,
)
from src.evaluate import rhyme_density, syllable_consistency, originality_score
from src.tools import get_similar_lines

_fallback_client = None

# Quality bars that decide (in code) whether a revision pass is worth doing.
TARGETS = {
    "rhyme_density": 0.30,
    "syllable_consistency": 0.70,
    "originality_score": 0.80,
}


def _strip_formatting(text):
    """Light cleanup only — strip a wrapping markdown code fence if present.
    There's no tool-JSON to scrub anymore because the model is never given
    tools to call, so this doesn't need to be a regex minefield."""
    if not text:
        return ""
    text = text.strip()
    text = re.sub(r'^```[a-zA-Z]*\n?', '', text)
    text = re.sub(r'\n?```$', '', text)
    return text.strip()


def _get_fallback_client():
    global _fallback_client
    if _fallback_client is None:
        kwargs = {"api_key": FALLBACK_API_KEY}
        if FALLBACK_BASE_URL:
            kwargs["base_url"] = FALLBACK_BASE_URL
        _fallback_client = OpenAI(**kwargs)
    return _fallback_client


def _call_fallback(messages, temperature, stream_callback=None):
    """Returns (text, error). Exactly one of them is set."""
    if not FALLBACK_API_KEY:
        return None, (
            "Local model unreachable AND no fallback configured "
            "(FALLBACK_API_KEY is missing in .env)."
        )

    try:
        client = _get_fallback_client()
        response = client.chat.completions.create(
            model=FALLBACK_MODEL,
            messages=messages,
            temperature=temperature,
            stream=bool(stream_callback),
        )
        if stream_callback:
            full_content = ""
            for chunk in response:
                delta = chunk.choices[0].delta
                if delta.content:
                    full_content += delta.content
                    stream_callback(delta.content, full_content)
            return full_content, None
        return response.choices[0].message.content or "", None
    except Exception as e:
        return None, f"Fallback LLM ({FALLBACK_PROVIDER}) also failed: {e}"


def call_llm(messages, temperature=1.0, stream_callback=None):
    """Call the local model; fall back to the cloud model if it's unreachable.
    Returns (text, error) — exactly one of them is set."""
    payload = {
        "provider": LOCAL_PROVIDER,
        "model": LOCAL_MODEL,
        "messages": messages,
        "stream": bool(stream_callback),
        "options": {"temperature": temperature},
    }

    try:
        response = requests.post(
            LOCAL_URL, json=payload, timeout=MODEL_TIMEOUT, stream=bool(stream_callback)
        )
        response.raise_for_status()

        if stream_callback:
            full_content = ""
            for line in response.iter_lines():
                if not line:
                    continue
                chunk = json.loads(line)
                content = chunk.get("message", {}).get("content", "")
                if content:
                    full_content += content
                    stream_callback(content, full_content)
            return full_content, None

        return response.json()["message"]["content"], None

    except (requests.exceptions.ConnectionError, requests.exceptions.Timeout) as e:
        print(f"Local model unreachable ({e}). Falling back to {FALLBACK_PROVIDER}...")
        return _call_fallback(messages, temperature, stream_callback)
    except Exception as e:
        print(f"Local model error ({e}). Falling back to {FALLBACK_PROVIDER}...")
        return _call_fallback(messages, temperature, stream_callback)


def _build_prompt(theme, artist_style, mood, max_words, context_lines):
    context_str = "\n".join(f"- {l}" for l in context_lines) or "(no reference lines found)"

    system_prompt = (
        "You are a skilled songwriter. Write lyrics that sound like a real song, "
        f"in the voice of {artist_style or 'a versatile modern artist'} — not like "
        "an AI describing one.\n\n"
        "GUIDELINES:\n"
        "1. Match the vocabulary, slang, and emotional tone in the reference lines "
        "below. Avoid generic, overly clean, or AI-sounding language.\n"
        "2. Rhyme with intent — end most lines on words that genuinely rhyme with "
        "a nearby line.\n"
        "3. Keep syllable count per line fairly consistent so it could actually be sung.\n"
        "4. Avoid clichés — if a line feels generic, make it more specific and personal.\n"
        "5. Output ONLY the finished lyrics in plain text. No JSON, no markdown, "
        "no explanations. Simple section labels like [Verse 1] or [Chorus] are fine."
    )

    user_prompt = (
        f"Theme: {theme}\n"
        f"Mood: {mood}\n"
        f"Artist style: {artist_style or 'Generic'}\n\n"
        "Reference lines from real songs in this style (for tone/vocabulary only — "
        f"don't copy them):\n{context_str}\n\n"
        f"Write a full song (verses + chorus) of about {max_words} words."
    )

    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]


def _metrics_for(lyrics):
    return {
        "rhyme_density": rhyme_density(lyrics),
        "syllable_consistency": syllable_consistency(lyrics),
        "originality_score": originality_score(lyrics),
    }


def _passes_targets(metrics):
    return all(metrics.get(k, 0.0) >= target for k, target in TARGETS.items())


def _build_revision_prompt(lyrics, metrics, max_words):
    word_count = len(lyrics.split())
    return (
        f"Here is the current draft:\n{lyrics}\n\n"
        "--- METRICS (target in brackets) ---\n"
        f"Rhyme density: {metrics['rhyme_density']:.0%} "
        f"[target >{TARGETS['rhyme_density']:.0%}]\n"
        f"Syllable consistency: {metrics['syllable_consistency']:.0%} "
        f"[target >{TARGETS['syllable_consistency']:.0%}]\n"
        f"Originality: {metrics['originality_score']:.0%} "
        f"[target >{TARGETS['originality_score']:.0%}]\n"
        f"Length: {word_count} words [target ~{max_words}]\n\n"
        "Revise the lyrics to close the gap on whichever metrics are below "
        "target, while keeping the theme and style. Output ONLY the revised "
        "lyrics in plain text."
    )


def generate_lyrics(theme, artist_style=None, mood="emotional", max_words=120,
                     temperature=1.0, stream_callback=None):
    """
    Single-pass generation with retrieval baked into the prompt, plus at most
    one revision pass — only if the metrics actually fall short of target.

    Returns (lyrics, generation_log, metrics). On failure, lyrics starts with
    "Error:" — callers should check for that rather than assuming success.
    """
    generation_log = []

    # Retrieval happens directly in Python, before we ever talk to the model.
    context_lines = get_similar_lines(theme, artist=artist_style, k=10)
    generation_log.append({
        "step": "Retrieval",
        "info": f"Pulled {len(context_lines)} reference lines for '{theme}'"
                + (f" ({artist_style} style)" if artist_style else ""),
    })

    messages = _build_prompt(theme, artist_style, mood, max_words, context_lines)

    lyrics, error = call_llm(messages, temperature=temperature, stream_callback=stream_callback)
    if error:
        return f"Error: {error}", generation_log, {}

    lyrics = _strip_formatting(lyrics)
    generation_log.append({"step": "Draft", "info": f"{len(lyrics.split())} words generated"})

    metrics = _metrics_for(lyrics)
    did_revise = False

    if not _passes_targets(metrics):
        messages.append({"role": "assistant", "content": lyrics})
        messages.append({"role": "user", "content": _build_revision_prompt(lyrics, metrics, max_words)})

        revised, error = call_llm(messages, temperature=temperature)
        if not error and revised and revised.strip():
            lyrics = _strip_formatting(revised)
            metrics = _metrics_for(lyrics)
            did_revise = True
            generation_log.append({"step": "Revision", "info": "Applied one revision pass to close metric gaps"})
        else:
            generation_log.append({"step": "Revision", "info": f"Skipped ({error or 'empty response'})"})
    else:
        generation_log.append({"step": "Revision", "info": "Not needed — draft already met all quality targets"})

    # The draft was already streamed live. If we revised afterward, stream the
    # final version too so the UI doesn't end up showing the pre-revision draft.
    if stream_callback and did_revise:
        full_content = ""
        for i in range(0, len(lyrics), 3):
            chunk = lyrics[i:i + 3]
            full_content += chunk
            stream_callback(chunk, full_content)
            time.sleep(0.02)

    return lyrics, generation_log, metrics

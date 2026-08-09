import requests
import json
import time
from openai import OpenAI
from tools import TOOL_SCHEMAS, TOOL_FUNCTIONS, get_similar_lines, count_syllables, get_rhymes
from config import LOCAL_URL, LOCAL_MODEL, LOCAL_PROVIDER, MODEL_TIMEOUT, FALLBACK_PROVIDER, FALLBACK_MODEL, FALLBACK_API_KEY
from evaluate import rhyme_density, syllable_consistency, originality_score

_fallback_client = None


def parse_tool_arguments(args):
    """Normalize tool arguments from provider-specific formats into a dict."""
    if isinstance(args, dict):
        return args
    if isinstance(args, str):
        try:
            parsed = json.loads(args)
            return parsed if isinstance(parsed, dict) else {}
        except Exception:
            return {}
    return {}


def normalize_tool_calls(response):
    """Ensure tool calls are in the 'tool_calls' list, even if provided as text."""
    if not response: return response
    if response.get("tool_calls"): return response

    content = (response.get("content") or "").strip()
    if content.startswith("{") and '"name"' in content:
        try:
            parsed = json.loads(content)
            if isinstance(parsed, dict) and "name" in parsed:
                response["tool_calls"] = [{"function": parsed}]
        except: pass
    return response

def execute_tool_call(func_name, args):
    """Execute one tool call safely and return a serializable result."""
    if func_name not in TOOL_FUNCTIONS:
        return f"Error: unknown tool '{func_name}'."
    if not isinstance(args, dict):
        return f"Error: args must be a dictionary, got {type(args)}"

    try:
        return TOOL_FUNCTIONS[func_name](**args)
    except Exception as e:
        return f"Error executing tool '{func_name}': {e}"


def resolve_response_with_tools(messages, temperature, tool_logs, max_rounds=4):
    """Run the model and resolve tool calls across multiple rounds."""
    response = call_local_llm(messages, tools=TOOL_SCHEMAS, temperature=temperature)
    if not response:
        return None

    response = normalize_tool_calls(response)
    seen_calls = set()

    for _ in range(max_rounds):
        tool_calls = response.get("tool_calls") or []
        if not tool_calls:
            return response

        messages.append(response)
        executed_any = False

        for tool_call in tool_calls:
            if not isinstance(tool_call, dict) or not isinstance(tool_call.get("function"), dict):
                continue

            func_name = tool_call["function"].get("name")
            args = parse_tool_arguments(tool_call["function"].get("arguments", {}))

            # Break repetitive tool loops where the model keeps requesting the same call.
            call_key = (func_name, json.dumps(args, sort_keys=True, ensure_ascii=True))
            if call_key in seen_calls:
                messages.append({
                    "role": "user",
                    "content": "You already called that same tool with identical arguments. Stop calling tools and provide plain text output only."
                })
                continue
            seen_calls.add(call_key)

            print(f"Calling tool: {func_name}({args})")
            result = execute_tool_call(func_name, args)
            tool_logs.append({"tool": func_name, "args": args, "result": result})
            executed_any = True

            messages.append({
                "role": "tool",
                "content": str(result),
                "name": func_name
            })

        if not executed_any:
            messages.append({
                "role": "user",
                "content": "Tool call payload was invalid. Continue without tools and return plain text only."
            })

        response = call_local_llm(messages, tools=TOOL_SCHEMAS, temperature=temperature)
        if not response:
            return None
        response = normalize_tool_calls(response)

    return response


def get_fallback_client():
    global _fallback_client
    if not _fallback_client:
        _fallback_client = OpenAI(api_key=FALLBACK_API_KEY)
    return _fallback_client

def call_fallback_llm(messages, tools=None, temperature=1.0, stream_callback=None):
    client = get_fallback_client()
    
    # Map tools to OpenAI format if provided
    formatted_tools = tools if tools else None

    try:
        if client == "":
            raise ValueError("Fallback LLM error: API key is not set. Please set FALLBACK_API_KEY.")
        response = client.chat.completions.create(
            model=FALLBACK_MODEL,
            messages=messages,
            temperature=temperature,
            tools=formatted_tools,
            stream=bool(stream_callback)
        )
        
        if stream_callback:
            full_content = ""
            for chunk in response:
                delta = chunk.choices[0].delta
                if delta.content:
                    full_content += delta.content
                    stream_callback(delta.content, full_content)
            # We don't support streaming tool calls in this basic implementation
            # Groq currently doesn't stream tool calls well anyway
            return {"role": "assistant", "content": full_content}
        else:
            msg = response.choices[0].message
            result = {"role": "assistant", "content": msg.content or ""}
            if msg.tool_calls:
                result["tool_calls"] = [
                    {"function": {"name": t.function.name, "arguments": t.function.arguments}} 
                    for t in msg.tool_calls
                ]
            return result
    except Exception as e:
        print(f"Fallback LLM error: {e}")
        return None

def call_local_llm(messages, tools=None, temperature=1.0, stream_callback=None):
    """Call local model with tool support."""
    payload = {
        "provider": LOCAL_PROVIDER,
        "model": LOCAL_MODEL,
        "messages": messages,
        "stream": bool(stream_callback),
        "options": {"temperature": temperature}
    }

    if tools:
        payload["tools"] = tools

    try:
        response = requests.post(LOCAL_URL, json=payload, timeout=MODEL_TIMEOUT, stream=bool(stream_callback))
        response.raise_for_status()
        
        if stream_callback:
            full_content = ""
            for line in response.iter_lines():
                if line:
                    chunk = json.loads(line)
                    if "message" in chunk and "content" in chunk["message"]:
                        content = chunk["message"]["content"]
                        full_content += content
                        stream_callback(content, full_content)
            return {"role": "assistant", "content": full_content}
        else:
            return response.json()["message"]
            
    except (requests.exceptions.ConnectionError, requests.exceptions.Timeout) as e:
        print(f"Local LLM failed/timed out: {e}. Switching to fallback ({FALLBACK_PROVIDER})...")
        return call_fallback_llm(messages, tools, temperature, stream_callback)
    except Exception as e:
        print(f"Unexpected error calling Local LLM: {e}")
        return call_fallback_llm(messages, tools, temperature, stream_callback)

def generate_lyrics_v2(theme, artist_style=None, mood="emotional", max_words=120, temperature=1.0, stream_callback=None):
    """Generate lyrics using a draft -> critique -> revise loop with tools."""

    # 1. Initial Context Retrieval
    print(f"Retrieving context for theme: {theme}...")
    similar_lines = get_similar_lines(theme, artist=artist_style, k=5)
    context_str = "\n".join([f"- {l}" for l in similar_lines])

    system_prompt = (
        "You are a world-class songwriter and lyrical architect. Your goal is to write lyrics that "
        f"perfectly capture the essence of {artist_style if artist_style else 'a versatile artist'}. "
        "You do not just write; you engineer lyrics for maximum emotional impact, rhyme precision, and rhythmic flow. "
        "\n\nCORE GUIDELINES:\n"
        "1. STYLE MATCHING: Use `get_similar_lines` to analyze the provided artist's vocabulary, lyrics written style, and thematic patterns. "
        "Do not copy verbatim, but emulate the 'soul' of their writing.\n"
        "2. RHYME & METER: Use `get_rhymes` to find sophisticated end-rhymes. Ensure the meter (syllable count per line) "
        "is consistent and intentional.\n"
        "3. AUTHENTICITY: Avoid clichés. If a line feels generic, use retrieval to find a more unique angle.\n"
        "4. ITERATION: You will be critiqued based on programmatic metrics (Rhyme Density, Syllable Consistency, Originality). "
        "Your goal is to satisfy these constraints before finalizing your draft.\n"
        "5. TOOLS: If you want to use a tool (e.g. get_rhymes), use the tool calling API provided. DO NOT output the tool call JSON in your text response.\n"
        "6. FORMAT: OUTPUT YOUR FINAL LYRICS IN PLAIN TEXT ONLY. DO NOT OUTPUT JSON. DO NOT WRAP IN MARKDOWN CODE BLOCKS. NO FORMATTING OTHER THAN NEWLINES."
    )

    user_prompt = (
        f"Theme: {theme}\nMood: {mood}\n"
        f"Artist Style: {artist_style if artist_style else 'Generic'}\n\n"
        f"Reference lines from the dataset:\n{context_str}\n\n"
        f"Please write a verse with approximately {max_words} words. Output the lyrics in plain text only."
    )

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ]

    # Loop for Draft -> Critique -> Revise
    max_revisions = 3
    current_lyrics = ""
    tool_logs = []

    for i in range(max_revisions):
        print(f"Iteration {i+1}...")
        
        # Generate/Regenerate and resolve tool calls before consuming text output.
        response = resolve_response_with_tools(messages, temperature, tool_logs)
        if not response:
            return "Error: Could not connect to LLM.", []

        current_lyrics = response["content"]
        messages.append({"role": "assistant", "content": current_lyrics})

        # Critique using tools and metrics
        print("Critiquing draft...")
        
        # Don't stream the critique to the UI to keep it clean, or stream it to a different element?
        # Actually, let's keep it clean and only stream the generation.
        # But for now, we won't pass stream_callback to the critique step.

        # Per-line syllable analysis
        syllables_per_line = count_syllables(current_lyrics)
        if isinstance(syllables_per_line, list):
            syllable_report = "\n".join([
                f"Line {i+1}: {s} syllables" 
                for i, s in enumerate(syllables_per_line)
            ])  
        else:
            syllable_report = f"Total: {syllables_per_line}"

        # High-level quality metrics
        rd = rhyme_density(current_lyrics)
        sc = syllable_consistency(current_lyrics)
        os = originality_score(current_lyrics)

        critique_prompt = (
            f"Here is the current draft:\n{current_lyrics}\n\n"
            f"--- METRIC ANALYSIS ---\n"
            f"Rhyme Density: {rd:.2%} (Target: >30%)\n"
            f"Syllable Consistency: {sc:.2%} (Target: >70%)\n"
            f"Originality Score: {os:.2%} (Target: >80%)\n\n"
            f"--- METER ANALYSIS ---\n"
            f"{syllable_report}\n\n"
            "Analyze these metrics. If the rhyme is weak, the meter is inconsistent, or the lyrics "
            "feel too close to the training data, suggest a specific line-by-line revision. "
            "If the draft meets all targets and feels authentic, respond with 'FINAL'."
        )

        critique_messages = [{"role": "user", "content": critique_prompt}]
        critique_res = call_local_llm(critique_messages, tools=None, temperature=temperature)
        if not critique_res or "FINAL" in critique_res["content"].upper():
            print("Draft finalized.")
            break

        print(f"Critique: {critique_res['content']}")
        messages.append({"role": "user", "content": f"Critique: {critique_res['content']}\nPlease revise the lyrics based on this critique. Output ONLY the revised lyrics in plain text. Do NOT use JSON, markdown blocks, or tool calls. Just output the lyrics directly as plain text."})

    # Final Guard: Ensure output is not JSON
    if current_lyrics.strip().startswith("{"):
        repair_messages = messages + [{
            "role": "user",
            "content": "Your last response was JSON. Please provide the FINAL lyrics in plain text only, no JSON."
        }]
        repaired = call_local_llm(repair_messages, tools=None, temperature=temperature)
        if repaired:
            current_lyrics = repaired.get("content", current_lyrics)

    # Final cleanup and streaming step
    if stream_callback:
        print("Streaming final polished lyrics to UI...")
        # We split by spaces but preserve newlines
        # A simpler way is to yield character by character, or chunk by chunk
        full_content = ""
        chunk_size = 3
        for i in range(0, len(current_lyrics), chunk_size):
            chunk = current_lyrics[i:i+chunk_size]
            full_content += chunk
            stream_callback(chunk, full_content)
            time.sleep(0.02)

    return current_lyrics, tool_logs

if __name__ == "__main__":
    # Simple test
    theme = "loneliness in a big city"
    artist = "Drake"
    mood = "melancholic"

    print(f"Generating lyrics for '{theme}' in style of {artist}...")
    lyrics, logs = generate_lyrics_v2(theme, artist, mood)

    print("\n--- FINAL LYRICS ---")
    print(lyrics)
    print("\n--- TOOL LOGS ---")
    for log in logs:
        print(log)

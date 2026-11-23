"""The HTTP generation endpoints (OpenAI, Anthropic, Ollama) on real tiny models with real tokenizers: chat templates,
incremental detokenization, stop strings, finish reasons, usage, reasoning budgets, tool calls, JSON-schema
constrained decoding, lanes, the removed fake knobs, API keys and the render endpoints. The models are tiny
randomly-initialised HF Qwen3s converted to Turing whose vocabulary is the tokenizer's, so every id decodes."""
import asyncio
import json

import pytest
import torch
import transformers as tf
from fastapi.testclient import TestClient

from turing.config import TuringConfig
from turing.models.hf_loader import RealHuggingFaceLoader
from turing.models.registry import get_model_config
from turing.serving import generation
from turing.serving.engine import ContinuousBatchEngine
from turing.serving.server import create_app
from turing.serving.tools import ToolCallingHandler

pytest.importorskip("llguidance")
pytest.importorskip("jsonschema")


def tokenizer(name):
    try:
        return tf.AutoTokenizer.from_pretrained(name, local_files_only=True)
    except Exception as e:  # noqa: BLE001
        pytest.skip(f"tokenizer {name} is not in the local Hugging Face cache: {e}")


def tiny_chat_model(tok, seed=0):
    """A 2-layer Qwen3 with random weights and the tokenizer's vocabulary, converted to SubspaceCausalLM."""
    torch.manual_seed(seed)
    hf = tf.Qwen3ForCausalLM(tf.Qwen3Config(
        vocab_size=len(tok), hidden_size=64, intermediate_size=128, num_hidden_layers=2, num_attention_heads=4,
        num_key_value_heads=2, head_dim=32, max_position_embeddings=4096, tie_word_embeddings=False)).eval()
    model = RealHuggingFaceLoader.convert_hf_model(hf, name="tiny-qwen3")
    model.config.kv_pool_gib = 0.02
    return model


def tiny_engine(tok_name="Qwen/Qwen3-0.6B", **kw):
    tok = tokenizer(tok_name)
    model = tiny_chat_model(tok)
    return ContinuousBatchEngine(model.config, TuringConfig(device="cpu", max_batch_size=8), model=model,
                                 tokenizer=tok, **kw)


@pytest.fixture(scope="module")
def qwen():
    engine = tiny_engine()
    with TestClient(create_app(engine)) as client:
        yield engine, client


@pytest.fixture(scope="module")
def deepseek():
    """A DeepSeek-R1-style template: the prompt ends inside <think>, so every answer starts with reasoning."""
    engine = tiny_engine("deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B")
    with TestClient(create_app(engine)) as client:
        yield engine, client


WEATHER = [{"type": "function", "function": {
    "name": "get_weather", "description": "Weather in a city",
    "parameters": {"type": "object", "properties": {"city": {"type": "string", "maxLength": 12},
                                                   "unit": {"enum": ["c", "f"]}},
                   "required": ["city", "unit"], "additionalProperties": False}}}]
TIME = [{"type": "function", "function": {
    "name": "get_time", "description": "Time in a zone",
    "parameters": {"type": "object", "properties": {"zone": {"enum": ["utc", "pst"]}}, "required": ["zone"],
                   "additionalProperties": False}}}]


def _sse(text):
    return [json.loads(line[6:]) for line in text.splitlines() if line.startswith("data: ") and line != "data: [DONE]"]


def _events(text):
    return [json.loads(line[6:]) for line in text.splitlines() if line.startswith("data: ")]


# ------------------------------------------------------------------------------------------- 1. prompts

def test_chat_prompt_is_the_tokenizers_chat_template(qwen):
    engine, client = qwen
    tok = engine.tokenizer
    messages = [
        {"role": "developer", "content": "Be brief."},
        {"role": "user", "content": [{"type": "text", "text": "Weather"}, {"type": "text", "text": "in Paris?"}]},
        {"role": "assistant", "content": None, "tool_calls": [{"id": "call_1", "type": "function", "function": {
            "name": "get_weather", "arguments": json.dumps({"city": "Paris", "unit": "c"})}}]},
        {"role": "tool", "tool_call_id": "call_1", "content": "sunny"},
    ]
    r = client.post("/v1/chat/completions/render", json={"model": "m", "messages": messages, "tools": WEATHER})
    assert r.status_code == 200, r.text
    expected = tok.apply_chat_template(
        [{"role": "system", "content": "Be brief."}, {"role": "user", "content": "Weather\nin Paris?"},
         {"role": "assistant", "content": "", "tool_calls": [{"id": "call_1", "type": "function", "function": {
             "name": "get_weather", "arguments": {"city": "Paris", "unit": "c"}}}]},
         {"role": "tool", "tool_call_id": "call_1", "content": "sunny"}],
        tools=WEATHER, add_generation_prompt=True, tokenize=False)
    assert r.json()["tokens"] == tok.encode(expected, add_special_tokens=False)
    text = tok.decode(r.json()["tokens"])
    assert "<tools>" in text and '"get_weather"' in text and "<tool_response>\nsunny" in text
    assert text.endswith("<|im_start|>assistant\n")

    # the generation endpoint feeds exactly those ids
    r2 = client.post("/v1/chat/completions", json={"model": "m", "messages": messages, "tools": WEATHER,
                                                   "max_tokens": 2, "temperature": 0})
    assert r2.status_code == 200 and r2.json()["usage"]["prompt_tokens"] == r.json()["count"]

    for bad in ([{"type": "image_url", "image_url": {"url": "data:,"}}], [{"type": "input_audio"}]):
        r = client.post("/v1/chat/completions", json={"model": "m", "messages": [{"role": "user", "content": bad}]})
        assert r.status_code == 400 and "not supported" in r.json()["error"]["message"]


def test_anthropic_and_ollama_prompts(qwen):
    engine, client = qwen
    tok = engine.tokenizer
    body = {"model": "m", "max_tokens": 2, "temperature": 0, "system": [{"type": "text", "text": "Be brief."}],
            "messages": [
                {"role": "user", "content": "Weather in Paris?"},
                {"role": "assistant", "content": [{"type": "text", "text": "Checking."}, {
                    "type": "tool_use", "id": "toolu_1", "name": "get_weather",
                    "input": {"city": "Paris", "unit": "c"}}]},
                {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "toolu_1",
                                              "content": [{"type": "text", "text": "sunny"}]}]}]}
    r = client.post("/v1/messages", json=body)
    assert r.status_code == 200, r.text
    expected = tok.apply_chat_template(
        [{"role": "system", "content": "Be brief."}, {"role": "user", "content": "Weather in Paris?"},
         {"role": "assistant", "content": "Checking.", "tool_calls": [{"id": "toolu_1", "type": "function",
                                                                        "function": {"name": "get_weather",
                                                                                     "arguments": {"city": "Paris", "unit": "c"}}}]},
         {"role": "tool", "tool_call_id": "toolu_1", "content": "sunny"}],
        add_generation_prompt=True, tokenize=False)
    assert r.json()["usage"]["input_tokens"] == len(tok.encode(expected, add_special_tokens=False))
    img = {**body, "messages": [{"role": "user", "content": [{"type": "image", "source": {}}]}]}
    r = client.post("/v1/messages", json=img)
    assert r.status_code == 400 and r.json()["error"]["type"] == "invalid_request_error"

    # Ollama /api/chat: the chat template; /api/generate: template, raw, system as Ollama defines them
    r = client.post("/api/chat", json={"model": "m", "stream": False, "options": {"num_predict": 1},
                                       "messages": [{"role": "system", "content": "S"},
                                                    {"role": "user", "content": "hi"}]})
    chat_ids = tok.encode(tok.apply_chat_template([{"role": "system", "content": "S"}, {"role": "user", "content": "hi"}],
                                                  add_generation_prompt=True, tokenize=False), add_special_tokens=False)
    assert r.json()["prompt_eval_count"] == len(chat_ids)
    r = client.post("/api/generate", json={"model": "m", "prompt": "hi", "system": "S", "stream": False,
                                           "options": {"num_predict": 1}})
    assert r.json()["context"][:len(chat_ids)] == chat_ids
    r = client.post("/api/generate", json={"model": "m", "prompt": "hi there", "raw": True, "stream": False,
                                           "options": {"num_predict": 1}})
    assert r.json()["context"][:-1] == tok.encode("hi there")
    r = client.post("/api/generate", json={"model": "m", "prompt": "hi", "system": "S", "stream": False,
                                           "template": "{{ if .System }}<<{{ .System }}>> {{ end }}Q: {{ .Prompt }} A:",
                                           "options": {"num_predict": 1}})
    assert r.json()["context"][:-1] == tok.encode("<<S>> Q: hi A:")
    r = client.post("/api/chat", json={"model": "m", "messages": [{"role": "user", "content": "x", "images": ["AA=="]}]})
    assert r.status_code == 400 and "images" in r.json()["error"]


def test_synthetic_model_without_tokenizer_keeps_a_plain_prompt():
    engine = ContinuousBatchEngine(get_model_config("test-tiny"), TuringConfig(device="cpu"))
    with TestClient(create_app(engine)) as client:
        r = client.post("/v1/chat/completions/render", json={"messages": [{"role": "system", "content": "S"},
                                                                          {"role": "user", "content": "hi"}]})
        assert r.json()["tokens"] == engine.encode_prompt("System: S\n\nUser: hi\n\nAssistant:")
        r = client.post("/v1/completions/render", json={"prompt": [1, 2, 3]})
        assert r.json()["tokens"] == [1, 2, 3]


# ------------------------------------------------------------------------------------------- 2. the shared core

def test_incremental_detokenization_never_splits_characters(qwen):
    engine, _ = qwen
    text = "héllo wörld 世界 🎉🎉 ok"
    ids = engine.tokenizer.encode(text, add_special_tokens=False)
    d = generation.Detokenizer(engine)
    pieces = [d.push(t) for t in ids] + [d.flush()]
    assert "".join(pieces) == text and not any("�" in p for p in pieces)


def test_tags_and_stop_strings_split_across_chunks():
    from turing.serving.reasoning import ReasoningStreamFilter
    f = ReasoningStreamFilter()
    out = [f.process_token(x) for x in ["Hi <th", "ink>abc</thi", "nk>\n\nDone"]] + [f.flush()]
    assert out == [(None, "Hi "), ("abc", None), (None, "Done"), (None, None)]
    s = generation.StopStrings(["END"])
    assert [s.push(x) for x in ["abcE", "N", "D more"]] == [("abc", None), ("", None), ("", "END")]
    g = generation.ToolGate("hermes")
    assert [g.push(x) for x in ["hi <to", "ol_call>{x}"]] == ["hi ", ""] and g.buf == "<tool_call>{x}"


def test_stop_strings_finish_reasons_and_cached_usage(qwen):
    engine, client = qwen
    msgs = [{"role": "user", "content": "Tell me a long story about the sea and the sky, please, in detail."}]
    base = {"model": "m", "messages": msgs, "temperature": 0, "max_tokens": 16}
    first = client.post("/v1/chat/completions", json=base).json()
    full = first["choices"][0]["message"]["content"]
    assert first["choices"][0]["finish_reason"] == "length" and first["usage"]["completion_tokens"] == 16
    # the same prompt again reuses the cached prompt pages
    again = client.post("/v1/chat/completions", json=base).json()
    assert again["choices"][0]["message"]["content"] == full
    assert again["usage"]["prompt_tokens_details"]["cached_tokens"] >= 16

    stop = full[len(full) // 2:len(full) // 2 + 3]
    cut = full[:full.find(stop)]
    r = client.post("/v1/chat/completions", json={**base, "stop": [stop]}).json()
    assert r["choices"][0]["message"]["content"] == cut and r["choices"][0]["finish_reason"] == "stop"
    assert r["usage"]["completion_tokens"] < 16

    s = client.post("/v1/chat/completions", json={**base, "stop": stop, "stream": True,
                                                  "stream_options": {"include_usage": True}})
    chunks = _sse(s.text)
    assert "".join(c["choices"][0]["delta"].get("content") or "" for c in chunks if c["choices"]) == cut
    assert [c["choices"][0]["finish_reason"] for c in chunks if c["choices"] and c["choices"][0]["finish_reason"]] == ["stop"]
    assert chunks[-1]["usage"]["completion_tokens"] == r["usage"]["completion_tokens"]

    a = client.post("/v1/messages", json={"model": "m", "messages": msgs, "temperature": 0, "max_tokens": 16,
                                          "stop_sequences": [stop]}).json()
    assert a["stop_reason"] == "stop_sequence" and a["stop_sequence"] == stop and a["content"][0]["text"] == cut
    a = client.post("/v1/messages", json={"model": "m", "messages": msgs, "temperature": 0, "max_tokens": 16}).json()
    assert a["stop_reason"] == "max_tokens" and a["usage"]["output_tokens"] == 16
    o = client.post("/api/chat", json={"model": "m", "messages": msgs, "stream": False,
                                       "options": {"temperature": 0, "num_predict": 16, "stop": [stop]}}).json()
    assert o["message"]["content"] == cut and o["done_reason"] == "stop"
    c = client.post("/v1/completions", json={"model": "m", "prompt": "Once upon a time", "max_tokens": 5,
                                             "temperature": 0}).json()
    assert c["choices"][0]["finish_reason"] == "length" and c["usage"]["completion_tokens"] == 5


# ------------------------------------------------------------------------------------------- 3. reasoning

def test_thinking_budget_forces_the_close_tag():
    engine = tiny_engine()
    tok = engine.tokenizer
    open_id, close_id = tok.convert_tokens_to_ids("<think>"), tok.convert_tokens_to_ids("</think>")

    def think_first(gen, logits):  # make the random model open a thinking block, never close it or end itself
        if not gen:
            logits.fill_(float("-inf"))
            logits[open_id] = 0.0
        else:
            logits[close_id] = float("-inf")
            for e in engine.eos_token_ids:
                logits[e] = float("-inf")

    async def go():
        await engine.start()
        try:
            ids, text = generation.render_chat(engine, [{"role": "user", "content": "2+2?"}])
            g = await generation.start(engine, ids, generation.GenerationParams(
                max_tokens=12, temperature=1.0, thinking_budget=5, logits_processor=think_first), prompt_text=text)
            return await g.collect()
        finally:
            await engine.stop()

    g = asyncio.run(go())
    assert g.output_ids[0] == open_id and g.output_ids[6] == close_id  # 5 thinking tokens, then </think> forced
    assert close_id not in g.output_ids[1:6]
    assert g.reasoning_text == tok.decode(g.output_ids[1:6])
    assert g.content == tok.decode(g.output_ids[7:]).lstrip("\n")


def test_reasoning_split_and_budget_over_http(deepseek):
    engine, client = deepseek
    close_id = engine.tokenizer.convert_tokens_to_ids("</think>")
    msgs = [{"role": "user", "content": "2+2?"}]
    a = client.post("/v1/messages", json={"model": "m", "messages": msgs, "max_tokens": 12, "temperature": 0,
                                          "thinking": {"type": "enabled", "budget_tokens": 4}}).json()
    assert a["content"][0]["type"] == "thinking" and a["content"][0]["thinking"]
    s = client.post("/v1/messages", json={"model": "m", "messages": msgs, "max_tokens": 12, "temperature": 0,
                                          "stream": True, "thinking": {"type": "enabled", "budget_tokens": 4}})
    deltas = [e["delta"] for e in _events(s.text) if e["type"] == "content_block_delta"]
    assert "".join(d["thinking"] for d in deltas if d["type"] == "thinking_delta") == a["content"][0]["thinking"]
    starts = [e["content_block"]["type"] for e in _events(s.text) if e["type"] == "content_block_start"]
    assert starts[0] == "thinking"

    # the prompt ends inside <think>: with budget 4, </think> is forced as the 5th token, so the answer has text
    assert a["content"][-1]["type"] == "text" and a["stop_reason"] == "max_tokens"
    o = client.post("/v1/chat/completions", json={"model": "m", "messages": msgs, "max_tokens": 12, "temperature": 0,
                                                  "reasoning_effort": "low"}).json()
    msg = o["choices"][0]["message"]
    assert msg["reasoning_content"] and "</think>" not in msg["reasoning_content"] and "<think>" not in (msg["content"] or "")
    s = client.post("/v1/chat/completions", json={"model": "m", "messages": msgs, "max_tokens": 12, "temperature": 0,
                                                  "reasoning_effort": "low", "stream": True})
    assert "".join(c["choices"][0]["delta"].get("reasoning_content") or "" for c in _sse(s.text)) == \
        msg["reasoning_content"]
    assert close_id not in engine.encode_prompt(msg["content"] or "")


# ------------------------------------------------------------------------------------------- 4. tools

def test_tool_call_parsers():
    for text, args_key in [('<tool_call>\n{"name": "f", "arguments": {"a": 1}}\n</tool_call>', "arguments"),
                           ('[TOOL_CALLS][{"name": "f", "arguments": {"a": 1}}]', "arguments"),
                           ('{"name": "f", "parameters": {"a": 1}}', "parameters")]:
        clean, calls = ToolCallingHandler.extract_tool_calls(text)
        assert clean == "" and len(calls) == 1, text
        assert calls[0]["function"] == {"name": "f", "arguments": '{"a": 1}'} and calls[0]["id"].startswith("call_")
    assert ToolCallingHandler.extract_tool_calls("just text {not json")[1] == []


def test_tool_choice_required_and_named_are_enforced(qwen):
    engine, client = qwen
    msgs = [{"role": "user", "content": "Weather in Paris?"}]
    for seed in range(3):
        torch.manual_seed(seed)
        r = client.post("/v1/chat/completions", json={"model": "m", "messages": msgs, "tools": WEATHER + TIME,
                                                      "tool_choice": "required", "max_tokens": 64,
                                                      "temperature": 1.0}).json()
        choice = r["choices"][0]
        assert choice["finish_reason"] == "tool_calls", r
        (call,) = choice["message"]["tool_calls"]
        assert call["function"]["name"] in ("get_weather", "get_time") and call["id"].startswith("call_")
        assert isinstance(json.loads(call["function"]["arguments"]), dict)

    torch.manual_seed(7)
    s = client.post("/v1/chat/completions", json={
        "model": "m", "messages": msgs, "tools": WEATHER + TIME, "max_tokens": 64, "temperature": 1.0,
        "tool_choice": {"type": "function", "function": {"name": "get_time"}}, "stream": True})
    chunks = _sse(s.text)
    calls = [tc for c in chunks for tc in c["choices"][0]["delta"].get("tool_calls") or []]
    assert len(calls) == 1 and calls[0]["function"]["name"] == "get_time" and calls[0]["index"] == 0
    assert json.loads(calls[0]["function"]["arguments"])["zone"] in ("utc", "pst")
    assert not any(c["choices"][0]["delta"].get("content") for c in chunks)
    assert chunks[-1]["choices"][0]["finish_reason"] == "tool_calls"

    torch.manual_seed(3)
    a = client.post("/v1/messages", json={
        "model": "m", "messages": msgs, "max_tokens": 64, "temperature": 1.0,
        "tools": [{"name": "get_weather", "description": "w",
                   "input_schema": WEATHER[0]["function"]["parameters"]}],
        "tool_choice": {"type": "tool", "name": "get_weather"}}).json()
    assert a["stop_reason"] == "tool_use"
    (use,) = [b for b in a["content"] if b["type"] == "tool_use"]
    assert use["name"] == "get_weather" and use["id"].startswith("toolu_") and use["input"]["unit"] in ("c", "f")

    r = client.post("/v1/chat/completions/render", json={"messages": msgs, "tools": WEATHER, "tool_choice": "none"})
    assert "get_weather" not in engine.tokenizer.decode(r.json()["tokens"])
    r = client.post("/v1/chat/completions", json={"model": "m", "messages": msgs, "tools": WEATHER,
                                                  "tool_choice": {"type": "function", "function": {"name": "nope"}}})
    assert r.status_code == 400 and "nope" in r.json()["error"]["message"]


# ------------------------------------------------------------------------------------------- 5. JSON schema

SCHEMA = {"type": "object", "properties": {
    "name": {"type": "string", "maxLength": 10},
    "age": {"type": "integer", "minimum": 0, "maximum": 120},
    "admin": {"type": "boolean"},
    "tags": {"type": "array", "items": {"enum": ["a", "b", "c"]}, "maxItems": 3}},
    "required": ["name", "age", "admin", "tags"], "additionalProperties": False}


def test_json_schema_output_is_always_valid_on_random_weights(qwen):
    import jsonschema
    _, client = qwen
    fmt = {"type": "json_schema", "json_schema": {"name": "person", "schema": SCHEMA}}
    for seed in range(20):
        torch.manual_seed(seed)
        r = client.post("/v1/chat/completions", json={"model": "m", "max_tokens": 96, "temperature": 1.0,
                                                      "response_format": fmt,
                                                      "messages": [{"role": "user", "content": "A person"}]}).json()
        choice = r["choices"][0]
        assert "schema_error" not in choice and choice["finish_reason"] == "stop", (seed, r)
        jsonschema.validate(json.loads(choice["message"]["content"]), SCHEMA)


def test_json_modes_errors_and_ollama_format(qwen):
    _, client = qwen
    torch.manual_seed(0)
    r = client.post("/api/generate", json={"model": "m", "prompt": "a person", "stream": False, "format": SCHEMA,
                                           "options": {"num_predict": 96}}).json()
    assert "schema_error" not in r and json.loads(r["response"]).keys() == SCHEMA["properties"].keys()
    # a schema llguidance cannot enforce: 400 with its message
    bad = {"type": "object", "properties": {"a": {"type": "string", "pattern": "(?=x)"}}}
    r = client.post("/v1/chat/completions", json={"model": "m", "messages": [{"role": "user", "content": "x"}],
                                                  "response_format": {"type": "json_schema",
                                                                      "json_schema": {"name": "b", "schema": bad}}})
    assert r.status_code == 400 and "look-around" in r.json()["error"]["message"]
    # cut short by max_tokens: the output is reported as not valid
    r = client.post("/v1/chat/completions", json={"model": "m", "messages": [{"role": "user", "content": "x"}],
                                                  "max_tokens": 3, "response_format": {"type": "json_object"}}).json()
    assert r["choices"][0]["finish_reason"] == "length" and "not valid JSON" in r["choices"][0]["schema_error"]


def test_structured_output_without_tokenizer_or_llguidance_is_a_400():
    engine = ContinuousBatchEngine(get_model_config("test-tiny"), TuringConfig(device="cpu"))
    with TestClient(create_app(engine)) as client:
        r = client.post("/v1/chat/completions", json={"model": "m", "messages": [{"role": "user", "content": "x"}],
                                                      "response_format": {"type": "json_object"}})
        assert r.status_code == 400 and "tokenizer" in r.json()["error"]["message"]


# ------------------------------------------------------------------------------------------- 6. lanes

def test_lanes_order_waiting_requests_and_show_in_metrics():
    import httpx
    engine = ContinuousBatchEngine(get_model_config("test-tiny"), TuringConfig(device="cpu", max_batch_size=1))
    app = create_app(engine)
    assert engine.admission is not None and engine.lane_policy is not None

    async def go():
        await engine.start()
        done = []
        try:
            async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://t") as c:
                async def post(name, lane, header=False):
                    body = {"model": "m", "prompt": "hi", "max_tokens": 12, "temperature": 0}
                    kw = {"headers": {"X-Turing-Lane": lane}} if header else {}
                    if not header:
                        body["lane"] = lane
                    r = await c.post("/v1/completions", json=body, **kw)
                    assert r.status_code == 200 and r.headers["x-turing-lane"] == lane
                    done.append(name)

                first = asyncio.create_task(post("running", "background"))
                while not engine.running_batch:
                    await asyncio.sleep(0.001)
                tasks = [asyncio.create_task(post(f"bg{i}", "background")) for i in range(3)]
                while len(engine.waiting_queue) < 3:
                    await asyncio.sleep(0.001)
                tasks.append(asyncio.create_task(post("interactive", "interactive", header=True)))
                await asyncio.gather(first, *tasks)
                m = (await c.get("/metrics?format=json")).json()["telemetry"]["lanes"]
                bad = await c.post("/v1/completions", json={"model": "m", "prompt": "hi", "lane": "vip"})
        finally:
            await engine.stop()
        return done, m, bad

    done, lanes, bad = asyncio.run(go())
    assert done[0] == "running" and done[1] == "interactive"  # queued last, scheduled first
    assert lanes["background"]["admitted"] == lanes["background"]["finished"] == 4
    assert lanes["interactive"]["admitted"] == 1 and lanes["batch"]["admitted"] == 0
    assert bad.status_code == 400


# ------------------------------------------------------------------------------------------- 7, 8. knobs and keys

def test_removed_knobs_are_400_and_headers_report_real_values(qwen):
    _, client = qwen
    body = {"model": "m", "messages": [{"role": "user", "content": "hi"}], "max_tokens": 2}
    for extra, header, needle in [({"use_svd_kv": True}, {}, "--kv-cache-dtype"),
                                  ({"sparsity_ratio": 0.5}, {}, "--sparsity"),
                                  ({}, {"X-Turing-SVD-KV": "1"}, "--kv-cache-dtype"),
                                  ({}, {"X-Turing-SVD-Rank": "8"}, "--kv-cache-dtype"),
                                  ({}, {"X-Turing-Sparsity": "0.5"}, "--sparsity")]:
        r = client.post("/v1/chat/completions", json={**body, **extra}, headers=header)
        assert r.status_code == 400 and needle in r.json()["error"]["message"], (extra, header)
    r = client.post("/v1/messages", json={**body, "sparsity_ratio": 0.5})
    assert r.status_code == 400 and r.json()["error"]["type"] == "invalid_request_error"
    r = client.post("/v1/chat/completions", json=body, headers={"X-Turing-Draft-Tokens": "3"})
    assert r.status_code == 200
    assert r.headers["x-turing-kv-cache-dtype"] == "native" and r.headers["x-turing-draft-tokens"] == "3"
    assert int(r.headers["x-turing-cached-tokens"]) == r.json()["usage"]["prompt_tokens_details"]["cached_tokens"]
    assert "x-turing-sparsity" not in r.headers and "x-turing-svd-kv" not in r.headers
    r = client.post("/v1/chat/completions", json={**body, "response_format": {"type": "json_object"}},
                    headers={"X-Turing-Draft-Tokens": "3"})
    assert r.headers["x-turing-draft-tokens"] == "0"  # constrained requests never speculate
    r = client.post("/v1/chat/completions", json=body, headers={"X-Turing-Adapter": "nope"})
    assert r.status_code == 400 and "nope" in r.json()["error"]["message"]
    assert client.post("/v1/chat/completions", json={**body, "model": "tiny:latest"}).status_code == 200


def test_api_key():
    engine = ContinuousBatchEngine(get_model_config("test-tiny"), TuringConfig(device="cpu"))
    with TestClient(create_app(engine, api_key="sk-test")) as client:
        assert client.get("/health").status_code == 200
        body = {"model": "m", "messages": [{"role": "user", "content": "hi"}], "max_tokens": 1}
        r = client.post("/v1/chat/completions", json=body)
        assert r.status_code == 401 and r.json()["error"]["code"] == "invalid_api_key"
        r = client.post("/v1/messages", json=body, headers={"x-api-key": "wrong"})
        assert r.status_code == 401 and r.json()["error"]["type"] == "authentication_error"
        assert client.get("/api/tags").json() == {"error": "invalid or missing API key"}
        assert client.get("/v1/models").status_code == 401
        assert client.post("/v1/chat/completions", json=body,
                           headers={"Authorization": "Bearer sk-test"}).status_code == 200
        assert client.post("/v1/messages", json=body, headers={"x-api-key": "sk-test"}).status_code == 200
        assert client.get("/api/tags", headers={"x-api-key": "sk-test"}).status_code == 200


def test_think_token_ids_reads_the_vocabulary_once_per_tokenizer():
    """Every chat request asked the tokenizer for its whole vocabulary twice (a new 151,936-entry dict each time for
    Qwen2.5: 205 ms a request on an L4 VM, during which the engine's event loop ran nothing else)."""
    from turing.serving.reasoning import think_token_ids

    class Tok:
        calls = 0

        def __init__(self, vocab):
            self.vocab = vocab

        def get_vocab(self):
            Tok.calls += 1
            return dict(self.vocab)

    thinker, plain = Tok({"<think>": 7, "</think>": 9, "a": 1}), Tok({"a": 1})
    for _ in range(3):
        assert think_token_ids(thinker) == ([7], [9])
        assert think_token_ids(plain) is None
    assert Tok.calls == 2
    got = think_token_ids(thinker)
    got[0].append(0)  # a caller's list is its own
    assert think_token_ids(thinker) == ([7], [9])
    assert think_token_ids(None) is None

"""
The standalone C++ server (turing-cli serve), driven over real HTTP: OpenAI chat `messages` with the GGUF chat
template, --host, max_tokens / temperature from the body, JSON-escaped output, and 400 / 404 / 405 status codes.
"""

import http.client
import json
import socket
import struct
import subprocess
import time

import numpy as np
import pytest

from turing.models.gguf_loader import DEFAULT_ALIGNMENT, GGMLType, create_test_gguf_file

CHATML = "{% for m in messages %}<|im_start|>{{ m.role }}\n{{ m.content }}<|im_end|>\n{% endfor %}"
# Same length as "<tok_" so the vocabulary can be rewritten in place: quote, backslash, newline, tab, invalid UTF-8.
ODD_PREFIX = b'"\\\n\t\xff'


def _tiny_gguf(path, chat_template=None, odd_tokens=False, eos=None):
    np.random.seed(99)
    create_test_gguf_file(str(path), hidden_dim=32, ffn_dim=64, num_layers=2, num_heads=2, num_kv_heads=2,
                          vocab_size=64, max_position_embeddings=256, quant_type=GGMLType.F16)
    data = bytearray(path.read_bytes())
    if odd_tokens:
        data = bytearray(data.replace(b"<tok_", ODD_PREFIX))
    entries = b""
    if eos is not None:
        key = b"tokenizer.ggml.eos_token_id"
        entries += struct.pack("<Q", len(key)) + key + struct.pack("<II", 4, eos)  # UINT32
    if chat_template is not None or entries:
        # Insert KVs right after the 24-byte header; the template string is padded so the insertion is a multiple
        # of the alignment and the data section stays aligned without touching the tensor table.
        key = b"tokenizer.chat_template"
        value = (chat_template or "").encode()
        value += b" " * (-(len(entries) + 8 + len(key) + 4 + 8 + len(value)) % DEFAULT_ALIGNMENT)
        entries += struct.pack("<Q", len(key)) + key + struct.pack("<I", 8) + struct.pack("<Q", len(value)) + value
        magic, version, n_tensors, n_kv = struct.unpack_from("<IIQQ", data, 0)
        n_new = 2 if eos is not None else 1
        data = bytearray(struct.pack("<IIQQ", magic, version, n_tensors, n_kv + n_new)) + entries + data[24:]
    path.write_bytes(bytes(data))
    return str(path)


def _free_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _request(port, method, path, body=None):
    conn = http.client.HTTPConnection("127.0.0.1", port, timeout=30)
    payload = body if isinstance(body, (bytes, str)) or body is None else json.dumps(body)
    conn.request(method, path, body=payload, headers={"Content-Type": "application/json"})
    resp = conn.getresponse()
    raw = resp.read()
    conn.close()
    return resp.status, json.loads(raw.decode("utf-8"))  # must always be valid UTF-8 JSON


@pytest.fixture(scope="module")
def servers(turing_cli_binary, tmp_path_factory):
    d = tmp_path_factory.mktemp("cpp_server")
    procs, ports = [], {}
    for name, kwargs in {"plain": {"odd_tokens": True}, "chatml": {"chat_template": CHATML}}.items():
        gguf = _tiny_gguf(d / f"{name}.gguf", **kwargs)
        port = _free_port()
        proc = subprocess.Popen([turing_cli_binary, "serve", "--model", gguf, "--host", "127.0.0.1", "--port", str(port),
                                 "--max-tokens", "6", "--temp", "0"], stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        procs.append(proc)
        deadline = time.time() + 20
        while True:
            try:
                if _request(port, "GET", "/health")[0] == 200:
                    break
            except OSError:
                pass
            if proc.poll() is not None or time.time() > deadline:
                out = proc.stdout.read().decode(errors="replace") if proc.poll() is not None else ""
                for p in procs:
                    p.kill()
                pytest.fail(f"server did not start: {out}")
            time.sleep(0.05)
        ports[name] = port
    yield ports
    for p in procs:
        p.terminate()
        p.wait(timeout=10)
        p.stdout.close()


def _complete(port, prompt, **body):
    status, out = _request(port, "POST", "/v1/completions", {"prompt": prompt, "temperature": 0, **body})
    assert status == 200, out
    return out


def test_chat_applies_gguf_chat_template(servers):
    port = servers["chatml"]
    messages = [{"role": "system", "content": "be brief"}, {"role": "user", "content": "hi there"}]
    status, chat = _request(port, "POST", "/v1/chat/completions", {"messages": messages, "temperature": 0, "max_tokens": 5})
    assert status == 200, chat
    assert chat["object"] == "chat.completion" and chat["choices"][0]["message"]["role"] == "assistant"
    templated = "<|im_start|>system\nbe brief<|im_end|>\n<|im_start|>user\nhi there<|im_end|>\n<|im_start|>assistant\n"
    plain = _complete(port, templated, max_tokens=5)
    assert chat["choices"][0]["message"]["content"] == plain["choices"][0]["text"]
    assert chat["usage"] == plain["usage"]


def test_chat_without_template_uses_role_content_lines(servers):
    port = servers["plain"]
    status, chat = _request(port, "POST", "/v1/chat/completions",
                            {"messages": [{"role": "user", "content": [{"type": "text", "text": "hello"}]}], "temperature": 0})
    assert status == 200, chat
    plain = _complete(port, "user: hello\nassistant:")
    assert chat["choices"][0]["message"]["content"] == plain["choices"][0]["text"]
    assert chat["usage"]["completion_tokens"] == 6  # the server's --max-tokens default


def test_generated_text_is_json_escaped(servers):
    out = _complete(servers["plain"], "abc", max_tokens=4)
    text = out["choices"][0]["text"]
    # every token of this vocabulary carries a quote, a backslash, a newline, a tab and an invalid UTF-8 byte
    assert text.count('"') == text.count("\\") == text.count("\t") == text.count("�") == 4
    assert text.count("\n") == 4


def test_max_tokens_and_temperature_from_body(servers):
    port = servers["chatml"]
    for n in (0, 1, 9):
        out = _complete(port, "abc", max_tokens=n)
        assert out["usage"]["completion_tokens"] == n  # this GGUF has no EOS token, so only length stops it
        assert out["choices"][0]["finish_reason"] == "length"
    greedy = {_complete(port, "abc", max_tokens=12)["choices"][0]["text"] for _ in range(2)}
    assert len(greedy) == 1
    hot = {_request(port, "POST", "/v1/completions", {"prompt": "abc", "max_tokens": 12, "temperature": 100.0})[1]
           ["choices"][0]["text"] for _ in range(3)}
    assert hot - greedy and len(hot) > 1


@pytest.mark.parametrize("method,path,body,status", [
    ("POST", "/v1/chat/completions", "{not json", 400),
    ("POST", "/v1/chat/completions", "[1, 2]", 400),
    ("POST", "/v1/chat/completions", {"messages": []}, 400),
    ("POST", "/v1/chat/completions", {"messages": [{"content": "no role"}]}, 400),
    ("POST", "/v1/chat/completions", {"messages": [{"role": "user", "content": "x"}], "max_tokens": -1}, 400),
    ("POST", "/v1/chat/completions", {"messages": [{"role": "user", "content": "x"}], "temperature": "hot"}, 400),
    ("POST", "/v1/completions", {"prompt": 5}, 400),
    ("POST", "/v1/completions", {"prompt": "x" * 300}, 400),  # past the 256-token context
    ("GET", "/v1/chat/completions", None, 405),
    ("POST", '/v2/"quoted\\path', {}, 404),
    ("GET", "/nope", None, 404),
])
def test_error_status_codes(servers, method, path, body, status):
    got, out = _request(servers["plain"], method, path, body)
    assert got == status
    assert out["error"]["code"] == status and out["error"]["message"]


def test_host_flag_binds_requested_address(turing_cli_binary, tmp_path):
    gguf = _tiny_gguf(tmp_path / "m.gguf")
    port = _free_port()
    proc = subprocess.run([turing_cli_binary, "serve", "--model", gguf, "--host", "203.0.113.7", "--port", str(port)],
                          capture_output=True, text=True, timeout=30)
    assert proc.returncode != 0  # an address this machine does not own cannot be bound
    assert "Bind failed on 203.0.113.7" in proc.stderr


def _generated(binary, gguf, *extra):
    res = subprocess.run([binary, "generate", "--model", gguf, "--prompt", "abc", "--temp", "0", *extra],
                         capture_output=True, text=True, timeout=60)
    assert res.returncode == 0, res.stderr
    return int(res.stderr.split("Generated ")[1].split(" tokens")[0]), res


def test_generation_stops_at_the_gguf_eos_token(turing_cli_binary, tmp_path):
    plain = _tiny_gguf(tmp_path / "plain.gguf")
    out = subprocess.run([turing_cli_binary, "logits", "--model", plain, "--tokens", "33,34,35"],  # "abc": byte % vocab
                         capture_output=True, text=True, check=True)
    first = int(np.argmax(json.loads(out.stdout)["logits"][-1]))  # greedy's first token after "abc"
    assert _generated(turing_cli_binary, plain, "--max-tokens", "8")[0] == 8  # no EOS in metadata: length stops it
    with_eos = _tiny_gguf(tmp_path / "eos.gguf", eos=first)
    assert _generated(turing_cli_binary, with_eos, "--max-tokens", "8")[0] == 1


def test_ctx_flag_caps_the_context(turing_cli_binary, tmp_path):
    gguf = _tiny_gguf(tmp_path / "m.gguf")
    n, _ = _generated(turing_cli_binary, gguf, "--ctx", "8", "--max-tokens", "20")
    assert n == 6  # 3 prompt tokens + 5 new fill the 8 positions (the last sampled token needs no position)
    res = subprocess.run([turing_cli_binary, "generate", "--model", gguf, "--prompt", "abcdefghij", "--ctx", "8"],
                         capture_output=True, text=True, timeout=60)
    assert res.returncode != 0 and "exceeds the model's context of 8 tokens" in res.stderr

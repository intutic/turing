#pragma once

#include <string>
#include <sstream>
#include <iostream>
#include <functional>
#include <stdexcept>
#include <vector>
#include <utility>
#include <cstring>
#include <cstdlib>
#include <cstdio>
#include <ctime>
#include <cctype>
#include <algorithm>

#if defined(_WIN32)
#include <winsock2.h>
#include <ws2tcpip.h>
#pragma comment(lib, "ws2_32.lib")
#else
#include <sys/socket.h>
#include <sys/time.h>
#include <netinet/in.h>
#include <netdb.h>
#include <unistd.h>
#endif

namespace turing {

// ============================================================================
// Minimal JSON: a parser for request bodies and an escaper for responses.
// ============================================================================
struct JsonValue {
    enum Type { Null, Bool, Number, String, Array, Object } type = Null;
    bool boolean = false;
    double number = 0.0;
    std::string str;
    std::vector<JsonValue> arr;
    std::vector<std::pair<std::string, JsonValue>> obj;

    const JsonValue* get(const std::string& key) const {
        for (const auto& kv : obj) if (kv.first == key) return &kv.second;
        return nullptr;
    }
};

class JsonParser {
public:
    // Throws std::invalid_argument on malformed input or nesting deeper than 64.
    static JsonValue parse(const std::string& text) {
        JsonParser p(text);
        JsonValue v = p.value(0);
        p.ws();
        if (p.i_ != p.s_.size()) p.fail("trailing characters");
        return v;
    }

private:
    const std::string& s_;
    size_t i_ = 0;
    explicit JsonParser(const std::string& s) : s_(s) {}

    [[noreturn]] void fail(const char* what) const {
        throw std::invalid_argument(std::string("invalid JSON: ") + what + " at byte " + std::to_string(i_));
    }
    void ws() { while (i_ < s_.size() && (s_[i_] == ' ' || s_[i_] == '\t' || s_[i_] == '\n' || s_[i_] == '\r')) ++i_; }
    bool eat(char c) { ws(); if (i_ < s_.size() && s_[i_] == c) { ++i_; return true; } return false; }
    bool lit(const char* w) {
        size_t n = std::strlen(w);
        if (s_.compare(i_, n, w) == 0) { i_ += n; return true; }
        return false;
    }

    JsonValue value(int depth) {
        if (depth > 64) fail("nesting too deep");
        ws();
        if (i_ >= s_.size()) fail("unexpected end");
        JsonValue v;
        char c = s_[i_];
        if (c == '{') {
            ++i_;
            v.type = JsonValue::Object;
            if (eat('}')) return v;
            do {
                ws();
                if (i_ >= s_.size() || s_[i_] != '"') fail("expected key");
                std::string key = string();
                if (!eat(':')) fail("expected ':'");
                v.obj.emplace_back(std::move(key), value(depth + 1));
            } while (eat(','));
            if (!eat('}')) fail("expected '}'");
        } else if (c == '[') {
            ++i_;
            v.type = JsonValue::Array;
            if (eat(']')) return v;
            do { v.arr.push_back(value(depth + 1)); } while (eat(','));
            if (!eat(']')) fail("expected ']'");
        } else if (c == '"') {
            v.type = JsonValue::String;
            v.str = string();
        } else if (lit("true")) {
            v.type = JsonValue::Bool; v.boolean = true;
        } else if (lit("false")) {
            v.type = JsonValue::Bool;
        } else if (lit("null")) {
            v.type = JsonValue::Null;
        } else if (c == '-' || (c >= '0' && c <= '9')) {
            size_t start = i_;
            if (s_[i_] == '-') ++i_;
            while (i_ < s_.size() && (std::isdigit(static_cast<unsigned char>(s_[i_])) || s_[i_] == '.' ||
                                       s_[i_] == 'e' || s_[i_] == 'E' || s_[i_] == '+' || s_[i_] == '-')) ++i_;
            std::string num = s_.substr(start, i_ - start);
            char* end = nullptr;
            v.type = JsonValue::Number;
            v.number = std::strtod(num.c_str(), &end);
            if (num.empty() || end != num.c_str() + num.size()) fail("bad number");
        } else {
            fail("unexpected character");
        }
        return v;
    }

    static void put_utf8(std::string& out, uint32_t cp) {
        if (cp < 0x80) { out += static_cast<char>(cp); }
        else if (cp < 0x800) { out += static_cast<char>(0xC0 | (cp >> 6)); out += static_cast<char>(0x80 | (cp & 0x3F)); }
        else if (cp < 0x10000) {
            out += static_cast<char>(0xE0 | (cp >> 12)); out += static_cast<char>(0x80 | ((cp >> 6) & 0x3F));
            out += static_cast<char>(0x80 | (cp & 0x3F));
        } else {
            out += static_cast<char>(0xF0 | (cp >> 18)); out += static_cast<char>(0x80 | ((cp >> 12) & 0x3F));
            out += static_cast<char>(0x80 | ((cp >> 6) & 0x3F)); out += static_cast<char>(0x80 | (cp & 0x3F));
        }
    }

    uint32_t hex4() {
        if (i_ + 4 > s_.size()) fail("short \\u escape");
        uint32_t v = 0;
        for (int k = 0; k < 4; ++k) {
            char h = s_[i_++];
            v <<= 4;
            if (h >= '0' && h <= '9') v |= h - '0';
            else if (h >= 'a' && h <= 'f') v |= h - 'a' + 10;
            else if (h >= 'A' && h <= 'F') v |= h - 'A' + 10;
            else fail("bad \\u escape");
        }
        return v;
    }

    std::string string() {
        ++i_;  // opening quote
        std::string out;
        while (true) {
            if (i_ >= s_.size()) fail("unterminated string");
            char c = s_[i_++];
            if (c == '"') return out;
            if (static_cast<unsigned char>(c) < 0x20) fail("control character in string");
            if (c != '\\') { out += c; continue; }
            if (i_ >= s_.size()) fail("unterminated escape");
            char e = s_[i_++];
            switch (e) {
                case '"': out += '"'; break;
                case '\\': out += '\\'; break;
                case '/': out += '/'; break;
                case 'b': out += '\b'; break;
                case 'f': out += '\f'; break;
                case 'n': out += '\n'; break;
                case 'r': out += '\r'; break;
                case 't': out += '\t'; break;
                case 'u': {
                    uint32_t cp = hex4();
                    if (cp >= 0xD800 && cp <= 0xDBFF && i_ + 1 < s_.size() && s_[i_] == '\\' && s_[i_ + 1] == 'u') {
                        i_ += 2;
                        uint32_t lo = hex4();
                        if (lo < 0xDC00 || lo > 0xDFFF) fail("bad surrogate pair");
                        cp = 0x10000 + ((cp - 0xD800) << 10) + (lo - 0xDC00);
                    } else if (cp >= 0xD800 && cp <= 0xDFFF) {
                        cp = 0xFFFD;
                    }
                    put_utf8(out, cp);
                    break;
                }
                default: fail("bad escape");
            }
        }
    }
};

// JSON string body (no quotes): escapes quotes, backslashes and control characters, and replaces invalid UTF-8
// (a byte-level token can end mid-character) with U+FFFD so the response is always valid JSON.
inline std::string json_escape(const std::string& s) {
    std::string out;
    out.reserve(s.size() + 8);
    size_t i = 0;
    while (i < s.size()) {
        unsigned char c = static_cast<unsigned char>(s[i]);
        if (c < 0x80) {
            switch (c) {
                case '"': out += "\\\""; break;
                case '\\': out += "\\\\"; break;
                case '\n': out += "\\n"; break;
                case '\r': out += "\\r"; break;
                case '\t': out += "\\t"; break;
                case '\b': out += "\\b"; break;
                case '\f': out += "\\f"; break;
                default:
                    if (c < 0x20) { char buf[8]; std::snprintf(buf, sizeof(buf), "\\u%04x", c); out += buf; }
                    else out += static_cast<char>(c);
            }
            ++i;
            continue;
        }
        size_t len = (c >> 5) == 0x6 ? 2 : (c >> 4) == 0xE ? 3 : (c >> 3) == 0x1E ? 4 : 0;
        bool ok = len > 0 && i + len <= s.size();
        for (size_t k = 1; ok && k < len; ++k) ok = (static_cast<unsigned char>(s[i + k]) & 0xC0) == 0x80;
        if (ok) { out.append(s, i, len); i += len; }
        else { out += "\xEF\xBF\xBD"; ++i; }
    }
    return out;
}

// ============================================================================
// Chat formatting. GGUF carries tokenizer.chat_template as Jinja; rather than run Jinja, the template's family is
// recognised by its markers (ChatML, Llama 3, Gemma, Mistral/Llama 2 [INST]) and rendered natively. An absent or
// unrecognised template falls back to plain "role: content" lines.
// ============================================================================
struct ChatMessage {
    std::string role;
    std::string content;
};

inline std::string apply_chat_template(const std::string& tmpl, const std::vector<ChatMessage>& messages) {
    std::string out;
    if (tmpl.find("<|im_start|>") != std::string::npos) {
        for (const auto& m : messages) out += "<|im_start|>" + m.role + "\n" + m.content + "<|im_end|>\n";
        return out + "<|im_start|>assistant\n";
    }
    if (tmpl.find("<|start_header_id|>") != std::string::npos) {
        for (const auto& m : messages) out += "<|start_header_id|>" + m.role + "<|end_header_id|>\n\n" + m.content + "<|eot_id|>";
        return out + "<|start_header_id|>assistant<|end_header_id|>\n\n";
    }
    if (tmpl.find("<start_of_turn>") != std::string::npos) {
        for (const auto& m : messages) {
            std::string role = m.role == "assistant" ? "model" : "user";  // Gemma has no system role
            out += "<start_of_turn>" + role + "\n" + m.content + "<end_of_turn>\n";
        }
        return out + "<start_of_turn>model\n";
    }
    if (tmpl.find("[INST]") != std::string::npos) {
        std::string system;
        for (const auto& m : messages) {
            if (m.role == "system") { system += m.content + "\n\n"; continue; }
            if (m.role == "assistant") { out += m.content + "</s>"; continue; }
            out += "[INST] " + system + m.content + " [/INST]";
            system.clear();
        }
        return out;
    }
    for (const auto& m : messages) out += m.role + ": " + m.content + "\n";
    return out + "assistant:";
}

struct GenerationResult {
    std::string text;
    size_t prompt_tokens = 0;
    size_t completion_tokens = 0;
    std::string finish_reason = "length";  // "stop" when the model emitted EOS
};

struct HttpResponse {
    int status = 200;
    std::string body;
};

/**
 * Single-threaded OpenAI-compatible HTTP server: POST /v1/chat/completions, POST /v1/completions, GET /health.
 * Requests are served one at a time (the engine holds one sequence's KV cache).
 */
class HttpServerCpp {
public:
    using GenerateFn = std::function<GenerationResult(const std::string& prompt, int max_tokens, float temperature)>;

    std::string host = "127.0.0.1";
    int port = 8000;
    std::string model_name = "turing-model";
    std::string chat_template;  // GGUF tokenizer.chat_template ("" if absent)
    int default_max_tokens = 32;
    float default_temperature = 0.7f;
    GenerateFn generate_handler;

    static constexpr size_t kMaxRequestBytes = 1 << 20;

    HttpServerCpp(std::string h, int p, GenerateFn handler) : host(std::move(h)), port(p), generate_handler(std::move(handler)) {}

    // Routes one parsed request. Public so the server logic does not depend on a socket.
    HttpResponse handle(const std::string& method, const std::string& path, const std::string& body) {
        if (path == "/health") {
            if (method != "GET") return error(405, "use GET");
            return {200, "{\"status\": \"healthy\", \"runtime\": \"turing-cpp-standalone\"}"};
        }
        bool chat = path == "/v1/chat/completions";
        if (!chat && path != "/v1/completions") return error(404, "unknown path: " + path);
        if (method != "POST") return error(405, "use POST");

        JsonValue req;
        try {
            req = JsonParser::parse(body);
        } catch (const std::invalid_argument& e) {
            return error(400, e.what());
        }
        if (req.type != JsonValue::Object) return error(400, "request body must be a JSON object");

        int max_tokens = default_max_tokens;
        float temperature = default_temperature;
        const JsonValue* mt = req.get("max_tokens");
        if (!mt) mt = req.get("max_completion_tokens");
        if (mt && mt->type != JsonValue::Null) {
            if (mt->type != JsonValue::Number || mt->number < 0 || mt->number > 1e9) return error(400, "max_tokens must be a non-negative integer");
            max_tokens = static_cast<int>(mt->number);
        }
        const JsonValue* temp = req.get("temperature");
        if (temp && temp->type != JsonValue::Null) {
            if (temp->type != JsonValue::Number || temp->number < 0) return error(400, "temperature must be a non-negative number");
            temperature = static_cast<float>(temp->number);
        }
        const JsonValue* stream = req.get("stream");
        if (stream && stream->type == JsonValue::Bool && stream->boolean) return error(400, "stream is not supported by the C++ server");

        std::string prompt;
        if (chat) {
            const JsonValue* msgs = req.get("messages");
            if (!msgs || msgs->type != JsonValue::Array || msgs->arr.empty()) return error(400, "messages must be a non-empty array");
            std::vector<ChatMessage> messages;
            for (const auto& m : msgs->arr) {
                const JsonValue* role = m.type == JsonValue::Object ? m.get("role") : nullptr;
                const JsonValue* content = m.type == JsonValue::Object ? m.get("content") : nullptr;
                if (!role || role->type != JsonValue::String) return error(400, "each message needs a string role");
                std::string text;
                if (content && content->type == JsonValue::String) {
                    text = content->str;
                } else if (content && content->type == JsonValue::Array) {  // [{"type": "text", "text": ...}, ...]
                    for (const auto& part : content->arr) {
                        const JsonValue* t = part.type == JsonValue::Object ? part.get("text") : nullptr;
                        if (!t || t->type != JsonValue::String) return error(400, "only text content parts are supported");
                        text += t->str;
                    }
                } else if (!content || content->type != JsonValue::Null) {
                    return error(400, "message content must be a string or an array of text parts");
                }
                messages.push_back({role->str, text});
            }
            prompt = apply_chat_template(chat_template, messages);
        } else {
            const JsonValue* p = req.get("prompt");
            if (!p || p->type != JsonValue::String) return error(400, "prompt must be a string");
            prompt = p->str;
        }

        GenerationResult r;
        try {
            r = generate_handler(prompt, max_tokens, temperature);
        } catch (const std::out_of_range& e) {
            return error(400, e.what());
        } catch (const std::exception& e) {
            return error(500, e.what());
        }

        std::ostringstream o;
        const long long created = static_cast<long long>(std::time(nullptr));
        const unsigned long long id = ++request_counter_;
        o << "{\"id\": \"" << (chat ? "chatcmpl-" : "cmpl-") << id << "\", \"object\": \""
          << (chat ? "chat.completion" : "text_completion") << "\", \"created\": " << created
          << ", \"model\": \"" << json_escape(model_name) << "\", \"choices\": [{\"index\": 0, ";
        if (chat) o << "\"message\": {\"role\": \"assistant\", \"content\": \"" << json_escape(r.text) << "\"}";
        else o << "\"text\": \"" << json_escape(r.text) << "\"";
        o << ", \"finish_reason\": \"" << r.finish_reason << "\"}], \"usage\": {\"prompt_tokens\": " << r.prompt_tokens
          << ", \"completion_tokens\": " << r.completion_tokens << ", \"total_tokens\": "
          << (r.prompt_tokens + r.completion_tokens) << "}}";
        return {200, o.str()};
    }

    // Binds host:port and serves until accept fails. Returns false if the address cannot be bound.
    bool run() {
#if defined(_WIN32)
        WSADATA wsaData;
        WSAStartup(MAKEWORD(2, 2), &wsaData);
        using sock_t = SOCKET;
        const sock_t bad = INVALID_SOCKET;
#else
        using sock_t = int;
        const sock_t bad = -1;
#endif
        addrinfo hints{};
        hints.ai_family = AF_UNSPEC;
        hints.ai_socktype = SOCK_STREAM;
        hints.ai_flags = AI_PASSIVE;
        addrinfo* res = nullptr;
        const std::string port_str = std::to_string(port);
        if (getaddrinfo(host.c_str(), port_str.c_str(), &hints, &res) != 0 || !res) {
            std::cerr << "[!] Cannot resolve host " << host << std::endl;
            return false;
        }
        sock_t server_fd = bad;
        for (addrinfo* a = res; a; a = a->ai_next) {
            server_fd = socket(a->ai_family, a->ai_socktype, a->ai_protocol);
            if (server_fd == bad) continue;
            int opt = 1;
            setsockopt(server_fd, SOL_SOCKET, SO_REUSEADDR, reinterpret_cast<const char*>(&opt), sizeof(opt));
            if (bind(server_fd, a->ai_addr, static_cast<int>(a->ai_addrlen)) == 0 && listen(server_fd, 16) == 0) break;
            close_socket(server_fd);
            server_fd = bad;
        }
        freeaddrinfo(res);
        if (server_fd == bad) {
            std::cerr << "[!] Bind failed on " << host << ":" << port << std::endl;
            return false;
        }
        std::cout << "[*] Turing Engine C++ HTTP Server listening on http://" << host << ":" << port << std::endl;

        while (true) {
            sock_t client = accept(server_fd, nullptr, nullptr);
            if (client == bad) break;
#if defined(_WIN32)
            DWORD timeout_ms = 10000;
            setsockopt(client, SOL_SOCKET, SO_RCVTIMEO, reinterpret_cast<const char*>(&timeout_ms), sizeof(timeout_ms));
#else
            timeval tv{10, 0};  // a stalled client cannot hold the single-threaded server forever
            setsockopt(client, SOL_SOCKET, SO_RCVTIMEO, &tv, sizeof(tv));
#endif
            HttpResponse resp = serve_one(client);
            const char* reason = resp.status == 200 ? "OK" : resp.status == 400 ? "Bad Request" : resp.status == 404 ? "Not Found"
                               : resp.status == 405 ? "Method Not Allowed" : resp.status == 413 ? "Payload Too Large" : "Internal Server Error";
            std::ostringstream oss;
            oss << "HTTP/1.1 " << resp.status << " " << reason << "\r\n"
                << "Content-Type: application/json\r\n"
                << "Content-Length: " << resp.body.size() << "\r\n"
                << "Connection: close\r\n\r\n"
                << resp.body;
            const std::string out = oss.str();
            size_t sent = 0;
            while (sent < out.size()) {
                auto n = send(client, out.data() + sent, static_cast<int>(out.size() - sent), 0);
                if (n <= 0) break;
                sent += static_cast<size_t>(n);
            }
            close_socket(client);
        }
        close_socket(server_fd);
#if defined(_WIN32)
        WSACleanup();
#endif
        return true;
    }

private:
    unsigned long long request_counter_ = 0;

    static HttpResponse error(int status, const std::string& msg) {
        return {status, "{\"error\": {\"message\": \"" + json_escape(msg) + "\", \"code\": " + std::to_string(status) + "}}"};
    }

    template <typename S>
    static void close_socket(S fd) {
#if defined(_WIN32)
        closesocket(fd);
#else
        close(fd);
#endif
    }

    // Reads one request (headers, then Content-Length bytes of body) and routes it.
    template <typename S>
    HttpResponse serve_one(S client) {
        std::string req;
        char buf[8192];
        size_t header_end = std::string::npos;
        while ((header_end = req.find("\r\n\r\n")) == std::string::npos) {
            if (req.size() > 65536) return error(413, "headers too large");
            auto n = recv(client, buf, sizeof(buf), 0);
            if (n <= 0) return error(400, "incomplete request");
            req.append(buf, static_cast<size_t>(n));
        }
        std::string head = req.substr(0, header_end);
        size_t content_length = 0;
        std::istringstream lines(head);
        std::string request_line, line;
        std::getline(lines, request_line);
        while (std::getline(lines, line)) {
            size_t colon = line.find(':');
            if (colon == std::string::npos) continue;
            std::string name = line.substr(0, colon);
            for (auto& ch : name) ch = static_cast<char>(std::tolower(static_cast<unsigned char>(ch)));
            if (name == "content-length") {
                char* end = nullptr;
                unsigned long long v = std::strtoull(line.c_str() + colon + 1, &end, 10);
                if (v > kMaxRequestBytes) return error(413, "request body larger than 1 MiB");
                content_length = static_cast<size_t>(v);
            }
        }
        std::string body = req.substr(header_end + 4);
        while (body.size() < content_length) {
            auto n = recv(client, buf, sizeof(buf), 0);
            if (n <= 0) return error(400, "incomplete request body");
            body.append(buf, static_cast<size_t>(n));
        }
        body.resize(std::min(body.size(), content_length));

        std::istringstream rl(request_line);
        std::string method, target;
        rl >> method >> target;
        if (method.empty() || target.empty()) return error(400, "malformed request line");
        size_t q = target.find('?');
        if (q != std::string::npos) target.resize(q);
        return handle(method, target, body);
    }
};

} // namespace turing

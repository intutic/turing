#pragma once

#include <string>
#include <sstream>
#include <iostream>
#include <functional>
#include <cstring>
#include <cstdlib>

#if defined(_WIN32)
#include <winsock2.h>
#include <ws2tcpip.h>
#pragma comment(lib, "ws2_32.lib")
#else
#include <sys/socket.h>
#include <netinet/in.h>
#include <unistd.h>
#endif

namespace turing {

class HttpServerCpp {
public:
    int port = 8000;
    std::function<std::string(const std::string& prompt, int max_tokens)> generate_handler;

    HttpServerCpp(int p, std::function<std::string(const std::string&, int)> handler)
        : port(p), generate_handler(handler) {}

    void run() {
#if defined(_WIN32)
        WSADATA wsaData;
        WSAStartup(MAKEWORD(2, 2), &wsaData);
#endif
        int server_fd = socket(AF_INET, SOCK_STREAM, 0);
        if (server_fd < 0) {
            std::cerr << "[!] Error creating socket" << std::endl;
            return;
        }

        int opt = 1;
#if defined(_WIN32)
        setsockopt(server_fd, SOL_SOCKET, SO_REUSEADDR, (const char*)&opt, sizeof(opt));
#else
        setsockopt(server_fd, SOL_SOCKET, SO_REUSEADDR, &opt, sizeof(opt));
#endif

        struct sockaddr_in address;
        std::memset(&address, 0, sizeof(address));
        address.sin_family = AF_INET;
        address.sin_addr.s_addr = INADDR_ANY;
        address.sin_port = htons(port);

        if (bind(server_fd, (struct sockaddr*)&address, sizeof(address)) < 0) {
            std::cerr << "[!] Bind failed on port " << port << std::endl;
#if defined(_WIN32)
            closesocket(server_fd);
#else
            close(server_fd);
#endif
            return;
        }

        if (listen(server_fd, 10) < 0) {
            std::cerr << "[!] Listen failed" << std::endl;
            return;
        }

        std::cout << "[*] Turing Engine C++ HTTP Server listening on http://0.0.0.0:" << port << std::endl;

        while (true) {
            struct sockaddr_in client_addr;
            socklen_t client_len = sizeof(client_addr);
#if defined(_WIN32)
            int client_fd = accept(server_fd, (struct sockaddr*)&client_addr, &client_len);
#else
            int client_fd = accept(server_fd, (struct sockaddr*)&client_addr, &client_len);
#endif
            if (client_fd < 0) break;

            char buffer[4096];
            std::memset(buffer, 0, sizeof(buffer));
#if defined(_WIN32)
            int bytes_read = recv(client_fd, buffer, sizeof(buffer) - 1, 0);
#else
            ssize_t bytes_read = read(client_fd, buffer, sizeof(buffer) - 1);
#endif

            if (bytes_read > 0) {
                std::string req(buffer);
                std::string response_body;

                if (req.find("GET /health") != std::string::npos) {
                    response_body = "{\"status\": \"healthy\", \"runtime\": \"turing-cpp-standalone\"}";
                } else {
                    // Extract prompt from JSON body if present
                    std::string prompt = "Hello from Turing C++";
                    size_t body_pos = req.find("\r\n\r\n");
                    if (body_pos != std::string::npos) {
                        std::string body = req.substr(body_pos + 4);
                        size_t p_pos = body.find("\"prompt\":");
                        if (p_pos != std::string::npos) {
                            size_t q_start = body.find("\"", p_pos + 9);
                            size_t q_end = body.find("\"", q_start + 1);
                            if (q_start != std::string::npos && q_end != std::string::npos) {
                                prompt = body.substr(q_start + 1, q_end - q_start - 1);
                            }
                        }
                    }

                    std::string gen_text = generate_handler ? generate_handler(prompt, 32) : "OK";
                    response_body = "{\"choices\": [{\"text\": \"" + gen_text + "\"}]}";
                }

                std::ostringstream oss;
                oss << "HTTP/1.1 200 OK\r\n"
                    << "Content-Type: application/json\r\n"
                    << "Content-Length: " << response_body.size() << "\r\n"
                    << "Connection: close\r\n\r\n"
                    << response_body;

                std::string resp = oss.str();
#if defined(_WIN32)
                send(client_fd, resp.c_str(), static_cast<int>(resp.size()), 0);
                closesocket(client_fd);
#else
                write(client_fd, resp.c_str(), resp.size());
                close(client_fd);
#endif
            }
        }

#if defined(_WIN32)
        closesocket(server_fd);
        WSACleanup();
#else
        close(server_fd);
#endif
    }
};

} // namespace turing

#include <iostream>
#include <string>
#include <thread>
#include <chrono>
#include "json.hpp"
#ifdef _WIN32
#include <winsock2.h>
#pragma comment(lib, "ws2_32.lib")
#else
#include <sys/socket.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#include <unistd.h>
#endif

int main() {
#ifdef _WIN32
    WSADATA wsaData;
    WSAStartup(MAKEWORD(2,2), &wsaData);
#endif

    int sock = socket(AF_INET, SOCK_DGRAM, 0);
    sockaddr_in addr{};
    addr.sin_family = AF_INET;
    addr.sin_port = htons(5000);
    addr.sin_addr.s_addr = inet_addr("239.255.42.99"); // Multicast address

    for (int i = 0; i < 10; ++i) {
        // Create a simple JSON message
        nlohmann::json j;
        j["t"] = i;
        j["objects"] = {
            { {"id","car1"}, {"brg",12.4}, {"rng",5.3}, {"spd",0.0} }
        };
        std::string msg = j.dump();

        // Send the message
        sendto(sock, msg.c_str(), msg.size(), 0, (sockaddr*)&addr, sizeof(addr));
        std::cout << "Sent: " << msg << std::endl;
        std::this_thread::sleep_for(std::chrono::seconds(1));
    }

#ifdef _WIN32
    closesocket(sock);
    WSACleanup();
#else
    close(sock);
#endif
    return 0;
}
import socket
import struct
import json

UDP_IP = '239.255.42.99'
UDP_PORT = 5000

# Create the UDP socket
sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM, socket.IPPROTO_UDP)
sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
sock.bind(('', UDP_PORT))  # Bind to all interfaces on the given port

# Join the multicast group
mreq = struct.pack("4sl", socket.inet_aton(UDP_IP), socket.INADDR_ANY)
sock.setsockopt(socket.IPPROTO_IP, socket.IP_ADD_MEMBERSHIP, mreq)

print("Listening for UDP packets on {}:{} ...".format(UDP_IP, UDP_PORT))

while True:
    data, addr = sock.recvfrom(65536)
    try:
        msg = json.loads(data.decode())
        print("Received from {}: {}".format(addr, msg))
    except Exception as e:
        print("Received invalid data:", data)
import socket
import FilterManager
import json

#--------필터데이터--------
m_Filter_brightness = 0
m_Filter_gamma = 0
m_Filter_Gaussian = 0
#--------------------------

HOST = '192.168.0.133'
PORT = 5487

server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
server_socket.bind((HOST, PORT))
server_socket.listen()

print(f"서버가 {HOST}:{PORT}에서 실행 중입니다.")


def TcpServer():
    print("서버생성")
    while True:
        client_socket, addr = server_socket.accept()
        print(f"{addr}에서 연결됨")

        try:
            while True:
                data = client_socket.recv(2048)
                if not data:
                    break
                if not data.decode().strip():
                    continue
                    
                #decoded_data = json.loads(data.decode())
                if data.decode().strip() == '{ \\"ID\\":0 }':
                    print('호출')

                elif data.decode().strip() ==  '{ \\"ID\\":1 }':
                    print('호출2')

                elif data.decode().strip() ==  '{ \\"ID\\":3 }':
                    print('호출3')


                #print(str(data))

                #client_socket.sendall(data)

        except Exception as e:
            print(f"에러 발생: {e}")

        # finally:
        #     client_socket.close()

TcpServer()

from robotiq.robotiq_gripper import RobotiqGripper

HOST = "192.168.0.2"
PORT = 30003

def main():
    #tcp_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    #tcp_socket.connect((HOST, PORT))
    gripper=RobotiqGripper()
    gripper.connect(HOST, 63352)
    #gripper.activate()
    gripper.move(255, 255, 255)


if __name__ == '__main__':
    main()

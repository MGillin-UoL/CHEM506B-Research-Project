
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
    # First number: How far it opens (0 fully closed) and closes (255 fully open)
    # Second number: How quickly it opens and closes (0 slowest, 255 fastest)
    # Third number: Force (0 minimum, 255 maximum)

if __name__ == '__main__':
    main()

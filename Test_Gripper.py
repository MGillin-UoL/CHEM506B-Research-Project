
from robotiq.robotiq_gripper import RobotiqGripper

HOST = "192.168.10.2"
PORT = 30003

def main():
    #tcp_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    #tcp_socket.connect((HOST, PORT))
    gripper=RobotiqGripper()
    gripper.connect(HOST, 63352)
    #gripper.activate()
    gripper.move(180, 255, 255)

    # First number: How far it opens and closes
    # - 0 = Fully open
    # - 125 = Large sample vial main body (40 ml)
    # - 140 = Normal sample vial with lid
    # - 155 = Normal sample vial without lid
    # - 180 = Between pipettes
    # - 200 = Holding pipette
    # - 240 = Dosing head
    # - 255 = Fully closed
    # Second number: How quickly it opens and closes (0 slowest, 255 fastest)
    # Third number: Force (0 minimum, 255 maximum)


if __name__ == '__main__':
    main()

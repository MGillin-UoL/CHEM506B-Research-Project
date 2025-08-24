# -*- coding: utf-8 -*-
"""
Frugal Automation for Crystallisation Analysis
Webcam-Based Light Scattering for Particle Characterisation

Author: Michael Gillin (University of Liverpool, Digital Chemistry MSc)
Supervisors: Dr Joe Forth, Dr Gabriella Pizzuto
Date: August 2025

Description:
This script controlsa Robotiq Hand-E adaptive gripper, enabling the user to
test the distance, speed, and force of the gripper.

Notes:
This script was developed as part of Michael Gillin's Digital Chemistry MSc
dissertation at the University of Liverpool, "Frugal Automation for Crystallisation
Analysis: Webcam-Based Light Scattering for Low-Cost Particle Characterisation"

GitHub: https://github.com/MGillin-UoL
Email: sgmgilli@liverpool.ac.uk
"""

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

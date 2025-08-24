# -*- coding: utf-8 -*-
"""
Frugal Automation for Crystallisation Analysis
Webcam-Based Light Scattering for Particle Characterisation

Author: Michael Gillin (University of Liverpool, Digital Chemistry MSc)
Supervisors: Dr Joe Forth, Dr Gabriella Pizzuto
Date: August 2025

Description:
This script enables the user to obtain the six-axis coordinates for a UR5e
robotic arm running off URScript. Once ran, the coordinates in the terminal
can be copied and pasted into the main robotic movements code.

Notes:
This script was developed as part of Michael Gillin's Digital Chemistry MSc
dissertation at the University of Liverpool, "Frugal Automation for Crystallisation
Analysis: Webcam-Based Light Scattering for Low-Cost Particle Characterisation"

GitHub: https://github.com/MGillin-UoL
Email: sgmgilli@liverpool.ac.uk
"""

import sys
import os

# Add the directory containing robotiq_preamble.py to the Python search path
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(current_dir, 'robotiq'))

from utils.UR_Functions import URfunctions as URControl

HOST = "192.168.0.2"
PORT = 30003

def main():
    robot = URControl(ip="192.168.0.2", port=30003)
    print(robot.get_current_joint_positions().tolist())
    print(robot.get_current_tcp())
if __name__ == '__main__':
     main()

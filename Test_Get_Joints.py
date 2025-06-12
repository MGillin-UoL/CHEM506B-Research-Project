import sys
import os
import time
import argparse
import math 
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

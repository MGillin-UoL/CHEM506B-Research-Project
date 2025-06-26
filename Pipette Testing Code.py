import os
import sys
import time
import math
import csv

import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

from utils.UR_Functions import URfunctions as URControl
from robotiq.robotiq_gripper import RobotiqGripper
from Kinetics import process_and_plot_absorbance


# === Robot Configuration ===
ROBOT_POSITIONS = {
    "Home_Position": [1.68345308303833, -1.508956865673401, 1.536279026662008, -1.6149097881712855, 4.729211807250977, 0.14349502325057983],
    "SVH1_P1": [1.7411258220672607, -1.1768915218165894, 1.8037975470172327, -2.2151204548277796, 4.726142883300781, 0.20422950387001038], # Sample Vial Holder 1, Position 1
    "Above_SVH1_P1": [1.7414016723632812, -1.3616285038045426, 1.3544829527484339, -1.581027170220846, 4.728183746337891, 0.20159339904785156],
    "Above_MTQ_P1": [1.6635525226593018, -1.6336456737914027, 1.6214864889727991, -1.5751520595946253, 4.7296552658081055, 0.12319839745759964], # Mettler Toledo Quantos, Position 1
    "MTQ_P1": [1.666123867034912, -1.4169096511653443, 2.1400330702411097, -2.3105293713011683, 4.7272443771362305, 0.1289839893579483],
    "Before_CH_P1": [1.6446585655212402, -0.9325866860202332, 1.7292874495135706, -2.3080955944456996, 3.1797618865966797, 1.6713024377822876], # Before Cartridge Holder, Position 1
    "CH_P1": [1.6021586656570435, -0.9525528711131592, 1.6396577993976038, -1.7278710804381312, 3.1662089824676514, 2.1333248615264893],
    "Before_MTQ_Cartidge": [2.098874568939209, -1.813380857507223, 2.2846065203296106, -0.3756905359080811, 2.06877064704895, 3.169734477996826],
    "MTQ_Cartridge": [1.9800453186035156, -1.718318601647848, 2.212010685597555, -0.403808669453003, 1.950730323791504, 3.1575405597686768],
    "Before_CH_P2": [1.6570565700531006, -1.055628017788269, 1.8491695562945765, -2.002738138238424, 3.1821682453155518, 1.9736840724945068],
    "CH_P2": [1.5977692604064941, -0.8127642434886475, 1.8110173384295862, -3.433835645715231, 3.199983835220337, 0.7463662624359131],
    "Above_Hotplate": [1.2258187532424927, -1.9161116085448207, 1.8315866629229944, -1.5137065213969727, 4.734868049621582, -0.4259908835040491],
    "On_Hotplate": [1.2316064834594727, -1.7949210606017054, 2.2526286284076136, -2.038816114465231, 4.734594345092773, -0.30711537996401006],
    "SVH1_P2": [1.734782099723816, -1.1264804166606446, 1.7398274580584925, -2.216509004632467, 4.726589202880859, 0.19797921180725098],
    "Above_SVH1_P2": [1.7350428104400635, -1.3055239182761689, 1.3550098578082483, -1.6525570354857386, 4.728509902954102, 0.19569134712219238],
    "Before_CH_P3": [1.6619412899017334, -1.1525814098170777, 1.895367447529928, -1.5512126323631783, 3.2425289154052734, 2.3269267082214355],
    "CH_P3": [1.6062703132629395, -1.1352652174285431, 2.0295231978045862, -2.2644511661925257, 3.2158920764923096, 1.7632675170898438],
    "Before_CH_P4": [1.6780502796173096, -1.173294411306717, 1.859753433858053, -0.8634331387332459, 3.228264808654785, 2.951382875442505],
    "CH_P4": [1.6071313619613647, -1.2490405601314087, 2.070871178303854, -1.6016117535033167, 3.1629626750946045, 2.3482446670532227],
    "Above_SVH2_P1": [1.8444747924804688, -1.9149447880186976, 1.8616092840777796, -1.5492754739573975, 4.715482234954834, 0.19258466362953186],
    "SVH2_P1": [1.8438289165496826, -1.6695581875243128, 2.4063966909991663, -2.3394586048521937, 4.713134765625, 0.19549989700317383],
    
    
    "Pipette_Above_Holder1": [1.740378499031067, -1.333426148896553, 1.4481218496905726, -1.696397443810934, 4.7199578285217285, 0.20239689946174622],
    "Pipette_In_Holder1": [1.7368545532226562, -1.2701246005347748, 1.6811278502093714, -1.9927717647948207, 4.719141960144043, 0.20026108622550964],
    "Pipette_Above_Holder2": [1.7326463460922241, -1.265038089161255, 1.4235413710223597, -1.7402202091612757, 4.719878673553467, 0.19503235816955566],
    "Pipette_In_Holder2": [1.7350941896438599, -1.2634319526008149, 1.5668423811541956, -1.891529222527975, 4.727492809295654, 0.1968572437763214],
    
    
    "in_front_of_white_bg": [0.9755, -1.2299, 1.5326, 1.2679, 1.5616, 2.5228],
    "above_home_holder": [1.7151, -0.9303, 0.9488, 1.4103, 1.4994, 3.2781],
    "readjustment": [1.7268, -0.8310, 0.8724, 1.5186, 1.5266, 3.2738],
    "return_to_holder": [1.7270, -0.7679, 1.0835, 1.2442, 1.5274, 3.2754]
}

MOVEMENT_PARAMS = {
    "speed": 0.25,
    "acceleration": 0.5,
    "blending": 0.02,
}

CSV_FILE_PATH = os.path.expanduser("~/Code/chem504-2425_GroupA/examples/blue_pixel_data.csv")


# === Helper Functions ===
def degreestorad(angles_deg):
    return [angle * math.pi / 180 for angle in angles_deg]

def move_robot(robot, position):
    robot.move_joint_list(
    position,
    MOVEMENT_PARAMS["speed"],
    MOVEMENT_PARAMS["acceleration"],
    MOVEMENT_PARAMS["blending"]
)


def operate_gripper(gripper, position):
    gripper.move(position, 125, 125)


# === Main Execution ===
def main():
    robot = URControl(ip="192.168.0.2", port=30003)
    gripper = RobotiqGripper()
    gripper.connect("192.168.0.2", 63352)

    # Initial position
    operate_gripper(gripper, 200)
    move_robot(robot, ROBOT_POSITIONS["Home_Position"])
    print("Starting in home position", "\n")
    operate_gripper(gripper, 200)
    
    move_robot(robot, ROBOT_POSITIONS["Pipette_Above_Holder1"])
    print("Above Holder 1")
    operate_gripper(gripper, 240)
    move_robot(robot, ROBOT_POSITIONS["Pipette_In_Holder1"])
    print("In Holder 1")
    operate_gripper(gripper, 235)
    
    move_robot(robot, ROBOT_POSITIONS["Pipette_In_Holder1"])
    print("Sucking up liquid")
    operate_gripper(gripper, 230)
    
    move_robot(robot, ROBOT_POSITIONS["Pipette_In_Holder1"])
    print("Sucking up liquid")
    operate_gripper(gripper, 225)
    
    move_robot(robot, ROBOT_POSITIONS["Pipette_In_Holder1"])
    print("Sucking up liquid")
    operate_gripper(gripper,220)
    
    move_robot(robot, ROBOT_POSITIONS["Pipette_In_Holder1"])
    print("Sucking up liquid")
    operate_gripper(gripper, 215)
    
    move_robot(robot, ROBOT_POSITIONS["Pipette_In_Holder1"])
    print("Sucking up liquid")
    operate_gripper(gripper, 210)
    
    move_robot(robot, ROBOT_POSITIONS["Pipette_In_Holder1"])
    print("Sucking up liquid")
    operate_gripper(gripper, 205)
    
    move_robot(robot, ROBOT_POSITIONS["Pipette_In_Holder1"])
    print("Sucking up liquid")
    operate_gripper(gripper, 200)
    
    move_robot(robot, ROBOT_POSITIONS["Pipette_Above_Holder1"])
    print("Above Holder 1")
    move_robot(robot, ROBOT_POSITIONS["Pipette_Above_Holder2"])
    print("Above Holder 2")
    
    
    for position in range(201, 241, 1):  # Loop from 201 to 240 inclusive
        move_robot(robot, ROBOT_POSITIONS["Pipette_In_Holder2"])
        print("In Holder 2")
        operate_gripper(gripper, position)
     
    
    move_robot(robot, ROBOT_POSITIONS["Pipette_Above_Holder2"])
    print("Unsqueezing pipette")
    operate_gripper(gripper, 239)
    
    move_robot(robot, ROBOT_POSITIONS["Pipette_Above_Holder2"])
    print("Unsqueezing pipette")
    operate_gripper(gripper, 238)
    
    move_robot(robot, ROBOT_POSITIONS["Pipette_Above_Holder2"])
    print("Unsqueezing pipette")
    operate_gripper(gripper, 237)
    
    move_robot(robot, ROBOT_POSITIONS["Pipette_Above_Holder2"])
    print("Unsqueezing pipette")
    operate_gripper(gripper, 236)
    
    move_robot(robot, ROBOT_POSITIONS["Pipette_Above_Holder2"])
    print("Unsqueezing pipette")
    operate_gripper(gripper, 235)

    move_robot(robot, ROBOT_POSITIONS["Pipette_Above_Holder2"])
    print("Unsqueezing pipette")
    operate_gripper(gripper, 230)

    move_robot(robot, ROBOT_POSITIONS["Pipette_Above_Holder2"])
    print("Unsqueezing pipette")
    operate_gripper(gripper, 225)
    
    move_robot(robot, ROBOT_POSITIONS["Pipette_Above_Holder2"])
    print("Unsqueezing pipette")
    operate_gripper(gripper, 220)
    
    move_robot(robot, ROBOT_POSITIONS["Pipette_Above_Holder2"])
    print("Unsqueezing pipette")
    operate_gripper(gripper, 215)
    
    move_robot(robot, ROBOT_POSITIONS["Pipette_Above_Holder2"])
    print("Unsqueezing pipette")
    operate_gripper(gripper, 210)
    
    move_robot(robot, ROBOT_POSITIONS["Pipette_Above_Holder2"])
    print("Unsqueezing pipette")
    operate_gripper(gripper, 205)
    
    move_robot(robot, ROBOT_POSITIONS["Pipette_Above_Holder2"])
    print("Unsqueezing pipette")
    operate_gripper(gripper, 200)
    
    move_robot(robot, ROBOT_POSITIONS["Home_Position"])
    print("Starting in home position", "\n")

    
    # move_robot(robot, ROBOT_POSITIONS["Home_Position"])
    # operate_gripper(gripper, 240)
    # move_robot(robot, ROBOT_POSITIONS["Home_Position"])
    # operate_gripper(gripper, 235)
    # move_robot(robot, ROBOT_POSITIONS["Home_Position"])
    # operate_gripper(gripper, 230)
    # move_robot(robot, ROBOT_POSITIONS["Home_Position"])
    # operate_gripper(gripper, 225)
    # move_robot(robot, ROBOT_POSITIONS["Home_Position"])
    # operate_gripper(gripper, 220)
    # move_robot(robot, ROBOT_POSITIONS["Home_Position"])
    # operate_gripper(gripper, 215)
    # move_robot(robot, ROBOT_POSITIONS["Home_Position"])
    # operate_gripper(gripper, 210)
    # move_robot(robot, ROBOT_POSITIONS["Home_Position"])
    # operate_gripper(gripper, 205)
    # move_robot(robot, ROBOT_POSITIONS["Home_Position"])
    # operate_gripper(gripper, 200)
    # move_robot(robot, ROBOT_POSITIONS["Home_Position"])
    # operate_gripper(gripper, 240)


if __name__ == "__main__":
    main()

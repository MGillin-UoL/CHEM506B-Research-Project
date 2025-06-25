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
    operate_gripper(gripper, 0)
    move_robot(robot, ROBOT_POSITIONS["Home_Position"])
    print("Starting in home position", "\n")

    # Loading Mettler Toledo Quantos with Sample Vial 1
    operate_gripper(gripper, 0)
    move_robot(robot, ROBOT_POSITIONS["SVH1_P1"])
    print("Moved to sample vial holder Position 1", "\n")
    operate_gripper(gripper, 155)

    move_robot(robot, ROBOT_POSITIONS["Above_SVH1_P1"])
    print("Moved to above sample vial holder Position 1", "\n")
    
    move_robot(robot, ROBOT_POSITIONS["Above_MTQ_P1"])
    print("Moved to above Mettler Toledo Quantos Position 1", "\n")
    
    move_robot(robot, ROBOT_POSITIONS["MTQ_P1"])
    operate_gripper(gripper, 0)
    print("Released sample into Mettler Toledo Quantos Position 1", "\n")

    move_robot(robot, ROBOT_POSITIONS["Above_MTQ_P1"])
    print("Moved to above Mettler Toledo Quantos Position 1", "\n")
    
    
    # Load Cartridge 1
    move_robot(robot, ROBOT_POSITIONS["Before_CH_P1"])
    print("Moved just in front of cartidge holder Position 1", "\n")
    
    move_robot(robot, ROBOT_POSITIONS["CH_P1"])
    print("Grabbed cartridge from Position 1", "\n")
    operate_gripper(gripper, 100)
    
    move_robot(robot, ROBOT_POSITIONS["Before_CH_P1"])
    print("Moved just in front of cartidge holder Position 1", "\n")
    
    move_robot(robot, ROBOT_POSITIONS["Before_MTQ_Cartidge"])
    print("Moved in front of Mettler Toledo Cartridge loading area", "\n")
    
    move_robot(robot, ROBOT_POSITIONS["MTQ_Cartridge"])
    operate_gripper(gripper, 0)
    print("Loaded Mettler Toledo cartridge", "\n")
    
    move_robot(robot, ROBOT_POSITIONS["Before_MTQ_Cartidge"])
    print("Moved back in front of Mettler Toledo Cartridge loading area", "\n")
    
    time.sleep(5)
    
    print("Dispensing copper (II) chloride into sample vial", "\n")
    time.sleep(5)
    
    move_robot(robot, ROBOT_POSITIONS["MTQ_Cartridge"])
    operate_gripper(gripper, 100)
    print("Grabbed Mettler Toledo cartridge", "\n")
    
    move_robot(robot, ROBOT_POSITIONS["Before_MTQ_Cartidge"])
    print("Unloaded and moved back in front of Mettler Toledo Cartridge loading area", "\n")

    move_robot(robot, ROBOT_POSITIONS["Before_CH_P1"])
    print("Moved just in front of cartidge holder Position 1", "\n")

    move_robot(robot, ROBOT_POSITIONS["CH_P1"])
    operate_gripper(gripper, 0)
    print("Returned cartridge to Position 1", "\n")
    
    move_robot(robot, ROBOT_POSITIONS["Before_CH_P1"])
    print("Moved just in front of cartidge holder Position 1", "\n")


    # Load Cartridge 2
    move_robot(robot, ROBOT_POSITIONS["Before_CH_P2"])
    print("Moved just in front of cartidge holder Position 2", "\n")
    
    move_robot(robot, ROBOT_POSITIONS["CH_P2"])
    print("Grabbed cartridge from Position 2", "\n")
    operate_gripper(gripper, 100)
    
    move_robot(robot, ROBOT_POSITIONS["Before_CH_P2"])
    print("Moved just in front of cartidge holder Position 2", "\n")
    
    move_robot(robot, ROBOT_POSITIONS["Before_MTQ_Cartidge"])
    print("Moved in front of Mettler Toledo Cartridge loading area", "\n")
    
    move_robot(robot, ROBOT_POSITIONS["MTQ_Cartridge"])
    operate_gripper(gripper, 0)
    print("Loaded Mettler Toledo cartridge", "\n")
    
    move_robot(robot, ROBOT_POSITIONS["Before_MTQ_Cartidge"])
    print("Moved back in front of Mettler Toledo Cartridge loading area", "\n")
    
    time.sleep(5)
    
    print("Dispensing water into sample vial", "\n")
    time.sleep(5)
    
    move_robot(robot, ROBOT_POSITIONS["MTQ_Cartridge"])
    operate_gripper(gripper, 100)
    print("Grabbed Mettler Toledo cartridge", "\n")
    
    move_robot(robot, ROBOT_POSITIONS["Before_MTQ_Cartidge"])
    print("Unloaded and moved back in front of Mettler Toledo Cartridge loading area", "\n")

    move_robot(robot, ROBOT_POSITIONS["Before_CH_P2"])
    print("Moved just in front of cartidge holder Position 2", "\n")

    move_robot(robot, ROBOT_POSITIONS["CH_P2"])
    operate_gripper(gripper, 0)
    print("Returned cartridge to Position 2", "\n")
    
    move_robot(robot, ROBOT_POSITIONS["Before_CH_P2"])
    print("Moved just in front of cartidge holder Position 2", "\n")


    # Putting Sample Vial 1 on the hotplate
    move_robot(robot, ROBOT_POSITIONS["Above_MTQ_P1"])
    print("Moved to above Mettler Toledo Quantos Position 1", "\n")
    
    move_robot(robot, ROBOT_POSITIONS["MTQ_P1"])
    operate_gripper(gripper, 155)
    print("Grabbed sample from Mettler Toledo Quantos Position 1", "\n")    

    move_robot(robot, ROBOT_POSITIONS["Above_MTQ_P1"])
    print("Moved to above Mettler Toledo Quantos Position 1", "\n")
    
    move_robot(robot, ROBOT_POSITIONS["Above_Hotplate"])
    print("Moved to above hotplate stirrer")
    
    move_robot(robot, ROBOT_POSITIONS["On_Hotplate"])
    print("Moved to hotplate stirrer", "\n")
    operate_gripper(gripper, 0)
    
    move_robot(robot, ROBOT_POSITIONS["Above_Hotplate"])
    print("Moved to above hotplate stirrer")
    
    
    # Loading Mettler Toledo Quantos with Sample Vial 2
    move_robot(robot, ROBOT_POSITIONS["Above_SVH1_P2"])
    print("Moved to above sample vial holder Position 2", "\n")

    move_robot(robot, ROBOT_POSITIONS["SVH1_P2"])
    print("Moved to sample vial holder Position 2", "\n")
    operate_gripper(gripper, 155)

    move_robot(robot, ROBOT_POSITIONS["Above_SVH1_P2"])
    print("Moved to above sample vial holder Position 2", "\n")
    
    move_robot(robot, ROBOT_POSITIONS["Above_MTQ_P1"])
    print("Moved to above Mettler Toledo Quantos Position 1", "\n")
    
    move_robot(robot, ROBOT_POSITIONS["MTQ_P1"])
    operate_gripper(gripper, 0)
    print("Released sample into Mettler Toledo Quantos Position 1", "\n")

    move_robot(robot, ROBOT_POSITIONS["Above_MTQ_P1"])
    print("Moved to above Mettler Toledo Quantos Position 1", "\n")

    
    # Load Cartridge 3
    move_robot(robot, ROBOT_POSITIONS["Before_CH_P3"])
    print("Moved just in front of cartidge holder Position 3", "\n")
    
    move_robot(robot, ROBOT_POSITIONS["CH_P3"])
    print("Grabbed cartridge from Position 3", "\n")
    operate_gripper(gripper, 100)
    
    move_robot(robot, ROBOT_POSITIONS["Before_CH_P3"])
    print("Moved just in front of cartidge holder Position 1", "\n")
    
    move_robot(robot, ROBOT_POSITIONS["Before_MTQ_Cartidge"])
    print("Moved in front of Mettler Toledo Cartridge loading area", "\n")
    
    move_robot(robot, ROBOT_POSITIONS["MTQ_Cartridge"])
    operate_gripper(gripper, 0)
    print("Loaded Mettler Toledo cartridge", "\n")
    
    move_robot(robot, ROBOT_POSITIONS["Before_MTQ_Cartidge"])
    print("Moved back in front of Mettler Toledo Cartridge loading area", "\n")
    
    time.sleep(5)
    
    print("Dispensing piroctone olamine into sample vial", "\n")
    time.sleep(5)
    
    move_robot(robot, ROBOT_POSITIONS["MTQ_Cartridge"])
    operate_gripper(gripper, 100)
    print("Grabbed Mettler Toledo cartridge", "\n")
    
    move_robot(robot, ROBOT_POSITIONS["Before_MTQ_Cartidge"])
    print("Unloaded and moved back in front of Mettler Toledo Cartridge loading area", "\n")

    move_robot(robot, ROBOT_POSITIONS["Before_CH_P3"])
    print("Moved just in front of cartidge holder Position 3", "\n")

    move_robot(robot, ROBOT_POSITIONS["CH_P3"])
    operate_gripper(gripper, 0)
    print("Returned cartridge to Position 3", "\n")
    
    move_robot(robot, ROBOT_POSITIONS["Before_CH_P3"])
    print("Moved just in front of cartidge holder Position 3", "\n")


    # Load Cartridge 4
    move_robot(robot, ROBOT_POSITIONS["Before_CH_P4"])
    print("Moved just in front of cartidge holder Position 4", "\n")
    
    move_robot(robot, ROBOT_POSITIONS["CH_P4"])
    print("Grabbed cartridge from Position 4", "\n")
    operate_gripper(gripper, 100)
    
    move_robot(robot, ROBOT_POSITIONS["Before_CH_P4"])
    print("Moved just in front of cartidge holder Position 4", "\n")
    
    move_robot(robot, ROBOT_POSITIONS["Before_MTQ_Cartidge"])
    print("Moved in front of Mettler Toledo Cartridge loading area", "\n")
    
    move_robot(robot, ROBOT_POSITIONS["MTQ_Cartridge"])
    operate_gripper(gripper, 0)
    print("Loaded Mettler Toledo cartridge", "\n")
    
    move_robot(robot, ROBOT_POSITIONS["Before_MTQ_Cartidge"])
    print("Moved back in front of Mettler Toledo Cartridge loading area", "\n")
    
    time.sleep(5)
    
    print("Dispensing ethanol into sample vial", "\n")
    time.sleep(5)
    
    move_robot(robot, ROBOT_POSITIONS["MTQ_Cartridge"])
    operate_gripper(gripper, 100)
    print("Grabbed Mettler Toledo cartridge", "\n")
    
    move_robot(robot, ROBOT_POSITIONS["Before_MTQ_Cartidge"])
    print("Unloaded and moved back in front of Mettler Toledo Cartridge loading area", "\n")

    move_robot(robot, ROBOT_POSITIONS["Before_CH_P4"])
    print("Moved just in front of cartidge holder Position 4", "\n")

    move_robot(robot, ROBOT_POSITIONS["CH_P4"])
    operate_gripper(gripper, 0)
    print("Returned cartridge to Position 4", "\n")
    
    move_robot(robot, ROBOT_POSITIONS["Before_CH_P4"])
    print("Moved just in front of cartidge holder Position 4", "\n")


    # Moving Sample Vial 1 from hotplate stirrer to Sample Vial Holder 2
    move_robot(robot, ROBOT_POSITIONS["Above_Hotplate"])
    print("Moved to above hotplate stirrer")
    
    move_robot(robot, ROBOT_POSITIONS["On_Hotplate"])
    operate_gripper(gripper, 155)
    print("Grabbed Sample Vial 1 from hotplate stirrer")
    
    move_robot(robot, ROBOT_POSITIONS["Above_Hotplate"])
    print("Moved to above hotplate stirrer")

    move_robot(robot, ROBOT_POSITIONS["Above_SVH2_P1"])
    print("Moved to above Sample Vial Holder 2 Position 1")
    
    move_robot(robot, ROBOT_POSITIONS["SVH2_P1"])
    operate_gripper(gripper, 0)
    print("Inserted Sample Vial 1 into Sample Vial Holder 2 Position 1")
    
    move_robot(robot, ROBOT_POSITIONS["Above_SVH2_P1"])
    print("Moved to above Sample Vial Holder 2 Position 1")


    # Putting Sample Vial 2 on the hotplate
    move_robot(robot, ROBOT_POSITIONS["Above_MTQ_P1"])
    print("Moved to above Mettler Toledo Quantos Position 1", "\n")
    
    move_robot(robot, ROBOT_POSITIONS["MTQ_P1"])
    operate_gripper(gripper, 155)
    print("Grabbed sample from Mettler Toledo Quantos Position 1", "\n")    

    move_robot(robot, ROBOT_POSITIONS["Above_MTQ_P1"])
    print("Moved to above Mettler Toledo Quantos Position 1", "\n")
    
    move_robot(robot, ROBOT_POSITIONS["Above_Hotplate"])
    print("Moved to above hotplate stirrer")
    
    move_robot(robot, ROBOT_POSITIONS["On_Hotplate"])
    operate_gripper(gripper, 0)
    print("Released sample vial onto hotplate stirrer", "\n")
    
    move_robot(robot, ROBOT_POSITIONS["Above_Hotplate"])
    print("Moved to above hotplate stirrer")

    # === Image Capture Setup ===
    cam = cv2.VideoCapture(0)
    cv2.namedWindow("test")
    cv2.namedWindow("Blue Mask")

    # Threshold settings
    reaction_threshold = 50
    blue_pixel_below_threshold_duration = 5
    blue_pixel_counts = []

    roi_x, roi_y, roi_w, roi_h = 200, 150, 260, 280
    start_time = time.time()
    last_logged_time = -1
    below_threshold_counter = 3

    # HSV sliders
    def nothing(x): pass
    for name, max_val in [("Lower Hue", 179), ("Upper Hue", 179),
                          ("Lower Sat", 255), ("Upper Sat", 255),
                          ("Lower Val", 255), ("Upper Val", 255)]:
        cv2.createTrackbar(name, "Blue Mask", 0 if "Lower" in name else max_val, max_val, nothing)

    # === Video Loop ===
    try:
        while True:
            if cv2.waitKey(1) & 0xFF == 27:
                print("Escape hit, closing...")
                break

            ret, frame = cam.read()
            if not ret:
                print("Failed to grab frame")
                break

            roi_frame = frame[roi_y:roi_y+roi_h, roi_x:roi_x+roi_w]
            roi_frame = cv2.GaussianBlur(roi_frame, (5, 5), 0)
            hsv_roi = cv2.cvtColor(roi_frame, cv2.COLOR_BGR2HSV)

            lower_h = cv2.getTrackbarPos("Lower Hue", "Blue Mask")
            upper_h = cv2.getTrackbarPos("Upper Hue", "Blue Mask")
            lower_s = cv2.getTrackbarPos("Lower Sat", "Blue Mask")
            upper_s = cv2.getTrackbarPos("Upper Sat", "Blue Mask")
            lower_v = cv2.getTrackbarPos("Lower Val", "Blue Mask")
            upper_v = cv2.getTrackbarPos("Upper Val", "Blue Mask")

            lower_blue = np.array([lower_h, lower_s, lower_v])
            upper_blue = np.array([upper_h, upper_s, upper_v])

            blue_mask = cv2.inRange(hsv_roi, lower_blue, upper_blue)
            kernel = np.ones((3, 3), np.uint8)
            blue_mask = cv2.morphologyEx(blue_mask, cv2.MORPH_OPEN, kernel)
            blue_mask = cv2.morphologyEx(blue_mask, cv2.MORPH_CLOSE, kernel)

            blue_pixel_count = cv2.countNonZero(blue_mask)
            elapsed_seconds = int(time.time() - start_time)

            if elapsed_seconds > last_logged_time:
                last_logged_time = elapsed_seconds
                blue_pixel_counts.append((elapsed_seconds, blue_pixel_count))
                print(f"Time: {elapsed_seconds}s, Blue Pixels: {blue_pixel_count}")

            if blue_pixel_count > 0:
                below_threshold_counter = 0
            else:
                below_threshold_counter += 1
                print(f"Blue below threshold for {below_threshold_counter} seconds.")
                if below_threshold_counter >= blue_pixel_below_threshold_duration:
                    print("The reaction is no longer blue. Stopping.")
                    break

            cv2.rectangle(frame, (roi_x, roi_y), (roi_x + roi_w, roi_y + roi_h), (0, 255, 0), 2)
            cv2.imshow("test", frame)
            cv2.imshow("Blue Mask", cv2.merge([blue_mask] * 3))

    except Exception as e:
        print(f"Error occurred: {e}")

    finally:
        cam.release()
        cv2.destroyAllWindows()

        try:
            with open(CSV_FILE_PATH, mode='w', newline='') as file:
                writer = csv.writer(file)
                writer.writerow(["Time (Seconds)", "Blue Pixel Count"])
                writer.writerows(blue_pixel_counts)
            print(f"Data successfully saved in: {CSV_FILE_PATH}")
        except Exception as e:
            print(f"Error while saving CSV file: {e}")

        total_blue = sum(count for _, count in blue_pixel_counts)
        print("Sample vial is blue" if total_blue >= reaction_threshold else "Sample vial did not reach the blue threshold")

    # === Return Sample ===
    operate_gripper(gripper, 0)
    move_robot(robot, ROBOT_POSITIONS["Home_Position"])
    print("Returning to home position", "\n")

    # === Process Data ===
    df_result = process_and_plot_absorbance(CSV_FILE_PATH)
    print(df_result)


if __name__ == "__main__":
    main()

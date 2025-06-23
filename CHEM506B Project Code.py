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
    "Facing_CH_P1": [1.660979986190796, -0.8530509036830445, 1.3222835699664515, -0.6348576110652466, 3.2073233127593994, -0.13249665895570928], # Facing Cartridge Holder, Position 1
    "CH_P1": [1.5990753173828125, -0.9024868172458191, 1.7017572561847132, -2.2899843655028285, 3.1535933017730713, -1.4146059195147913],
    "just_above_stirrer": [1.4119, -1.2327, 1.5361, 1.2655, 1.5617, 2.9592],
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
    print("Starting in home position")
    time.sleep(0.1)

    # Sample acquisition
    operate_gripper(gripper, 0)
    move_robot(robot, ROBOT_POSITIONS["SVH1_P1"])
    print("Moved to sample vial holder Position 1")
    operate_gripper(gripper, 140)
    time.sleep(0.1)

    # Move to Mettler Toledo Quantos
    move_robot(robot, ROBOT_POSITIONS["Above_SVH1_P1"])
    print("Moved to above sample vial holder Position 1")
    time.sleep(0.1)
    move_robot(robot, ROBOT_POSITIONS["Above_MTQ_P1"])
    print("Moved to above Mettler Toledo Quantos Position 1")
    time.sleep(0.1)
    move_robot(robot, ROBOT_POSITIONS["MTQ_P1"])
    operate_gripper(gripper, 0)
    print("Released sample into Mettler Toledo Quantos Position 1")
    time.sleep(0.1)
    move_robot(robot, ROBOT_POSITIONS["Above_MTQ_P1"])
    print("Moved to above Mettler Toledo Quantos Position 1")
    time.sleep(0.1)

    # Load Sample Vial 1
    move_robot(robot, ROBOT_POSITIONS["Facing_CH_P1"])
    print("Moved in front of catridge holder Position 1")
    time.sleep(0.1)
    move_robot(robot, ROBOT_POSITIONS["CH_P1"])
    print("Grabbed cartridge holder from Position 1")
    operate_gripper(gripper, 100)
    time.sleep(1)
    move_robot(robot, ROBOT_POSITIONS["in_front_of_white_bg"])
    print("Moved to white background")

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
    time.sleep(1)
    move_robot(robot, ROBOT_POSITIONS["readjustment"])
    print("Readjustment position")

    time.sleep(1)
    move_robot(robot, ROBOT_POSITIONS["return_to_holder"])
    print("Returned to holder")

    operate_gripper(gripper, 0)
    
    time.sleep(1)
    move_robot(robot, ROBOT_POSITIONS["home"])
    print("Returned to home position")

    # === Process Data ===
    df_result = process_and_plot_absorbance(CSV_FILE_PATH)
    print(df_result)


if __name__ == "__main__":
    main()

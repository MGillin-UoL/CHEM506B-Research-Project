import time
import math
import os
import sys
from utils.UR_Functions import URfunctions as URControl
from robotiq.robotiq_gripper import RobotiqGripper

import cv2
import numpy as np
import time
import csv
import os

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import os

from Kinetics import process_and_plot_absorbance

# Robot position constants
ROBOT_POSITIONS = {
    "home": [1.9598381519317627, -1.8197394810118617, 2.0655067602740687, -0.2610810560039063, 1.9489684104919434, 3.1007790565490723],
    "sample_vial": [1.8568476438522339, -1.0523559612086792, 2.228867832814352, -1.175378904943802, 1.8485941886901855, 3.121866464614868],
    "up_vertical": [1.8578236103057861, -1.447265812461712, 2.110725227986471, -0.6615637105754395, 1.847927212715149, 3.124692440032959],
    "above_stirrer": [1.168654441833496, -2.158050676385397, 2.3632450739489954, -0.20627863824877934, 1.1556280851364136, 3.1256730556488037],
    "in_stirrer": [1.1673400402069092, -1.8471819363036097, 2.67733341852297, -0.8302822870067139, 1.157050371170044, 3.12343168258667],
    "gripper_vertical": [1.418738842010498, -1.2759456199458619, 1.2870314756976526, 1.557830734843872, 1.5609979629516602, 2.964608669281006],
    "gripper_down": [1.412574291229248, -1.2245715421489258, 1.5707538763629358, 1.2227379518696289, 1.5618265867233276, 2.960045576095581],
    "just_above_stirrer": [1.4119385480880737, -1.232708917265274, 1.5361388365374964, 1.2655228811451416, 1.5617036819458008, 2.9591991901397705],
    "in_front_of_white_bg": [0.9755352735519409, -1.2298626464656373, 1.5326388517962855, 1.2679366308399658, 1.5615804195404053, 2.5227737426757812],
    "above_home_holder": [1.7151169776916504, -0.9303394120982666, 0.9488051573382776, 1.4103073316761474, 1.499406337738037, 3.27805757522583],
    "readjustment": [1.7267637252807617, -0.8310088676265259, 0.8723586241351526, 1.5185677248188476, 1.5266249179840088, 3.2738213539123535],
    "return_to_holder": [1.7270193099975586, -0.7678989332965394, 1.0835111776935022, 1.2442163664051513, 1.527403473854065, 3.275404453277588]
    }

# Constants for movement
MOVEMENT_PARAMS = {
    "speed": 0.25,
    "acceleration": 0.5,
    "blending": 0.02,
}

CSV_FILE_PATH = os.path.expanduser("~/Code/chem504-2425_GroupA/examples/blue_pixel_data.csv")


def degreestorad(angles_deg):
    """Convert a list of angles from degrees to radians."""
    return [angle * (math.pi / 180) for angle in angles_deg]

def move_robot(robot, position):
    """Move robot to a specific joint position."""
    robot.move_joint_list(position, MOVEMENT_PARAMS["speed"], MOVEMENT_PARAMS["acceleration"], MOVEMENT_PARAMS["blending"])

def operate_gripper(gripper, position):
    """Operate the gripper: 0 for open, 140 for picking up, 255 for fully closed."""
    gripper.move(position, 125, 125)

def main():
    robot = URControl(ip="192.168.0.2", port=30003)
    gripper = RobotiqGripper()
    gripper.connect("192.168.0.2", 63352)

    # Open gripper initially
    operate_gripper(gripper, 0)  # Open the gripper
    
    # Perform robot movements and gripper actions
    move_robot(robot, ROBOT_POSITIONS["home"])
    print("Moved to home position")

    time.sleep(1)
    
    # Sample handling process
    operate_gripper(gripper, 0)  # Open gripper
    move_robot(robot, ROBOT_POSITIONS["sample_vial"])
    print("Moved to sample vial position")

    operate_gripper(gripper, 140)  # Close the gripper to pick up the sample (gripper at 140)

    time.sleep(1)
    move_robot(robot, ROBOT_POSITIONS["up_vertical"])
    print("Moved to up vertical position")

    time.sleep(1)
    move_robot(robot, ROBOT_POSITIONS["above_stirrer"])
    print("Moved to above stirrer position")

    time.sleep(1)
    move_robot(robot, ROBOT_POSITIONS["in_stirrer"])
    operate_gripper(gripper, 0)  # Open gripper to release the sample
    print("Moved to in stirrer position")

    time.sleep(1)
    move_robot(robot, ROBOT_POSITIONS["above_stirrer"])
    print("Moved back above stirrer")
    ######## Stirring process (30 seconds) update this please when it comes to it
    time.sleep(30)

    move_robot(robot, ROBOT_POSITIONS["gripper_down"])
    operate_gripper(gripper, 140)  # Close the gripper to pick up sample
    print("Moved to gripper down position")

    time.sleep(1)
    move_robot(robot, ROBOT_POSITIONS["just_above_stirrer"])
    print("Moved just above stirrer")
    
    time.sleep(1)
    move_robot(robot, ROBOT_POSITIONS["in_front_of_white_bg"])
    print("Moved to white background")

    #starts video capture
    cam = cv2.VideoCapture(0)

    cv2.namedWindow("test")
    cv2.namedWindow("Blue Mask")

    # Thresholds
    reaction_threshold = 50  # Total required blue pixels for successful detection
    blue_pixel_below_threshold_duration = 5  # Seconds below threshold before stopping

    # Track blue pixel count over time
    blue_pixel_counts = []

    # Define Region of Interest (ROI)
    roi_x, roi_y, roi_w, roi_h = 200, 150, 260, 280  

    # Track time
    start_time = time.time()
    last_logged_time = -1
    below_threshold_counter = 3

    # Optional: for dynamic threshold tuning
    def nothing(x): pass
    cv2.createTrackbar("Lower Hue", "Blue Mask", 100, 179, nothing)
    cv2.createTrackbar("Upper Hue", "Blue Mask", 130, 179, nothing)
    cv2.createTrackbar("Lower Sat", "Blue Mask", 150, 255, nothing)
    cv2.createTrackbar("Upper Sat", "Blue Mask", 255, 255, nothing)
    cv2.createTrackbar("Lower Val", "Blue Mask", 50, 255, nothing)
    cv2.createTrackbar("Upper Val", "Blue Mask", 255, 255, nothing)

    CSV_FILE_PATH = "blue_pixel_data.csv"

    try:
        while True:
            k = cv2.waitKey(1) & 0xFF
            if k == 27:  # ESC key
                print("Escape hit, closing...")
                break

            ret, frame = cam.read()
            if not ret:
                print("Failed to grab frame")
                break

            # Crop ROI and apply Gaussian blur
            roi_frame = frame[roi_y:roi_y+roi_h, roi_x:roi_x+roi_w]
            roi_frame = cv2.GaussianBlur(roi_frame, (5, 5), 0)

            # Convert to HSV
            hsv_roi = cv2.cvtColor(roi_frame, cv2.COLOR_BGR2HSV)

            # Get current HSV threshold values from trackbars
            lower_h = cv2.getTrackbarPos("Lower Hue", "Blue Mask")
            upper_h = cv2.getTrackbarPos("Upper Hue", "Blue Mask")
            lower_s = cv2.getTrackbarPos("Lower Sat", "Blue Mask")
            upper_s = cv2.getTrackbarPos("Upper Sat", "Blue Mask")
            lower_v = cv2.getTrackbarPos("Lower Val", "Blue Mask")
            upper_v = cv2.getTrackbarPos("Upper Val", "Blue Mask")

            lower_blue = np.array([lower_h, lower_s, lower_v])
            upper_blue = np.array([upper_h, upper_s, upper_v])

            # Create blue mask
            blue_mask = cv2.inRange(hsv_roi, lower_blue, upper_blue)

            # Morphological operations
            kernel = np.ones((3, 3), np.uint8)
            blue_mask = cv2.morphologyEx(blue_mask, cv2.MORPH_OPEN, kernel)
            blue_mask = cv2.morphologyEx(blue_mask, cv2.MORPH_CLOSE, kernel)

            # Count blue pixels
            blue_pixel_count = cv2.countNonZero(blue_mask)

            # Track time
            elapsed_seconds = int(time.time() - start_time)
            if elapsed_seconds > last_logged_time:
                last_logged_time = elapsed_seconds
                blue_pixel_counts.append((elapsed_seconds, blue_pixel_count))
                print(f"Time: {elapsed_seconds}s, Blue Pixels: {blue_pixel_count}")

            # Below-threshold handling
            if blue_pixel_count > 0:
                below_threshold_counter = 0
            else:
                below_threshold_counter += 1
                print(f"Blue below threshold for {below_threshold_counter} seconds.")
                if below_threshold_counter >= blue_pixel_below_threshold_duration:
                    print("The reaction is no longer blue. Stopping.")
                    break

            # Draw ROI and display
            cv2.rectangle(frame, (roi_x, roi_y), (roi_x + roi_w, roi_y + roi_h), (0, 255, 0), 2)
            cv2.imshow("test", frame)

            # Show the blue mask
            blue_mask_display = cv2.merge([blue_mask] * 3)
            cv2.imshow("Blue Mask", blue_mask_display)

    except Exception as e:
        print(f"Error occurred: {e}")

    finally:
        cam.release()
        cv2.destroyAllWindows()

        # Save data
        try:
            with open(CSV_FILE_PATH, mode='w', newline='') as file:
                writer = csv.writer(file)
                writer.writerow(["Time (Seconds)", "Blue Pixel Count"])
                writer.writerows(blue_pixel_counts)
            print(f"Data successfully saved in: {CSV_FILE_PATH}")
        except Exception as e:
            print(f"Error while saving CSV file: {e}")

        # Final decision
        if sum(count for _, count in blue_pixel_counts) >= reaction_threshold:
            print("Sample vial is blue")
        else:
            print("Sample vial did not reach the blue threshold")
            

    time.sleep(1)
    move_robot(robot, ROBOT_POSITIONS["readjustment"])
    print("readjustment")
    
    time.sleep(1)
    move_robot(robot, ROBOT_POSITIONS["return_to_holder"])
    print("Moved to return holder position")

    # Open the gripper after completion
    operate_gripper(gripper, 0)  # Open the gripper to finish


    # Call the function with the file path and other parameters
    df_result = process_and_plot_absorbance(CSV_FILE_PATH)

    # print the resulting DataFrame to see the processed data
    print(df_result)

if __name__ == "__main__":
    main()

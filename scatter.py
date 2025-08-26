# -*- coding: utf-8 -*-
"""
Created on Fri Aug 15 12:03:45 2025

@author: manny
"""
""" Real-Time CV-Laser Diffraction Analyser """
import cv2
import time
import random
import numpy as np
from collections import deque
import matplotlib.pyplot as plt
# from scipy.ndimage import center_of_mass
from skimage.measure import regionprops, label
from cv2_enumerate_cameras import enumerate_cameras
# from cv2.videoio_registry import getBackendName
# from cv2_enumerate_cameras import supported_backendspytho

# Variable declaration
# For region of interest
roi_w, roi_h = 640, 480     # Size of region to extract
brightness_threshold = 250  # Threshold for bright region (0–255)
history_length = 5  # Number of frames for moving average
history = deque(maxlen=history_length)  # History buffer for smoothing

# OPTICAL PARAMETERS
PIXEL_SIZE = 3.98          # in microns
DISTANCE_MM = 275          # distance to screen in mm

# --- Helper functions ---
# random colour generator for multiple plots
def random_color():
    return (random.random(), random.random(), random.random())

# Check connected USB cameras and their indexes
def camera_index():
    # For Windows OS
    devices = enumerate_cameras(cv2.CAP_DSHOW)
    # For Linux OS
    # devices = enumerate_cameras(cv2.CAP_GSTREAMER)
    for device in devices:
        print(f'{device.index}: {device.name}')
        # Search for specific device name, change depending on camera or USB device
        # if device.name == "Integrated Webcam":
        if device.name == "HD Pro Webcam C920":
            print(f'Found {device.name} on index {device.index}')
            return device.index
        else:
            print('No suitable camera found.')

# Find the diffraction center using intensity centroid
def find_center(gray):
    threshold = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY)[1]
    labeled = label(threshold)
    props = regionprops(labeled)
    if props:
        c = np.array(props[0].centroid[::-1])  # Reverse to (x, y)
    else:
        c = np.array(gray.shape[::-1]) / 2  # Default to image center
    return c

# Compute radial intensity profile
def plot_radial_intensity(image, center_x, center_y, angle_step=1):
   """
   Plots the radial intensity profile of an image.
   Args:
       image: A 2D NumPy array representing the image.
       center_x: The x-coordinate of the image center.
       center_y: The y-coordinate of the image center.
       angle_step: The step size for angles in degrees (e.g., 1 for 1-degree increments).
   """
   height, width = image.shape
   angles = np.arange(-180, 180, angle_step)
   radial_intensities = []
   
   for angle in angles:
       # Convert angle to radians
       angle_rad = np.radians(angle)
       
       # Calculate coordinates of a point on the radial line
       x = center_x + np.cos(angle_rad) * max(width, height)  # Extend to the image boundary
       y = center_y + np.sin(angle_rad) * max(width, height)
       
       # Extract intensity values along the line using interpolation 
       # (more robust for non-integer coordinates)
       x_coords, y_coords = np.linspace(center_x, x, num=max(width,height)), np.linspace(center_y, y, num=max(width,height)) # Changed num to max(width, height)
       try:
         # Linear interpolation using map_coordinates (better for non-integer coordinates)
         from scipy.ndimage import map_coordinates
         interpolated_values = map_coordinates(image, [y_coords, x_coords], order=1, mode='nearest')
         
         # Calculate the distance from the center to each point along the line
         distances = np.sqrt((x_coords - center_x)**2 + (y_coords - center_y)**2)
         
         # Filter out points beyond the image boundary
         valid_indices = distances <= min(width, height)
         interpolated_values = interpolated_values[valid_indices]
         distances = distances[valid_indices]
        #  print(distances)
         
         # Calculate the average intensity
         avg_intensity = np.mean(interpolated_values)
         
       except:
         avg_intensity = 0  # Handle cases where interpolation fails
        
       radial_intensities.append(avg_intensity)
       
   # Plot the radial intensity profile
   plt.figure(figsize=(10, 6))
   plt.plot(angles, radial_intensities)
   plt.xlabel("Angle (degrees)")
   plt.ylabel("Average Intensity")
   plt.title("Radial Intensity Profile")
   plt.grid(True)
   plt.show()
   
def compute_radial_profile(img):
    # Get image dimensions
    h, w = img.shape
    
    # Find center coordinates
    center = find_center(img)
    
    plot_radial_intensity(img, center[0], center[1], angle_step=1)
    
    # Create coordinate grid from the center
    y, x = np.indices((h, w))
    # center = center_of_mass(img)
    r = np.sqrt((x - center[1])**2 + (y - center[0])**2)
    r = r.astype(int)
    
    # Calculate the mean intensity for each radius
    tbin = np.bincount(r.ravel(), img.ravel())
    nr = np.bincount(r.ravel())
    radialprofile = tbin / np.maximum(nr, 1)
    
    # Pixel radii in meters
    pixel_r = np.arange(len(radialprofile))
    
    # Convert to scattering angles using θ = arctan(r/L)
    angle = np.arctan((pixel_r * PIXEL_SIZE) / DISTANCE_MM)
    angles_deg = np.degrees(angle)
    
    # Cross-sectional average intensity
    distribution = np.mean(img, axis=0)
    
    return angles_deg, radialprofile, distribution

# Visualise original image and pre-processing effects
def visualise(a, b):
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.title("Original Diffraction pattern")
    plt.imshow(cv2.cvtColor(a, cv2.COLOR_BGR2RGB))
    plt.axis()
    
    plt.subplot(1, 2, 2)
    plt.title("Centered Bright Region")
    plt.imshow(cv2.cvtColor(b, cv2.COLOR_BGR2RGB))
    plt.axis()
    plt.tight_layout()
    plt.show

# Plot update
def intensity_plots(angles, I, distro):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    ax1.clear()
    ax2.clear()
    
    # Cross-sectional intensity distribution
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    ax1.plot(distro, label='Avg. Intensity')
    ax1.set_title("Cross-sectional Intensity Profile", fontsize=14)
    ax1.set_xlabel("Pixels", fontsize=13)
    ax1.set_ylabel("Intensity", fontsize=13)
    
    ax1.set_xlim(xmin=0)
    ax1.legend()
    ax1.grid()
    
    # Angle-dependent Intensity distribution
    plt.subplot(1, 2, 1)
    ax2.plot(angles, I, label='Radial Intensity')
    ax2.set_title("Angular Intensity Profile", fontsize=14)
    ax2.set_xlabel("Scattering Angle (degrees)", fontsize=13)
    ax2.set_ylabel("Intensity", fontsize=13)
    
    ax2.set_xlim(xmin=0)
    ax2.set_ylim(ymin=0)
    ax2.legend()
    ax2.grid()
    plt.pause(0.01)
    plt.show

def cv_diffraction():
    cam_index = camera_index()  # RGB camera index

    # Initialize the video capture
    cap = cv2.VideoCapture(cam_index, cv2.CAP_DSHOW)
    if not cap.isOpened():
        print("Camera not accessible.")
        cap.release()
        cv2.destroyAllWindows()
        exit()

    # Initialize the plots
    # plt.ion()   # Turn on interactive mode

    start_time = time.time()
    
    try:
        print("Laser difrraction in progress...")
        while time.time() - start_time < 10:
            ret, frame = cap.read()
            if not ret:
                break
    
            # Convert to grayscale and apply blur to reduce pixel noise
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # Threshold bright regions
            _, thresh = cv2.threshold(gray, 
                brightness_threshold, 255, cv2.THRESH_BINARY)
        
            # Find centroid of bright region
            M = cv2.moments(thresh)
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
            else:
                # fallback: use max intensity pixel
                _, _, _, max_loc = cv2.minMaxLoc(gray)
                cx, cy = max_loc
        
            # Add to history and compute average position
            history.append((cx, cy))
            avg_cx = int(np.mean([pt[0] for pt in history]))
            avg_cy = int(np.mean([pt[1] for pt in history]))
        
            # Define ROI bounds around smoothed center
            x1 = max(avg_cx - roi_w // 2, 0)
            y1 = max(avg_cy - roi_h // 2, 0)
            x2 = min(x1 + roi_w, frame.shape[1])
            y2 = min(y1 + roi_h, frame.shape[0])
            roi = frame[y1:y2, x1:x2]
        
            # Create black canvas
            canvas = np.zeros_like(frame)
        
            # Compute center position on canvas
            canvas_cx, canvas_cy = canvas.shape[1] // 2, canvas.shape[0] // 2
            start_x = canvas_cx - roi.shape[1] // 2
            start_y = canvas_cy - roi.shape[0] // 2
        
            # Insert ROI in center of canvas
            canvas[start_y:start_y + roi.shape[0], start_x:start_x + roi.shape[1]] = roi
        
            # Draw rectangle on original frame (for visualization/debug)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            # Coordinates for the center of the dark spot (x, y)
            dark_spot_center = (cx,cy)
            dark_spot_radius = 25  # radius in pixels
        
            # Draw a solid black circle on the frame to block the beam spot
            cv2.circle(frame, dark_spot_center, dark_spot_radius,
                       (0, 0, 0), thickness=-1)
            
            gray1 = cv2.cvtColor(canvas, cv2.COLOR_BGR2GRAY)
            
            # get current time
            # current_time = time.time()
            
            # Run analysis every 5 seconds (can be changed)
            # if current_time - last_update_time >= 5:
  
 
            # Show webcam feed
            # Stack original and processed frames for side-by-side view
            stacked = np.hstack((frame, canvas))
            cv2.imshow("Laser Diffraction (Left) | Centered Bright Region (Right)", stacked)

            if cv2.waitKey(1) & 0xFF == 27:
                break
          
    finally:
        # Release resources and close windows
        cap.release()
        cv2.destroyAllWindows()
        # Turn off interactive mode
        # plt.ioff()
        # plt.show()

    return gray1, frame, canvas

# Main loop with webcam
def main():
    gray1, frame, canvas = cv_diffraction()

    # Step 1: Visualise image preprocessing effects
    visualise(frame, canvas)
    
    # Step 2: Compute angular intensity profile
    angles, intensity, distribution = compute_radial_profile(gray1)

    # Step 3: Plot results
    intensity_plots(angles, intensity, distribution)
    # last_update_time = current_time

def run_scatter():
    """Wrapper function so other scripts can run the whole workflow"""
    return main()   # just calls your existing main()

if __name__ == "__main__":
    run_scatter()


# Main workflow
if __name__ == "__main__":
    main()


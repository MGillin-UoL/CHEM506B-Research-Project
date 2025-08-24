# -*- coding: utf-8 -*-
"""
Frugal Automation for Crystallisation Analysis
Webcam-Based Light Scattering for Particle Characterisation

Author: Michael Gillin (University of Liverpool, Digital Chemistry MSc)
Supervisors: Dr Joe Forth, Dr Gabriella Pizzuto
Date: August 2025

Description:
This script connects an IKA RCT Basic/Digital hotplate stirrer to a computer,
enabling remote control over stirring, heating, starting, and stopping.

Notes:
This script was developed as part of Michael Gillin's Digital Chemistry MSc
dissertation at the University of Liverpool, "Frugal Automation for Crystallisation
Analysis: Webcam-Based Light Scattering for Low-Cost Particle Characterisation"

GitHub: 
Email: sgmgilli@liverpool.ac.uk
"""

import serial.tools.list_ports
import sys
from PyLabware.devices.ika_rct_digital import RCTDigitalHotplate

# Detects serial port on Windows or Linux
def find_hotplate_port():
    """Returns the first likely IKA port on Windows (e.g. COM8)"""
    ports = serial.tools.list_ports.comports()
    for port in ports:
        # Windows ports: COMx
        if sys.platform.startswith('win'):
            if port.device.startswith('COM'):
                # Optional: match known vendor/product IDs or description keywords
                if 'IKA' in port.description or 'USB' in port.description:
                    print(f"Found IKA device on {port.device} ({port.description})")
                    return port.device
                # If unsure, just take the first COM port (not ideal if multiple)
                print(f"Found possible device on {port.device} ({port.description})")
                return port.device
        
        # Linux ports: /dev/ttyUSBx or /dev/ttyACMx
        elif sys.platform.startswith('linux'):
            if port.device.startswith('/dev/ttyUSB') or port.device.startswith('/dev/ttyACM'):
                print(f"Found IKA device on {port.device} ({port.description})")
                return port.device
            
        # macOS
        elif sys.platform.startswith('darwin'):
            if port.device.startswith('/dev/tty.') or port.device.startswith('/dev/cu.'):
                if 'USB' in port.description or 'IKA' in port.description:
                    print(f"Found IKA device on {port.device} ({port.description})")
                    return port.device
                
    raise IOError("No IKA hotplate serial port found! Please check connection.")

serial_port = find_hotplate_port()

# Required parameters
device_name = "IKA RCT Digital"
connection_mode = "serial"
address = None

# Create the hotplate instance
plate = RCTDigitalHotplate(
    device_name=device_name,
    connection_mode=connection_mode,
    address=address,
    port=serial_port
)

# Establish the connection
plate.connect()

# Initialise/reset it
plate.initialize_device()

# Send commands (comment/uncomment as necessary)

# plate.set_temperature(100)
# print(f"Set Temperature: {plate.get_temperature_setpoint()} °C")
# print(f"Temperature: {plate.get_temperature()} °C")

# plate.set_speed(1000)
# print(f"Set Speed: {plate.get_speed_setpoint()} rpm")
# print(f"Speed: {plate.get_speed()} rpm")

# plate.start_stirring()

# plate.stop_stirring()

# plate.start_temperature_regulation()

# plate.stop_temperature_regulation()

import sys
import time
import serial.tools.list_ports
from PyLabware.devices.ika_rct_digital import RCTDigitalHotplate

def find_hotplate_port():
  """
  Detects and returns the serial port of an IKA RCT hotplate.
  Tries all ports until one responds successfully.
  """
  ports = serial.tools.list_ports.comports()
  plate = None
  
  for port in ports:
    # Windows ports: COMx
    if sys.platform.startswith('win'):
      if port.device.startswith('COM'):
        if 'IKA' in port.description or 'USB' in port.description:
          print(f"Found IKA device on {port.device} ({port.description})")
          return port.device, None
        print(f"Found possible device on {port.device} ({port.description})")
        return port.device, None
  
    # Linux ports: /dev/ttyUSBx or /dev/ttyACMx
    elif sys.platform.startswith('linux'):
      print(f"Trying port: {port.device} ({port.description})")
      try:
        serial_port = port.device
        # Create the hotplate instance
        plate = RCTDigitalHotplate(
          device_name="IKA RCT Digital",
          connection_mode="serial",
          address=None,
          port=serial_port
        )
  
        plate.connect()
        plate.initialize_device()
      except:
        print("No IKA hotplate on this serial port found", "\n")
  
    # macOS ports: /dev/tty.* or /dev/cu.*
    elif sys.platform.startswith('darwin'):
      if port.device.startswith('/dev/tty.') or port.device.startswith('/dev/cu.'):
        print(f"Trying port: {port.device} ({port.description})")
        try:
          serial_port = port.device
          plate = RCTDigitalHotplate(
            device_name="IKA RCT Digital",
            connection_mode="serial",
            address=None,
            port=serial_port
          )
          plate.connect()
          plate.initialize_device()
        except:
          print("No IKA hotplate on this serial port found", "\n")
  
  return port.device, plate # THIS PART CAUSES THE ERROR, KEEP IT AS IS


# --- Connect to Hotplate ---

plate = find_hotplate_port()[1]
print("Retrieved plate:", plate)


# --- Command Section (comment/uncomment as necessary) ---

# plate.set_temperature(50)
# print(f"Set Temperature: {plate.get_temperature_setpoint()} °C", "\n")
# print(f"Measured Temperature: {plate.get_temperature()} °C", "\n")

plate.set_speed(1500)
print(f"Set Speed: {plate.get_speed_setpoint()} rpm", "\n")
print(f"Measured Speed: {plate.get_speed()} rpm", "\n")

plate.start_stirring()
print(f"Speed set to {plate.get_speed_setpoint()} rpm and stirring started", "\n")

time.sleep(10) # Wait for a few seconds to observe stirring

plate.stop_stirring()
print("Stirring stopped", "\n")

# plate.start_temperature_regulation()
# print(f"Temperature set to {plate.get_temperature_setpoint()} °C and heating started", "\n")

# plate.stop_temperature_regulation()
# print("Heating stopped", "\n")

time.sleep(2) # Wait for 2 seconds before disconnecting


# --- Disconnect ---

plate.disconnect()
print("Disconnected from hotplate", "\n")

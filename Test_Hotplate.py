import serial.tools.list_ports
from PyLabware.devices.ika_rct_digital import RCTDigitalHotplate

# Detects serial port
def find_hotplate_port():
    """Return the first likely IKA port e.g. /dev/ttyACM0."""
    ports = serial.tools.list_ports.comports()
    for port in ports:
        if port.device.startswith('/dev/ttyACM') or port.device.startswith('/dev/ttyUSB'):
            print(f"Found device on {port.device}")
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

# Send commands
plate.set_temperature(50)
print(f"Temperature: {plate.get_temperature()} °C")

plate.set_speed(1500)
print(f"Speed: {plate.get_speed()} rpm")

plate.stop_stirring()

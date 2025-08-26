# -*- coding: utf-8 -*-
"""
Frugal Automation for Crystallisation Analysis
Webcam-Based Light Scattering for Particle Characterisation

Author: Michael Gillin (University of Liverpool, Digital Chemistry MSc)
Supervisors: Dr Joe Forth, Dr Gabriella Pizzuto
Date: August 2025

Description:
This script controls a UR5e robotic arm, a Robotiq Hand-E adaptive gripper, and
an IKA RCT Digital/Basic hotplate stirrer to conduct a solvent synthesis, which can
then be analysed using laser diffraction. This was designed using the workspace
available in the Stephenson Institute for Renewable Energy at the University of
Liverpool.

Notes:
This script was developed as part of Michael Gillin's Digital Chemistry MSc
dissertation at the University of Liverpool, "Frugal Automation for Crystallisation
Analysis: Webcam-Based Light Scattering for Low-Cost Particle Characterisation"

GitHub: https://github.com/MGillin-UoL
Email: sgmgilli@liverpool.ac.uk
"""

import os
import sys
import time
import math
import serial.tools.list_ports

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

import SIRE_Scatter2

from utils.UR_Functions import URfunctions as URControl
from robotiq.robotiq_gripper import RobotiqGripper
from PyLabware.devices.ika_rct_digital import RCTDigitalHotplate

 
# === Robot Configuration ===
ROBOT_POSITIONS = {
    "Front_Home_Position": [-0.09376556078066045, -0.9369700712016602, -1.6996105909347534, -3.5858966312804164, -1.6568916479693812, 3.1620473861694336],
    "Left_Home_Position": [1.5178043842315674, -0.937005953197815, -1.6996502876281738, -3.5857583485045375, -1.6568644682513636, 3.162015914916992],
    "SVH1_P1": [-0.34178239503969365, -2.517654081384176, -1.3193950653076172, -2.308610578576559, -0.3194788138019007, 3.0161139965057373], # Sample Vial Holder 1, Position 1
    "Above_SVH1_P1": [-0.3417742888080042, -2.2367712459959925, -1.1180524826049805, -2.7905599079527796, -0.3184884230243128, 3.015852928161621],
    "Just_Above_SVH1_P1": [-0.34179479280580694, -2.4304715595641078, -1.309723138809204, -2.4052349529662074, -0.3192938009845179, 3.015932321548462],
    "Next_To_Above_MTQ_P1": [-0.2731235663043421, -1.2992018324187775, -2.4014391899108887, -2.5230938396849574, -2.582552496586935, 3.184636116027832],
    "Above_MTQ_P1": [0.4903833270072937, -1.8673612080016078, -1.8238451480865479, -2.559404512444967, -1.8204062620746058, 3.141796112060547], # Mettler Toledo Quantos, Position 1
    "MTQ_P1": [0.4903712272644043, -2.0565864048399867, -1.9301269054412842, -2.263874670068258, -1.821090046559469, 3.1417758464813232],
    "Before_PCH_P1": [2.3522355556488037, -1.1998017591289063, -2.7459702491760254, -2.2998153171935023, -0.7677553335772913, 3.181354284286499], # Before Powder Cartridge Holder, Position 1
    "PCH_P1": [1.9516382217407227, -1.6935316524901332, -2.3674488067626953, -2.1936947307982386, -1.1689417997943323, 3.196838140487671],
    "Lift_PCH_P1": [1.9516072273254395, -1.6834732494749964, -2.3638932704925537, -2.2073213062682093, -1.1689093748675745, 3.196845293045044],
    "Far_Before_MTQ_Cartridge": [1.775646448135376, -2.1505099735655726, -2.2893424034118652, 1.2703043657490234, -0.19677478471864873, 0.061104536056518555],
    "Before_MTQ_Cartridge": [1.3382258415222168, -1.9477249584593714, -1.8765450716018677, -2.3511506519713343, -0.2623027006732386, 3.0951857566833496],
    "Immediately_Before_MTQ_Cartridge": [1.3061299324035645, -1.9782973728575648, -1.8608148097991943, -2.3477589092650355, -0.29426271120180303, 3.107100248336792],
    "Lift_MTQ_Cartridge": [1.2809028625488281, -1.9899574718871058, -1.8422791957855225, -2.362013956109518, -0.3193929831134241, 3.1148033142089844],
    "MTQ_Cartridge": [1.2809269428253174, -1.9944311581053675, -1.8446779251098633, -2.35512699703359, -0.319406811391012, 3.114818811416626],
    
    "Before_LCH_P1": [0.37961721420288086, -1.4504522916725655, -2.105588674545288, -2.72361483196401, -1.1972549597369593, 3.1729578971862793], # Before Liquid Cartridge Holder, Position 1
    "LCH_P1": [0.3450068235397339, -1.530487374668457, -2.0417842864990234, -2.7073332271971644, -1.2320283094989222, 3.172990083694458],
    "Lift_LCH_P1": [0.3450068235397339, -1.5271324676326294, -2.036325216293335, -2.716217180291647, -1.2320039908038538, 3.1730246543884277],
    
    "Just_Above_Hotplate": [-0.6458004156695765, -1.5638938637315114, -2.576573610305786, -2.068869730035299, -0.620941464100973, 3.086587905883789],
    "Above_Hotplate": [-0.6424158255206507, -1.1912538570216675, -2.304549217224121, -2.713623186151022, -0.6162975470172327, 3.086707592010498],
    "On_Hotplate": [-0.660140339528219, -1.625033517877096, -2.583313465118408, -2.002408643762106, -0.6353533903705042, 3.0883703231811523],
    "SVH1_P2": [-0.3924616018878382, -2.489788671533102, -1.3760035037994385, -2.3608724079527796, -0.7791598478900355, 3.0564565658569336],
    "Above_SVH1_P2": [-0.3852294127093714, -2.180908342401022, -1.14850914478302, -2.8968755207457484, -0.7708385626422327, 3.0558924674987793],
    "Just_Above_SVH1_P2": [-0.3853057066546839, -2.39510502437734, -1.376378059387207, -2.4547921619811, -0.7717660109149378, 3.0557847023010254],

    "Before_PCH_P2": [1.9712352752685547, -1.3218897146037598, -2.6846671104431152, -2.2479621372618617, -1.1490677038775843, 3.228564739227295],
    "Just_Before_PCH_P2": [1.7870339155197144, -1.679598947564596, -2.369499921798706, -2.20719637493276, -1.3337057272540491, 3.2336931228637695],
    "PCH_P2": [1.7769079208374023, -1.7120305500426234, -2.345625877380371, -2.198662420312399, -1.3438738028155726, 3.233949899673462],
    "Lift_PCH_P2": [1.7769079208374023, -1.7022444210448207, -2.342111587524414, -2.21199955562734, -1.3438380400287073, 3.233937978744507],
    "Down_PCH_P2": [1.7741049528121948, -1.715409894982809, -2.3471932411193848, -2.1937619648375453, -1.3467028776751917, 3.234041452407837],

    "Before_LCH_P2": [0.23276247084140778, -1.409643517141678, -2.1396472454071045, -2.73058619121694, -1.3440454641925257, 3.1735215187072754],
    "LCH_P2": [0.2011663168668747, -1.540644984026887, -2.0346217155456543, -2.7051907978453578, -1.37622577348818, 3.2222726345062256],
    "Lift_LCH_P2": [0.2011781930923462, -1.531740554874279, -2.019920825958252, -2.7289029560484828, -1.376162354146139, 3.222304105758667],
    "Down_LCH_P2": [0.20120887458324432, -1.5430927959135552, -2.0390331745147705, -2.69774927715444, -1.3758490721331995, 3.1734976768493652],

    "Above_SVH2_P1": [-0.48005420366396123, -1.8583980999388636, -1.6833781003952026, -2.643320222894186, -0.4554827849017542, 3.0585689544677734],
    "Just_Above_SVH2_P1": [-0.48005849519838506, -2.132143636743063, -1.8702583312988281, -2.1826840839781703, -0.4564884344684046, 3.0585689544677734],
    "SVH2_P1": [-0.4799750486956995, -2.256341596642965, -1.8823070526123047, -2.0465690098204554, -0.45677215257753545, 3.058764696121216],

    "Pipette_Holder_P1": [-0.41819841066469365, -2.5290938816466273, -1.0543980598449707, -2.6606956921019496, -1.9625042120562952, 3.1501612663269043],
    "Above_Pipette_Holder_P1": [-0.4181745688067835, -2.41787113765859, -0.9318845272064209, -2.8944245777525843, -1.9620550314532679, 3.1502132415771484],
    "Pipette_Above_SVH2_P1": [-1.0519540945636194, -2.086891313592428, -1.5786707401275635, -2.5696493587889613, -2.2993937174426478, 3.167163610458374],
    "Pipette_Just_In_SVH2_P1": [-1.041694466267721, -2.1203700504698695, -1.6124547719955444, -2.502833505670065, -2.2892852465258997, 3.1665141582489014],
    "Pipette_In_SVH2_P1": [-1.051969353352682, -2.234718462029928, -1.6549360752105713, -2.3455053768553675, -2.299900833760397, 3.1671910285949707],
    "Pipette_Near_Hotplate_Vial": [-0.91943866411318, -1.57354797939443, -2.139638900756836, -2.526546140710348, -2.166326347981588, 3.1596767902374268],
    "Pipette_Above_Hotplate_Vial": [-0.880564037953512, -1.3417670887759705, -2.201249122619629, -2.695799490014547, -0.9592760244952601, 3.110018730163574],
    "Pipette_In_Hotplate_Vial": [-0.8806036154376429, -1.3924806875041504, -2.271242618560791, -2.5750004253783167, -0.9595149199115198, 3.109926462173462],
    "Pipette_Above_Pipette_Bin": [-0.6214655081378382, -1.6172100506224574, -1.9516651630401611, -2.657548566857809, -0.700897518788473, 3.0920045375823975],
    "Pipette_In_Pipette_Bin": [-0.6215084234820765, -1.934718748132223, -2.1650853157043457, -2.126490732232565, -0.7020323912249964, 3.09183669090271],

    "Bend_1": [-0.07924396196474248, -1.9365974865355433, -1.1166322231292725, -2.9463712177672328, -1.6542423407184046, 3.162107467651367],
    "Bend_2": [-0.08581430116762334, -2.880160471002096, 0.7212498823748987, -2.901539464990133, -1.6539271513568323, 3.1620712280273438],
    "Bend_3": [-0.08561593690981084, -3.23695768932485, 2.5191178957568567, -2.90075745205068, -1.6536272207843226, 3.1620934009552],
    "Bend_4": [0.4484091103076935, -1.8120438061156214, 1.707728687916891, -2.9004746876158656, -1.6536029020892542, 3.1620712280273438],
    "Bend_5": [0.44842472672462463, -1.6935836277403773, 1.890376392995016, -4.7773479423918666, -1.6531713644610804, 3.1620712280273438],
    "Bend_6": [0.44826388359069824, -1.6931635342040003, 1.8634966055499476, -1.6312023601927699, -1.651621166859762, 3.1621789932250977],
    "Bend_7": [0.4482363164424896, -1.6931482754149378, 1.863542381917135, -1.6310907802977503, 1.635554313659668, 3.1621949672698975],
    "Bend_8": [0.44792458415031433, -1.6758815250792445, 1.8636601606952112, 0.21694104253735347, 1.635502576828003, 3.1621551513671875],
    "Bend_9": [0.525568962097168, -1.3377635043910523, 1.6177495161639612, -0.30527766168627934, -1.4652331511126917, 3.2085859775543213],
    "Bend_10": [0.5225677490234375, -1.3361430627158661, 1.6181829611407679, -0.346909837131836, -4.338824812565939, 4.872586727142334],
    "Bend_11": [0.5698254108428955, -1.4623436492732544, 2.079040829335348, -0.6842791003039856, -4.196372334157125, 3.0396065711975098],
    "Facing_Closed_Box": [-0.08608705202211553, -1.5961557827391566, 2.6305986086474817, -1.0374980133822937, -4.805425349866049, 3.096189260482788],
    "Facing_Open_Box": [0.5680867433547974, -1.0889963668635865, 1.9173124472247522, -0.832726852302887, -4.1507607142077845, 3.0929715633392334],

    "Box_Fully_Open": [0.507939338684082, -0.9684929412654419, 1.7020848433123987, -0.7379414600184937, -4.210777346287863, 3.092982530593872],
    "Box_3/4_Open": [0.3772026300430298, -1.0522657197764893, 1.852778736745016, -0.8043583196452637, -4.341596905385153, 3.093782424926758],
    "Box_1/2_Open": [0.25989431142807007, -1.1158338350108643, 1.9637325445758265, -0.8514501613429566, -4.458983246480123, 3.0943703651428223],
    "Box_1/4_Open": [0.09338236600160599, -1.1722186368754883, 2.059188191090719, -0.8901782792857666, -4.625538770352499, 3.095069646835327],
    "Box_Fully_Closed": [-0.02497417131532842, -1.206587867145874, 2.1157739798175257, -0.9122558397105713, -4.743896547948019, 3.0954971313476562],

    "SVH3_P1": [0.5987365245819092, -0.9929319185069581, 2.0656450430499476, -1.0797048074058075, -5.670722607766287, 3.1288177967071533],
    "Just_Above_SVH3_P1": [0.5987414717674255, -1.1386118990233918, 2.04608661333193, -0.9144613903811951, -5.670835498963491, 3.128770112991333],
    "Above_SVH3_P1": [0.5987197756767273, -1.329923854475357, 1.9546454588519495, -0.6314733189395447, -5.671102348958151, 3.1285338401794434],

    "Dilute_Sample_In_Box_Holder": [0.2244097888469696, -0.6542285245708008, 1.1413634459124964, -0.5238320392421265, -4.497173253689901, 3.1417722702026367],
    "Dilute_Sample_Above_Box_Holder": [0.2243943214416504, -0.6777724784663697, 1.1300938765155237, -0.4889703553966065, -4.497149531041281, 3.1417806148529053],
    "Dilute_Sample_Before_Box": [0.3238547146320343, -1.1832003456405182, 1.9276488463031214, -0.7960153383067627, -4.4144150654422205, 3.1422996520996094],

    "Near_Pipette_Holder_P2": [2.1543357372283936, -1.685329099694723, 2.0619681517230433, -0.3881797355464478, -4.806321326886312, 3.0961053371429443],
    "Pipette_Holder_P2": [2.307023048400879, -0.5393737119487305, 0.8936064879046839, -0.35599930704150395, -5.53933817545046, 3.0959291458129883],
    "Above_Pipette_Holder_P2": [2.3070194721221924, -0.6414576333812256, 0.6142099539386194, 0.025749965304992628, -5.539062563573019, 3.095592975616455],
    "Dilution_Pipette_Above_Hotplate_Sample_Vial": [1.4398818016052246, -1.7634946308531703, 2.190577809010641, -0.430060939197876, -4.824704710637228, 3.0955810546875],
    "Dilution_Pipette_Just_In_Hotplate_Sample_Vial": [1.4399192333221436, -1.7445036373534144, 2.218029324208395, -0.47649045408282475, -4.824666444455282, 3.0956292152404785],
    "Dilution_Pipette_In_Hotplate_Sample_Vial": [1.4400067329406738, -1.6957513294615687, 2.2733753363238733, -0.5806353849223633, -4.824500624333517, 3.095740795135498],
    "Dilution_Pipette_Above_SVH3_P1": [1.2315895557403564, -1.0465870064548035, 1.5367682615863245, -0.5023830694011231, -3.6871798674212855, 3.112401008605957],
    "Dilution_Pipette_Just_In_SVH3_P1": [1.24106764793396, -0.9889148038676758, 1.5025599638568323, -0.52619822443042, -3.677673403416769, 3.1120660305023193],
    "Dilution_Pipette_In_SVH3_P1": [1.2316241264343262, -1.003032462005951, 1.5729783217059534, -0.5821421903422852, -3.687171522771017, 3.1124773025512695],
    "Dilution_Pipette_Above_Pipette_Bin": [1.9177758693695068, -1.4740468201092263, 1.8912518660174769, -0.4210222524455567, -4.346394364033834, 3.093782424926758],
    "Dilution_Pipette_In_Pipette_Bin": [1.9179952144622803, -1.1700433057597657, 2.1019633452044886, -0.9356538218310853, -4.345923964177267, 3.094118595123291],

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


# Choices
# 1. Full synthesis using Mettler Toledo Quantos
# 2. Copper chloride and piroctone olamine solutions already prepared

# Choices once syngthesis is ready
# 1. Synthesise and measure one sample of piroctone olamine
# 2. Measure particle size distribution over time
# 3. Measure impact of copper chloride concentration on particle size distribution


# # === IKA RCT Digial Hotplate Configuration

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
                
    return port.device, plate   # THIS PART CAUSES THE ERROR, KEEP IT AS IS

# --- Connect to Hotplate ---

plate = find_hotplate_port()[1]
print("Retrieved plate:", plate)


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
    robot = URControl(ip="192.168.10.2", port=30003)
    gripper = RobotiqGripper()
    gripper.connect("192.168.10.2", 63352)

    # # Initial position
    # operate_gripper(gripper, 0)
    # move_robot(robot, ROBOT_POSITIONS["Front_Home_Position"])
    # print("Starting in home position", "\n")
    

    # # Loading Mettler Toledo Quantos with Sample Vial 1

    # operate_gripper(gripper, 0)
    # move_robot(robot, ROBOT_POSITIONS["Above_SVH1_P1"])
    # print("Moved to above Sample Vial Holder 1 Position 1", "\n")
    # time.sleep(1)

    # move_robot(robot, ROBOT_POSITIONS["Just_Above_SVH1_P1"])
    # print("Moved to just above Sample Vial Holder 1 Position 1", "\n")
    # time.sleep(1)

    # move_robot(robot, ROBOT_POSITIONS["SVH1_P1"])
    # print("Picked up sample vial from Sample Vial Holder 1 Position 1", "\n")
    # time.sleep(1)
    # operate_gripper(gripper, 125)
    # time.sleep(1)

    # move_robot(robot, ROBOT_POSITIONS["Just_Above_SVH1_P1"])
    # print("Moved to just above Sample Vial Holder 1 Position 1", "\n")
    # time.sleep(1)

    # move_robot(robot, ROBOT_POSITIONS["Above_SVH1_P1"])
    # print("Moved to above Sample Vial Holder 1 Position 1", "\n")
    # time.sleep(1)

    # move_robot(robot, ROBOT_POSITIONS["Front_Home_Position"])
    # print("Moving to home position", "\n")
    # time.sleep(1)
    
    # move_robot(robot, ROBOT_POSITIONS["Next_To_Above_MTQ_P1"])
    # print("Moving to next to above Mettler Toledo Quantos Position 1", "\n")
    # time.sleep(1)

    # move_robot(robot, ROBOT_POSITIONS["Above_MTQ_P1"])
    # print("Moved to above Mettler Toledo Quantos Position 1", "\n")
    # time.sleep(1)

    # move_robot(robot, ROBOT_POSITIONS["MTQ_P1"])
    # operate_gripper(gripper, 0)
    # print("Released sample into Mettler Toledo Quantos Position 1", "\n")
    # time.sleep(1)

    # move_robot(robot, ROBOT_POSITIONS["Above_MTQ_P1"])
    # print("Moved to above Mettler Toledo Quantos Position 1", "\n")
    # time.sleep(1)

    # move_robot(robot, ROBOT_POSITIONS["Next_To_Above_MTQ_P1"])
    # print("Moving to next to above Mettler Toledo Quantos Position 1", "\n")
    # time.sleep(1)

    # move_robot(robot, ROBOT_POSITIONS["Front_Home_Position"])
    # print("Moving to home position", "\n")
    # time.sleep(1)

    
    # Load Powder Cartridge 1 (Copper(II) Chloride)

    # move_robot(robot, ROBOT_POSITIONS["Left_Home_Position"])
    # print("Moved to left home position", "\n")
    # time.sleep(1)

    # move_robot(robot, ROBOT_POSITIONS["Before_PCH_P1"])
    # print("Moved just in front of Powder Cartridge Holder Position 1", "\n")
    # time.sleep(1)

    # move_robot(robot, ROBOT_POSITIONS["PCH_P1"])
    # print("Grabbed cartridge from Powder Cartridge Holder Position 1", "\n")
    # operate_gripper(gripper, 240)
    # time.sleep(1)

    # move_robot(robot, ROBOT_POSITIONS["Lift_PCH_P1"])
    # print("Lifted cartridge from Powder Cartridge Holder Position 1", "\n")
    # time.sleep(1)

    # move_robot(robot, ROBOT_POSITIONS["Before_PCH_P1"])
    # print("Moved just in front of Powder Cartridge Holder Position 1", "\n")
    # time.sleep(1)

    # move_robot(robot, ROBOT_POSITIONS["Left_Home_Position"])
    # print("Moved to left home position", "\n")
    # time.sleep(1)

    # move_robot(robot, ROBOT_POSITIONS["Before_MTQ_Cartridge"])
    # print("Moved in front of Mettler Toledo Quantos Cartridge loading area", "\n")
    # time.sleep(1)

    # move_robot(robot, ROBOT_POSITIONS["Immediately_Before_MTQ_Cartridge"])
    # print("Moved immediately in front of Mettler Toledo Quantos Cartridge loading area", "\n")
    # time.sleep(1)

    # move_robot(robot, ROBOT_POSITIONS["Lift_MTQ_Cartridge"])
    # print("Just above Mettler Toledo Quantos loading position", "\n")
    # time.sleep(1)

    # move_robot(robot, ROBOT_POSITIONS["MTQ_Cartridge"])
    # operate_gripper(gripper, 0)
    # print("Loaded Mettler Toledo Quantos cartridge", "\n")
    # time.sleep(1)

    # move_robot(robot, ROBOT_POSITIONS["Lift_MTQ_Cartridge"])
    # print("Just above Mettler Toledo Quantos loading position", "\n")
    # time.sleep(1)

    # move_robot(robot, ROBOT_POSITIONS["Immediately_Before_MTQ_Cartridge"])
    # print("Moved immediately in front of Mettler Toledo Quantos Cartridge loading area", "\n")
    # time.sleep(1)

    # move_robot(robot, ROBOT_POSITIONS["Before_MTQ_Cartridge"])
    # print("Moved back in front of Mettler Toledo Quantos Cartridge loading area", "\n")
    
    # time.sleep(5)
    
    # print("Dispensing copper (II) chloride powder into sample vial", "\n")
    # time.sleep(5)
    
    # move_robot(robot, ROBOT_POSITIONS["Before_MTQ_Cartridge"])
    # print("Moved in front of Mettler Toledo Quantos Cartridge loading area", "\n")
    # time.sleep(1)

    # move_robot(robot, ROBOT_POSITIONS["Immediately_Before_MTQ_Cartridge"])
    # print("Moved immediately in front of Mettler Toledo Quantos Cartridge loading area", "\n")
    # time.sleep(1)

    # move_robot(robot, ROBOT_POSITIONS["Lift_MTQ_Cartridge"])
    # print("Just above Mettler Toledo Quantos loading position", "\n")
    # time.sleep(1)

    # move_robot(robot, ROBOT_POSITIONS["MTQ_Cartridge"])
    # operate_gripper(gripper, 240)
    # print("Loaded Mettler Toledo Quantos cartridge", "\n")
    # time.sleep(1)

    # move_robot(robot, ROBOT_POSITIONS["Lift_MTQ_Cartridge"])
    # print("Just above Mettler Toledo Quantos loading position", "\n")
    # time.sleep(1)

    # move_robot(robot, ROBOT_POSITIONS["Immediately_Before_MTQ_Cartridge"])
    # print("Moved immediately in front of Mettler Toledo Quantos Cartridge loading area", "\n")
    # time.sleep(1)

    # move_robot(robot, ROBOT_POSITIONS["Before_MTQ_Cartridge"])
    # print("Moved back in front of Mettler Toledo Quantos Cartridge loading area", "\n")
    # time.sleep(1)

    # move_robot(robot, ROBOT_POSITIONS["Left_Home_Position"])
    # print("Moved to left home position", "\n")

    # move_robot(robot, ROBOT_POSITIONS["Before_PCH_P1"])
    # print("Moved just in front of Powder Cartridge Holder Position 1", "\n")

    # move_robot(robot, ROBOT_POSITIONS["Lift_PCH_P1"])
    # print("Lifted cartridge from Powder Cartridge Holder Position 1", "\n")
    # time.sleep(1)

    # move_robot(robot, ROBOT_POSITIONS["PCH_P1"])
    # operate_gripper(gripper, 0)
    # print("Returned cartridge to Powder Cartridge Holder Position 1", "\n")
    # time.sleep(1)

    # move_robot(robot, ROBOT_POSITIONS["Lift_PCH_P1"])
    # print("Lifted cartridge from Powder Cartridge Holder Position 1", "\n")
    # time.sleep(1)

    # move_robot(robot, ROBOT_POSITIONS["Before_PCH_P1"])
    # print("Moved just in front of Powder Cartridge Holder Position 1", "\n")

    # move_robot(robot, ROBOT_POSITIONS["Left_Home_Position"])
    # print("Moved to left home position", "\n")

    # # Load Liquid Cartridge 1

    # move_robot(robot, ROBOT_POSITIONS["Front_Home_Position"])
    # print("Moving to home position", "\n")

    # move_robot(robot, ROBOT_POSITIONS["Before_LCH_P1"])
    # print("Moved just in front of Liquid Cartridge Holder Position 1", "\n")
    # time.sleep(1)

    # move_robot(robot, ROBOT_POSITIONS["LCH_P1"])
    # print("Grabbed cartridge from Liquid Cartridge Holder Position 1", "\n")
    # operate_gripper(gripper, 240)
    # time.sleep(1)

    # move_robot(robot, ROBOT_POSITIONS["Lift_LCH_P1"])
    # print("Lifted cartridge from Liquid Cartridge Holder Position 1", "\n")
    # time.sleep(1)
    
    # move_robot(robot, ROBOT_POSITIONS["Before_LCH_P1"])
    # print("Moved just in front of Liquid Cartridge Holder Position 1", "\n")
    # time.sleep(1)

    # move_robot(robot, ROBOT_POSITIONS["Far_Before_MTQ_Cartridge"])
    # print("Moved far in front of Mettler Toledo Quantos Cartridge loading area", "\n")
    # time.sleep(1)

    # move_robot(robot, ROBOT_POSITIONS["Before_MTQ_Cartridge"])
    # print("Moved in front of Mettler Toledo Quantos Cartridge loading area", "\n")
    
    # move_robot(robot, ROBOT_POSITIONS["Immediately_Before_MTQ_Cartridge"])
    # print("Moved immediately in front of Mettler Toledo Quantos Cartridge loading area", "\n")
    # time.sleep(1)

    # move_robot(robot, ROBOT_POSITIONS["Lift_MTQ_Cartridge"])
    # print("Just above Mettler Toledo Quantos loading position", "\n")
    # time.sleep(1)

    # move_robot(robot, ROBOT_POSITIONS["MTQ_Cartridge"])
    # print("Loaded Mettler Toledo Quantos cartridge", "\n")
    # operate_gripper(gripper, 0)
    # time.sleep(1)

    # move_robot(robot, ROBOT_POSITIONS["Lift_MTQ_Cartridge"])
    # print("Just above Mettler Toledo Quantos loading position", "\n")
    # time.sleep(1)

    # move_robot(robot, ROBOT_POSITIONS["Immediately_Before_MTQ_Cartridge"])
    # print("Moved immediately in front of Mettler Toledo Quantos Cartridge loading area", "\n")
    # time.sleep(1)

    # move_robot(robot, ROBOT_POSITIONS["Before_MTQ_Cartridge"])
    # print("Moved back in front of Mettler Toledo Quantos Cartridge loading area", "\n")
    # time.sleep(1)

    # print("Dispensing water into sample vial", "\n")
    # time.sleep(5)

    # move_robot(robot, ROBOT_POSITIONS["Immediately_Before_MTQ_Cartridge"])
    # print("Moved immediately in front of Mettler Toledo Quantos Cartridge loading area", "\n")
    # time.sleep(1)

    # move_robot(robot, ROBOT_POSITIONS["MTQ_Cartridge"])
    # print("Grabbed Mettler Toledo Quantos cartridge", "\n")
    # operate_gripper(gripper, 240)
    # time.sleep(1)

    # move_robot(robot, ROBOT_POSITIONS["Lift_MTQ_Cartridge"])
    # print("Just above Mettler Toledo Quantos loading position", "\n")
    # time.sleep(1)

    # move_robot(robot, ROBOT_POSITIONS["Immediately_Before_MTQ_Cartridge"])
    # print("Moved immediately in front of Mettler Toledo Quantos Cartridge loading area", "\n")
    # time.sleep(1)

    # move_robot(robot, ROBOT_POSITIONS["Before_MTQ_Cartridge"])
    # print("Moved back in front of Mettler Toledo Quantos Cartridge loading area", "\n")
    # time.sleep(1)

    # move_robot(robot, ROBOT_POSITIONS["Far_Before_MTQ_Cartridge"])
    # print("Moved far in front of Mettler Toledo Quantos Cartridge loading area", "\n")
    # time.sleep(1)

    # move_robot(robot, ROBOT_POSITIONS["Before_LCH_P1"])
    # print("Moved just in front of Liquid Cartridge Holder Position 1", "\n")
    # time.sleep(1)

    # move_robot(robot, ROBOT_POSITIONS["Lift_LCH_P1"])
    # print("Moved just above Liquid Cartridge Holder Position 1", "\n")
    # time.sleep(1)

    # move_robot(robot, ROBOT_POSITIONS["LCH_P1"])
    # print("Returned cartridge to Liquid Cartridge Holder Position 1", "\n")
    # operate_gripper(gripper, 0)
    # time.sleep(1)

    # move_robot(robot, ROBOT_POSITIONS["Before_LCH_P1"])
    # print("Moved just in front of Liquid Cartridge Holder Position 1", "\n")
    # time.sleep(1)


    # # Putting Sample Vial 1 on the hotplate

    # move_robot(robot, ROBOT_POSITIONS["Front_Home_Position"])
    # print("Moving to home position", "\n")
    # operate_gripper(gripper, 0)
    # time.sleep(1)

    # move_robot(robot, ROBOT_POSITIONS["Next_To_Above_MTQ_P1"])
    # print("Moving to next to above Mettler Toledo Quantos Position 1", "\n")
    # time.sleep(1)

    # move_robot(robot, ROBOT_POSITIONS["Above_MTQ_P1"])
    # print("Moved to above Mettler Toledo Quantos Position 1", "\n")
    # time.sleep(1)

    # move_robot(robot, ROBOT_POSITIONS["MTQ_P1"])
    # operate_gripper(gripper, 125)
    # print("Grabbed sample from Mettler Toledo Quantos Position 1", "\n")
    # time.sleep(1)

    # move_robot(robot, ROBOT_POSITIONS["Above_MTQ_P1"])
    # print("Moved to above Mettler Toledo Quantos Position 1", "\n")
    # time.sleep(1)

    # move_robot(robot, ROBOT_POSITIONS["Next_To_Above_MTQ_P1"])
    # print("Moving to next to above Mettler Toledo Quantos Position 1", "\n")
    # time.sleep(1)

    # move_robot(robot, ROBOT_POSITIONS["Front_Home_Position"])
    # print("Moving to home position", "\n")
    # time.sleep(1)

    # move_robot(robot, ROBOT_POSITIONS["Above_Hotplate"])
    # print("Moved to above hotplate stirrer", "\n")
    # time.sleep(1)
    
    # move_robot(robot, ROBOT_POSITIONS["Just_Above_Hotplate"])
    # print("Moved to just above hotplate stirrer", "\n")
    # time.sleep(1)

    # move_robot(robot, ROBOT_POSITIONS["On_Hotplate"])
    # operate_gripper(gripper, 0)
    # print("Placed sample vial on hotplate stirrer", "\n")
    # time.sleep(1)
    
    # move_robot(robot, ROBOT_POSITIONS["Just_Above_Hotplate"])
    # print("Moved to just above hotplate stirrer", "\n")
    # time.sleep(1)

    # move_robot(robot, ROBOT_POSITIONS["Above_Hotplate"])
    # print("Moved to above hotplate stirrer", "\n")
    # time.sleep(1)
    
    # plate.set_speed(1000)
    # print(f"Speed: {plate.get_speed()} rpm")
    # plate.start_stirring()
    
    # print("Dissolving copper (II) chloride in water on hotplate")
    
    
    # # Loading Mettler Toledo Quantos with Sample Vial 2

    # operate_gripper(gripper, 0)
    # move_robot(robot, ROBOT_POSITIONS["Front_Home_Position"])
    # print("Starting in home position", "\n")

    # operate_gripper(gripper, 0)
    # move_robot(robot, ROBOT_POSITIONS["Above_SVH1_P2"])
    # print("Moved to above Sample Vial Holder 1 Position 2", "\n")
    # time.sleep(1)

    # move_robot(robot, ROBOT_POSITIONS["Just_Above_SVH1_P2"])
    # print("Moved to just above Sample Vial Holder 1 Position 2", "\n")
    # time.sleep(1)

    # move_robot(robot, ROBOT_POSITIONS["SVH1_P2"])
    # print("Picked up sample vial from Sample Vial Holder 1 Position 2", "\n")
    # time.sleep(1)
    # operate_gripper(gripper, 125)
    # time.sleep(1)

    # move_robot(robot, ROBOT_POSITIONS["Just_Above_SVH1_P2"])
    # print("Moved to just above Sample Vial Holder 1 Position 2", "\n")
    # time.sleep(1)

    # move_robot(robot, ROBOT_POSITIONS["Above_SVH1_P2"])
    # print("Moved to above Sample Vial Holder 1 Position 2", "\n")
    # time.sleep(1)

    # move_robot(robot, ROBOT_POSITIONS["Front_Home_Position"])
    # print("Moving to home position", "\n")
    # time.sleep(1)
    
    # move_robot(robot, ROBOT_POSITIONS["Next_To_Above_MTQ_P1"])
    # print("Moving to next to above Mettler Toledo Quantos Position 1", "\n")
    # time.sleep(1)

    # move_robot(robot, ROBOT_POSITIONS["Above_MTQ_P1"])
    # print("Moved to above Mettler Toledo Quantos Position 1", "\n")
    # time.sleep(1)
    
    # move_robot(robot, ROBOT_POSITIONS["MTQ_P1"])
    # time.sleep(1)
    # operate_gripper(gripper, 0)
    # print("Released sample into Mettler Toledo Quantos Position 1", "\n")
    # time.sleep(1)

    # move_robot(robot, ROBOT_POSITIONS["Above_MTQ_P1"])
    # print("Moved to above Mettler Toledo Quantos Position 1", "\n")
    # time.sleep(1)

    # move_robot(robot, ROBOT_POSITIONS["Next_To_Above_MTQ_P1"])
    # print("Moving to next to above Mettler Toledo Quantos Position 1", "\n")
    # time.sleep(1)

    # move_robot(robot, ROBOT_POSITIONS["Front_Home_Position"])
    # print("Moving to home position", "\n")
    # time.sleep(1)


    ## Load Powder Cartridge 2 (Piroctone Olamine)

    # move_robot(robot, ROBOT_POSITIONS["Left_Home_Position"])
    # print("Moved to left home position", "\n")
    # time.sleep(1)

    # move_robot(robot, ROBOT_POSITIONS["Before_PCH_P2"])
    # print("Moved in front of Powder Cartridge Holder Position 2", "\n")
    # time.sleep(1)

    # move_robot(robot, ROBOT_POSITIONS["Just_Before_PCH_P2"])
    # print("Moved just in front of Powder Cartridge Holder Position 2", "\n")
    # time.sleep(1)

    # move_robot(robot, ROBOT_POSITIONS["PCH_P2"])
    # print("Grabbed cartridge from Powder Cartridge Holder Position 2", "\n")
    # operate_gripper(gripper, 240)
    # time.sleep(1)

    # move_robot(robot, ROBOT_POSITIONS["Lift_PCH_P2"])
    # print("Lifted cartridge from Powder Cartridge Holder Position 2", "\n")
    # time.sleep(1)

    # move_robot(robot, ROBOT_POSITIONS["Just_Before_PCH_P2"])
    # print("Moved just in front of Powder Cartridge Holder Position 2", "\n")
    # time.sleep(1)

    # move_robot(robot, ROBOT_POSITIONS["Before_PCH_P2"])
    # print("Moved just in front of Powder Cartridge Holder Position 2", "\n")
    # time.sleep(1)

    # move_robot(robot, ROBOT_POSITIONS["Left_Home_Position"])
    # print("Moved to left home position", "\n")

    # move_robot(robot, ROBOT_POSITIONS["Before_MTQ_Cartridge"])
    # print("Moved in front of Mettler Toledo Quantos Cartridge loading area", "\n")
    # time.sleep(1)

    # move_robot(robot, ROBOT_POSITIONS["Immediately_Before_MTQ_Cartridge"])
    # print("Moved immediately in front of Mettler Toledo Quantos Cartridge loading area", "\n")
    # time.sleep(1)

    # move_robot(robot, ROBOT_POSITIONS["Lift_MTQ_Cartridge"])
    # print("Just above Mettler Toledo Quantos loading position", "\n")
    # time.sleep(1)

    # move_robot(robot, ROBOT_POSITIONS["MTQ_Cartridge"])
    # operate_gripper(gripper, 0)
    # print("Loaded Mettler Toledo Quantos cartridge", "\n")
    # time.sleep(1)

    # move_robot(robot, ROBOT_POSITIONS["Lift_MTQ_Cartridge"])
    # print("Just above Mettler Toledo Quantos loading position", "\n")
    # time.sleep(1)

    # move_robot(robot, ROBOT_POSITIONS["Immediately_Before_MTQ_Cartridge"])
    # print("Moved immediately in front of Mettler Toledo Quantos Cartridge loading area", "\n")
    # time.sleep(1)

    # move_robot(robot, ROBOT_POSITIONS["Before_MTQ_Cartridge"])
    # print("Moved back in front of Mettler Toledo Quantos Cartridge loading area", "\n")
    
    # time.sleep(5)
    
    # print("Dispensing piroctone olamine powder into sample vial", "\n")
    # time.sleep(5)
    
    # move_robot(robot, ROBOT_POSITIONS["Before_MTQ_Cartridge"])
    # print("Moved in front of Mettler Toledo Quantos Cartridge loading area", "\n")
    # time.sleep(1)

    # move_robot(robot, ROBOT_POSITIONS["Immediately_Before_MTQ_Cartridge"])
    # print("Moved immediately in front of Mettler Toledo Quantos Cartridge loading area", "\n")
    # time.sleep(1)

    # move_robot(robot, ROBOT_POSITIONS["Lift_MTQ_Cartridge"])
    # print("Just above Mettler Toledo Quantos loading position", "\n")
    # time.sleep(1)

    # move_robot(robot, ROBOT_POSITIONS["MTQ_Cartridge"])
    # operate_gripper(gripper, 240)
    # print("Loaded Mettler Toledo Quantos cartridge", "\n")
    # time.sleep(1)

    # move_robot(robot, ROBOT_POSITIONS["Lift_MTQ_Cartridge"])
    # print("Just above Mettler Toledo Quantos loading position", "\n")
    # time.sleep(1)

    # move_robot(robot, ROBOT_POSITIONS["Immediately_Before_MTQ_Cartridge"])
    # print("Moved immediately in front of Mettler Toledo Quantos Cartridge loading area", "\n")
    # time.sleep(1)

    # move_robot(robot, ROBOT_POSITIONS["Before_MTQ_Cartridge"])
    # print("Moved back in front of Mettler Toledo Quantos Cartridge loading area", "\n")
    # time.sleep(1)

    # move_robot(robot, ROBOT_POSITIONS["Left_Home_Position"])
    # print("Moved to left home position", "\n")
    # time.sleep(1)

    # move_robot(robot, ROBOT_POSITIONS["Before_PCH_P2"])
    # print("Moved in front of Powder Cartridge Holder Position 2", "\n")
    # time.sleep(1)

    # move_robot(robot, ROBOT_POSITIONS["Just_Before_PCH_P2"])
    # print("Moved just in front of Powder Cartridge Holder Position 2", "\n")
    # time.sleep(1)

    # move_robot(robot, ROBOT_POSITIONS["Lift_PCH_P2"])
    # print("Lifted cartridge from Powder Cartridge Holder Position 2", "\n")
    # time.sleep(1)

    # move_robot(robot, ROBOT_POSITIONS["PCH_P2"])
    # print("Grabbed cartridge from Powder Cartridge Holder Position 2", "\n")
    # time.sleep(1)

    # move_robot(robot, ROBOT_POSITIONS["Down_PCH_P2"])
    # print("Gave the cartridge a little shimmy")
    # operate_gripper(gripper, 0)
    # time.sleep(1)

    # move_robot(robot, ROBOT_POSITIONS["Lift_PCH_P2"])
    # print("Lifted cartridge from Powder Cartridge Holder Position 2", "\n")
    # time.sleep(1)

    # move_robot(robot, ROBOT_POSITIONS["Just_Before_PCH_P2"])
    # print("Moved just in front of Powder Cartridge Holder Position 2", "\n")
    # time.sleep(1)

    # move_robot(robot, ROBOT_POSITIONS["Before_PCH_P2"])
    # print("Moved just in front of Powder Cartridge Holder Position 2", "\n")
    # time.sleep(1)

    # move_robot(robot, ROBOT_POSITIONS["Left_Home_Position"])
    # print("Moved to left home position", "\n")


    # # Load Liquid Cartridge 2

    # move_robot(robot, ROBOT_POSITIONS["Front_Home_Position"])
    # print("Moving to home position", "\n")

    # move_robot(robot, ROBOT_POSITIONS["Before_LCH_P2"])
    # print("Moved just in front of Liquid Cartridge Holder Position 2", "\n")
    # time.sleep(1)

    # move_robot(robot, ROBOT_POSITIONS["LCH_P2"])
    # print("Grabbed cartridge from Liquid Cartridge Holder Position 2", "\n")
    # operate_gripper(gripper, 240)
    # time.sleep(1)

    # move_robot(robot, ROBOT_POSITIONS["Lift_LCH_P2"])
    # print("Lifted cartridge from Liquid Cartridge Holder Position 2", "\n")
    # time.sleep(1)
    
    # move_robot(robot, ROBOT_POSITIONS["Before_LCH_P2"])
    # print("Moved just in front of Liquid Cartridge Holder Position 2", "\n")
    # time.sleep(1)

    # move_robot(robot, ROBOT_POSITIONS["Far_Before_MTQ_Cartridge"])
    # print("Moved far in front of Mettler Toledo Quantos Cartridge loading area", "\n")
    # time.sleep(1)

    # move_robot(robot, ROBOT_POSITIONS["Before_MTQ_Cartridge"])
    # print("Moved in front of Mettler Toledo Quantos Cartridge loading area", "\n")
    
    # move_robot(robot, ROBOT_POSITIONS["Immediately_Before_MTQ_Cartridge"])
    # print("Moved immediately in front of Mettler Toledo Quantos Cartridge loading area", "\n")
    # time.sleep(1)

    # move_robot(robot, ROBOT_POSITIONS["Lift_MTQ_Cartridge"])
    # print("Just above Mettler Toledo Quantos loading position", "\n")
    # time.sleep(1)

    # move_robot(robot, ROBOT_POSITIONS["MTQ_Cartridge"])
    # print("Loaded Mettler Toledo Quantos cartridge", "\n")
    # operate_gripper(gripper, 0)
    # time.sleep(1)

    # move_robot(robot, ROBOT_POSITIONS["Lift_MTQ_Cartridge"])
    # print("Just above Mettler Toledo Quantos loading position", "\n")
    # time.sleep(1)

    # move_robot(robot, ROBOT_POSITIONS["Immediately_Before_MTQ_Cartridge"])
    # print("Moved immediately in front of Mettler Toledo Quantos Cartridge loading area", "\n")
    # time.sleep(1)

    # move_robot(robot, ROBOT_POSITIONS["Before_MTQ_Cartridge"])
    # print("Moved back in front of Mettler Toledo Quantos Cartridge loading area", "\n")
    # time.sleep(1)

    # print("Dispensing water into sample vial", "\n")
    # time.sleep(5)

    # move_robot(robot, ROBOT_POSITIONS["Immediately_Before_MTQ_Cartridge"])
    # print("Moved immediately in front of Mettler Toledo Quantos Cartridge loading area", "\n")
    # time.sleep(1)

    # move_robot(robot, ROBOT_POSITIONS["MTQ_Cartridge"])
    # print("Grabbed Mettler Toledo Quantos cartridge", "\n")
    # operate_gripper(gripper, 240)
    # time.sleep(1)

    # move_robot(robot, ROBOT_POSITIONS["Lift_MTQ_Cartridge"])
    # print("Just above Mettler Toledo Quantos loading position", "\n")
    # time.sleep(1)

    # move_robot(robot, ROBOT_POSITIONS["Immediately_Before_MTQ_Cartridge"])
    # print("Moved immediately in front of Mettler Toledo Quantos Cartridge loading area", "\n")
    # time.sleep(1)

    # move_robot(robot, ROBOT_POSITIONS["Before_MTQ_Cartridge"])
    # print("Moved back in front of Mettler Toledo Quantos Cartridge loading area", "\n")
    # time.sleep(1)

    # move_robot(robot, ROBOT_POSITIONS["Far_Before_MTQ_Cartridge"])
    # print("Moved far in front of Mettler Toledo Quantos Cartridge loading area", "\n")
    # time.sleep(1)

    # move_robot(robot, ROBOT_POSITIONS["Before_LCH_P2"])
    # print("Moved just in front of Liquid Cartridge Holder Position 2", "\n")
    # time.sleep(1)

    # move_robot(robot, ROBOT_POSITIONS["Lift_LCH_P2"])
    # print("Moved just above Liquid Cartridge Holder Position 2", "\n")
    # time.sleep(1)

    # move_robot(robot, ROBOT_POSITIONS["LCH_P2"])
    # print("Returned cartridge to Liquid Cartridge Holder Position 2", "\n")
    # time.sleep(1)

    # move_robot(robot, ROBOT_POSITIONS["Down_LCH_P2"])
    # print("Yoinked cartridge into position")
    # operate_gripper(gripper, 0)
    # time.sleep(1)

    # move_robot(robot, ROBOT_POSITIONS["Before_LCH_P2"])
    # print("Moved just in front of Liquid Cartridge Holder Position 2", "\n")
    # time.sleep(1)


    ## Moving Sample Vial 1 from hotplate stirrer to Sample Vial Holder 2

    # plate.stop_stirring()

    # move_robot(robot, ROBOT_POSITIONS["Front_Home_Position"])
    # print("Moving to home position", "\n")

    # move_robot(robot, ROBOT_POSITIONS["Above_Hotplate"])
    # print("Moved to above hotplate stirrer", "\n")
    # time.sleep(1)
    
    # move_robot(robot, ROBOT_POSITIONS["Just_Above_Hotplate"])
    # print("Moved to just above hotplate stirrer", "\n")
    # time.sleep(1)

    # move_robot(robot, ROBOT_POSITIONS["On_Hotplate"])
    # operate_gripper(gripper, 125)
    # print("Grabbed sample vial from hotplate stirrer", "\n")
    # time.sleep(1)
    
    # move_robot(robot, ROBOT_POSITIONS["Just_Above_Hotplate"])
    # print("Moved to just above hotplate stirrer", "\n")
    # time.sleep(1)

    # move_robot(robot, ROBOT_POSITIONS["Above_Hotplate"])
    # print("Moved to above hotplate stirrer", "\n")
    # time.sleep(1)

    # move_robot(robot, ROBOT_POSITIONS["Above_SVH2_P1"])
    # print("Moved to above Sample Vial Holder 2 Position 1", "\n")
    # time.sleep(1)

    # move_robot(robot, ROBOT_POSITIONS["Just_Above_SVH2_P1"])
    # print("Moved to just above Sample Vial Holder 2 Position 1", "\n")
    # time.sleep(1)

    # move_robot(robot, ROBOT_POSITIONS["SVH2_P1"])
    # operate_gripper(gripper, 0)
    # print("Inserted Sample Vial 1 into Sample Vial Holder 2 Position 1", "\n")
    # time.sleep(1)

    # move_robot(robot, ROBOT_POSITIONS["Just_Above_SVH2_P1"])
    # print("Moved to just above Sample Vial Holder 2 Position 1", "\n")
    # time.sleep(1)

    # move_robot(robot, ROBOT_POSITIONS["Above_SVH2_P1"])
    # print("Moved to above Sample Vial Holder 2 Position 1", "\n")
    # time.sleep(1)

    # # Putting Sample Vial 2 on the hotplate

    # move_robot(robot, ROBOT_POSITIONS["Front_Home_Position"])
    # print("Moving to home position", "\n")
    # operate_gripper(gripper, 0)
    # time.sleep(1)

    # move_robot(robot, ROBOT_POSITIONS["Next_To_Above_MTQ_P1"])
    # print("Moving to next to above Mettler Toledo Quantos Position 1", "\n")
    # time.sleep(1)

    # move_robot(robot, ROBOT_POSITIONS["Above_MTQ_P1"])
    # print("Moved to above Mettler Toledo Quantos Position 1", "\n")
    # time.sleep(1)

    # move_robot(robot, ROBOT_POSITIONS["MTQ_P1"])
    # operate_gripper(gripper, 125)
    # print("Grabbed sample from Mettler Toledo Quantos Position 1", "\n")
    # time.sleep(1)

    # move_robot(robot, ROBOT_POSITIONS["Above_MTQ_P1"])
    # print("Moved to above Mettler Toledo Quantos Position 1", "\n")
    # time.sleep(1)

    # move_robot(robot, ROBOT_POSITIONS["Next_To_Above_MTQ_P1"])
    # print("Moving to next to above Mettler Toledo Quantos Position 1", "\n")
    # time.sleep(1)

    # move_robot(robot, ROBOT_POSITIONS["Front_Home_Position"])
    # print("Moving to home position", "\n")
    # time.sleep(1)

    # move_robot(robot, ROBOT_POSITIONS["Above_Hotplate"])
    # print("Moved to above hotplate stirrer", "\n")
    # time.sleep(1)
    
    # move_robot(robot, ROBOT_POSITIONS["Just_Above_Hotplate"])
    # print("Moved to just above hotplate stirrer", "\n")
    # time.sleep(1)

    # move_robot(robot, ROBOT_POSITIONS["On_Hotplate"])
    # operate_gripper(gripper, 0)
    # print("Placed sample vial on hotplate stirrer", "\n")
    # time.sleep(1)
    
    # move_robot(robot, ROBOT_POSITIONS["Just_Above_Hotplate"])
    # print("Moved to just above hotplate stirrer", "\n")
    # time.sleep(1)

    # move_robot(robot, ROBOT_POSITIONS["Above_Hotplate"])
    # print("Moved to above hotplate stirrer", "\n")
    # time.sleep(1)
    
    # plate.set_speed(1000)
    # print(f"Speed: {plate.get_speed()} rpm")
    # plate.start_stirring()
    
    # # print("Dissolving piroctone olamine in ethanol on hotplate")

    
    # # Liquid dispensing
    
    # move_robot(robot, ROBOT_POSITIONS["Above_Pipette_Holder_P1"])
    # print("Above Pipette Holder Position 1", "\n")
    # time.sleep(1)
    # operate_gripper(gripper, 180)
    
    # move_robot(robot, ROBOT_POSITIONS["Pipette_Holder_P1"])
    # print("Picked up pipette from Pipette Holder Position 1", "\n")
    # time.sleep(1)
    # operate_gripper(gripper, 200)
    # time.sleep(1)

    # move_robot(robot, ROBOT_POSITIONS["Above_Pipette_Holder_P1"])
    # print("Above Pipette Holder Position 1", "\n")
    # time.sleep(1)

    # for cycle in range(5):  # Repeat the full pipetting routine 1,2,3,4,5 times
    #     print(f"\n--- Pipetting cycle {cycle + 1}/5 ---")

    #     move_robot(robot, ROBOT_POSITIONS["Pipette_Above_SVH2_P1"])
    #     print("Above Sample Vial Holder 2 Position 1", "\n")
    #     time.sleep(1)

    #     move_robot(robot, ROBOT_POSITIONS["Pipette_Just_In_SVH2_P1"])
    #     print("Just inside Sample Vial Holder 2 Position 1", "\n")
    #     time.sleep(1)
    #     operate_gripper(gripper, 240)
    #     time.sleep(1)
        
    #     for position in range(235, 199, -5):  # Loop from 235 to 200 inclusive
    #         move_robot(robot, ROBOT_POSITIONS["Pipette_In_SVH2_P1"])
    #         print("Sucking up copper chloride solution", "\n")
    #         operate_gripper(gripper, position)
    #         time.sleep(1)
        
        
    #     move_robot(robot, ROBOT_POSITIONS["Pipette_Above_SVH2_P1"])
    #     print("Above Sample Vial Holder 2 Position 1", "\n")
    #     time.sleep(1)

    #     move_robot(robot, ROBOT_POSITIONS["Pipette_Near_Hotplate_Vial"])
    #     print("Pipette on its way to the hotplate sample vial")
    #     time.sleep(1)

    #     move_robot(robot, ROBOT_POSITIONS["Pipette_Above_Hotplate_Vial"])
    #     print("Above sample vial on the hotplate", "\n")
    #     time.sleep(1)
        
    #     move_robot(robot, ROBOT_POSITIONS["Pipette_In_Hotplate_Vial"])
    #     print("In sample vial on the hotplate", "\n")
    #     time.sleep(1)
        
    #     for position in range(201, 241, 1):  # Loop from 201 to 240 inclusive
    #         move_robot(robot, ROBOT_POSITIONS["Pipette_In_Hotplate_Vial"])
    #         print("Dispensing copper chloride solution into piroctone olamine solution", "\n")
    #         operate_gripper(gripper, position)
    #         time.sleep(3)
        
        
    #     for position in range(240, 199, -5):  # Loop from 240 to 200 inclusive
    #         move_robot(robot, ROBOT_POSITIONS["Pipette_Above_Hotplate_Vial"])
    #         print("Unsqueezing pipette", "\n")
    #         operate_gripper(gripper, position)
    #         time.sleep(1)
    
    #     move_robot(robot, ROBOT_POSITIONS["Pipette_Near_Hotplate_Vial"])
    #     print("Pipette on its way to the sample vial")
    #     time.sleep(1)
    

    
    # move_robot(robot, ROBOT_POSITIONS["Pipette_Above_Pipette_Bin"])
    # print("Say goodbye to this pipette!", "\n")
    # time.sleep(1)
    
    # move_robot(robot, ROBOT_POSITIONS["Pipette_In_Pipette_Bin"])
    # print("Yeet!", "\n")
    # operate_gripper(gripper, 0)
    # time.sleep(1)
    
    # move_robot(robot, ROBOT_POSITIONS["Pipette_Above_Pipette_Bin"])
    # print("Moved above pipette bin", "\n")
    # time.sleep(1)

    # move_robot(robot, ROBOT_POSITIONS["Front_Home_Position"])
    # print("Starting in home position", "\n")
    # time.sleep(1)

    # time.sleep(7200)
    # plate.stop_stirring()


    # Move to box

    # move_robot(robot, ROBOT_POSITIONS["Bend_1"])
    # move_robot(robot, ROBOT_POSITIONS["Bend_2"])
    # move_robot(robot, ROBOT_POSITIONS["Bend_3"])
    # move_robot(robot, ROBOT_POSITIONS["Bend_4"])
    # move_robot(robot, ROBOT_POSITIONS["Bend_5"])
    # move_robot(robot, ROBOT_POSITIONS["Bend_6"])
    # move_robot(robot, ROBOT_POSITIONS["Bend_7"])
    # move_robot(robot, ROBOT_POSITIONS["Bend_8"])
    # move_robot(robot, ROBOT_POSITIONS["Bend_9"])
    # move_robot(robot, ROBOT_POSITIONS["Bend_10"])
    # move_robot(robot, ROBOT_POSITIONS["Bend_11"])
    # operate_gripper(gripper, 0)
    # move_robot(robot, ROBOT_POSITIONS["Facing_Closed_Box"])
    # time.sleep(1)


    # # Diluting sample
    
    # move_robot(robot, ROBOT_POSITIONS["Near_Pipette_Holder_P2"])
    # print("Near Pipette Holder Position 2", "\n")
    # time.sleep(1)

    # move_robot(robot, ROBOT_POSITIONS["Above_Pipette_Holder_P2"])
    # print("Above Pipette Holder Position 2", "\n")
    # time.sleep(1)

    # operate_gripper(gripper, 180)
    # move_robot(robot, ROBOT_POSITIONS["Pipette_Holder_P2"])
    # print("Picked up pipette from Pipette Holder Position 2", "\n")
    # time.sleep(1)

    # operate_gripper(gripper, 200)
    # time.sleep(1)

    # move_robot(robot, ROBOT_POSITIONS["Above_Pipette_Holder_P2"])
    # print("Above Pipette Holder Position 2", "\n")
    # time.sleep(1)

    # move_robot(robot, ROBOT_POSITIONS["Near_Pipette_Holder_P2"])
    # print("Near Pipette Holder Position 2", "\n")
    # time.sleep(1)

    # move_robot(robot, ROBOT_POSITIONS["Dilution_Pipette_Above_Hotplate_Sample_Vial"])
    # print("Above sample vial on the hotplate", "\n")
    # time.sleep(1)

    # move_robot(robot, ROBOT_POSITIONS["Dilution_Pipette_Just_In_Hotplate_Sample_Vial"])
    # print("Just in sample vial on the hotplate", "\n")
    # time.sleep(1)
    # operate_gripper(gripper, 220)
    # time.sleep(1)

    # move_robot(robot, ROBOT_POSITIONS["Dilution_Pipette_In_Hotplate_Sample_Vial"])
    # time.sleep(1)
    # operate_gripper(gripper, 219)
    # print("Sucking up copper-piroctone solution", "\n")
    # time.sleep(1)
    
    # move_robot(robot, ROBOT_POSITIONS["Dilution_Pipette_Above_Hotplate_Sample_Vial"])
    # print("Above sample vial on the hotplate", "\n")
    # time.sleep(1)

    # move_robot(robot, ROBOT_POSITIONS["Dilution_Pipette_Above_SVH3_P1"])
    # print("Above Sample Vial Holder 3 Position 1", "\n")
    # time.sleep(1)

    # move_robot(robot, ROBOT_POSITIONS["Dilution_Pipette_Just_In_SVH3_P1"])
    # print("In Sample Vial Holder 3 Position 1", "\n")
    # time.sleep(1)

    # move_robot(robot, ROBOT_POSITIONS["Dilution_Pipette_In_SVH3_P1"])
    # print("In Sample Vial Holder 3 Position 1", "\n")
    # time.sleep(1)
    # operate_gripper(gripper, 220)
    # print("Dispensing copper piroctone solution in ethanol-water dispersant", "\n")

    # for position in range(220, 199, -5):  # Loop from 220 to 200 inclusive
    #     move_robot(robot, ROBOT_POSITIONS["Dilution_Pipette_Above_SVH3_P1"])
    #     print("Unsqueezing pipette", "\n")
    #     operate_gripper(gripper, position)
    #     time.sleep(1)

    # move_robot(robot, ROBOT_POSITIONS["Dilution_Pipette_Above_SVH3_P1"])
    # print("Above Sample Vial Holder 3 Position 1", "\n")
    # time.sleep(1)

    # move_robot(robot, ROBOT_POSITIONS["Dilution_Pipette_Above_Pipette_Bin"])
    # print("Dilution pipette is above pipette bin", "\n")
    # time.sleep(1)

    # move_robot(robot, ROBOT_POSITIONS["Dilution_Pipette_In_Pipette_Bin"])
    # print("Dilution pipette is in pipette bin", "\n")
    # operate_gripper(gripper, 0)
    # time.sleep(1)
    
    # move_robot(robot, ROBOT_POSITIONS["Dilution_Pipette_Above_Pipette_Bin"])
    # print("Moved above pipette bin", "\n")
    # time.sleep(1)


    # # Opening box for laser diffraction

    # operate_gripper(gripper, 0)
    # move_robot(robot, ROBOT_POSITIONS["Facing_Closed_Box"])
    # time.sleep(1)

    # move_robot(robot, ROBOT_POSITIONS["Box_Fully_Closed"])
    # time.sleep(1)
    # operate_gripper(gripper, 255)
    # print("Box is fully closed", "\n")
    # time.sleep(1)

    # move_robot(robot, ROBOT_POSITIONS["Box_1/4_Open"])
    # print("Box is 1/4 open", "\n")
    # time.sleep(1)

    # move_robot(robot, ROBOT_POSITIONS["Box_1/2_Open"])
    # print("Box is 1/2 open", "\n")
    # time.sleep(1)

    # move_robot(robot, ROBOT_POSITIONS["Box_3/4_Open"])
    # print("Box is 3/4 open", "\n")
    # time.sleep(1)

    # move_robot(robot, ROBOT_POSITIONS["Box_Fully_Open"])
    # print("Box is fully open", "\n")
    # time.sleep(1)
    # operate_gripper(gripper, 0)
    # time.sleep(1)

    # move_robot(robot, ROBOT_POSITIONS["Facing_Open_Box"])
    # time.sleep(1)

    # operate_gripper(gripper, 0)
    # move_robot(robot, ROBOT_POSITIONS["Facing_Closed_Box"])
    # time.sleep(1)


    # # Putting sample vial in box

    # move_robot(robot, ROBOT_POSITIONS["Above_SVH3_P1"])
    # time.sleep(1)
    # move_robot(robot, ROBOT_POSITIONS["Just_Above_SVH3_P1"])
    # time.sleep(1)
    # move_robot(robot, ROBOT_POSITIONS["SVH3_P1"])
    # time.sleep(1)
    # operate_gripper(gripper, 125)
    # time.sleep(1)
    # move_robot(robot, ROBOT_POSITIONS["Just_Above_SVH3_P1"])
    # time.sleep(1)
    # move_robot(robot, ROBOT_POSITIONS["Above_SVH3_P1"])
    # time.sleep(1)
    # move_robot(robot, ROBOT_POSITIONS["Facing_Closed_Box"])
    # time.sleep(1)
    # move_robot(robot, ROBOT_POSITIONS["Dilute_Sample_Before_Box"])
    # time.sleep(1)
    # move_robot(robot, ROBOT_POSITIONS["Dilute_Sample_Above_Box_Holder"])
    # time.sleep(1)
    # move_robot(robot, ROBOT_POSITIONS["Dilute_Sample_In_Box_Holder"])
    # time.sleep(1)
    # operate_gripper(gripper, 0)
    # time.sleep(1)
    # move_robot(robot, ROBOT_POSITIONS["Dilute_Sample_Above_Box_Holder"])
    # time.sleep(1)
    # move_robot(robot, ROBOT_POSITIONS["Dilute_Sample_Before_Box"])
    # time.sleep(1)
    # move_robot(robot, ROBOT_POSITIONS["Facing_Closed_Box"])
    # time.sleep(1)

    # # Closing box for laser diffraction

    # move_robot(robot, ROBOT_POSITIONS["Facing_Open_Box"])
    # time.sleep(1)

    # move_robot(robot, ROBOT_POSITIONS["Box_Fully_Open"])
    # print("Box is fully open", "\n")
    # time.sleep(1)
    # operate_gripper(gripper, 255)
    # time.sleep(1)

    # move_robot(robot, ROBOT_POSITIONS["Box_3/4_Open"])
    # print("Box is 3/4 open", "\n")
    # time.sleep(1)

    # move_robot(robot, ROBOT_POSITIONS["Box_1/2_Open"])
    # print("Box is 1/2 open", "\n")
    # time.sleep(1)

    # move_robot(robot, ROBOT_POSITIONS["Box_1/4_Open"])
    # print("Box is 1/4 open", "\n")
    # time.sleep(1)

    # move_robot(robot, ROBOT_POSITIONS["Box_Fully_Closed"])
    # print("Box is fully closed", "\n")
    # time.sleep(1)
    # operate_gripper(gripper, 0)
    # time.sleep(1)

    # move_robot(robot, ROBOT_POSITIONS["Facing_Closed_Box"])
    # time.sleep(1)


    # Run the scatter workflow
    scatter.run_scatter()


    # # Opening box after laser diffraction

    # operate_gripper(gripper, 0)
    # move_robot(robot, ROBOT_POSITIONS["Facing_Closed_Box"])
    # time.sleep(1)
    # move_robot(robot, ROBOT_POSITIONS["Box_Fully_Closed"])
    # time.sleep(1)
    # operate_gripper(gripper, 255)
    # print("Box is fully closed", "\n")
    # time.sleep(1)
    # move_robot(robot, ROBOT_POSITIONS["Box_1/4_Open"])
    # print("Box is 1/4 open", "\n")
    # time.sleep(1)

    # move_robot(robot, ROBOT_POSITIONS["Box_1/2_Open"])
    # print("Box is 1/2 open", "\n")
    # time.sleep(1)

    # move_robot(robot, ROBOT_POSITIONS["Box_3/4_Open"])
    # print("Box is 3/4 open", "\n")
    # time.sleep(1)

    # move_robot(robot, ROBOT_POSITIONS["Box_Fully_Open"])
    # print("Box is fully open", "\n")
    # time.sleep(1)
    # operate_gripper(gripper, 0)
    # time.sleep(1)

    # move_robot(robot, ROBOT_POSITIONS["Facing_Open_Box"])
    # time.sleep(1)


    # # Removing sample vial from laser diffraction box
    
    # move_robot(robot, ROBOT_POSITIONS["Facing_Closed_Box"])
    # time.sleep(1)
    # move_robot(robot, ROBOT_POSITIONS["Dilute_Sample_Before_Box"])
    # time.sleep(1)
    # move_robot(robot, ROBOT_POSITIONS["Dilute_Sample_Above_Box_Holder"])
    # time.sleep(1)
    # move_robot(robot, ROBOT_POSITIONS["Dilute_Sample_In_Box_Holder"])
    # time.sleep(1)
    # operate_gripper(gripper, 125)
    # time.sleep(1)
    # move_robot(robot, ROBOT_POSITIONS["Dilute_Sample_Above_Box_Holder"])
    # time.sleep(1)
    # move_robot(robot, ROBOT_POSITIONS["Dilute_Sample_Before_Box"])
    # time.sleep(1)
    # move_robot(robot, ROBOT_POSITIONS["Facing_Closed_Box"])
    # time.sleep(1)
    # move_robot(robot, ROBOT_POSITIONS["Above_SVH3_P1"])
    # time.sleep(1)
    # move_robot(robot, ROBOT_POSITIONS["Just_Above_SVH3_P1"])
    # time.sleep(1)
    # move_robot(robot, ROBOT_POSITIONS["SVH3_P1"])
    # time.sleep(1)
    # operate_gripper(gripper, 0)
    # time.sleep(1)
    # move_robot(robot, ROBOT_POSITIONS["Just_Above_SVH3_P1"])
    # time.sleep(1)
    # move_robot(robot, ROBOT_POSITIONS["Above_SVH3_P1"])
    # time.sleep(1)
    # move_robot(robot, ROBOT_POSITIONS["Facing_Closed_Box"])
    # time.sleep(1)

if __name__ == "__main__":
    main()

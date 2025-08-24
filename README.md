# CHEM506B-Research-Project
This GitHub repository contains the code associated with Michael Gillin's Digital Chemistry: AI, Machine Learning, Robotics, and Automation MSc dissertation project at the University of Liverpool, titled "Frugal Automation for Crystallisation Analysis: Webcam-Based Light Scattering for Particle Characterisation".

## Description

### Project Overview

This project develops a low-cost, automated system for synthesising a metal-organic complex, analysing crystallisation, and determining particle size distributions. A Logitch HD webcam and Class 1 laser are used to capture scattering patterns, which are then processed with Python to extract radial intensity profiles and recover particle size distributions. The system provides a frugal alternative to commercial instruments like the Malvern Panalytical Mastersizer 3000, with applications in crystallisation, soft matter formulations, and materials characterisation.

The specific reaction performed over this research is the synthesis of bis(piroctone)copper(II) from the dropwise addition of aqueous copper(II) chloride to ethanolic piroctone olamine, as shown in the reaction scheme below:

insert here

## Prerequisites

The following software is required to run the code and acquire the appropriate data:

- Visual Studio Code (or other code editor) - https://code.visualstudio.com/download
- Python 3.9+ (Anaconda is recommended. To install, click "Free Download using the URL") - https://www.anaconda.com/

Required Python packages:
- opencv-python
- numpy
- matplotlib
- scikit-image
- scipy
- miepython
- cv2-enumerate-cameras

Install dependencies with:
pip install -r requirements.txt

Hardware:

- UR5e robotic arm
- Robotiq Hand-E adaptive gripper
- HD Web camera (Logitech C920 HD Pro Webcam)
- IKA RCT basic/digital hotplate stirrer
- Sample vial(s)
- 20 mm magnetic stirrer bar(s)
- White/black background
- Mettler Toledo Quantos
- Class 1 red laser (635 nm)

## Usage

Clone the repository:
git clone https://github.com/MGillin-UoL/CHEM506B-Research-Project.git
cd CHEM506B-Research-Project

Run the scattering analysis:
python SIRE_Scatter.py


## Authors
If you have any questions or would like additional information, please contact:
- Michael Gillin (sgmgilli@liverpool.ac.uk)

## License
This project is not currently licensed.

## Acknowledgements
The author would like to thank his supervisors, Dr Joe Forth and Gabriella Pizzuto for supervising his project. He would also like to thank Dr Marco Giardello for cosupervising his project. He would also like to thank Emmanuel Ombo, who supported Michael throughout his project. He would also like to thank Jakub Glowacki for supporting him with his robotic workflow development. He would also like to thank all members of the Forth Group, who taught him how to 3D print, introduced him to the university's disposal mechanisms, and saw him day-to-date in the office.

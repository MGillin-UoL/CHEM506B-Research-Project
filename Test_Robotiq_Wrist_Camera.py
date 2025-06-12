import requests
from PIL import ImageTk, Image
import numpy as np
import matplotlib.image
import shutil
import io
def main():
    resp = requests.get("http://"+"192.168.0.2"+":4242/current.jpg?type=color",stream=True)
    with open('img.png', 'wb') as out_file:
        shutil.copyfileobj(resp.raw, out_file)
    

if __name__=="__main__":
    main()

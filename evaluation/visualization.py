import cv2
import json
import numpy as np
#img_name = "fb2ebd179d2715f604e18d2aa0ed08f2_d3d3LnN0YXRpc3RpcXVlcy5kZXZlbG9wcGVtZW50LWR1cmFibGUuZ291di5mcgkzNy4yMzUuODkuMTA3.xls-0-0.png"
img_name = "fa8527e1883a153f7101c6882c7ba16e_d3d3LmhwYy5nby50aAkxMjIuMTU0LjczLjI2-2-0.png"
image = cv2.imread(img_name)
color = [(255,0,0), (0,0,255), (0,0,0)]
radius = 4
thickness = 1
with open("KPDetection10000.json", 'r') as file:
    data = json.load(file)
for point in data[img_name][0]['2']:
    print((int)(point[2]), (int)(point[3]))
    cv2.circle(image, ((int)(point[2]), (int)(point[3])), radius, color[1], -1)
for point in data[img_name][1]['2']:
    print((int)(point[2]), (int)(point[3]))
    cv2.circle(image, ((int)(point[2]), (int)(point[3])), radius, color[0], -1)
cv2.imwrite("visualized.png", image)
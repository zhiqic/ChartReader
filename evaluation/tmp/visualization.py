import cv2
import json
import numpy as np
img_name = "8763f43082006c32874b58150a0bc2de_d3d3LnNpZ3JoLnNwLmdvdi5icgkyMDEuNTUuOC4xOA==.xls-3-0.png"
image = cv2.imread(img_name)
color = [(255,0,0), (0,0,255), (0,0,0)]
radius = 4
thickness = 1
with open("KPDetection5000.json", 'r') as file:
    data = json.load(file)
for point in data[img_name][0]['2']:
    print((int)(point[2]), (int)(point[3]))
    cv2.circle(image, ((int)(point[2]), (int)(point[3])), radius, color[1], -1)
for point in data[img_name][1]['2']:
    print((int)(point[2]), (int)(point[3]))
    cv2.circle(image, ((int)(point[2]), (int)(point[3])), radius, color[0], -1)
cv2.imwrite("visualized.png", image)
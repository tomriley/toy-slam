#!/usr/bin/env python3
from glumpy import app, glm, gloo, gl, __version__, data
import glfw
import numpy as np
import cv2 as cv
import display
import display3d
import slam
import math

cap = cv.VideoCapture('video/test_kitti984.mp4')

width = cap.get(cv.CAP_PROP_FRAME_WIDTH)
height = cap.get(cv.CAP_PROP_FRAME_HEIGHT)

F = 718.856
P = np.array([
    [F, 0, 607.1928, 0],
    [0, F, 185.2157000, 0],
    [0, 0, 1, 0],
])
K = P[0:3, 0:3]

# P (K is 3x3 subset)
# 7.188560000000e+02 0.000000000000e+00 6.071928000000e+02 0.000000000000e+00
# 0.000000000000e+00 7.188560000000e+02 185.2157000000     0.000000000000e+00
# 0.000000000000e+00 0.000000000000e+00 1.000000000000e+00 0.000000000000e+00

# 1242 × 375
# 607.1928
# 1242/2

# aa = math.atan2(height, 2 * F)
# fovy = math.degrees(2 * aa)
# print("yfov = ", fovy)

# far = 10000.0
# near = 0.1
# aspect = float(width)/height
# cot = 1.0 / math.tan(fovy / 2)
# P = np.array([ # projection matrix
#     [cot/aspect, 0, 0, 0],
#     [0, cot, 0, 0],
#     [0, 0, (far + near) / (near - far), (2 * far * near) / (near - far)],
#     [0, 0, -1, 0]
# ], dtype=float)

print(P)

success, image = cap.read() # get first frame

disp = display.Display(width, height)
disp3d = display3d.Display3D(1000, 1000)
for i in range(11):
    disp3d.add_point([0.1*i+0.1, 0, 0], [1,0,0])
    disp3d.add_point([0, 0.1*i+0.1, 0], [1,1,0])
    disp3d.add_point([0, 0, 0.1*i+0.1], [0,1,0])

flow = slam.SLAM(K, P)

print(image.shape, image.dtype)

# upload initial frame
disp.set_image(cv.cvtColor(image, cv.COLOR_BGR2RGB))

@disp.window.timer(1/3.0)
def timer(fps):
    if not cap.isOpened(): return
    
    _, image = cap.read()
    image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    frame = flow.process_image(image)
    for pt in frame.points:
        disp3d.add_point(pt[0:3], [0.7,0.3,0])
    pos = frame.pose @ [0,0,0,1]
    disp3d.add_point(pos[0:3], [1,1,1])
    print("pose", frame.pose)

    if frame.lines is not None:
        # show the future direction (previous frame)
        bg = cv.cvtColor(image, cv.COLOR_GRAY2BGR)
        comp = cv.add(bg, frame.lines)
        disp.set_image(cv.cvtColor(comp, cv.COLOR_BGR2RGB))

app.use("glfw")
app.run()

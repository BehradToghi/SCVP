# Copyright 2020 Connected & Autonomous Vehicle REsearch Lab (CAVREL)
# at University of Central Florida (UCF).

# run Carla Server
# SDL_VIDEODRIVER=offscreen ./CarlaUE4.sh -opengl -quality-level=Low

"""
Example of full autonomous vehicle driving pipeline.

    L            : toggle next light type
    SHIFT + L    : toggle high beam
    Z/X          : toggle right/left blinker
    I            : toggle interior light

    TAB          : change sensor position
    ` or N       : next sensor
    [1-9]        : change to sensor [1-9]
    G            : toggle radar visualization
    C            : change weather (Shift+C reverse)

    F1           : toggle HUD
    H/?          : toggle help
    ESC          : quit

"""

from __future__ import print_function

import glob
import os
import sys
import argparse
import collections
import datetime
import logging
import math
import random
import re
import weakref
import pickle
import imutils
import cv2
import networkx as nx
from enum import Enum
from collections import deque
import matplotlib.pyplot as plt
from pathlib import Path
import matplotlib.image as mpimg
from imutils.video import FPS

try:
    import pygame
    from pygame.locals import KMOD_CTRL
    from pygame.locals import KMOD_SHIFT
    from pygame.locals import K_0
    from pygame.locals import K_9
    from pygame.locals import K_ESCAPE
    from pygame.locals import K_F1
    from pygame.locals import K_TAB
    from pygame.locals import K_c
    from pygame.locals import K_g
    from pygame.locals import K_h
    from pygame.locals import K_n
    from pygame.locals import K_q
    from pygame.locals import K_l
    from pygame.locals import K_i
    from pygame.locals import K_z
    from pygame.locals import K_x

except ImportError:
    raise RuntimeError('cannot import pygame, make sure pygame package is installed')

try:
    import numpy as np
except ImportError:
    raise RuntimeError(
        'cannot import numpy, make sure numpy package is installed')

# ==============================================================================
# -- Find CARLA module ---------------------------------------------------------
# ==============================================================================
try:
    sys.path.append(glob.glob('../carla/dist/carla-*%d.%d-%s.egg' % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError:
    pass

# ==============================================================================
# -- Add PythonAPI for release mode --------------------------------------------
# ==============================================================================
try:
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + '/carla')
except IndexError:
    pass

import carla
from carla import ColorConverter as cc
# ==============================================================================
# -- Global functions ----------------------------------------------------------
# ==============================================================================
def find_weather_presets():
    """Method to find weather presets"""
    rgx = re.compile('.+?(?:(?<=[a-z])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])|$)')
    def name(x): return ' '.join(m.group(0) for m in rgx.finditer(x))
    presets = [x for x in dir(carla.WeatherParameters) if re.match('[A-Z].+', x)]
    return [(getattr(carla.WeatherParameters, x), name(x)) for x in presets]

def get_actor_display_name(actor, truncate=250):
    """Method to get actor display name"""
    name = ' '.join(actor.type_id.replace('_', '.').title().split('.')[1:])
    return (name[:truncate - 1] + u'\u2026') if len(name) > truncate else name

def get_speed(vehicle):
    """
    Compute speed of a vehicle in Km/h.

        :param vehicle: the vehicle for which speed is calculated
        :return: speed as a float in Km/h
    """
    vel = vehicle.get_velocity()

    return 3.6 * math.sqrt(vel.x ** 2 + vel.y ** 2 + vel.z ** 2)

def vector(location_1, location_2):
    """
    Returns the unit vector from location_1 to location_2

        :param location_1, location_2: carla.Location objects
    """
    x = location_2.x - location_1.x
    y = location_2.y - location_1.y
    z = location_2.z - location_1.z
    norm = np.linalg.norm([x, y, z]) + np.finfo(float).eps

    return [x / norm, y / norm, z / norm]

def is_within_distance(target_location, current_location, orientation, max_distance, d_angle_th_up, d_angle_th_low=0):
    """
    Check if a target object is within a certain distance from a reference object.
    A vehicle in front would be something around 0 deg, while one behind around 180 deg.

        :param target_location: location of the target object
        :param current_location: location of the reference object
        :param orientation: orientation of the reference object
        :param max_distance: maximum allowed distance
        :param d_angle_th_up: upper thereshold for angle
        :param d_angle_th_low: low thereshold for angle (optional, default is 0)
        :return: True if target object is within max_distance ahead of the reference object
    """
    target_vector = np.array([target_location.x - current_location.x, target_location.y - current_location.y])
    norm_target = np.linalg.norm(target_vector)

    # If the vector is too short, we can simply stop here
    if norm_target < 0.001:
        return True

    if norm_target > max_distance:
        return False

    forward_vector = np.array(
        [math.cos(math.radians(orientation)), math.sin(math.radians(orientation))])
    d_angle = math.degrees(math.acos(np.clip(np.dot(forward_vector, target_vector) / norm_target, -1., 1.)))

    return d_angle_th_low < d_angle < d_angle_th_up

def is_within_distance_ahead(target_transform, current_transform, max_distance):
    """
    Check if a target object is within a certain distance in front of a reference object.

    :param target_transform: location of the target object
    :param current_transform: location of the reference object
    :param orientation: orientation of the reference object
    :param max_distance: maximum allowed distance
    :return: True if target object is within max_distance ahead of the reference object
    """
    target_vector = np.array([target_transform.location.x - current_transform.location.x, target_transform.location.y - current_transform.location.y])
    norm_target = np.linalg.norm(target_vector)

    # If the vector is too short, we can simply stop here
    if norm_target < 0.001:
        return True

    if norm_target > max_distance:
        return False

    fwd = current_transform.get_forward_vector()
    forward_vector = np.array([fwd.x, fwd.y])
    d_angle = math.degrees(math.acos(np.clip(np.dot(forward_vector, target_vector) / norm_target, -1., 1.)))

    return d_angle < 90.0

def draw_waypoints(world, waypoints, z=0.01):
    """
    Draw a list of waypoints at a certain height given in z.

        :param world: carla.world object
        :param waypoints: list or iterable container with the waypoints to draw
        :param z: height in meters
    """
    for wpt in waypoints:
        wpt_t = wpt.transform
        begin = wpt_t.location + carla.Location(z=z)
        angle = math.radians(wpt_t.rotation.yaw)
        end = begin + carla.Location(x=math.cos(angle), y=math.sin(angle))
        world.debug.draw_arrow(begin, end, arrow_size=0.1, life_time=1.0)

def compute_distance(location_1, location_2):
    """
    Euclidean distance between 3D points

        :param location_1, location_2: 3D points
    """
    x = location_2.x - location_1.x
    y = location_2.y - location_1.y
    z = location_2.z - location_1.z
    norm = np.linalg.norm([x, y, z]) + np.finfo(float).eps
    return norm

def _retrieve_options(list_waypoints, current_waypoint):
    """
    Compute the type of connection between the current active waypoint and the multiple waypoints present in
    list_waypoints. The result is encoded as a list of RoadOption enums.

    :param list_waypoints: list with the possible target waypoints in case of multiple options
    :param current_waypoint: current active waypoint
    :return: list of RoadOption enums representing the type of connection from the active waypoint to each
             candidate in list_waypoints
    """
    options = []
    for next_waypoint in list_waypoints:
        # this is needed because something we are linking to
        # the beggining of an intersection, therefore the
        # variation in angle is small
        next_next_waypoint = next_waypoint.next(3.0)[0]
        link = _compute_connection(current_waypoint, next_next_waypoint)
        options.append(link)

    return options

def _compute_connection(current_waypoint, next_waypoint, threshold=35):
    """
    Compute the type of topological connection between an active waypoint (current_waypoint) and a target waypoint
    (next_waypoint).

    :param current_waypoint: active waypoint
    :param next_waypoint: target waypoint
    :return: the type of topological connection encoded as a RoadOption enum:
             RoadOption.STRAIGHT
             RoadOption.LEFT
             RoadOption.RIGHT
    """
    n = next_waypoint.transform.rotation.yaw
    n = n % 360.0

    c = current_waypoint.transform.rotation.yaw
    c = c % 360.0

    diff_angle = (n - c) % 180.0
    if diff_angle < threshold or diff_angle > (180 - threshold):
        return RoadOption.STRAIGHT
    elif diff_angle > 90.0:
        return RoadOption.LEFT
    else:
        return RoadOption.RIGHT

def positive(num):
    """
    Return the given number if positive, else 0

        :param num: value to check
    """
    return num if num > 0.0 else 0.0

def compute_magnitude_angle(target_location, current_location, orientation):
    """
    Compute relative angle and distance between a target_location and a current_location

        :param target_location: location of the target object
        :param current_location: location of the reference object
        :param orientation: orientation of the reference object
        :return: a tuple composed by the distance to the object and the angle between both objects
    """
    target_vector = np.array([target_location.x - current_location.x, target_location.y - current_location.y])
    norm_target = np.linalg.norm(target_vector)

    forward_vector = np.array([math.cos(math.radians(orientation)), math.sin(math.radians(orientation))])
    d_angle = math.degrees(math.acos(np.clip(np.dot(forward_vector, target_vector) / norm_target, -1., 1.)))

    return (norm_target, d_angle)

def distance_vehicle(waypoint, vehicle_transform):
    """
    Returns the 2D distance from a waypoint to a vehicle

        :param waypoint: actual waypoint
        :param vehicle_transform: transform of the target vehicle
    """
    loc = vehicle_transform.location
    x = waypoint.transform.location.x - loc.x
    y = waypoint.transform.location.y - loc.y

    return math.sqrt(x * x + y * y)
# ==============================================================================
# -- Perception functions ----------------------------------------------------------
# ==============================================================================
def get_project_root():
    """Returns project root folder."""
    return str(Path(__file__).parent.parent.parent.parent)
#ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR=get_project_root()

Camera_image = np.zeros(shape=(480, 640, 3), dtype=np.uint8)

def sumMatrix(A, B):
    A = np.array(A)
    B = np.array(B)
    answer = A + B
    return answer.tolist()

test_con = 0  # test function

#V1
def lane_detectionv1(RGB_Camera_im):
    pt1_sum_ri = (0, 0)
    pt2_sum_ri = (0, 0)
    pt1_avg_ri = (0, 0)
    count_posi_num_ri = 0

    pt1_sum_le = (0, 0)
    pt2_sum_le = (0, 0)
    pt1_avg_le = (0, 0)

    count_posi_num_le = 0

    #################################################
    # Now image resolution is 720x1280x3
    size_im = cv2.resize(RGB_Camera_im, dsize=(640, 480))  # VGA resolution
    # size_im = cv2.resize(test_im, dsize=(800, 600))  # SVGA resolution
    # size_im = cv2.resize(test_im, dsize=(1028, 720))  # HD resolution
    # size_im = cv2.resize(test_im, dsize=(1920, 1080))  # Full-HD resolution
    # cv2.imshow("size_im", size_im)
    #################################################

    #################################################
    # ROI Coordinates Set-up
    # roi = size_im[320:480, 213:426]  # [380:430, 330:670]   [y:y+b, x:x+a]
    # roi_im = cv2.resize(roi, (213, 160))  # x,y
    # cv2.imshow("roi_im", roi_im)
    roi = size_im[240:480, 108:532]  # [380:430, 330:670]   [y:y+b, x:x+a]
    roi_im = cv2.resize(roi, (424, 240))  # (a of x, b of y)
    # cv2.imshow("roi_im", roi_im)
    #################################################

    #################################################
    # Gaussian Blur Filter
    Blur_im = cv2.bilateralFilter(roi_im, d=-1, sigmaColor=5, sigmaSpace=5)
    #################################################

    #################################################
    # Canny edge detector
    edges = cv2.Canny(Blur_im, 50, 100)
    # cv2.imshow("edges", edges)
    #################################################

    #################################################
    # Hough Transformation
    # lines = cv2.HoughLinesP(edges, rho=1, theta=np.pi / 180.0, threshold=80, minLineLength=30, maxLineGap=50)
    lines = cv2.HoughLinesP(edges, rho=1, theta=np.pi / 180.0, threshold=25, minLineLength=10, maxLineGap=20)

    if lines is None:
        lines = [[0, 0, 0, 0]]
        return None, None
    else:
        for line in lines:

            x1, y1, x2, y2 = line[0]

            if x2 == x1:
                a = 1
            else:
                a = x2 - x1

            b = y2 - y1

            radi = b / a
            # print('radi=', radi)

            theta_atan = math.atan(radi) * 180.0 / math.pi
            # print('theta_atan=', theta_atan)

            pt1_ri = (x1 + 108, y1 + 240)
            pt2_ri = (x2 + 108, y2 + 240)
            pt1_le = (x1 + 108, y1 + 240)
            pt2_le = (x2 + 108, y2 + 240)

            if theta_atan > 20.0 and theta_atan < 90.0:
                # cv2.line(size_im, (x1+108, y1+240), (x2+108, y2+240), (0, 255, 0), 2)

                count_posi_num_ri += 1
                pt1_sum_ri = sumMatrix(pt1_ri, pt1_sum_ri)
                pt2_sum_ri = sumMatrix(pt2_ri, pt2_sum_ri)

            if theta_atan < -20.0 and theta_atan > -90.0:
                # cv2.line(size_im, (x1+108, y1+240), (x2+108, y2+240), (0, 0, 255), 2)

                count_posi_num_le += 1
                pt1_sum_le = sumMatrix(pt1_le, pt1_sum_le)
                pt2_sum_le = sumMatrix(pt2_le, pt2_sum_le)

        pt1_avg_ri = pt1_sum_ri // np.array(count_posi_num_ri)
        pt2_avg_ri = pt2_sum_ri // np.array(count_posi_num_ri)
        pt1_avg_le = pt1_sum_le // np.array(count_posi_num_le)
        pt2_avg_le = pt2_sum_le // np.array(count_posi_num_le)


        #################################################
        # right-----------------------------------------------------------
        x1_avg_ri, y1_avg_ri = pt1_avg_ri
        x2_avg_ri, y2_avg_ri = pt2_avg_ri

        a_avg_ri = ((y2_avg_ri - y1_avg_ri) / (x2_avg_ri - x1_avg_ri))
        b_avg_ri = (y2_avg_ri - (a_avg_ri * x2_avg_ri))

        pt2_y2_fi_ri = 480

        if a_avg_ri > 0:
            pt2_x2_fi_ri = int((pt2_y2_fi_ri - b_avg_ri) // a_avg_ri)
        else:
            pt2_x2_fi_ri = 0

        pt2_fi_ri = (pt2_x2_fi_ri, pt2_y2_fi_ri)
        # pt2_fi_ri = (int(pt2_x2_fi_ri), pt2_y2_fi_ri)

        # left------------------------------------------------------------
        x1_avg_le, y1_avg_le = pt1_avg_le
        x2_avg_le, y2_avg_le = pt2_avg_le


        a_avg_le = ((y2_avg_le - y1_avg_le) / (x2_avg_le - x1_avg_le))
        b_avg_le = (y2_avg_le - (a_avg_le * x2_avg_le))


        pt1_y1_fi_le = 480
        if a_avg_le < 0:
            pt1_x1_fi_le = int((pt1_y1_fi_le - b_avg_le) // a_avg_le)
        else:
            pt1_x1_fi_le = 0
        # pt1_x1_fi_le = ((pt1_y1_fi_le - b_avg_le) // a_avg_le)


        pt1_fi_le = (pt1_x1_fi_le, pt1_y1_fi_le)
          #################################################

        #################################################
        # lane painting
        # right-----------------------------------------------------------
        # cv2.line(size_im, tuple(pt1_avg_ri), tuple(pt2_avg_ri), (0, 255, 0), 2) # right lane
        cv2.line(size_im, tuple(pt1_avg_ri), tuple(pt2_fi_ri), (0, 255, 0), 2)  # right lane
        # left-----------------------------------------------------------
        # cv2.line(size_im, tuple(pt1_avg_le), tuple(pt2_avg_le), (0, 255, 0), 2) # left lane
        cv2.line(size_im, tuple(pt1_fi_le), tuple(pt2_avg_le), (0, 255, 0), 2)  # left lane
        # center-----------------------------------------------------------
        cv2.line(size_im, (320, 480), (320, 360), (0, 228, 255), 1)  # middle lane
        #################################################

        #################################################
        # possible lane
        # FCP = np.array([pt1_avg_ri, pt2_avg_ri, pt1_avg_le, pt2_avg_le])
        # cv2.fillConvexPoly(size_im, FCP, color=(255, 242, 213)) # BGR
        #################################################
        FCP_img = np.zeros(shape=(480, 640, 3), dtype=np.uint8) + 0
        # FCP = np.array([pt1_avg_ri, pt2_avg_ri, pt1_avg_le, pt2_avg_le])
        # FCP = np.array([(100,100), (100,200), (200,200), (200,100)])
        FCP = np.array([pt2_avg_le, pt1_fi_le, pt2_fi_ri, pt1_avg_ri])
        cv2.fillConvexPoly(FCP_img, FCP, color=(255, 242, 213))  # BGR
        alpha = 0.9
        size_im = cv2.addWeighted(size_im, alpha, FCP_img, 1 - alpha, 0)

        # alpha = 0.4
        # size_im = cv2.addWeighted(size_im, alpha, FCP, 1 - alpha, 0)
        #################################################

        #################################################
        # lane center steering  (320, 360)
        lane_center_y_ri = 360
        if a_avg_ri > 0:
            lane_center_x_ri = int((lane_center_y_ri - b_avg_ri) // a_avg_ri)
        else:
            lane_center_x_ri = 0

        lane_center_y_le = 360
        if a_avg_le < 0:
            lane_center_x_le = int((lane_center_y_le - b_avg_le) // a_avg_le)
        else:
            lane_center_x_le = 0

        # caenter left lane (255, 90, 185)
        cv2.line(size_im, (lane_center_x_le, lane_center_y_le - 10), (lane_center_x_le, lane_center_y_le + 10),
                 (0, 228, 255), 1)
        # caenter right lane
        cv2.line(size_im, (lane_center_x_ri, lane_center_y_ri - 10), (lane_center_x_ri, lane_center_y_ri + 10),
                 (0, 228, 255), 1)
        # caenter middle lane
        lane_center_x = ((lane_center_x_ri - lane_center_x_le) // 2) + lane_center_x_le
        cv2.line(size_im, (lane_center_x, lane_center_y_ri - 10), (lane_center_x, lane_center_y_le + 10),
                 (0, 228, 255), 1)

        text_left = 'Turn Left'
        text_right = 'Turn Right'
        text_center = 'Center'
        text_non = ''
        org = (320, 440)
        font = cv2.FONT_HERSHEY_SIMPLEX

        if 0 < lane_center_x <= 318:
            cv2.putText(size_im, text_left, org, font, 0.7, (0, 0, 255), 2)
        elif 318 < lane_center_x < 322:
            # elif lane_center_x > 318 and lane_center_x < 322 :
            cv2.putText(size_im, text_center, org, font, 0.7, (0, 0, 255), 2)
        elif lane_center_x >= 322:
            cv2.putText(size_im, text_right, org, font, 0.7, (0, 0, 255), 2)
        elif lane_center_x == 0:
            cv2.putText(size_im, text_non, org, font, 0.7, (0, 0, 255), 2)
        #################################################

        global test_con
        test_con = 1
        count_posi_num_ri = 0

        pt1_sum_ri = (0, 0)
        pt2_sum_ri = (0, 0)
        pt1_avg_ri = (0, 0)
        pt2_avg_ri = (0, 0)

        count_posi_num_le = 0

        pt1_sum_le = (0, 0)
        pt2_sum_le = (0, 0)
        pt1_avg_le = (0, 0)
        pt2_avg_le = (0, 0)

        return lines,size_im

def do_canny(frame):
    # Converts frame to grayscale because we only need the luminance channel for detecting edges - less computationally expensive
    gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    # Applies a 5x5 gaussian blur with deviation of 0 to frame - not mandatory since Canny will do this for us
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    # Applies Canny edge detector with minVal of 50 and maxVal of 150
    canny = cv2.Canny(blur, 50, 150)
    return canny

def do_segment(frame):
    # Since an image is a multi-directional array containing the relative intensities of each pixel in the image, we can use frame.shape to return a tuple: [number of rows, number of columns, number of channels] of the dimensions of the frame
    # frame.shape[0] give us the number of rows of pixels the frame has. Since height begins from 0 at the top, the y-coordinate of the bottom of the frame is its height
    height = frame.shape[0]
    # Creates a triangular polygon for the mask defined by three (x, y) coordinates
    polygons = np.array([
                            [(0, height), (800, height), (380, 290)]
                        ])
    # Creates an image filled with zero intensities with the same dimensions as the frame
    mask = np.zeros_like(frame)
    # Allows the mask to be filled with values of 1 and the other areas to be filled with values of 0
    cv2.fillPoly(mask, polygons, 255)
    # A bitwise and operation between the mask and frame keeps only the triangular area of the frame
    segment = cv2.bitwise_and(frame, mask)
    return segment

def calculate_lines(frame, lines):
    # Empty arrays to store the coordinates of the left and right lines
    left = []
    right = []
    # Loops through every detected line
    for line in lines:
        # Reshapes line from 2D array to 1D array
        x1, y1, x2, y2 = line.reshape(4)
        # Fits a linear polynomial to the x and y coordinates and returns a vector of coefficients which describe the slope and y-intercept
        parameters = np.polyfit((x1, x2), (y1, y2), 1)
        slope = parameters[0]
        y_intercept = parameters[1]
        # If slope is negative, the line is to the left of the lane, and otherwise, the line is to the right of the lane
        if slope < 0:
            left.append((slope, y_intercept))
        else:
            right.append((slope, y_intercept))
    # Averages out all the values for left and right into a single slope and y-intercept value for each line
    left_avg = np.average(left, axis = 0)
    right_avg = np.average(right, axis = 0)
    # Calculates the x1, y1, x2, y2 coordinates for the left and right lines
    left_line = calculate_coordinates(frame, left_avg)
    right_line = calculate_coordinates(frame, right_avg)
    return np.array([left_line, right_line])

def calculate_coordinates(frame, parameters):
    slope, intercept = parameters
    # Sets initial y-coordinate as height from top down (bottom of the frame)
    y1 = frame.shape[0]
    # Sets final y-coordinate as 150 above the bottom of the frame
    y2 = int(y1 - 150)
    # Sets initial x-coordinate as (y1 - b) / m since y1 = mx1 + b
    x1 = int((y1 - intercept) / slope)
    # Sets final x-coordinate as (y2 - b) / m since y2 = mx2 + b
    x2 = int((y2 - intercept) / slope)
    return np.array([x1, y1, x2, y2])

def visualize_lines(frame, lines):
    # Creates an image filled with zero intensities with the same dimensions as the frame
    lines_visualize = np.zeros_like(frame)
    # Checks if any lines are detected
    if lines is not None:
        for x1, y1, x2, y2 in lines:
            # Draws lines between two coordinates with green color and 5 thickness
            cv2.line(lines_visualize, (x1, y1), (x2, y2), (0, 255, 0), 5)

        x1mid=int((lines[0][0]+lines[1][0])/2)
        y1mid = int((lines[0][1] + lines[1][1]) / 2)
        x2mid = int((lines[0][2] + lines[1][2]) / 2)
        y2mid = int((lines[0][3] + lines[1][3]) / 2)
        cv2.line(lines_visualize, (x1mid, y1mid), (x2mid, y2mid),  (0, 0, 255), 5)

        xm_per_pix=3/640
        vehicle_midpoint_p=410 #frame.shape[0]
        vehicle_offset_m= (vehicle_midpoint_p - x1mid) * xm_per_pix


    return lines_visualize,vehicle_offset_m

#V2
font = cv2.FONT_HERSHEY_SIMPLEX
fontColor = (0, 0, 0)
fontSize = 0.5
def lane_detectionv2(RGB_Camera_im):
    size_im = cv2.resize(RGB_Camera_im, dsize=(640, 480))  # VGA resolution

    canny = do_canny(size_im)
    # cv2.imshow("canny", canny)
    # plt.imshow(frame)
    # plt.show()
    segment = do_segment(canny)
    hough = cv2.HoughLinesP(segment, 2, np.pi / 180, 100, np.array([]), minLineLength=100, maxLineGap=50)
    # Averages multiple detected lines from hough into one line for left border of lane and one line for right border of lane

    if hough is None:
        return None, None

    lines = calculate_lines(size_im, hough)
    # Visualizes the lines
    if lines is None:
        return None, None
    else:
        lines_visualize, vehicle_offset_m = visualize_lines(size_im, lines)
        # cv2.imshow("hough", lines_visualize)
        # Overlays lines on frame by taking their weighted sums and adding an arbitrary scalar value of 1 as the gamma argument
        output = cv2.addWeighted(size_im, 0.9, lines_visualize, 1, 1)

        cv2.putText(output, 'Vehicle offset : {:.4f} m'.format(vehicle_offset_m), (410, 450), font, fontSize,
                   fontColor, 2)

    return 1, output

#v3
def grayscale(img):
    """Applies the Grayscale transform
    This will return an image with only one color channel
    but NOTE: to see the returned image as grayscale
    (assuming your grayscaled image is called 'gray')
    you should call plt.imshow(gray, cmap='gray')"""
    return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # Or use BGR2GRAY if you read an image with cv2.imread()
    # return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

def canny(img, low_threshold, high_threshold):
    """Applies the Canny transform"""
    return cv2.Canny(img, low_threshold, high_threshold)

def gaussian_blur(img, kernel_size):
    """Applies a Gaussian Noise kernel"""
    return cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)

def region_of_interest(img, vertices):
    """
    Applies an image mask.
    Only keeps the region of the image defined by the polygon
    formed from `vertices`. The rest of the image is set to black.
    `vertices` should be a numpy array of integer points.
    """
    # defining a blank mask to start with
    mask = np.zeros_like(img)

    # defining a 3 channel or 1 channel color to fill the mask with depending on the input image
    if len(img.shape) > 2:
        channel_count = img.shape[2]  # i.e. 3 or 4 depending on your image
        ignore_mask_color = (255,) * channel_count
    else:
        ignore_mask_color = 255

    # filling pixels inside the polygon defined by "vertices" with the fill color
    cv2.fillPoly(mask, vertices, ignore_mask_color)

    # returning the image only where mask pixels are nonzero
    masked_image = cv2.bitwise_and(img, mask)
    return masked_image

def draw_lines(img, lines, color=[255, 0, 0], thickness=8):
    """
    NOTE: this is the function you might want to use as a starting point once you want to
    average/extrapolate the line segments you detect to map out the full
    extent of the lane (going from the result shown in raw-lines-example.mp4
    to that shown in P1_example.mp4).
    Think about things like separating line segments by their
    slope ((y2-y1)/(x2-x1)) to decide which segments are part of the left
    line vs. the right line.  Then, you can average the position of each of
    the lines and extrapolate to the top and bottom of the lane.
    This function draws `lines` with `color` and `thickness`.
    Lines are drawn on the image inplace (mutates the image).
    If you want to make the lines semi-transparent, think about combining
    this function with the weighted_img() function below
    """
    (height, widh, chanels) = img.shape
    # number of lanes
    left_lines = 0
    right_lines = 0

    # lanes coordinates
    x1_l, y1_l, x2_l, y2_l = 0, 0, 0, 0
    x1_r, y1_r, x2_r, y2_r = 0, 0, 0, 0

    ymin = height

    # find the slope to check right and left lines, compute the sum of the coordinates for average
    if lines is None:
        return None

    for line in lines:
        for x1, y1, x2, y2 in line:
            slope = (y2 - y1) / (x2 - x1)
            slope_deg = 180 * math.atan(slope) / np.pi

            # filterning lines that are not lanes based on slope deg
            if 20 < abs(slope_deg) < 70:
                if slope < 0:
                    left_lines += 1
                    x1_l, y1_l, x2_l, y2_l = x1 + x1_l, y1 + y1_l, x2 + x2_l, y2 + y2_l
                    ymin = min(ymin, y2)

                else:
                    right_lines += 1
                    x1_r, y1_r, x2_r, y2_r = x1 + x1_r, y1 + y1_r, x2 + x2_r, y2 + y2_r
                    ymin = min(ymin, y2)

    #  average the position of each of the lines and extrapolate to the top and bottom of the lane.
    if left_lines:
        #  average the position of each of the lines
        x1, y1, x2, y2 = x1_l / left_lines, y1_l / left_lines, x2_l / left_lines, y2_l / left_lines
        slope = (y2 - y1) / (x2 - x1)
        intercept = ((y1 - slope * x1))

        # extrapolate to the top
        y2 = ymin
        x2 = int((y2 - intercept) / slope)

        # extrapolate to the bottom
        xi = int((height - intercept) / slope)
        cv2.line(img, (xi, height), (int(x2), int(y2)), color, thickness)

    # for right lines
    if right_lines:
        x1, y1, x2, y2 = x1_r / right_lines, y1_r / right_lines, x2_r / right_lines, y2_r / right_lines
        slope = (y2 - y1) / (x2 - x1)
        intercept = ((y1 - slope * x1))

        y2 = ymin
        x2 = int((y2 - intercept) / slope)
        xi = int((height - intercept) / slope)
        cv2.line(img, (xi, height), (int(x2), int(y2)), [0, 0, 255], thickness)

def hough_lines(img, rho, theta, threshold=50, min_line_len=50, max_line_gap=100):
    """
    `img` should be the output of a Canny transform.
    Returns an image with hough lines drawn.
    """
    lines = cv2.HoughLinesP(img, rho, theta, threshold, np.array([]), minLineLength=min_line_len,
                            maxLineGap=max_line_gap)
    line_img = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)
    draw_lines(line_img, lines)
    return line_img

def weighted_img(img, initial_img):
    """
    `img` is the output of the hough_lines(), An image with lines drawn on it.
    Should be a blank image (all black) with lines drawn on it.
    `initial_img` should be the image before any processing.
    The result image is computed as follows:
    initial_img * alfa + img * beta + gamma
    NOTE: initial_img and img must be the same shape!
    """
    return cv2.addWeighted(initial_img, 0.8, img, 1., 0)

def process_image(image):
    # NOTE: The output you return should be a color image (3 channel) for processing video below
    # TODO: put your pipeline here,
    # you should return the final output (image where lines are drawn on lanes)
    img_gray = grayscale(image)
    (widh, height) = img_gray.shape
    img_gaus = gaussian_blur(img_gray, 3)
    img_can = canny(img_gaus, 50, 150)
    vertices = np.array([[[80, widh - 30], [(height / 2) - 50, (widh / 2) + 60], [(height / 2) + 50, (widh / 2) + 60], [height - 80, widh - 30]]], dtype=np.int32)
    img_mask = region_of_interest(img_can,vertices)
    img_hough = hough_lines(img_mask, 2, np.pi/180,50,100,100)
    result = weighted_img(img_hough, image)
    return img_hough,result

def lane_detectionv3(RGB_Camera_im):
    frame = cv2.resize(RGB_Camera_im, (640, 480))
    lines, size_im = process_image(frame)
    return lines, size_im

def object_detection_SSD(RGB_Camera_im):
    # construct the argument parse and parse the arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("-p", "--prototxt", default=ROOT_DIR + "/Models/opencv-ssd-cuda/MobileNetSSD_deploy.prototxt",
                    help="path to Caffe 'deploy' prototxt file")
    ap.add_argument("-m", "--model", default=ROOT_DIR + "/Models/opencv-ssd-cuda/MobileNetSSD_deploy.caffemodel",
                    help="path to Caffe pre-trained model")
    ap.add_argument("-i", "--input", type=str, default="../example_videos/janie.mp4",
                    help="path to (optional) input video file")
    ap.add_argument("-o", "--output", type=str, default="../output_videos/yolo_janie.avi",
                    help="path to (optional) output video file")
    ap.add_argument("-d", "--display", type=int, default=1,
                    help="whether or not output frame should be displayed")
    ap.add_argument("-c", "--confidence", type=float, default=0.2,
                    help="minimum probability to filter weak detections")
    ap.add_argument("-u", "--use-gpu", type=bool, default=True,
                    help="boolean indicating if CUDA GPU should be used")
    args = vars(ap.parse_args())

    # initialize the list of class labels MobileNet SSD was trained to
    # detect, then generate a set of bounding box colors for each class
    CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
               "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
               "dog", "horse", "motorbike", "person", "pottedplant", "sheep",
               "sofa", "train", "tvmonitor"]
    COLORS = np.random.uniform(0, 255, size=(len(CLASSES), 3))

    # load our serialized model from disk
    net = cv2.dnn.readNetFromCaffe(args["prototxt"], args["model"])

    # check if we are going to use GPU
    if args["use_gpu"]:
        # set CUDA as the preferable backend and target
        print("[INFO] setting preferable backend and target to CUDA...")
        net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
        net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)

    frame=RGB_Camera_im

    # resize the frame, grab the frame dimensions, and convert it to
    # a blob
    frame = imutils.resize(frame, width=400)
    (h, w) = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(frame, 0.007843, (300, 300), 127.5)

    # pass the blob through the network and obtain the detections and
    # predictions
    net.setInput(blob)
    detections = net.forward()

    # loop over the detections
    for i in np.arange(0, detections.shape[2]):
        # extract the confidence (i.e., probability) associated with
        # the prediction
        confidence = detections[0, 0, i, 2]

        # filter out weak detections by ensuring the `confidence` is
        # greater than the minimum confidence
        if confidence > args["confidence"]:
            # extract the index of the class label from the
            # `detections`, then compute the (x, y)-coordinates of
            # the bounding box for the object
            idx = int(detections[0, 0, i, 1])
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")

            # draw the prediction on the frame
            label = "{}: {:.2f}%".format(CLASSES[idx],
                                         confidence * 100)


            cv2.rectangle(frame, (startX, startY), (endX, endY),
                          COLORS[idx], 2)
            y = startY - 15 if startY - 15 > 15 else startY + 15
            cv2.putText(frame, label, (startX, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLORS[idx], 2)
    return 1,frame

def object_detection_Yolo(RGB_Camera_im):
    # construct the argument parse and parse the arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--input", type=str, default="../example_videos/janie.mp4",
                    help="path to (optional) input video file")
    ap.add_argument("-o", "--output", type=str, default="../output_videos/yolo_janie.avi",
                    help="path to (optional) output video file")
    ap.add_argument("-d", "--display", type=int, default=1,
                    help="whether or not output frame should be displayed")
    ap.add_argument("-y", "--yolo", type=str, default=ROOT_DIR + "/Models/opencv-yolo-cuda",
                    help="base path to YOLO directory")
    ap.add_argument("-c", "--confidence", type=float, default=0.5,
                    help="minimum probability to filter weak detections")
    ap.add_argument("-t", "--threshold", type=float, default=0.3,
                    help="threshold when applyong non-maxima suppression")
    ap.add_argument("-u", "--use-gpu", type=bool, default=1,
                    help="boolean indicating if CUDA GPU should be used")
    args = vars(ap.parse_args())

    # load the COCO class labels our YOLO model was trained on
    labelsPath = os.path.sep.join([args["yolo"], "coco.names"])
    LABELS = open(labelsPath).read().strip().split("\n")

    # initialize a list of colors to represent each possible class label
    np.random.seed(42)
    COLORS = np.random.randint(0, 255, size=(len(LABELS), 3),
                               dtype="uint8")

    # derive the paths to the YOLO weights and model configuration
    weightsPath = os.path.sep.join([args["yolo"], "yolov3.weights"])
    configPath = os.path.sep.join([args["yolo"], "yolov3.cfg"])

    # load our YOLO object detector trained on COCO dataset (80 classes)
    print("[INFO] loading YOLO from disk...")
    net = cv2.dnn.readNetFromDarknet(configPath, weightsPath)

    # check if we are going to use GPU
    if args["use_gpu"]:
        # set CUDA as the preferable backend and target
        print("[INFO] setting preferable backend and target to CUDA...")
        net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
        net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)

    # determine only the *output* layer names that we need from YOLO
    ln = net.getLayerNames()
    ln = [ln[i[0] - 1] for i in net.getUnconnectedOutLayers()]

    # initialize the width and height of the frames in the video file
    W = None
    H = None

    frame = RGB_Camera_im

    # if the frame dimensions are empty, grab them
    if W is None or H is None:
        (H, W) = frame.shape[:2]

    # construct a blob from the input frame and then perform a forward
    # pass of the YOLO object detector, giving us our bounding boxes
    # and associated probabilities
    blob = cv2.dnn.blobFromImage(frame, 1 / 255.0, (416, 416),
                                 swapRB=True, crop=False)
    net.setInput(blob)
    layerOutputs = net.forward(ln)

    # initialize our lists of detected bounding boxes, confidences,
    # and class IDs, respectively
    boxes = []
    confidences = []
    classIDs = []

    # loop over each of the layer outputs
    for output in layerOutputs:
        # loop over each of the detections
        for detection in output:
            # extract the class ID and confidence (i.e., probability)
            # of the current object detection
            scores = detection[5:]
            classID = np.argmax(scores)
            confidence = scores[classID]

            # filter out weak predictions by ensuring the detected
            # probability is greater than the minimum probability
            if confidence > args["confidence"]:
                # scale the bounding box coordinates back relative to
                # the size of the image, keeping in mind that YOLO
                # actually returns the center (x, y)-coordinates of
                # the bounding box followed by the boxes' width and
                # height
                box = detection[0:4] * np.array([W, H, W, H])
                (centerX, centerY, width, height) = box.astype("int")

                # use the center (x, y)-coordinates to derive the top
                # and and left corner of the bounding box
                x = int(centerX - (width / 2))
                y = int(centerY - (height / 2))

                # update our list of bounding box coordinates,
                # confidences, and class IDs
                boxes.append([x, y, int(width), int(height)])
                confidences.append(float(confidence))
                classIDs.append(classID)

    # apply non-maxima suppression to suppress weak, overlapping
    # bounding boxes
    idxs = cv2.dnn.NMSBoxes(boxes, confidences, args["confidence"],
                            args["threshold"])

    # ensure at least one detection exists
    if len(idxs) > 0:
        # loop over the indexes we are keeping
        for i in idxs.flatten():
            # extract the bounding box coordinates
            (x, y) = (boxes[i][0], boxes[i][1])
            (w, h) = (boxes[i][2], boxes[i][3])

            # draw a bounding box rectangle and label on the frame
            color = [int(c) for c in COLORS[classIDs[i]]]
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            text = "{}: {:.4f}".format(LABELS[classIDs[i]],
                                       confidences[i])
            cv2.putText(frame, text, (x, y - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    return 1,frame

def object_detection_mask(RGB_Camera_im):
    # construct the argument parse and parse the arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--input", type=str, default="../example_videos/janie.mp4",
                    help="path to (optional) input video file")
    ap.add_argument("-o", "--output", type=str, default="../output_videos/mask_janie.avi",
                    help="path to (optional) output video file")
    ap.add_argument("-d", "--display", type=int, default=1,
                    help="whether or not output frame should be displayed")
    ap.add_argument("-m", "--mask-rcnn", type=str, default=ROOT_DIR + "/Models/mask-rcnn-coco/",
                    help="base path to mask-rcnn directory")
    ap.add_argument("-c", "--confidence", type=float, default=0.5,
                    help="minimum probability to filter weak detections")
    ap.add_argument("-t", "--threshold", type=float, default=0.3,
                    help="minimum threshold for pixel-wise mask segmentation")
    ap.add_argument("-u", "--use-gpu", type=bool, default=1,
                    help="boolean indicating if CUDA GPU should be used")
    args = vars(ap.parse_args())

    # load the COCO class labels our Mask R-CNN was trained on
    labelsPath = os.path.sep.join([args["mask_rcnn"],
                                   "object_detection_classes_coco.txt"])
    LABELS = open(labelsPath).read().strip().split("\n")

    # initialize a list of colors to represent each possible class label
    np.random.seed(42)
    COLORS = np.random.randint(0, 255, size=(len(LABELS), 3),
                               dtype="uint8")

    # derive the paths to the Mask R-CNN weights and model configuration
    weightsPath = os.path.sep.join([args["mask_rcnn"],
                                    "frozen_inference_graph.pb"])
    configPath = os.path.sep.join([args["mask_rcnn"],
                                   "mask_rcnn_inception_v2_coco_2018_01_28.pbtxt"])

    # load our Mask R-CNN trained on the COCO dataset (90 classes)
    # from disk
    print("[INFO] loading Mask R-CNN from disk...")
    net = cv2.dnn.readNetFromTensorflow(weightsPath, configPath)

    # check if we are going to use GPU
    if args["use_gpu"]:
        # set CUDA as the preferable backend and target
        print("[INFO] setting preferable backend and target to CUDA...")
        net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
        net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)

    frame = RGB_Camera_im

    blob = cv2.dnn.blobFromImage(frame, swapRB=True, crop=False)
    net.setInput(blob)
    (boxes, masks) = net.forward(["detection_out_final",
                                  "detection_masks"])

    # loop over the number of detected objects
    for i in range(0, boxes.shape[2]):
        # extract the class ID of the detection along with the
        # confidence (i.e., probability) associated with the
        # prediction
        classID = int(boxes[0, 0, i, 1])
        confidence = boxes[0, 0, i, 2]

        # filter out weak predictions by ensuring the detected
        # probability is greater than the minimum probability
        if confidence > args["confidence"]:
            # scale the bounding box coordinates back relative to the
            # size of the frame and then compute the width and the
            # height of the bounding box
            (H, W) = frame.shape[:2]
            box = boxes[0, 0, i, 3:7] * np.array([W, H, W, H])
            (startX, startY, endX, endY) = box.astype("int")
            boxW = endX - startX
            boxH = endY - startY

            # extract the pixel-wise segmentation for the object,
            # resize the mask such that it's the same dimensions of
            # the bounding box, and then finally threshold to create
            # a *binary* mask
            mask = masks[i, classID]
            mask = cv2.resize(mask, (boxW, boxH),
                              interpolation=cv2.INTER_CUBIC)
            mask = (mask > args["threshold"])

            # extract the ROI of the image but *only* extracted the
            # masked region of the ROI
            roi = frame[startY:endY, startX:endX][mask]

            # grab the color used to visualize this particular class,
            # then create a transparent overlay by blending the color
            # with the ROI
            color = COLORS[classID]
            blended = ((0.4 * color) + (0.6 * roi)).astype("uint8")

            # store the blended ROI in the original frame
            frame[startY:endY, startX:endX][mask] = blended

            # draw the bounding box of the instance on the frame
            color = [int(c) for c in color]
            cv2.rectangle(frame, (startX, startY), (endX, endY),
                          color, 2)

            # draw the predicted label and associated probability of
            # the instance segmentation on the frame
            text = "{}: {:.4f}".format(LABELS[classID], confidence)
            cv2.putText(frame, text, (startX, startY - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        return 1, frame
# ==============================================================================
# -- Agent ---------------------------------------------------------
# ==============================================================================
from Agents.Agent import Agent
from Agents.AgentState import AgentState
from Agents.BasicAgent import BasicAgent
from Agents.RoamingAgent import RoamingAgent
from Agents.BehaviorAgent import BehaviorAgent
# ==============================================================================
# -- Typeofbehavior ---------------------------------------------------------
# ==============================================================================
from Agents.Typeofbehavior import Cautious, Normal, Aggressive
# ==============================================================================
# -- World ---------------------------------------------------------------
# ==============================================================================
from World.World import World
# ==============================================================================
# -- WorldRepresentation -------------------------------------------------------
# ==============================================================================
from World.World import WorldRepresentation
# ==============================================================================
# -- KeyboardControl -----------------------------------------------------------
# ==============================================================================
from simulator.KeyboardControl import KeyboardControl
# ==============================================================================
# -- HUD -----------------------------------------------------------------------
# ==============================================================================
from simulator.HUD import HUD
# ==============================================================================
# -- FadingText ----------------------------------------------------------------
# ==============================================================================
from simulator.FadingText import FadingText
# ==============================================================================
# -- HelpText ------------------------------------------------------------------
# ==============================================================================
from simulator.HelpText import HelpText
# ==============================================================================
# -- CollisionSensor -----------------------------------------------------------
# ==============================================================================
from Perception.CollisionSensor import CollisionSensor
# ==============================================================================
# -- LaneInvasionSensor --------------------------------------------------------
# ==============================================================================
from Perception.LaneInvasionSensor import LaneInvasionSensor
# ==============================================================================
# -- GnssSensor --------------------------------------------------------
# ==============================================================================
from Perception.GnssSensor import GnssSensor
# ==============================================================================
# -- RadarSensor -------------------------------------------------------------
# ==============================================================================
from Perception.RadarSensor import RadarSensor
# ==============================================================================
# -- CameraManager -------------------------------------------------------------
# ==============================================================================
from Perception.CameraManager import CameraManager
# ==============================================================================
# -- GlobalRoutePlanner ---------------------------------------------------------
# ==============================================================================
from GlobalRoutePlanner.RoadOption import RoadOption
from GlobalRoutePlanner.GlobalRoutePlanner import GlobalRoutePlanner
from GlobalRoutePlanner.GlobalRoutePlannerDAO import GlobalRoutePlannerDAO
# ==============================================================================
# -- LocalPlanner ---------------------------------------------------------
# ==============================================================================
from LocalPlanner.LocalPlanner import LocalPlanner
from LocalPlanner.LocalPlannerBehavior import LocalPlannerBehavior
# ==============================================================================
# -- Controller ---------------------------------------------------------
# ==============================================================================
from Controller.VehiclePIDController import VehiclePIDController
from Controller.PIDLongitudinalController import PIDLongitudinalController
from Controller.PIDLateralController import PIDLateralController
# ==============================================================================
# -- Game Loop ---------------------------------------------------------
# ==============================================================================

def game_loop(args):
    """ Main loop for agent"""

    pygame.init()
    pygame.font.init()
    world = None
    tot_target_reached = 0
    num_min_waypoints = 21
    counter=0

    try:
        client = carla.Client(args.host, args.port)
        client.set_timeout(4.0)

        display = pygame.display.set_mode(
            (args.width, args.height),
            pygame.HWSURFACE | pygame.DOUBLEBUF)

        hud = HUD(args.width, args.height)
        world = World(client.load_world('Town01'), hud, args)
         # Changing The Map
        #world = World(client.load_world('Town03'), hud, args)
        # Town04 ,Town06 is highway | Town07 is country |Town03 default
        controller = KeyboardControl(world)

        if args.agent == "Roaming":
            agent = RoamingAgent(world.player)
        elif args.agent == "Basic":
            agent = BasicAgent(world.player)
            spawn_point = world.map.get_spawn_points()[0]
            agent.set_destination((spawn_point.location.x,
                                   spawn_point.location.y,
                                   spawn_point.location.z))
        else:
            agent = BehaviorAgent(world.player, behavior=args.behavior)

            spawn_points = world.map.get_spawn_points()
            random.shuffle(spawn_points)

            if spawn_points[0].location != agent.vehicle.get_location():
                destination = spawn_points[0].location
            else:
                destination = spawn_points[1].location

            agent.set_destination(agent.vehicle.get_location(), destination, clean=True)

        clock = pygame.time.Clock()

        while True:
            clock.tick_busy_loop(60)
            if controller.parse_events(client, world, clock):
                return

            # As soon as the server is ready continue!
            if not world.world.wait_for_tick(10.0):
                continue

            if args.agent == "Roaming" or args.agent == "Basic":
                if controller.parse_events(client, world, clock):
                    return

                # as soon as the server is ready continue!
                world.world.wait_for_tick(10.0)

                world.tick(clock)
                world.render(display)
                pygame.display.flip()
                control = agent.run_step(world.player)
                control.manual_gear_shift = False
                world.player.apply_control(control)
            else:
                agent.update_information(world)

                world.tick(clock)
                world.render(display)
                pygame.display.flip()

                # Set new destination when target has been reached
                if len(agent.get_local_planner()._waypoints_queue) < num_min_waypoints and args.loop:
                    agent.reroute(spawn_points)
                    tot_target_reached += 1
                    world.hud.notification("The target has been reached " +
                                           str(tot_target_reached) + " times.", seconds=4.0)

                elif len(agent.get_local_planner()._waypoints_queue) == 0 and not args.loop:
                    print("Target reached, mission accomplished...")
                    break

                speed_limit = world.player.get_speed_limit()
                agent.get_local_planner().set_speed(speed_limit)

                control = agent.run_step()
                world.player.apply_control(control)

            # #################################################
            # # it's my code
            # pt1_sum_ri = (0, 0)
            # pt2_sum_ri = (0, 0)
            # pt1_avg_ri = (0, 0)
            # count_posi_num_ri = 0
            #
            # pt1_sum_le = (0, 0)
            # pt2_sum_le = (0, 0)
            # pt1_avg_le = (0, 0)
            #
            # count_posi_num_le = 0
            #
            #
            # global Camera_image
            # RGB_Camera_im = cv2.cvtColor(Camera_image, cv2.COLOR_BGR2RGB)
            #
            # # Test lane dectection ,object detecion based on SSD, Yolo and Semantic Segmentation
            # #lines,size_im= lane_detectionv3(RGB_Camera_im)
            # #lines,size_im=object_detection_SSD(RGB_Camera_im)
            # #lines, size_im = object_detection_Yolo(RGB_Camera_im)
            # #lines, size_im = object_detection_mask(RGB_Camera_im)
            # #lines, size_im = lane_detectionv2(RGB_Camera_im)
            #
            # if lines is None: #in case HoughLinesP fails to return a set of lines
            #         #make sure that this is the right shape [[ ]] and ***not*** []
            #         lines = [[0,0,0,0]]
            # else:
            #
            #     cv2.imshow('frame_size_im', size_im)
            #     cv2.waitKey(1)
            #     #cv2.imshow("test_im", test_im) # original size image
            #     #cv2.waitKey(1)

            #####################################################3
            # test= WorldRepresentation(world.world, world.player, args)
            # counter += 1
            # if ((counter % 10) == 0):
            #     print(test.dynamic_objects())
            ##########################################################3


    finally:
        if world is not None:
            world.destroy()

        pygame.quit()

# ==============================================================================
# -- main() --------------------------------------------------------------
# ==============================================================================

def main():
    """Main method"""

    argparser = argparse.ArgumentParser(
        description='CARLA Automatic Control Client')
    argparser.add_argument(
        '-v', '--verbose',
        action='store_true',
        dest='debug',
        help='Print debug information')
    argparser.add_argument(
        '--host',
        metavar='H',
        default='127.0.0.1',
        help='IP of the host server (default: 127.0.0.1)')
    argparser.add_argument(
        '-p', '--port',
        metavar='P',
        default=2000,
        type=int,
        help='TCP port to listen to (default: 2000)')
    argparser.add_argument(
        '--res',
        metavar='WIDTHxHEIGHT',
        default='640x480',#'1280x720',
        help='Window resolution (default: 640x480)')
    argparser.add_argument(
        '--filter',
        metavar='PATTERN',
        default='vehicle.tesla.model3',
        help='Actor filter (default: "vehicle.tesla.model3"))')
    argparser.add_argument(
        '--gamma',
        default=2.2,
        type=float,
        help='Gamma correction of the camera (default: 2.2)')
    argparser.add_argument(
        '-l', '--loop',
        action='store_true',
        dest='loop',
        help='Sets a new random destination upon reaching the previous one (default: False)')
    argparser.add_argument(
        '-b', '--behavior', type=str,
        choices=["cautious", "normal", "aggressive"],
        help='Choose one of the possible agent behaviors (default: normal) ',
        default='normal')
    argparser.add_argument("-a", "--agent", type=str,
                           choices=["Behavior", "Roaming", "Basic"],
                           help="select which agent to run",
                           default="Basic")
    argparser.add_argument(
        '-s', '--seed',
        help='Set seed for repeating executions (default: None)',
        default=None,
        type=int)

    global args
    args = argparser.parse_args()

    args.width, args.height = [int(x) for x in args.res.split('x')]

    log_level = logging.DEBUG if args.debug else logging.INFO
    logging.basicConfig(format='%(levelname)s: %(message)s', level=log_level)

    logging.info('listening to server %s:%s', args.host, args.port)

    print(__doc__)

    try:
        game_loop(args)

    except KeyboardInterrupt:
        print('\nCancelled by user. Bye!')


if __name__ == '__main__':
    main()

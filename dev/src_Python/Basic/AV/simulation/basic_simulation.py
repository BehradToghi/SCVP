from carla.settings   import CarlaSettings
from carla.client     import make_carla_client, VehicleControl
import configparser
import time
import os
import csv
import numpy as np
"""
Configurable params
"""
ITER_FOR_SIM_TIMESTEP  = 10     # no. iterations to compute approx sim timestep
WAIT_TIME_BEFORE_START = 1.00   # game seconds (time before controller start)
TOTAL_RUN_TIME         = 100.00 # game seconds (total runtime before sim end)
TOTAL_FRAME_BUFFER     = 300    # number of frames to buffer after total runtime
NUM_PEDESTRIANS        = 0      # total number of pedestrians to spawn
NUM_VEHICLES           = 2      # total number of vehicles to spawn
SEED_PEDESTRIANS       = 0      # seed for pedestrian spawn randomizer
SEED_VEHICLES          = 0      # seed for vehicle spawn randomizer
CLIENT_WAIT_TIME       = 3      # wait time for client before starting episode
                                # used to make sure the server loads
                                # consistently


WEATHERID = {
    "DEFAULT": 0,
    "CLEARNOON": 1,
    "CLOUDYNOON": 2,
    "WETNOON": 3,
    "WETCLOUDYNOON": 4,
    "MIDRAINYNOON": 5,
    "HARDRAINNOON": 6,
    "SOFTRAINNOON": 7,
    "CLEARSUNSET": 8,
    "CLOUDYSUNSET": 9,
    "WETSUNSET": 10,
    "WETCLOUDYSUNSET": 11,
    "MIDRAINSUNSET": 12,
    "HARDRAINSUNSET": 13,
    "SOFTRAINSUNSET": 14,
}
SIMWEATHER = WEATHERID["CLEARNOON"]     # set simulation weather

PLAYER_START_INDEX = 1      # spawn index for player (keep to 1)




WAYPOINTS_FILENAME = '/home/cavrel/PycharmProjects/Autonomous_Vehicle/Coursera AV/Carla/PythonClient/AV/simulation/course4_waypoints.txt'  # waypoint file to load
DIST_THRESHOLD_TO_LAST_WAYPOINT = 2.0  # some distance from last position before
                                       # simulation ends


# Course 4 specific parameters
C4_STOP_SIGN_FILE        = '/home/cavrel/PycharmProjects/Autonomous_Vehicle/Coursera AV/Carla/PythonClient/AV/simulation/stop_sign_params.txt'
C4_STOP_SIGN_FENCELENGTH = 5        # m
C4_PARKED_CAR_FILE       = '/home/cavrel/PycharmProjects/Autonomous_Vehicle/Coursera AV/Carla/PythonClient/AV/simulation/parked_vehicle_params.txt'


# Path interpolation parameters
INTERP_MAX_POINTS_PLOT    = 10   # number of points used for displaying
                                 # selected path
INTERP_DISTANCE_RES       = 0.01 # distance between interpolated points

def make_carla_settings(args):
    """Make a CarlaSettings object with the settings we need.
    """
    settings = CarlaSettings()

    # There is no need for non-agent info requests if there are no pedestrians
    # or vehicles.
    get_non_player_agents_info = False
    if (NUM_PEDESTRIANS > 0 or NUM_VEHICLES > 0):
        get_non_player_agents_info = True

    # Base level settings
    settings.set(
        SynchronousMode=True,
        SendNonPlayerAgentsInfo=get_non_player_agents_info,
        NumberOfVehicles=NUM_VEHICLES,
        NumberOfPedestrians=NUM_PEDESTRIANS,
        SeedVehicles=SEED_VEHICLES,
        SeedPedestrians=SEED_PEDESTRIANS,
        WeatherId=SIMWEATHER,
        QualityLevel=args.quality_level)
    return settings


class Timer(object):
    """ Timer Class

    The steps are used to calculate FPS, while the lap or seconds since lap is
    used to compute elapsed time.
    """

    def __init__(self, period):
        self.step = 0
        self._lap_step = 0
        self._lap_time = time.time()
        self._period_for_lap = period

    def tick(self):
        self.step += 1

    def has_exceeded_lap_period(self):
        if self.elapsed_seconds_since_lap() >= self._period_for_lap:
            return True
        else:
            return False

    def lap(self):
        self._lap_step = self.step
        self._lap_time = time.time()

    def ticks_per_second(self):
        return float(self.step - self._lap_step) / \
               self.elapsed_seconds_since_lap()

    def elapsed_seconds_since_lap(self):
        return time.time() - self._lap_time


def send_control_command(client, throttle, steer, brake,
                         hand_brake=False, reverse=False):
    """Send control command to CARLA client.

    Send control command to CARLA client.

    Args:
        client: The CARLA client object
        throttle: Throttle command for the sim car [0, 1]
        steer: Steer command for the sim car [-1, 1]
        brake: Brake command for the sim car [0, 1]
        hand_brake: Whether the hand brake is engaged
        reverse: Whether the sim car is in the reverse gear
    """
    control = VehicleControl()
    # Clamp all values within their limits
    steer = np.fmax(np.fmin(steer, 1.0), -1.0)
    throttle = np.fmax(np.fmin(throttle, 1.0), 0)
    brake = np.fmax(np.fmin(brake, 1.0), 0)

    control.steer = steer
    control.throttle = throttle
    control.brake = brake
    control.hand_brake = hand_brake
    control.reverse = reverse
    client.send_control(control)

class Simulation2(object):
    def __init__(self,args):

        self.SIMULATION_TIME_STEP = None
        self.TOTAL_EPISODE_FRAMES = None
        self.stopsign_fences=None
        self.parkedcar_box_pts=None
        self.waypoints=None
        self.waypoints_np =None
        # self.client=self.create_simulation(args)

    def send_control_command(self, client,throttle, steer, brake,
                             hand_brake=False, reverse=False):
        """Send control command to CARLA client.

        Send control command to CARLA client.

        Args:
            client: The CARLA client object
            throttle: Throttle command for the sim car [0, 1]
            steer: Steer command for the sim car [-1, 1]
            brake: Brake command for the sim car [0, 1]
            hand_brake: Whether the hand brake is engaged
            reverse: Whether the sim car is in the reverse gear
        """
        control = VehicleControl()
        # Clamp all values within their limits
        steer = np.fmax(np.fmin(steer, 1.0), -1.0)
        throttle = np.fmax(np.fmin(throttle, 1.0), 0)
        brake = np.fmax(np.fmin(brake, 1.0), 0)

        control.steer = steer
        control.throttle = throttle
        control.brake = brake
        control.hand_brake = hand_brake
        control.reverse = reverse
        client.send_control(control)

    def create_simulation(self, args):

        with make_carla_client(args.host, args.port) as client:
            print('Carla client connected.')

            settings = make_carla_settings(args)

            # Now we load these settings into the server. The server replies
            # with a scene description containing the available start spots for
            # the player. Here we can provide a CarlaSettings object or a
            # CarlaSettings.ini file as string.
            scene = client.load_settings(settings)

            # Refer to the player start folder in the WorldOutliner to see the
            # player start information
            player_start = PLAYER_START_INDEX

            client.start_episode(player_start)

            time.sleep(CLIENT_WAIT_TIME);

            # Notify the server that we want to start the episode at the
            # player_start index. This function blocks until the server is ready
            # to start the episode.
            # print('Starting new episode at %r...' % scene.map_name)
            client.start_episode(player_start)


            #############################################
            # Load stop sign and parked vehicle parameters
            # Convert to input params for LP
            #############################################
            # Stop sign (X(m), Y(m), Z(m), Yaw(deg))
            stopsign_data = None
            stopsign_fences = []  # [x0, y0, x1, y1]
            with open(C4_STOP_SIGN_FILE, 'r') as stopsign_file:
                next(stopsign_file)  # skip header
                stopsign_reader = csv.reader(stopsign_file,
                                             delimiter=',',
                                             quoting=csv.QUOTE_NONNUMERIC)
                stopsign_data = list(stopsign_reader)
                # convert to rad
                for i in range(len(stopsign_data)):
                    stopsign_data[i][3] = stopsign_data[i][3] * np.pi / 180.0

                    # obtain stop sign fence points for LP
            for i in range(len(stopsign_data)):
                x = stopsign_data[i][0]
                y = stopsign_data[i][1]
                z = stopsign_data[i][2]
                yaw = stopsign_data[i][3] + np.pi / 2.0  # add 90 degrees for fence
                spos = np.array([
                    [0, 0],
                    [0, C4_STOP_SIGN_FENCELENGTH]])
                rotyaw = np.array([
                    [np.cos(yaw), np.sin(yaw)],
                    [-np.sin(yaw), np.cos(yaw)]])
                spos_shift = np.array([
                    [x, x],
                    [y, y]])
                spos = np.add(np.matmul(rotyaw, spos), spos_shift)
                stopsign_fences.append([spos[0, 0], spos[1, 0], spos[0, 1], spos[1, 1]])

            # Parked car(s) (X(m), Y(m), Z(m), Yaw(deg), RADX(m), RADY(m), RADZ(m))
            parkedcar_data = None
            parkedcar_box_pts = []  # [x,y]
            with open(C4_PARKED_CAR_FILE, 'r') as parkedcar_file:
                next(parkedcar_file)  # skip header
                parkedcar_reader = csv.reader(parkedcar_file,
                                              delimiter=',',
                                              quoting=csv.QUOTE_NONNUMERIC)
                parkedcar_data = list(parkedcar_reader)
                # convert to rad
                for i in range(len(parkedcar_data)):
                    parkedcar_data[i][3] = parkedcar_data[i][3] * np.pi / 180.0

                    # obtain parked car(s) box points for LP
            for i in range(len(parkedcar_data)):
                x = parkedcar_data[i][0]
                y = parkedcar_data[i][1]
                z = parkedcar_data[i][2]
                yaw = parkedcar_data[i][3]
                xrad = parkedcar_data[i][4]
                yrad = parkedcar_data[i][5]
                zrad = parkedcar_data[i][6]
                cpos = np.array([
                    [-xrad, -xrad, -xrad, 0, xrad, xrad, xrad, 0],
                    [-yrad, 0, yrad, yrad, yrad, 0, -yrad, -yrad]])
                rotyaw = np.array([
                    [np.cos(yaw), np.sin(yaw)],
                    [-np.sin(yaw), np.cos(yaw)]])
                cpos_shift = np.array([
                    [x, x, x, x, x, x, x, x],
                    [y, y, y, y, y, y, y, y]])
                cpos = np.add(np.matmul(rotyaw, cpos), cpos_shift)
                for j in range(cpos.shape[1]):
                    parkedcar_box_pts.append([cpos[0, j], cpos[1, j]])

            #############################################
            # Load Waypoints
            #############################################
            # Opens the waypoint file and stores it to "waypoints"
            waypoints_file = WAYPOINTS_FILENAME
            waypoints_filepath = \
                os.path.join(os.path.dirname(os.path.realpath(__file__)),
                             WAYPOINTS_FILENAME)
            waypoints_np = None
            with open(waypoints_filepath) as waypoints_file_handle:
                waypoints = list(csv.reader(waypoints_file_handle,
                                            delimiter=',',
                                            quoting=csv.QUOTE_NONNUMERIC))
                waypoints_np = np.array(waypoints)


            #############################################
            # Determine simulation average timestep (and total frames)
            #############################################
            # Ensure at least one frame is used to compute average timestep
            num_iterations = ITER_FOR_SIM_TIMESTEP
            if (ITER_FOR_SIM_TIMESTEP < 1):
                num_iterations = 1

            # Gather current data from the CARLA server. This is used to get the
            # simulator starting game time. Note that we also need to
            # send a command back to the CARLA server because synchronous mode
            # is enabled.
            measurement_data, sensor_data = client.read_data()
            sim_start_stamp = measurement_data.game_timestamp / 1000.0
            # Send a control command to proceed to next iteration.
            # This mainly applies for simulations that are in synchronous mode.
            self.send_control_command(client, throttle=0.0, steer=0, brake=1.0)
            # Computes the average timestep based on several initial iterations
            sim_duration = 0
            for i in range(num_iterations):
                # Gather current data
                measurement_data, sensor_data = client.read_data()
                # Send a control command to proceed to next iteration
                self.send_control_command(client, throttle=0.0, steer=0, brake=1.0)
                # Last stamp
                if i == num_iterations - 1:
                    sim_duration = measurement_data.game_timestamp / 1000.0 - \
                                   sim_start_stamp

                    # Outputs average simulation timestep and computes how many frames
            # will elapse before the simulation should end based on various
            # parameters that we set in the beginning.
            SIMULATION_TIME_STEP = sim_duration / float(num_iterations)
            print("SERVER SIMULATION STEP APPROXIMATION: " + \
                  str(SIMULATION_TIME_STEP))
            TOTAL_EPISODE_FRAMES = int((TOTAL_RUN_TIME + WAIT_TIME_BEFORE_START) / \
                                       SIMULATION_TIME_STEP) + TOTAL_FRAME_BUFFER

            self.SIMULATION_TIME_STEP=SIMULATION_TIME_STEP
            self.TOTAL_EPISODE_FRAMES=TOTAL_EPISODE_FRAMES

            self.stopsign_fences = stopsign_fences
            self.parkedcar_box_pts = parkedcar_box_pts
            self.waypoints = waypoints
            self.waypoints_np = waypoints_np

            return client


class Simulation(object):
    def __init__(self,SIMULATION_TIME_STEP, TOTAL_EPISODE_FRAMES, stopsign_fences, stopsign_data, parkedcar_box_pts,
                   parkedcar_data, waypoints,
                   waypoints_np):

        self.SIMULATION_TIME_STEP = SIMULATION_TIME_STEP
        self.TOTAL_EPISODE_FRAMES = TOTAL_EPISODE_FRAMES
        self.stopsign_fences=stopsign_fences
        self.parkedcar_box_pts=parkedcar_box_pts
        self.stopsign_fences_raw = stopsign_data
        self.parkedcar_box_pts_raw = parkedcar_data

        self.waypoints=waypoints
        self.waypoints_np =waypoints_np
        # self.client=self.create_simulation(args)


def create_simulation(args,client):
    settings = make_carla_settings(args)

    # Now we load these settings into the server. The server replies
    # with a scene description containing the available start spots for
    # the player. Here we can provide a CarlaSettings object or a
    # CarlaSettings.ini file as string.
    scene = client.load_settings(settings)

    # Refer to the player start folder in the WorldOutliner to see the
    # player start information
    player_start = PLAYER_START_INDEX

    client.start_episode(player_start)

    time.sleep(CLIENT_WAIT_TIME);

    # Notify the server that we want to start the episode at the
    # player_start index. This function blocks until the server is ready
    # to start the episode.
    # print('Starting new episode at %r...' % scene.map_name)
    client.start_episode(player_start)

    #############################################
    # Load stop sign and parked vehicle parameters
    # Convert to input params for LP
    #############################################
    # Stop sign (X(m), Y(m), Z(m), Yaw(deg))
    stopsign_data = None
    stopsign_fences = []  # [x0, y0, x1, y1]
    with open(C4_STOP_SIGN_FILE, 'r') as stopsign_file:
        next(stopsign_file)  # skip header
        stopsign_reader = csv.reader(stopsign_file,
                                     delimiter=',',
                                     quoting=csv.QUOTE_NONNUMERIC)
        stopsign_data = list(stopsign_reader)
        # convert to rad
        for i in range(len(stopsign_data)):
            stopsign_data[i][3] = stopsign_data[i][3] * np.pi / 180.0

            # obtain stop sign fence points for LP
    for i in range(len(stopsign_data)):
        x = stopsign_data[i][0]
        y = stopsign_data[i][1]
        z = stopsign_data[i][2]
        yaw = stopsign_data[i][3] + np.pi / 2.0  # add 90 degrees for fence
        spos = np.array([
            [0, 0],
            [0, C4_STOP_SIGN_FENCELENGTH]])
        rotyaw = np.array([
            [np.cos(yaw), np.sin(yaw)],
            [-np.sin(yaw), np.cos(yaw)]])
        spos_shift = np.array([
            [x, x],
            [y, y]])
        spos = np.add(np.matmul(rotyaw, spos), spos_shift)
        stopsign_fences.append([spos[0, 0], spos[1, 0], spos[0, 1], spos[1, 1]])

    # Parked car(s) (X(m), Y(m), Z(m), Yaw(deg), RADX(m), RADY(m), RADZ(m))
    parkedcar_data = None
    parkedcar_box_pts = []  # [x,y]
    with open(C4_PARKED_CAR_FILE, 'r') as parkedcar_file:
        next(parkedcar_file)  # skip header
        parkedcar_reader = csv.reader(parkedcar_file,
                                      delimiter=',',
                                      quoting=csv.QUOTE_NONNUMERIC)
        parkedcar_data = list(parkedcar_reader)
        # convert to rad
        for i in range(len(parkedcar_data)):
            parkedcar_data[i][3] = parkedcar_data[i][3] * np.pi / 180.0

            # obtain parked car(s) box points for LP
    for i in range(len(parkedcar_data)):
        x = parkedcar_data[i][0]
        y = parkedcar_data[i][1]
        z = parkedcar_data[i][2]
        yaw = parkedcar_data[i][3]
        xrad = parkedcar_data[i][4]
        yrad = parkedcar_data[i][5]
        zrad = parkedcar_data[i][6]
        cpos = np.array([
            [-xrad, -xrad, -xrad, 0, xrad, xrad, xrad, 0],
            [-yrad, 0, yrad, yrad, yrad, 0, -yrad, -yrad]])
        rotyaw = np.array([
            [np.cos(yaw), np.sin(yaw)],
            [-np.sin(yaw), np.cos(yaw)]])
        cpos_shift = np.array([
            [x, x, x, x, x, x, x, x],
            [y, y, y, y, y, y, y, y]])
        cpos = np.add(np.matmul(rotyaw, cpos), cpos_shift)
        for j in range(cpos.shape[1]):
            parkedcar_box_pts.append([cpos[0, j], cpos[1, j]])

    #############################################
    # Load Waypoints
    #############################################
    # Opens the waypoint file and stores it to "waypoints"
    waypoints_file = WAYPOINTS_FILENAME
    waypoints_filepath = \
        os.path.join(os.path.dirname(os.path.realpath(__file__)),
                     WAYPOINTS_FILENAME)
    waypoints_np = None
    with open(waypoints_filepath) as waypoints_file_handle:
        waypoints = list(csv.reader(waypoints_file_handle,
                                    delimiter=',',
                                    quoting=csv.QUOTE_NONNUMERIC))
        waypoints_np = np.array(waypoints)

    #############################################
    # Determine simulation average timestep (and total frames)
    #############################################
    # Ensure at least one frame is used to compute average timestep
    num_iterations = ITER_FOR_SIM_TIMESTEP
    if (ITER_FOR_SIM_TIMESTEP < 1):
        num_iterations = 1

    # Gather current data from the CARLA server. This is used to get the
    # simulator starting game time. Note that we also need to
    # send a command back to the CARLA server because synchronous mode
    # is enabled.
    measurement_data, sensor_data = client.read_data()
    sim_start_stamp = measurement_data.game_timestamp / 1000.0
    # Send a control command to proceed to next iteration.
    # This mainly applies for simulations that are in synchronous mode.
    send_control_command(client, throttle=0.0, steer=0, brake=1.0)
    # Computes the average timestep based on several initial iterations
    sim_duration = 0
    for i in range(num_iterations):
        # Gather current data
        measurement_data, sensor_data = client.read_data()
        # Send a control command to proceed to next iteration
        send_control_command(client, throttle=0.0, steer=0, brake=1.0)
        # Last stamp
        if i == num_iterations - 1:
            sim_duration = measurement_data.game_timestamp / 1000.0 - \
                           sim_start_stamp

            # Outputs average simulation timestep and computes how many frames
    # will elapse before the simulation should end based on various
    # parameters that we set in the beginning.
    SIMULATION_TIME_STEP = sim_duration / float(num_iterations)
    print("SERVER SIMULATION STEP APPROXIMATION: " + \
          str(SIMULATION_TIME_STEP))
    TOTAL_EPISODE_FRAMES = int((TOTAL_RUN_TIME + WAIT_TIME_BEFORE_START) / \
                               SIMULATION_TIME_STEP) + TOTAL_FRAME_BUFFER

    #############################################
    # Frame-by-Frame Iteration and Initialization
    #############################################
    measurement_data, sensor_data = client.read_data()
    send_control_command(client, throttle=0.0, steer=0, brake=1.0)

    sim = Simulation(SIMULATION_TIME_STEP, TOTAL_EPISODE_FRAMES, stopsign_fences, stopsign_data, parkedcar_box_pts,parkedcar_data, waypoints,
                     waypoints_np)

    return measurement_data,sim
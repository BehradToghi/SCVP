import math

from world.basic_world import Vehicle, History



class Perception(object):
    def __init__(self,measurement_data,sim):
        self.objects=None
        self.lane_waypoint=None
        self.drivable_area=None
        self.ego_vehicle = Vehicle()



        self.ego_vehicle.localization = None  # (x, y, yaw)
        #self.ego_vehicle.start_pos=self.get_start_pos(scene)
        self.ego_vehicle.history=History()
        self.current_timestamp=None


        self.player_collided_flag = None
        # Initialize collision history
        self.prev_collision_vehicles    = 0
        self.prev_collision_pedestrians = 0
        self.prev_collision_other       = 0

        self.lead_car_pos = []
        self.lead_car_length = []
        self.lead_car_speed = []

        self.stopsign_fences = sim.stopsign_fences
        self.parkedcar_box_pts = sim.parkedcar_box_pts
        self.waypoints = sim.waypoints
        self.waypoints_np = sim.waypoints_np

        self.lane_waypoints = sim.waypoints
        self.stopsign_fences_raw = sim.stopsign_fences_raw
        self.parkedcar_box_pts_raw = sim.parkedcar_box_pts_raw

        self.first_step(measurement_data)

    def get_start_pos(self, scene):
        """Obtains player start x,y, yaw pose from the scene

        Obtains the player x,y, and yaw pose from the scene.

        Args:
            scene: The CARLA scene object

        Returns: (x, y, yaw)
            x: X position in meters
            y: Y position in meters
            yaw: Yaw position in radians
        """
        x = scene.player_start_spots[0].location.x
        y = scene.player_start_spots[0].location.y
        yaw = math.radians(scene.player_start_spots[0].rotation.yaw)

        return (x, y, yaw)

    def first_step(self,measurement):
        self.current_timestamp = measurement.game_timestamp / 1000.0
        self.ego_vehicle.localization = self.get_current_pose(measurement)
        # Store pose history starting from the start position
        x, y, yaw = self.ego_vehicle.localization
        self.ego_vehicle.history.x_history=[x]
        self.ego_vehicle.history.y_history=[y]
        self.ego_vehicle.history.yaw_history=[yaw]
        self.ego_vehicle.history.time_history=[0]
        self.ego_vehicle.history.speed_history=[0]
        self.ego_vehicle.history.collided_flag_history = [False]  # assume player starts off non-collided

        # Initialize collision history
        self.prev_collision_vehicles=0
        self.prev_collision_pedestrians=0
        self.prev_collision_other = 0



    def run_step(self,measurement):

        self.ego_vehicle.localization=self.get_current_pose(measurement)


        self.player_collided_flag=self.get_player_collided_flag(measurement,
                                 self.prev_collision_vehicles,
                                 self.prev_collision_pedestrians,
                                 self.prev_collision_other)
        self.collided_flag, \
        self.prev_collision_vehicles, \
        self.prev_collision_pedestrians, \
        self.prev_collision_other = self.player_collided_flag

        # Store collision history

        self.ego_vehicle.history.collided_flag_history.append(
            self.collided_flag)  # assume player starts off non-collided



        self.ego_vehicle.forward_speed=measurement.player_measurements.forward_speed
        self.current_timestamp = float(measurement.game_timestamp) / 1000.0


        # Store history
        x, y, yaw = self.ego_vehicle.localization
        self.ego_vehicle.history.x_history.append(x)
        self.ego_vehicle.history.y_history.append(y)
        self.ego_vehicle.history.yaw_history.append(yaw)
        self.ego_vehicle.history.time_history.append(self.current_timestamp)
        self.ego_vehicle.history.speed_history.append(self.ego_vehicle.forward_speed)

        # Obtain Lead Vehicle information.
        self.lead_car_pos = []
        self.lead_car_length = []
        self.lead_car_speed = []
        for agent in measurement.non_player_agents:
            agent_id = agent.id
            if agent.HasField('vehicle'):
                self.lead_car_pos.append(
                    [agent.vehicle.transform.location.x,
                     agent.vehicle.transform.location.y])
                self.lead_car_length.append(agent.vehicle.bounding_box.extent.x)
                self.lead_car_speed.append(agent.vehicle.forward_speed)



    def get_current_pose(self, measurement):
        """Obtains current x,y,yaw pose from the client measurements

        Obtains the current x,y, and yaw pose from the client measurements.

        Args:
            measurement: The CARLA client measurements (from read_data())

        Returns: (x, y, yaw)
            x: X position in meters
            y: Y position in meters
            yaw: Yaw position in radians
        """
        x = measurement.player_measurements.transform.location.x
        y = measurement.player_measurements.transform.location.y
        yaw = math.radians(measurement.player_measurements.transform.rotation.yaw)

        return (x, y, yaw)

    def get_player_collided_flag(self, measurement,
                                 prev_collision_vehicles,
                                 prev_collision_pedestrians,
                                 prev_collision_other):
        """Obtains collision flag from player. Check if any of the three collision
        metrics (vehicles, pedestrians, others) from the player are true, if so the
        player has collided to something.

        Note: From the CARLA documentation:

        "Collisions are not annotated if the vehicle is not moving (<1km/h) to avoid
        annotating undesired collision due to mistakes in the AI of non-player
        agents."
        """
        player_meas = measurement.player_measurements
        current_collision_vehicles = player_meas.collision_vehicles
        current_collision_pedestrians = player_meas.collision_pedestrians
        current_collision_other = player_meas.collision_other

        collided_vehicles = current_collision_vehicles > prev_collision_vehicles
        collided_pedestrians = current_collision_pedestrians > \
                               prev_collision_pedestrians
        collided_other = current_collision_other > prev_collision_other

        return (collided_vehicles or collided_pedestrians or collided_other,
                current_collision_vehicles,
                current_collision_pedestrians,
                current_collision_other)

class WorldRepresentation(object):
    def __index__(self,perception):
        self.localization=None
        self.objects = None
        self.lane_waypoint = None
        self.drivable_area = None

    def run_step(self,perception):
        self.localization=perception.localization
        self.objects = perception.objects
        self.lane_waypoint = perception.lane_waypoint
        self.drivable_area = perception.drivable_area


def get_current_pose(measurement):
    """Obtains current x,y,yaw pose from the client measurements

    Obtains the current x,y, and yaw pose from the client measurements.

    Args:
        measurement: The CARLA client measurements (from read_data())

    Returns: (x, y, yaw)
        x: X position in meters
        y: Y position in meters
        yaw: Yaw position in radians
    """
    x = measurement.player_measurements.transform.location.x
    y = measurement.player_measurements.transform.location.y
    yaw = math.radians(measurement.player_measurements.transform.rotation.yaw)

    return (x, y, yaw)


def get_player_collided_flag( measurement,
                                 prev_collision_vehicles,
                                 prev_collision_pedestrians,
                                 prev_collision_other):
        """Obtains collision flag from player. Check if any of the three collision
        metrics (vehicles, pedestrians, others) from the player are true, if so the
        player has collided to something.

        Note: From the CARLA documentation:

        "Collisions are not annotated if the vehicle is not moving (<1km/h) to avoid
        annotating undesired collision due to mistakes in the AI of non-player
        agents."
        """
        player_meas = measurement.player_measurements
        current_collision_vehicles = player_meas.collision_vehicles
        current_collision_pedestrians = player_meas.collision_pedestrians
        current_collision_other = player_meas.collision_other

        collided_vehicles = current_collision_vehicles > prev_collision_vehicles
        collided_pedestrians = current_collision_pedestrians > \
                               prev_collision_pedestrians
        collided_other = current_collision_other > prev_collision_other

        return (collided_vehicles or collided_pedestrians or collided_other,
                current_collision_vehicles,
                current_collision_pedestrians,
                current_collision_other)
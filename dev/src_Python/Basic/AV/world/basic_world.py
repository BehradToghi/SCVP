
class History():
    def __init__(self):
        self.x_history=None
        self.y_history =None
        self.yaw_history=None
        self.time_history=None
        self.speed_history=None
        self.collided_flag_history=None

class Vehicle():
    def __init__(self):
        self.localization = None  # (x, y, yaw)
        self.start_pos=None
        self.history=History()
        self.forward_speed=None

class WorldRepresentation(object):
    """ Class representing the surrounding environment from perception stack """
    def __index__(self,perception):
        self.dynamic_objects = None                 # type list ; list of 3D BBs (X(m), Y(m), Z(m), Yaw(deg), RADX(m), RADY(m), RADZ(m))
        self.occupancy_grid  = None         # occupancy grid
        self.drivable_area = None           # grid of drivable area

        self.lane_waypoint = None           # type list: [[x1,y1]; ... [xn,yn]] ; list of lane points for center lane
        self.global_planner_waypoints=None  # type list: [[x1,y1,v1]; ... [xn,yn,vn]] ; current waypoints to track (global frame).
                                            # length and speed in m and m/s. (includes speed to track at each x,y location.)
        self.ego_vehicle = Vehicle()        # type Vehicle ;ego vehicle data

        self.current_timestamp = None       # type float ;timestamp

        self.player_collided_flag = None    # type bool ;collided flag

        # Initialize collision history
        self.prev_collision_vehicles = 0
        self.prev_collision_pedestrians = 0
        self.prev_collision_other = 0

        self.stopsign_raw=None              # type list ; (X(m), Y(m), Z(m), Yaw(deg))
        self.static_object_raw=None         # type list ; list of 3D BBs (X(m), Y(m), Z(m), Yaw(deg), RADX(m), RADY(m), RADZ(m))

        self.stopsign_fences = None
        self.parkedcar_box_pts = None
        self.waypoints = None               # global_planner_waypoints for this test
        self.waypoints_np = None


        self.lead_car_pos = []
        self.lead_car_length = []
        self.lead_car_speed = []

    def run_step(self,perception):
        self.ego_vehicle = perception.ego_vehicle


        self.dynamic_objects = perception.objects
        self.static_objects = perception.objects
        self.lane_waypoint = perception.lane_waypoint
        self.drivable_area = perception.drivable_area

        self.current_timestamp = perception.current_timestamp

        self.player_collided_flag = perception.player_collided_flag
        self.prev_collision_vehicles = perception.prev_collision_vehicles
        self.prev_collision_pedestrians = perception.prev_collision_pedestrians
        self.prev_collision_other = perception.prev_collision_other


        self.stopsign_fences = perception.stopsign_fences
        self.parkedcar_box_pts = perception.parkedcar_box_pts
        self.waypoints = perception.waypoints

        self.stopsign_raw = perception.stopsign_fences_raw
        self.static_object_raw = perception.parkedcar_box_pts_raw
        # self.waypoints = perception.waypoints
        # self.waypoints_np = perception.waypoints_np

        self.lead_car_pos = perception.lead_car_pos
        self.lead_car_length = perception.lead_car_length
        self.lead_car_speed = perception.lead_car_speed


    def get_ego_vehicle(self):
        return self.ego_vehicle

    def get_lane_waypoints(self):
        return self.lane_waypoint

    def get_drivable_area(self):
        return self.drivable_area

    def get_current_timestamp(self):
        return self.current_timestamp

    def get_dynamic_objects(self):
        return self.dynamic_objects

    def get_static_objects(self):
        return self.static_objects
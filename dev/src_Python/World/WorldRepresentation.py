class WorldRepresentation(object):
    """ Class representing the surrounding environment from perception stack """

    def __init__(self, carla_world, vehicle, args):
        """Constructor method"""
        self.world = carla_world
        self.vehicle = vehicle
        self.vehicle_list = self.world.get_actors().filter('vehicle.*')
        self.camera_manager = CameraManager(vehicle, HUD(args.width, args.height), args.gamma)

        VIEW_WIDTH = 1920//2
        VIEW_HEIGHT = 1080//2
        VIEW_FOV = 90
        calibration = np.identity(3)
        calibration[0, 2] = VIEW_WIDTH / 2.0
        calibration[1, 2] = VIEW_HEIGHT / 2.0
        calibration[0, 0] = calibration[1, 1] = VIEW_WIDTH / (2.0 * np.tan(VIEW_FOV * np.pi / 360.0))
        self.calibration = calibration

        try:
            self.map = self.world.get_map()
        except RuntimeError as error:
            print('RuntimeError: {}'.format(error))
            print('  The server could not send the OpenDRIVE (.xodr) file:')
            print('  Make sure it exists, has the same name of your town, and is correct.')
            sys.exit(1)

    def carla_map(self):
        # Approximate distance between the waypoints
        WAYPOINT_DISTANCE = 5.0  # in meters

        # GET WAYPOINTS IN THE MAP ##########################################
        # Returns a list of waypoints positioned on the center of the lanes
        # all over the map with an approximate distance between them.
        waypoint_list = self.map.generate_waypoints(WAYPOINT_DISTANCE)
        map=[(x,y)for x in [wp.transform.location.x for wp in waypoint_list] for y in [wp.transform.location.y for wp in waypoint_list]]
        return map


    def carla_road(self):
        # GET WAYPOINTS IN THE MAP ##########################################
        # It provides a minimal graph of the topology of the current OpenDRIVE file.
        # It is constituted by a list of pairs of waypoints, where the first waypoint
        # is the origin and the second one is the destination.
        # It can be loaded into NetworkX.
        # A valid output could be: [ (w0, w1), (w0, w2), (w1, w3), (w2, w3), (w0, w4) ]
        topology = self.map.get_topology()
        road_list = []

        # Approximate distance between the waypoints
        WAYPOINT_DISTANCE = 5.0  # in meters

        for wp_pair in topology:
            current_wp = wp_pair[0]
            # Check if there is a road with no previus road, this can happen
            # in opendrive. Then just continue.
            if current_wp is None:
                continue
            # First waypoint on the road that goes from wp_pair[0] to wp_pair[1].
            current_road_id = current_wp.road_id
            wps_in_single_road = [current_wp]
            # While current_wp has the same road_id (has not arrived to next road).
            while current_wp.road_id == current_road_id:
                # Check for next waypoints in aprox distance.
                available_next_wps = current_wp.next(WAYPOINT_DISTANCE)
                # If there is next waypoint/s?
                if available_next_wps:
                    # We must take the first ([0]) element because next(dist) can
                    # return multiple waypoints in intersections.
                    current_wp = available_next_wps[0]
                    wps_in_single_road.append(current_wp)
                else: # If there is no more waypoints we can stop searching for more.
                    break
            road_list.append(wps_in_single_road)
            return road_list

    @staticmethod
    def get_bounding_box(calibration, vehicle, camera):
        """
        Returns 3D bounding box for a vehicle based on camera view.
        """

        bb_cords = WorldRepresentation.create_bb_points(vehicle)
        cords_x_y_z = WorldRepresentation.vehicle_to_sensor(bb_cords, vehicle, camera)[:3, :]
        cords_y_minus_z_x = np.concatenate([cords_x_y_z[1, :], -cords_x_y_z[2, :], cords_x_y_z[0, :]])
        bbox = np.transpose(np.dot(calibration, cords_y_minus_z_x))
        camera_bbox = np.concatenate([bbox[:, 0] / bbox[:, 2], bbox[:, 1] / bbox[:, 2], bbox[:, 2]], axis=1)
        return camera_bbox

    @staticmethod
    def create_bb_points(vehicle):
        """
        Returns 3D bounding box for a vehicle.
        """

        cords = np.zeros((8, 4))
        extent = vehicle.bounding_box.extent
        cords[0, :] = np.array([extent.x, extent.y, -extent.z, 1])
        cords[1, :] = np.array([-extent.x, extent.y, -extent.z, 1])
        cords[2, :] = np.array([-extent.x, -extent.y, -extent.z, 1])
        cords[3, :] = np.array([extent.x, -extent.y, -extent.z, 1])
        cords[4, :] = np.array([extent.x, extent.y, extent.z, 1])
        cords[5, :] = np.array([-extent.x, extent.y, extent.z, 1])
        cords[6, :] = np.array([-extent.x, -extent.y, extent.z, 1])
        cords[7, :] = np.array([extent.x, -extent.y, extent.z, 1])
        return cords

    @staticmethod
    def vehicle_to_sensor(cords, vehicle, sensor):
        """
        Transforms coordinates of a vehicle bounding box to sensor.
        """

        world_cord = WorldRepresentation.vehicle_to_world(cords, vehicle)
        sensor_cord = WorldRepresentation.world_to_sensor(world_cord, sensor)
        return sensor_cord

    @staticmethod
    def vehicle_to_world(cords, vehicle):
        """
        Transforms coordinates of a vehicle bounding box to world.
        """

        bb_transform = carla.Transform(vehicle.bounding_box.location)
        bb_vehicle_matrix = WorldRepresentation.get_matrix(bb_transform)
        vehicle_world_matrix = WorldRepresentation.get_matrix(vehicle.get_transform())
        bb_world_matrix = np.dot(vehicle_world_matrix, bb_vehicle_matrix)
        world_cords = np.dot(bb_world_matrix, np.transpose(cords))
        return world_cords

    @staticmethod
    def world_to_sensor(cords, sensor):
        """
        Transforms world coordinates to sensor.
        """

        sensor_world_matrix = WorldRepresentation.get_matrix(sensor._camera_transforms[sensor.transform_index][0],)
        world_sensor_matrix = np.linalg.inv(sensor_world_matrix)
        sensor_cords = np.dot(world_sensor_matrix, cords)
        return sensor_cords

    @staticmethod
    def get_matrix(transform):
        """
        Creates matrix from carla transform.
        """

        rotation = transform.rotation
        location = transform.location
        c_y = np.cos(np.radians(rotation.yaw))
        s_y = np.sin(np.radians(rotation.yaw))
        c_r = np.cos(np.radians(rotation.roll))
        s_r = np.sin(np.radians(rotation.roll))
        c_p = np.cos(np.radians(rotation.pitch))
        s_p = np.sin(np.radians(rotation.pitch))
        matrix = np.matrix(np.identity(4))
        matrix[0, 3] = location.x
        matrix[1, 3] = location.y
        matrix[2, 3] = location.z
        matrix[0, 0] = c_p * c_y
        matrix[0, 1] = c_y * s_p * s_r - s_y * c_r
        matrix[0, 2] = -c_y * s_p * c_r - s_y * s_r
        matrix[1, 0] = s_y * c_p
        matrix[1, 1] = s_y * s_p * s_r + c_y * c_r
        matrix[1, 2] = -s_y * s_p * c_r + c_y * s_r
        matrix[2, 0] = s_p
        matrix[2, 1] = -c_p * s_r
        matrix[2, 2] = c_p * c_r
        return matrix

    def dynamic_objects(self):
        dynamic_object_list = []
        for vehicle in self.vehicle_list:
            dynamic_object ={
            'id' : vehicle.id,
            'acceleration_x' : vehicle.get_acceleration().x,
            'acceleration_y' : vehicle.get_acceleration().y,
            'acceleration_z' : vehicle.get_acceleration().z,
            'location_x' : vehicle.get_location().x,
            'location_y' : vehicle.get_location().y,
            'location_z' : vehicle.get_location().z,
            'velocity_x' : vehicle.get_velocity().x,
            'velocity_y' : vehicle.get_velocity().y,
            'velocity_z' : vehicle.get_velocity().z,
            'bounding_box' : WorldRepresentation.get_bounding_box(self.calibration, vehicle,self.camera_manager)}
            dynamic_object_list.append(dynamic_object)
        return dynamic_object_list


    def reference_point(self):
        """ Nearest waypoint on the center of a Driving lane."""
        reference_point = self.map.get_waypoint(self.vehicle.get_location(),project_to_road=True, lane_type=(carla.LaneType.Driving))
        return reference_point.transform.location

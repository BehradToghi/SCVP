"""
Copyright 2020 Connected & Autonomous Vehicle REsearch Lab (CAVREL)
at University of Central Florida (UCF).

run Carla Server
SDL_VIDEODRIVER=offscreen ./CarlaUE4.sh -opengl -quality-level=Low
./CarlaUE4.sh /Game/Maps/Course4 -windowed -carla-server -benchmark -fps=30
./CarlaUE4.sh /Game/Maps/Course4 -windowed -carla-server -benchmark -fps=30 -quality-level=Low​

-quality-level=<level>  //Sets the rendering quality of the simulator, which is ​ Low​ or Epic​
"""

from utils.cutils import *

from control import basic_controller
from planning import basic_planner
from simulation import basic_simulation
from perception import basic_perception
from world.basic_world import WorldRepresentation

# Script level imports
sys.path.append(os.path.abspath(sys.path[0] + '/..'))

from carla.tcp import TCPConnectionError


def run_pipeline(args):

    with basic_simulation.make_carla_client(args.host, args.port) as client:
        print('Carla client connected.')

        measurement_data,sim=basic_simulation.create_simulation(args, client)

        #############################################
        # Perception
        #############################################
        perception=basic_perception.Perception(measurement_data,sim)

        #############################################
        # Planner
        #############################################
        planner=basic_planner.BasicPlanner(perception)

        #############################################
        # Controller
        #############################################
        controller = basic_controller.Controller2D(planner)

        #############################################
        # WorldRepresentation
        #############################################
        world_representation=WorldRepresentation()

        #plot
        plotter=Plotter(sim,perception)


        #############################################
        # Scenario Execution Loop
        #############################################
        # Iterate the frames until the end of the trajectory is reached or the TOTAL_EPISODE_FRAMES is reached.
        reached_the_end = False
        skip_first_frame = True

        # Initialize the current timestamp.
        current_timestamp = perception.current_timestamp

        for frame in range(sim.TOTAL_EPISODE_FRAMES):
            # Gather current data from the CARLA server
            measurement_data, sensor_data = client.read_data()

            # Update pose and timestamp
            prev_timestamp = current_timestamp
            perception.run_step(measurement_data)
            world_representation.run_step(perception)

            current_timestamp = world_representation.current_timestamp
            current_x, current_y, yaw = world_representation.ego_vehicle.localization
            # Wait for some initial time before starting
            if current_timestamp <= basic_simulation.WAIT_TIME_BEFORE_START:
                basic_simulation.send_control_command(client, throttle=0.0, steer=0, brake=1.0)
                continue
            else:
                current_timestamp = current_timestamp - basic_simulation.WAIT_TIME_BEFORE_START
                world_representation.current_timestamp=current_timestamp
                perception.current_timestamp = current_timestamp

                # Execute the behaviour and local planning in the current instance
                # Note that updating the local path during every controller update  produces issues with the tracking performance (imagine everytime
                # the controller tried to follow the path, a new path appears).
            if frame % basic_planner.LP_FREQUENCY_DIVISOR == 0:
                   local_waypoints, wp_interp=planner.run_step(world_representation, prev_timestamp)

                   if local_waypoints != None:
                       # Update the other controller values and controls
                       controller.update_waypoints(wp_interp)

            # Controller Update
            if local_waypoints != None and local_waypoints != []:
                controller.update_values(world_representation,frame)
                controller.update_controls()
                cmd_throttle, cmd_steer, cmd_brake = controller.get_commands()
            else:
                cmd_throttle = 0.0
                cmd_steer = 0.0
                cmd_brake = 0.0


            # Skip the first frame or if there exists no local paths
            if skip_first_frame and frame == 0:
                pass
            elif local_waypoints == None:
                pass
            else:
                plotter.update(perception,controller,planner,frame,cmd_throttle,cmd_steer,cmd_brake)


            # Output controller command to CARLA server
            basic_simulation.send_control_command(client,throttle=cmd_throttle,steer=cmd_steer,brake=cmd_brake)


            # Find if reached the end , so the simulation will end.
            dist_to_last_waypoint = np.linalg.norm(np.array([perception.waypoints[-1][0] - current_x,perception.waypoints[-1][1] - current_y]))
            if  dist_to_last_waypoint < basic_simulation.DIST_THRESHOLD_TO_LAST_WAYPOINT:
                reached_the_end = True
            if reached_the_end:
                break

        # End - Stop vehicle and Store outputs to the controller output
        if reached_the_end:
            print("Reached the end of path. Writing to output...")
        else:
            print("Exceeded assessment time. Writing to output...")
        # Stop the car
        basic_simulation.send_control_command(client,throttle=0.0, steer=0.0, brake=1.0)

        # Store the various outputs
        plotter.save_plots(perception)

def main():
    """Main function.

    Args:
        -v, --verbose: print debug information
        --host: IP of the host server (default: localhost)
        -p, --port: TCP port to listen to (default: 2000)
        -a, --autopilot: enable autopilot
        -q, --quality-level: graphics quality level [Low or Epic]
        -i, --images-to-disk: save images to disk
        -c, --carla-settings: Path to CarlaSettings.ini file
    """
    argparser = argparse.ArgumentParser(description=__doc__)
    argparser.add_argument(
        '-v', '--verbose',
        action='store_true',
        dest='debug',
        help='print debug information')
    argparser.add_argument(
        '--host',
        metavar='H',
        default='localhost',
        help='IP of the host server (default: localhost)')
    argparser.add_argument(
        '-p', '--port',
        metavar='P',
        default=2000,
        type=int,
        help='TCP port to listen to (default: 2000)')
    argparser.add_argument(
        '-a', '--autopilot',
        action='store_true',
        help='enable autopilot')
    argparser.add_argument(
        '-q', '--quality-level',
        choices=['Low', 'Epic'],
        type=lambda s: s.title(),
        default='Low',
        help='graphics quality level.')
    argparser.add_argument(
        '-c', '--carla-settings',
        metavar='PATH',
        dest='settings_filepath',
        default=None,
        help='Path to a "CarlaSettings.ini" file')
    args = argparser.parse_args()

    # Logging startup info
    log_level = logging.DEBUG if args.debug else logging.INFO
    logging.basicConfig(format='%(levelname)s: %(message)s', level=log_level)
    logging.info('listening to server %s:%s', args.host, args.port)

    args.out_filename_format = '_out/episode_{:0>4d}/{:s}/{:0>6d}'

    # Execute when server connection is established
    while True:

        try:
            run_pipeline(args)
            print('Done.')
            return

        except TCPConnectionError as error:
            logging.error(error)
            time.sleep(1)

if __name__ == '__main__':

    try:
        main()
    except KeyboardInterrupt:
        print('\nCancelled by user. Bye!')


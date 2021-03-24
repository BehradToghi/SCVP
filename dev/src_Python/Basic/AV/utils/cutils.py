from __future__ import print_function
from __future__ import division

# System level imports
import sys
import argparse
import logging
import os
import numpy as np
import time

import configparser
from utils import live_plotter as lv   # Custom live plotting library

from simulation import basic_simulation
from planning import basic_planner


class CUtils(object):
    def __init__(self):
        pass

    def create_var(self, var_name, value):
        if not var_name in self.__dict__:
            self.__dict__[var_name] = value


class Plotter(object):
    def __init__(self,sim, perception):

        self.trajectory_fig, self.forward_speed_fig, self.throttle_fig, self.brake_fig, \
        self.steer_fig, self.enable_live_plot, self.live_plot_timer, self.lp_traj, self.lp_1d = self.ini_plot(sim, perception )


    def ini_plot(self,sim, perception):
        FIGSIZE_X_INCHES = 8  # x figure size of feedback in inches
        FIGSIZE_Y_INCHES = 8  # y figure size of feedback in inches
        PLOT_LEFT = 0.1  # in fractions of figure width and height
        PLOT_BOT = 0.1
        PLOT_WIDTH = 0.8
        PLOT_HEIGHT = 0.8

        waypoints_np = perception.waypoints_np
        TOTAL_EPISODE_FRAMES = sim.TOTAL_EPISODE_FRAMES
        start_x, start_y, start_yaw = perception.ego_vehicle.localization
        stopsign_fences = perception.stopsign_fences
        parkedcar_box_pts = perception.parkedcar_box_pts

        #############################################
        # Load Configurations
        #############################################

        # Load configuration file (options.cfg) and then parses for the various
        # options. Here we have two main options:
        # live_plotting and live_plotting_period, which controls whether
        # live plotting is enabled or how often the live plotter updates
        # during the simulation run.
        config = configparser.ConfigParser()
        config.read(os.path.join(
            os.path.dirname(os.path.realpath(__file__)), 'options.cfg'))
        demo_opt = config['Demo Parameters']

        # Get options
        enable_live_plot = demo_opt.get('live_plotting', 'true').capitalize()
        enable_live_plot = enable_live_plot == 'True'
        live_plot_period = float(demo_opt.get('live_plotting_period', 0))

        # Set options
        live_plot_timer = basic_simulation.Timer(live_plot_period)

        #############################################
        # Vehicle Trajectory Live Plotting Setup
        #############################################
        # Uses the live plotter to generate live feedback during the simulation
        # The two feedback includes the trajectory feedback and
        # the controller feedback (which includes the speed tracking).
        lp_traj = lv.LivePlotter(tk_title="Trajectory Trace")
        lp_1d = lv.LivePlotter(tk_title="Controls Feedback")

        ###
        # Add 2D position / trajectory plot
        ###
        trajectory_fig = lp_traj.plot_new_dynamic_2d_figure(
            title='Vehicle Trajectory',
            figsize=(FIGSIZE_X_INCHES, FIGSIZE_Y_INCHES),
            edgecolor="black",
            rect=[PLOT_LEFT, PLOT_BOT, PLOT_WIDTH, PLOT_HEIGHT])

        trajectory_fig.set_invert_x_axis()  # Because UE4 uses left-handed
        # coordinate system the X
        # axis in the graph is flipped
        trajectory_fig.set_axis_equal()  # X-Y spacing should be equal in size

        # Add waypoint markers
        trajectory_fig.add_graph("waypoints", window_size=waypoints_np.shape[0],
                                 x0=waypoints_np[:, 0], y0=waypoints_np[:, 1],
                                 linestyle="-", marker="", color='g')
        # Add trajectory markers
        trajectory_fig.add_graph("trajectory", window_size=TOTAL_EPISODE_FRAMES,
                                 x0=[start_x] * TOTAL_EPISODE_FRAMES,
                                 y0=[start_y] * TOTAL_EPISODE_FRAMES,
                                 color=[1, 0.5, 0])
        # Add starting position marker
        trajectory_fig.add_graph("start_pos", window_size=1,
                                 x0=[start_x], y0=[start_y],
                                 marker=11, color=[1, 0.5, 0],
                                 markertext="Start", marker_text_offset=1)
        # Add end position marker
        trajectory_fig.add_graph("end_pos", window_size=1,
                                 x0=[waypoints_np[-1, 0]],
                                 y0=[waypoints_np[-1, 1]],
                                 marker="D", color='r',
                                 markertext="End", marker_text_offset=1)
        # Add car marker
        trajectory_fig.add_graph("car", window_size=1,
                                 marker="s", color='b', markertext="Car",
                                 marker_text_offset=1)
        # Add lead car information
        trajectory_fig.add_graph("leadcar", window_size=1,
                                 marker="s", color='g', markertext="Lead Car",
                                 marker_text_offset=1)
        # Add stop sign position
        trajectory_fig.add_graph("stopsign", window_size=1,
                                 x0=[stopsign_fences[0][0]], y0=[stopsign_fences[0][1]],
                                 marker="H", color="r",
                                 markertext="Stop Sign", marker_text_offset=1)
        # Add stop sign "stop line"
        trajectory_fig.add_graph("stopsign_fence", window_size=1,
                                 x0=[stopsign_fences[0][0], stopsign_fences[0][2]],
                                 y0=[stopsign_fences[0][1], stopsign_fences[0][3]],
                                 color="r")

        # Load parked car points
        parkedcar_box_pts_np = np.array(parkedcar_box_pts)
        trajectory_fig.add_graph("parkedcar_pts", window_size=parkedcar_box_pts_np.shape[0],
                                 x0=parkedcar_box_pts_np[:, 0], y0=parkedcar_box_pts_np[:, 1],
                                 linestyle="", marker="+", color='b')

        # Add lookahead path
        trajectory_fig.add_graph("selected_path",
                                 window_size=basic_simulation.INTERP_MAX_POINTS_PLOT,
                                 x0=[start_x] * basic_simulation.INTERP_MAX_POINTS_PLOT,
                                 y0=[start_y] * basic_simulation.INTERP_MAX_POINTS_PLOT,
                                 color=[1, 0.5, 0.0],
                                 linewidth=3)

        # Add local path proposals
        for i in range(basic_planner.NUM_PATHS):
            trajectory_fig.add_graph("local_path " + str(i), window_size=200,
                                     x0=None, y0=None, color=[0.0, 0.0, 1.0])

        ###
        # Add 1D speed profile updater
        ###
        forward_speed_fig = \
            lp_1d.plot_new_dynamic_figure(title="Forward Speed (m/s)")
        forward_speed_fig.add_graph("forward_speed",
                                    label="forward_speed",
                                    window_size=TOTAL_EPISODE_FRAMES)
        forward_speed_fig.add_graph("reference_signal",
                                    label="reference_Signal",
                                    window_size=TOTAL_EPISODE_FRAMES)

        # Add throttle signals graph
        throttle_fig = lp_1d.plot_new_dynamic_figure(title="Throttle")
        throttle_fig.add_graph("throttle",
                               label="throttle",
                               window_size=TOTAL_EPISODE_FRAMES)
        # Add brake signals graph
        brake_fig = lp_1d.plot_new_dynamic_figure(title="Brake")
        brake_fig.add_graph("brake",
                            label="brake",
                            window_size=TOTAL_EPISODE_FRAMES)
        # Add steering signals graph
        steer_fig = lp_1d.plot_new_dynamic_figure(title="Steer")
        steer_fig.add_graph("steer",
                            label="steer",
                            window_size=TOTAL_EPISODE_FRAMES)

        # live plotter is disabled, hide windows
        if not enable_live_plot:
            lp_traj._root.withdraw()
            lp_1d._root.withdraw()


        return trajectory_fig,forward_speed_fig,throttle_fig,brake_fig,steer_fig,enable_live_plot,live_plot_timer,lp_traj,lp_1d


    def update(self,perception,controller,planner,frame,cmd_throttle,cmd_steer,cmd_brake):
        local_waypoints=planner.local_waypoints
        wp_interp= planner.wp_interp
        best_index= planner.best_index
        paths= planner.paths
        ego_state= planner.ego_state
        path_validity= planner.path_validity
        collision_check_array= planner.collision_check_array

        current_x, current_y, yaw = perception.ego_vehicle.localization
        current_timestamp=perception.current_timestamp
        current_speed=perception.ego_vehicle.forward_speed
        # Update live plotter with new feedback
        self.trajectory_fig.roll("trajectory", current_x, current_y)
        self.trajectory_fig.roll("car", current_x, current_y)
        if perception.lead_car_pos:  # If there exists a lead car, plot it
            self.trajectory_fig.roll("leadcar", perception.lead_car_pos[1][0],
                                perception.lead_car_pos[1][1])
        self.forward_speed_fig.roll("forward_speed",
                               current_timestamp,
                               current_speed)
        self.forward_speed_fig.roll("reference_signal",
                               current_timestamp,
                               controller._desired_speed)
        self.throttle_fig.roll("throttle", current_timestamp, cmd_throttle)
        self.brake_fig.roll("brake", current_timestamp, cmd_brake)
        self.steer_fig.roll("steer", current_timestamp, cmd_steer)

        # Local path plotter update
        if frame % basic_planner.LP_FREQUENCY_DIVISOR == 0:
            path_counter = 0
            for i in range(basic_planner.NUM_PATHS):
                # If a path was invalid in the set, there is no path to plot.
                if path_validity[i]:
                    # Colour paths according to collision checking.
                    if not collision_check_array[path_counter]:
                        colour = 'r'
                    elif i == best_index:
                        colour = 'k'
                    else:
                        colour = 'b'
                    self.trajectory_fig.update("local_path " + str(i), paths[path_counter][0], paths[path_counter][1],
                                          colour)
                    path_counter += 1
                else:
                    self.trajectory_fig.update("local_path " + str(i), [ego_state[0]], [ego_state[1]], 'r')
        # When plotting lookahead path, only plot a number of points
        # (INTERP_MAX_POINTS_PLOT amount of points). This is meant
        # to decrease load when live plotting
        wp_interp_np = np.array(wp_interp)
        path_indices = np.floor(np.linspace(0,
                                            wp_interp_np.shape[0] - 1,
                                            basic_simulation.INTERP_MAX_POINTS_PLOT))
        self.trajectory_fig.update("selected_path",
                              wp_interp_np[path_indices.astype(int), 0],
                              wp_interp_np[path_indices.astype(int), 1],
                              new_colour=[1, 0.5, 0.0])

        # Refresh the live plot based on the refresh rate
        # set by the options
        if self.enable_live_plot and \
                self.live_plot_timer.has_exceeded_lap_period():
            self.lp_traj.refresh()
            self.lp_1d.refresh()
            self.live_plot_timer.lap()

    def save_plots(self,perception):
        self.store_trajectory_plot(self.trajectory_fig.fig, 'trajectory.png')
        self.store_trajectory_plot(self.forward_speed_fig.fig, 'forward_speed.png')
        self.store_trajectory_plot(self.throttle_fig.fig, 'throttle_output.png')
        self.store_trajectory_plot(self.brake_fig.fig, 'brake_output.png')
        self.store_trajectory_plot(self.steer_fig.fig, 'steer_output.png')
        self.write_trajectory_file(perception)
        self.write_collisioncount_file(perception.ego_vehicle.history.collided_flag_history)

    def store_trajectory_plot(self, graph, fname):
        """ Store the resulting plot.
        """
        create_controller_output_dir(OUTPUT_FOLDER)

        file_name = os.path.join(OUTPUT_FOLDER, fname)
        graph.savefig(file_name)

    def write_trajectory_file(self, perception):
        x_list = perception.ego_vehicle.history.x_history
        y_list = perception.ego_vehicle.history.y_history
        v_list = perception.ego_vehicle.history.speed_history
        t_list = perception.ego_vehicle.history.time_history
        collided_list = perception.ego_vehicle.history.collided_flag_history

        create_controller_output_dir(OUTPUT_FOLDER)
        file_name = os.path.join(OUTPUT_FOLDER, 'trajectory.txt')

        with open(file_name, 'w') as trajectory_file:
            for i in range(len(x_list)):
                trajectory_file.write('%3.3f, %3.3f, %2.3f, %6.3f %r\n' % \
                                      (x_list[i], y_list[i], v_list[i], t_list[i],
                                       collided_list[i]))

    def write_collisioncount_file(self, collided_list):
        create_controller_output_dir(OUTPUT_FOLDER)
        file_name = os.path.join(OUTPUT_FOLDER, 'collision_count.txt')

        with open(file_name, 'w') as collision_file:
            collision_file.write(str(sum(collided_list)))
# output directory
OUTPUT_FOLDER = os.path.dirname(os.path.realpath(__file__)) +\
                           '/output/'

def create_controller_output_dir(output_folder):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)



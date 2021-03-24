import numpy as np
import copy
import scipy.spatial
import scipy.optimize
import scipy.integrate
import sys
from math import sin, cos, pi, sqrt
import math


# Planning Constants
NUM_PATHS = 7
BP_LOOKAHEAD_BASE      = 8.0              # m
BP_LOOKAHEAD_TIME      = 2.0              # s
PATH_OFFSET            = 1.5              # m
CIRCLE_OFFSETS         = [-1.0, 1.0, 3.0] # m
CIRCLE_RADII           = [1.5, 1.5, 1.5]  # m
TIME_GAP               = 1.0              # s
PATH_SELECT_WEIGHT     = 10
A_MAX                  = 1.5              # m/s^2
SLOW_SPEED             = 2.0              # m/s
STOP_LINE_BUFFER       = 3.5              # m
LEAD_VEHICLE_LOOKAHEAD = 20.0             # m
LP_FREQUENCY_DIVISOR   = 2                # Frequency divisor to make the
                                          # local planner operate at a lower
                                          # frequency than the controller
                                          # (which operates at the simulation
                                          # frequency). Must be a natural
                                          # number.
PREV_BEST_PATH         = []


# Path interpolation parameters
INTERP_MAX_POINTS_PLOT    = 10   # number of points used for displaying
                                 # selected path
INTERP_DISTANCE_RES       = 0.01 # distance between interpolated points

# CollisionChecker
class CollisionChecker:
    def __init__(self, circle_offsets, circle_radii, weight):
        self._circle_offsets = circle_offsets
        self._circle_radii = circle_radii
        self._weight = weight
        self.print_verbose=False

    ######################################################
    ######################################################
    # MODULE 7: CHECKING FOR COLLISSIONS
    #   Read over the function comments to familiarize yourself with the
    #   arguments and necessary variables to return. Then follow the TODOs
    #   (top-down) and use the surrounding comments as a guide.
    ######################################################
    ######################################################
    # Takes in a set of paths and obstacles, and returns an array
    # of bools that says whether or not each path is collision free.
    def collision_check(self, paths, obstacles):
        """Returns a bool array on whether each path is collision free.

        args:
            paths: A list of paths in the global frame.
                A path is a list of points of the following format:
                    [x_points, y_points, t_points]:
                        x_points: List of x values (m)
                        y_points: List of y values (m)
                        t_points: List of yaw values (rad)
                    Example of accessing the ith path, jth point's t value:
                        paths[i][2][j]
            obstacles: A list of [x, y] points that represent points along the
                border of obstacles, in the global frame.
                Format: [[x0, y0],
                         [x1, y1],
                         ...,
                         [xn, yn]]
                , where n is the number of obstacle points and units are [m, m]

        returns:
            collision_check_array: A list of boolean values which classifies
                whether the path is collision-free (true), or not (false). The
                ith index in the collision_check_array list corresponds to the
                ith path in the paths list.
        """
        collision_check_array = np.zeros(len(paths), dtype=bool)
        for i in range(len(paths)):
            collision_free = True
            path = paths[i]

            # Iterate over the points in the path.
            for j in range(len(path[0])):
                # Compute the circle locations along this point in the path.
                # These circle represent an approximate collision
                # border for the vehicle, which will be used to check
                # for any potential collisions along each path with obstacles.

                # The circle offsets are given by self._circle_offsets.
                # The circle offsets need to placed at each point along the path,
                # with the offset rotated by the yaw of the vehicle.
                # Each path is of the form [[x_values], [y_values],
                # [theta_values]], where each of x_values, y_values, and
                # theta_values are in sequential order.

                # Thus, we need to compute:
                # circle_x = point_x + circle_offset*cos(yaw)
                # circle_y = point_y circle_offset*sin(yaw)
                # for each point along the path.
                # point_x is given by path[0][j], and point _y is given by
                # path[1][j].
                circle_locations = np.zeros((len(self._circle_offsets), 2))

                # TODO: INSERT YOUR CODE BETWEEN THE DASHED LINES
                # --------------------------------------------------------------
                circle_locations[:, 0] = [i * int(np.cos(path[2][j])) for i in self._circle_offsets] + path[0][j]
                circle_locations[:, 1] = [i * int(np.sin(path[2][j])) for i in self._circle_offsets] + path[1][j]
                # --------------------------------------------------------------

                # Assumes each obstacle is approximated by a collection of
                # points of the form [x, y].
                # Here, we will iterate through the obstacle points, and check
                # if any of the obstacle points lies within any of our circles.
                # If so, then the path will collide with an obstacle and
                # the collision_free flag should be set to false for this flag
                for k in range(len(obstacles)):
                    collision_dists = \
                        scipy.spatial.distance.cdist(obstacles[k],
                                                     circle_locations)
                    collision_dists = np.subtract(collision_dists,
                                                  self._circle_radii)
                    collision_free = collision_free and \
                                     not np.any(collision_dists < 0)

                    if not collision_free:
                        break
                if not collision_free:
                    break

            collision_check_array[i] = collision_free

        return collision_check_array

    ######################################################
    ######################################################
    # MODULE 7: SELECTING THE BEST PATH INDEX
    #   Read over the function comments to familiarize yourself with the
    #   arguments and necessary variables to return. Then follow the TODOs
    #   (top-down) and use the surrounding comments as a guide.
    ######################################################
    ######################################################
    # Selects the best path in the path set, according to how closely
    # it follows the lane centerline, and how far away it is from other
    # paths that are in collision.
    # Disqualifies paths that collide with obstacles from the selection
    # process.
    # collision_check_array contains True at index i if paths[i] is
    # collision-free, otherwise it contains False.
    def select_best_path_index(self, paths, collision_check_array, goal_state):
        """Returns the path index which is best suited for the vehicle to
        traverse.

        Selects a path index which is closest to the center line as well as far
        away from collision paths.

        args:
            paths: A list of paths in the global frame.
                A path is a list of points of the following format:
                    [x_points, y_points, t_points]:
                        x_points: List of x values (m)
                        y_points: List of y values (m)
                        t_points: List of yaw values (rad)
                    Example of accessing the ith path, jth point's t value:
                        paths[i][2][j]
            collision_check_array: A list of boolean values which classifies
                whether the path is collision-free (true), or not (false). The
                ith index in the collision_check_array list corresponds to the
                ith path in the paths list.
            goal_state: Goal state for the vehicle to reach (centerline goal).
                format: [x_goal, y_goal, v_goal], unit: [m, m, m/s]
        useful variables:
            self._weight: Weight that is multiplied to the best index score.
        returns:
            best_index: The path index which is best suited for the vehicle to
                navigate with.
        """
        best_index = None
        best_score = float('Inf')
        for i in range(len(paths)):
            # Handle the case of collision-free paths.
            if collision_check_array[i]:
                # Compute the "distance from centerline" score.
                # The centerline goal is given by goal_state.
                # The exact choice of objective function is up to you.
                # A lower score implies a more suitable path.
                # TODO: INSERT YOUR CODE BETWEEN THE DASHED LINES
                # --------------------------------------------------------------
                score = np.sqrt((goal_state[0] - paths[i][0][len(paths[i][0]) - 1]) ** 2 + (
                            goal_state[1] - paths[i][1][len(paths[i][0]) - 1]) ** 2)
                # --------------------------------------------------------------

                # Compute the "proximity to other colliding paths" score and
                # add it to the "distance from centerline" score.
                # The exact choice of objective function is up to you.
                for j in range(len(paths)):
                    if j == i:
                        continue
                    else:
                        if not collision_check_array[j]:
                            # TODO: INSERT YOUR CODE BETWEEN THE DASHED LINES
                            # --------------------------------------------------
                            if self.print_verbose:
                                print("Adding score")
                            score += self._weight * paths[i][2][j]
                            # --------------------------------------------------
                            pass

            # Handle the case of colliding paths.
            else:
                score = float('Inf')

            if self.print_verbose:
                print("score = %f" % score)

            # Set the best index to be the path index with the lowest score
            if score < best_score:
                best_score = score
                best_index = i

        if self.print_verbose:
            print("--------------------")

        return best_index

# PathOptimizer
class PathOptimizer:
    def __init__(self):
        self._xf = 0.0
        self._yf = 0.0
        self._tf = 0.0

    ######################################################
    ######################################################
    # MODULE 7: PARAMETER OPTIMIZATION FOR POLYNOMIAL SPIRAL
    #   Read over the function comments to familiarize yourself with the
    #   arguments and necessary variables to return. Then follow the TODOs
    #   (top-down) and use the surrounding comments as a guide.
    ######################################################
    ######################################################
    # Sets up the optimization problem to compute a spiral to a given
    # goal point, (xf, yf, tf).
    def optimize_spiral(self, xf, yf, tf):
        """Optimization function used for finding the optimization parameters.

        Assumptions:
            1. The first point in the spiral is in origin of the vehicle frame
            2. Assumes that the curvature for the endpoints to be zero
               (i.e. p0 and p3 are zero of the vector p = [p0, p1, p2, p3, sf])

        args:
            xf: Final x position (m) for the given goal state.
            yf: Final y position (m) for the given goal state.
            tf: Final yaw position (rad) for the given goal state.

        returns:
            spiral: The resulting optimized path that best fits the goal state.
                The path is a list of points of the following format:
                    [x_points, y_points, t_points]:
                        x_points: List of x values (m) along the spiral
                        y_points: List of y values (m) along the spiral
                        t_points: List of yaw values (rad) along the spiral
        """
        # Save the terminal x, y, and theta.
        self._xf = xf
        self._yf = yf
        self._tf = tf
        # The straight line distance serves as a lower bound on any path's
        # arc length to the goal.
        sf_0 = np.linalg.norm([xf, yf])
        max = sys.maxsize
        # The initial variables correspond to a straight line with arc length
        # sf_0.  Recall that p here is defined as:
        #    [p1, p2, sf]
        #, where p1 and p2 are the curvatures at points p1 and p2
        #, and sf is the final arc length for the spiral.
        # Since we already set p0 and p4 (being the curvature of
        # the initial and final points) to be zero.
        p0 = [0.0, 0.0, sf_0]

        # Here we will set the bounds [lower, upper] for each optimization
        # variable.
        # The first two variables correspond to the curvature 1/3rd of the
        # way along the path and 2/3rds of the way along the path, respectively.
        # As a result, their curvature needs to lie within [-0.5, 0.5].
        # The third variable is the arc length, it has no upper limit, and it
        # has a lower limit of the straight line arc length.
        # TODO: INSERT YOUR CODE BETWEEN THE DASHED LINES
        # ------------------------------------------------------------------
        bounds = [[-0.5, 0.5], [-0.5, 0.5], [sf_0, max]]
        # ------------------------------------------------------------------

        # Here we will call scipy.optimize.minimize to optimize our spiral.
        # The objective and gradient are given to you by self.objective, and
        # self.objective_grad. The bounds are computed above, and the inital
        # variables for the optimizer are set by p0. You should use the L-BFGS-B
        # optimization methods.
        # TODO: INSERT YOUR CODE BETWEEN THE DASHED LINES
        # ------------------------------------------------------------------
        res = scipy.optimize.minimize(fun=self.objective, x0=p0, method='L-BFGS-B', jac=self.objective_grad, bounds=bounds)
        # ------------------------------------------------------------------

        spiral = self.sample_spiral(res.x)
        return spiral

    ######################################################
    ######################################################
    # MODULE 7: COMPUTE LIST OF THETAS
    #   Read over the function comments to familiarize yourself with the
    #   arguments and necessary variables to return. Then follow the TODOs
    #   (top-down) and use the surrounding comments as a guide.
    ######################################################
    ######################################################
    # This function computes the theta values for a given list of
    # arc lengths, and spiral parameters a, b, c, d.
    # Recall that the equation of a cubic spiral is
    # kappa(s) = a + b*s + c*s^2 + d*s^3
    # and since theta(s) is the integral of kappa(s) with respect to
    # arc length, then theta(s) = a*s + b/2*s^2 + c/3*s^3 + d/4*s^4.
    # Try to vectorize this function using numpy for speed, if you can.
    # Inputs: a - the first term of kappa(s).
    #         b - the second term of kappa(s).
    #         c - the third term of kappa(s).
    #         d - the fourth term of kappa(s).
    def thetaf(self, a, b, c, d, s):
        pass

        # TODO: INSERT YOUR CODE BETWEEN THE DASHED LINES
        # ------------------------------------------------------------------
        # Remember that a, b, c, d and s are lists
        thetas = a*s + (b/2)*(s**2) + (c/3)*(s**3) + (d/4)*(s**4)
        return thetas
        # ------------------------------------------------------------------

    ######################################################
    ######################################################
    # MODULE 7: SAMPLE SPIRAL PATH
    #   Read over the function comments to familiarize yourself with the
    #   arguments and necessary variables to return. Then follow the TODOs
    #   (top-down) and use the surrounding comments as a guide.
    ######################################################
    ######################################################
    # This function samples the spiral along its arc length to generate
    # a discrete set of x, y, and theta points for a path.
    def sample_spiral(self, p):
        """Samples a set of points along the spiral given the optimization
        parameters.

        args:
            p: The resulting optimization parameters that minimizes the
                objective function given a goal state.
                Format: [p1, p2, sf], Unit: [1/m, 1/m, m]
                , where p1 and p2 are the curvatures at points p1 and p2
                  and sf is the final arc length for the spiral.
        returns:
            [x_points, y_points, t_points]:
                x_points: List of x values (m) along the spiral
                y_points: List of y values (m) along the spiral
                t_points: List of yaw values (rad) along the spiral
        """
        # These equations map from the optimization parameter space
        # to the spiral parameter space.
        p = [0.0, p[0], p[1], 0.0, p[2]]    # recall p0 and p3 are set to 0
                                            # and p4 is the final arc length
        a = p[0]
        b = -(11.0*p[0]/2.0 - 9.0*p[1] + 9.0*p[2]/2.0 - p[3])/p[4]
        c = (9.0*p[0] - 45.0*p[1]/2.0 + 18.0*p[2] - 9.0*p[3]/2.0)/p[4]**2
        d = -(9.0*p[0]/2.0 - 27.0*p[1]/2.0 + 27.0*p[2]/2.0 - 9.0*p[3]/2.0)/p[4]**3


        # Set the s_points (list of s values along the spiral) to be from 0.0
        # to p[4] (final arc length)
        s_points = np.linspace(0.0, p[4])

        # Compute the theta, x, and y points from the uniformly sampled
        # arc length points s_points (p[4] is the spiral arc length).
        # Use self.thetaf() to compute the theta values from the s values.
        # Recall that x = integral cos(theta(s)) ds and
        #             y = integral sin(theta(s)) ds.
        # You will find the scipy.integrate.cumtrapz() function useful.
        # Try to vectorize the code using numpy functions for speed if you can.

        # Try to vectorize the code using numpy functions for speed if you can.
        # TODO: INSERT YOUR CODE BETWEEN THE DASHED LINES
        # ------------------------------------------------------------------
        t_points = self.thetaf(a, b, c, d, s_points)
        x_points = scipy.integrate.cumtrapz(np.cos(t_points), s_points, initial=None)
        y_points = scipy.integrate.cumtrapz(np.sin(t_points), s_points, initial=None)
        return [x_points, y_points, t_points]
        # ------------------------------------------------------------------

    ######################################################
    ######################################################
    # BELOW ARE THE FUNCTIONS USED FOR THE OPTIMIZER.
    ######################################################
    ######################################################

    def objective(self, p):
        """
        The optimizer can freely move 3 of the spiral parameter variables.
        The other two are fixed due to boundary conditions.
        """
        p = [0.0, p[0], p[1], 0.0, p[2]]
        return self.fbe(p) + 25*(self.fxf(p) + self.fyf(p)) + 30*self.ftf(p)

    def objective_grad(self, p):
        """
        The optimizer can freely move 3 of the spiral parameter variables.
        The other two are fixed due to boundary conditions.
        """
        p = [0.0, p[0], p[1], 0.0, p[2]]
        return np.add(np.add(np.add(self.fbe_grad(p), np.multiply(25, self.fxf_grad(p))), \
            np.multiply(25, self.fyf_grad(p))), np.multiply(30, self.ftf_grad(p)))

    def fxf(self, p):
        t2 = p[0]*(1.1E1/2.0)
        t3 = p[1]*9.0
        t4 = p[2]*(9.0/2.0)
        t5 = p[0]*(9.0/2.0)
        t6 = p[1]*(2.7E1/2.0)
        t7 = p[2]*(2.7E1/2.0)
        t8 = p[3]*(9.0/2.0)
        t9 = t5-t6+t7-t8
        t10 = p[0]*9.0
        t11 = p[1]*(4.5E1/2.0)
        t12 = p[2]*1.8E1
        t13 = t8-t10+t11-t12
        t14 = p[3]-t2+t3-t4
        t15 = self._xf-p[4]*(cos(p[0]*p[4]-p[4]*t9*(1.0/4.0)-p[4]*t13*(1.0/3.0)+p[4]*t14*(1.0/2.0))+cos(p[0]*p[4]*(1.0/2.0)-p[4]*t9*(1.0/6.4E1)-p[4]*t13*(1.0/2.4E1)+p[4]*t14*(1.0/8.0))*2.0+cos(p[0]*p[4]*(3.0/4.0)-p[4]*t9*7.91015625E-2-p[4]*t13*(9.0/6.4E1)+p[4]*t14*(9.0/3.2E1))*2.0+cos(p[0]*p[4]*(1.0/4.0)-p[4]*t9*9.765625E-4-p[4]*t13*(1.0/1.92E2)+p[4]*t14*(1.0/3.2E1))*2.0+cos(p[0]*p[4]*(3.0/8.0)-p[4]*t9*4.94384765625E-3-p[4]*t13*(9.0/5.12E2)+p[4]*t14*(9.0/1.28E2))*4.0+cos(p[0]*p[4]*(1.0/8.0)-p[4]*t9*6.103515625E-5-p[4]*t13*6.510416666666667E-4+p[4]*t14*(1.0/1.28E2))*4.0+cos(p[0]*p[4]*(5.0/8.0)-p[4]*t9*3.814697265625E-2-p[4]*t13*8.138020833333333E-2+p[4]*t14*(2.5E1/1.28E2))*4.0+cos(p[0]*p[4]*(7.0/8.0)-p[4]*t9*1.4654541015625E-1-p[4]*t13*2.233072916666667E-1+p[4]*t14*(4.9E1/1.28E2))*4.0+1.0)*(1.0/2.4E1)
        t0 = t15*t15
        return t0

    def fxf_grad(self, p):
        grad = [0.0, 0.0, 0.0]

        t2 = p[0]*(1.1E1/2.0)
        t3 = p[1]*9.0
        t4 = p[2]*(9.0/2.0)
        t5 = p[0]*(9.0/2.0)
        t6 = p[1]*(2.7E1/2.0)
        t7 = p[2]*(2.7E1/2.0)
        t8 = p[3]*(9.0/2.0)
        t9 = t5-t6+t7-t8
        t10 = p[0]*9.0
        t11 = p[1]*(4.5E1/2.0)
        t12 = p[2]*1.8E1
        t13 = t8-t10+t11-t12
        t14 = p[3]-t2+t3-t4
        t15 = p[0]*p[4]
        t16 = p[0]*p[4]*(1.0/2.0)
        t17 = p[0]*p[4]*(3.0/4.0)
        t18 = p[0]*p[4]*(1.0/4.0)
        t19 = p[0]*p[4]*(3.0/8.0)
        t20 = p[0]*p[4]*(1.0/8.0)
        t21 = p[0]*p[4]*(5.0/8.0)
        t22 = p[0]*p[4]*(7.0/8.0)
        t0 = p[4]*(self._xf-p[4]*(cos(t15-p[4]*t9*(1.0/4.0)-p[4]*t13*(1.0/3.0)+p[4]*t14*(1.0/2.0))+cos(t16-p[4]*t9*(1.0/6.4E1)-p[4]*t13*(1.0/2.4E1)+p[4]*t14*(1.0/8.0))*2.0+cos(t17-p[4]*t9*7.91015625E-2-p[4]*t13*(9.0/6.4E1)+p[4]*t14*(9.0/3.2E1))*2.0+cos(t18-p[4]*t9*9.765625E-4-p[4]*t13*(1.0/1.92E2)+p[4]*t14*(1.0/3.2E1))*2.0+cos(t19-p[4]*t9*4.94384765625E-3-p[4]*t13*(9.0/5.12E2)+p[4]*t14*(9.0/1.28E2))*4.0+cos(t20-p[4]*t9*6.103515625E-5-p[4]*t13*6.510416666666667E-4+p[4]*t14*(1.0/1.28E2))*4.0+cos(t21-p[4]*t9*3.814697265625E-2-p[4]*t13*8.138020833333333E-2+p[4]*t14*(2.5E1/1.28E2))*4.0+cos(t22-p[4]*t9*1.4654541015625E-1-p[4]*t13*2.233072916666667E-1+p[4]*t14*(4.9E1/1.28E2))*4.0+1.0)*(1.0/2.4E1))*(p[4]*sin(t15-p[4]*t9*(1.0/4.0)-p[4]*t13*(1.0/3.0)+p[4]*(p[3]-t2+t3-t4)*(1.0/2.0))*(3.0/8.0)+p[4]*sin(t16-p[4]*t9*(1.0/6.4E1)-p[4]*t13*(1.0/2.4E1)+p[4]*(p[3]-t2+t3-t4)*(1.0/8.0))*(5.1E1/6.4E1)+p[4]*sin(t17-p[4]*t9*7.91015625E-2-p[4]*t13*(9.0/6.4E1)+p[4]*(p[3]-t2+t3-t4)*(9.0/3.2E1))*8.701171875E-1+p[4]*sin(t18-p[4]*t9*9.765625E-4-p[4]*t13*(1.0/1.92E2)+p[4]*(p[3]-t2+t3-t4)*(1.0/3.2E1))*3.544921875E-1+p[4]*sin(t19-p[4]*t9*4.94384765625E-3-p[4]*t13*(9.0/5.12E2)+p[4]*(p[3]-t2+t3-t4)*(9.0/1.28E2))*1.2161865234375+p[4]*sin(t20-p[4]*t9*6.103515625E-5-p[4]*t13*6.510416666666667E-4+p[4]*(p[3]-t2+t3-t4)*(1.0/1.28E2))*2.259521484375E-1+p[4]*sin(t21-p[4]*t9*3.814697265625E-2-p[4]*t13*8.138020833333333E-2+p[4]*(p[3]-t2+t3-t4)*(2.5E1/1.28E2))*1.7669677734375+p[4]*sin(t22-p[4]*t9*1.4654541015625E-1-p[4]*t13*2.233072916666667E-1+p[4]*(p[3]-t2+t3-t4)*(4.9E1/1.28E2))*1.5970458984375)*(1.0/1.2E1)
        grad[0] = t0

        t2 = p[0]*(1.1E1/2.0)
        t3 = p[1]*9.0
        t4 = p[2]*(9.0/2.0)
        t5 = p[0]*(9.0/2.0)
        t6 = p[1]*(2.7E1/2.0)
        t7 = p[2]*(2.7E1/2.0)
        t8 = p[3]*(9.0/2.0)
        t9 = t5-t6+t7-t8
        t10 = p[0]*9.0
        t11 = p[1]*(4.5E1/2.0)
        t12 = p[2]*1.8E1
        t13 = t8-t10+t11-t12
        t14 = p[3]-t2+t3-t4
        t15 = p[0]*p[4]
        t16 = p[0]*p[4]*(1.0/2.0)
        t17 = p[4]*t14*(1.0/8.0)
        t18 = t16+t17-p[4]*t9*(1.0/6.4E1)-p[4]*t13*(1.0/2.4E1)
        t19 = p[0]*p[4]*(3.0/4.0)
        t20 = p[0]*p[4]*(1.0/4.0)
        t21 = p[4]*t14*(1.0/3.2E1)
        t22 = t20+t21-p[4]*t9*9.765625E-4-p[4]*t13*(1.0/1.92E2)
        t23 = p[0]*p[4]*(3.0/8.0)
        t24 = p[4]*t14*(9.0/1.28E2)
        t25 = t23+t24-p[4]*t9*4.94384765625E-3-p[4]*t13*(9.0/5.12E2)
        t26 = p[0]*p[4]*(1.0/8.0)
        t27 = p[4]*t14*(1.0/1.28E2)
        t28 = t26+t27-p[4]*t9*6.103515625E-5-p[4]*t13*6.510416666666667E-4
        t29 = p[0]*p[4]*(5.0/8.0)
        t30 = p[0]*p[4]*(7.0/8.0)
        t0 = p[4]*(self._xf-p[4]*(cos(t15-p[4]*t9*(1.0/4.0)-p[4]*t13*(1.0/3.0)+p[4]*t14*(1.0/2.0))+cos(t19-p[4]*t9*7.91015625E-2-p[4]*t13*(9.0/6.4E1)+p[4]*t14*(9.0/3.2E1))*2.0+cos(t29-p[4]*t9*3.814697265625E-2-p[4]*t13*8.138020833333333E-2+p[4]*t14*(2.5E1/1.28E2))*4.0+cos(t30-p[4]*t9*1.4654541015625E-1-p[4]*t13*2.233072916666667E-1+p[4]*t14*(4.9E1/1.28E2))*4.0+cos(t18)*2.0+cos(t22)*2.0+cos(t25)*4.0+cos(t28)*4.0+1.0)*(1.0/2.4E1))*(p[4]*sin(t15-p[4]*t9*(1.0/4.0)-p[4]*t13*(1.0/3.0)+p[4]*(p[3]-t2+t3-t4)*(1.0/2.0))*(3.0/8.0)+p[4]*sin(t19-p[4]*t9*7.91015625E-2-p[4]*t13*(9.0/6.4E1)+p[4]*(p[3]-t2+t3-t4)*(9.0/3.2E1))*3.955078125E-1+p[4]*sin(t29-p[4]*t9*3.814697265625E-2-p[4]*t13*8.138020833333333E-2+p[4]*(p[3]-t2+t3-t4)*(2.5E1/1.28E2))*2.838134765625E-1+p[4]*sin(t30-p[4]*t9*1.4654541015625E-1-p[4]*t13*2.233072916666667E-1+p[4]*(p[3]-t2+t3-t4)*(4.9E1/1.28E2))*1.2740478515625-p[4]*sin(t18)*(3.0/6.4E1)-p[4]*sin(t22)*1.201171875E-1-p[4]*sin(t25)*2.669677734375E-1-p[4]*sin(t28)*9.70458984375E-2)*(1.0/1.2E1)
        grad[1] = t0

        t2 = p[0]*(1.1E1/2.0)
        t3 = p[1]*9.0
        t4 = p[2]*(9.0/2.0)
        t5 = p[0]*(9.0/2.0)
        t6 = p[1]*(2.7E1/2.0)
        t7 = p[2]*(2.7E1/2.0)
        t8 = p[3]*(9.0/2.0)
        t9 = t5-t6+t7-t8
        t10 = p[0]*9.0
        t11 = p[1]*(4.5E1/2.0)
        t12 = p[2]*1.8E1
        t13 = t8-t10+t11-t12
        t14 = p[3]-t2+t3-t4
        t15 = p[0]*p[4]
        t16 = p[0]*p[4]*(1.0/2.0)
        t17 = p[0]*p[4]*(3.0/4.0)
        t18 = p[0]*p[4]*(1.0/4.0)
        t19 = p[0]*p[4]*(3.0/8.0)
        t20 = p[0]*p[4]*(1.0/8.0)
        t21 = p[0]*p[4]*(5.0/8.0)
        t22 = p[0]*p[4]*(7.0/8.0)
        t23 = p[4]*(p[3]-t2+t3-t4)*(1.0/2.0)
        t39 = p[4]*t9*(1.0/4.0)
        t40 = p[4]*t13*(1.0/3.0)
        t24 = t15+t23-t39-t40
        t25 = p[4]*(p[3]-t2+t3-t4)*(1.0/8.0)
        t41 = p[4]*t9*(1.0/6.4E1)
        t42 = p[4]*t13*(1.0/2.4E1)
        t26 = t16+t25-t41-t42
        t27 = p[4]*(p[3]-t2+t3-t4)*(1.0/3.2E1)
        t45 = p[4]*t9*9.765625E-4
        t46 = p[4]*t13*(1.0/1.92E2)
        t28 = t18+t27-t45-t46
        t29 = p[4]*(p[3]-t2+t3-t4)*(9.0/3.2E1)
        t43 = p[4]*t9*7.91015625E-2
        t44 = p[4]*t13*(9.0/6.4E1)
        t30 = t17+t29-t43-t44
        t31 = p[4]*(p[3]-t2+t3-t4)*(1.0/1.28E2)
        t49 = p[4]*t9*6.103515625E-5
        t50 = p[4]*t13*6.510416666666667E-4
        t32 = t20+t31-t49-t50
        t33 = p[4]*(p[3]-t2+t3-t4)*(9.0/1.28E2)
        t47 = p[4]*t9*4.94384765625E-3
        t48 = p[4]*t13*(9.0/5.12E2)
        t34 = t19+t33-t47-t48
        t35 = p[4]*(p[3]-t2+t3-t4)*(2.5E1/1.28E2)
        t51 = p[4]*t9*3.814697265625E-2
        t52 = p[4]*t13*8.138020833333333E-2
        t36 = t21+t35-t51-t52
        t37 = p[4]*(p[3]-t2+t3-t4)*(4.9E1/1.28E2)
        t53 = p[4]*t9*1.4654541015625E-1
        t54 = p[4]*t13*2.233072916666667E-1
        t38 = t22+t37-t53-t54
        t0 = (self._xf-p[4]*(cos(t15-t39-t40+p[4]*t14*(1.0/2.0))+cos(t16-t41-t42+p[4]*t14*(1.0/8.0))*2.0+cos(t18-t45-t46+p[4]*t14*(1.0/3.2E1))*2.0+cos(t17-t43-t44+p[4]*t14*(9.0/3.2E1))*2.0+cos(t20-t49-t50+p[4]*t14*(1.0/1.28E2))*4.0+cos(t19-t47-t48+p[4]*t14*(9.0/1.28E2))*4.0+cos(t21-t51-t52+p[4]*t14*(2.5E1/1.28E2))*4.0+cos(t22-t53-t54+p[4]*t14*(4.9E1/1.28E2))*4.0+1.0)*(1.0/2.4E1))*(cos(t24)*(1.0/2.4E1)+cos(t26)*(1.0/1.2E1)+cos(t28)*(1.0/1.2E1)+cos(t30)*(1.0/1.2E1)+cos(t32)*(1.0/6.0)+cos(t34)*(1.0/6.0)+cos(t36)*(1.0/6.0)+cos(t38)*(1.0/6.0)-p[4]*(sin(t24)*(p[0]*(1.0/8.0)+p[1]*(3.0/8.0)+p[2]*(3.0/8.0)+p[3]*(1.0/8.0))+sin(t26)*(p[0]*(1.5E1/1.28E2)+p[1]*(5.1E1/1.28E2)-p[2]*(3.0/1.28E2)+p[3]*(1.0/1.28E2))*2.0+sin(t28)*(p[0]*1.2060546875E-1+p[1]*1.7724609375E-1-p[2]*6.005859375E-2+p[3]*1.220703125E-2)*2.0+sin(t30)*(p[0]*1.1279296875E-1+p[1]*4.3505859375E-1+p[2]*1.9775390625E-1+p[3]*4.39453125E-3)*2.0+sin(t32)*(p[0]*8.7615966796875E-2+p[1]*5.6488037109375E-2-p[2]*2.4261474609375E-2+p[3]*5.157470703125E-3)*4.0+sin(t34)*(p[0]*1.24237060546875E-1+p[1]*3.04046630859375E-1-p[2]*6.6741943359375E-2+p[3]*1.3458251953125E-2)*4.0+sin(t36)*(p[0]*1.11541748046875E-1+p[1]*4.41741943359375E-1+p[2]*7.0953369140625E-2+p[3]*7.62939453125E-4)*4.0+sin(t38)*(p[0]*1.19842529296875E-1+p[1]*3.99261474609375E-1+p[2]*3.18511962890625E-1+p[3]*3.7384033203125E-2)*4.0)*(1.0/2.4E1)+1.0/2.4E1)*-2.0
        grad[2] = t0

        return grad

    def fyf(self, p):
        t2 = p[0]*(1.1E1/2.0)
        t3 = p[1]*9.0
        t4 = p[2]*(9.0/2.0)
        t5 = p[0]*(9.0/2.0)
        t6 = p[1]*(2.7E1/2.0)
        t7 = p[2]*(2.7E1/2.0)
        t8 = p[3]*(9.0/2.0)
        t9 = t5-t6+t7-t8
        t10 = p[0]*9.0
        t11 = p[1]*(4.5E1/2.0)
        t12 = p[2]*1.8E1
        t13 = t8-t10+t11-t12
        t14 = p[3]-t2+t3-t4
        t15 = self._yf-p[4]*(sin(p[0]*p[4]-p[4]*t9*(1.0/4.0)-p[4]*t13*(1.0/3.0)+p[4]*t14*(1.0/2.0))+sin(p[0]*p[4]*(1.0/2.0)-p[4]*t9*(1.0/6.4E1)-p[4]*t13*(1.0/2.4E1)+p[4]*t14*(1.0/8.0))*2.0+sin(p[0]*p[4]*(3.0/4.0)-p[4]*t9*7.91015625E-2-p[4]*t13*(9.0/6.4E1)+p[4]*t14*(9.0/3.2E1))*2.0+sin(p[0]*p[4]*(1.0/4.0)-p[4]*t9*9.765625E-4-p[4]*t13*(1.0/1.92E2)+p[4]*t14*(1.0/3.2E1))*2.0+sin(p[0]*p[4]*(3.0/8.0)-p[4]*t9*4.94384765625E-3-p[4]*t13*(9.0/5.12E2)+p[4]*t14*(9.0/1.28E2))*4.0+sin(p[0]*p[4]*(1.0/8.0)-p[4]*t9*6.103515625E-5-p[4]*t13*6.510416666666667E-4+p[4]*t14*(1.0/1.28E2))*4.0+sin(p[0]*p[4]*(5.0/8.0)-p[4]*t9*3.814697265625E-2-p[4]*t13*8.138020833333333E-2+p[4]*t14*(2.5E1/1.28E2))*4.0+sin(p[0]*p[4]*(7.0/8.0)-p[4]*t9*1.4654541015625E-1-p[4]*t13*2.233072916666667E-1+p[4]*t14*(4.9E1/1.28E2))*4.0)*(1.0/2.4E1)
        t0 = t15*t15
        return t0

    def fyf_grad(self, p):
        grad = [0.0, 0.0, 0.0]

        t2 = p[0]*(1.1E1/2.0)
        t3 = p[1]*9.0
        t4 = p[2]*(9.0/2.0)
        t5 = p[0]*(9.0/2.0)
        t6 = p[1]*(2.7E1/2.0)
        t7 = p[2]*(2.7E1/2.0)
        t8 = p[3]*(9.0/2.0)
        t9 = t5-t6+t7-t8
        t10 = p[0]*9.0
        t11 = p[1]*(4.5E1/2.0)
        t12 = p[2]*1.8E1
        t13 = t8-t10+t11-t12
        t14 = p[3]-t2+t3-t4
        t15 = p[0]*p[4]
        t16 = p[0]*p[4]*(1.0/2.0)
        t17 = p[0]*p[4]*(3.0/4.0)
        t18 = p[0]*p[4]*(1.0/4.0)
        t19 = p[0]*p[4]*(3.0/8.0)
        t20 = p[0]*p[4]*(1.0/8.0)
        t21 = p[0]*p[4]*(5.0/8.0)
        t22 = p[0]*p[4]*(7.0/8.0)
        t23 = p[4]*t14*(1.0/2.0)
        t24 = t15+t23-p[4]*t9*(1.0/4.0)-p[4]*t13*(1.0/3.0)
        t25 = p[4]*t14*(1.0/8.0)
        t26 = t16+t25-p[4]*t9*(1.0/6.4E1)-p[4]*t13*(1.0/2.4E1)
        t27 = p[4]*t14*(9.0/3.2E1)
        t28 = t17+t27-p[4]*t9*7.91015625E-2-p[4]*t13*(9.0/6.4E1)
        t29 = p[4]*t14*(1.0/3.2E1)
        t30 = t18+t29-p[4]*t9*9.765625E-4-p[4]*t13*(1.0/1.92E2)
        t31 = p[4]*t14*(9.0/1.28E2)
        t32 = t19+t31-p[4]*t9*4.94384765625E-3-p[4]*t13*(9.0/5.12E2)
        t33 = p[4]*t14*(1.0/1.28E2)
        t34 = t20+t33-p[4]*t9*6.103515625E-5-p[4]*t13*6.510416666666667E-4
        t35 = p[4]*t14*(2.5E1/1.28E2)
        t36 = t21+t35-p[4]*t9*3.814697265625E-2-p[4]*t13*8.138020833333333E-2
        t37 = p[4]*t14*(4.9E1/1.28E2)
        t38 = t22+t37-p[4]*t9*1.4654541015625E-1-p[4]*t13*2.233072916666667E-1
        t0 = p[4]*(self._yf-p[4]*(sin(t24)+sin(t26)*2.0+sin(t28)*2.0+sin(t30)*2.0+sin(t32)*4.0+sin(t34)*4.0+sin(t36)*4.0+sin(t38)*4.0)*(1.0/2.4E1))*(p[4]*cos(t24)*(3.0/8.0)+p[4]*cos(t26)*(5.1E1/6.4E1)+p[4]*cos(t28)*8.701171875E-1+p[4]*cos(t30)*3.544921875E-1+p[4]*cos(t32)*1.2161865234375+p[4]*cos(t34)*2.259521484375E-1+p[4]*cos(t36)*1.7669677734375+p[4]*cos(t38)*1.5970458984375)*(-1.0/1.2E1)
        grad[0] = t0


        t2 = p[0]*(1.1E1/2.0)
        t3 = p[1]*9.0
        t4 = p[2]*(9.0/2.0)
        t5 = p[0]*(9.0/2.0)
        t6 = p[1]*(2.7E1/2.0)
        t7 = p[2]*(2.7E1/2.0)
        t8 = p[3]*(9.0/2.0)
        t9 = t5-t6+t7-t8
        t10 = p[0]*9.0
        t11 = p[1]*(4.5E1/2.0)
        t12 = p[2]*1.8E1
        t13 = t8-t10+t11-t12
        t14 = p[3]-t2+t3-t4
        t15 = p[0]*p[4]
        t16 = p[0]*p[4]*(1.0/2.0)
        t17 = p[4]*t14*(1.0/8.0)
        t18 = t16+t17-p[4]*t9*(1.0/6.4E1)-p[4]*t13*(1.0/2.4E1)
        t19 = p[0]*p[4]*(3.0/4.0)
        t20 = p[0]*p[4]*(1.0/4.0)
        t21 = p[4]*t14*(1.0/3.2E1)
        t22 = t20+t21-p[4]*t9*9.765625E-4-p[4]*t13*(1.0/1.92E2)
        t23 = p[0]*p[4]*(3.0/8.0)
        t24 = p[4]*t14*(9.0/1.28E2)
        t25 = t23+t24-p[4]*t9*4.94384765625E-3-p[4]*t13*(9.0/5.12E2)
        t26 = p[0]*p[4]*(1.0/8.0)
        t27 = p[4]*t14*(1.0/1.28E2)
        t28 = t26+t27-p[4]*t9*6.103515625E-5-p[4]*t13*6.510416666666667E-4
        t29 = p[0]*p[4]*(5.0/8.0)
        t30 = p[0]*p[4]*(7.0/8.0)
        t31 = p[4]*t14*(1.0/2.0)
        t32 = t15+t31-p[4]*t9*(1.0/4.0)-p[4]*t13*(1.0/3.0)
        t33 = p[4]*t14*(9.0/3.2E1)
        t34 = t19+t33-p[4]*t9*7.91015625E-2-p[4]*t13*(9.0/6.4E1)
        t35 = p[4]*t14*(2.5E1/1.28E2)
        t36 = t29+t35-p[4]*t9*3.814697265625E-2-p[4]*t13*8.138020833333333E-2
        t37 = p[4]*t14*(4.9E1/1.28E2)
        t38 = t30+t37-p[4]*t9*1.4654541015625E-1-p[4]*t13*2.233072916666667E-1
        t0 = p[4]*(self._yf-p[4]*(sin(t18)*2.0+sin(t22)*2.0+sin(t25)*4.0+sin(t28)*4.0+sin(t32)+sin(t34)*2.0+sin(t36)*4.0+sin(t38)*4.0)*(1.0/2.4E1))*(p[4]*cos(t18)*(3.0/6.4E1)+p[4]*cos(t22)*1.201171875E-1+p[4]*cos(t25)*2.669677734375E-1+p[4]*cos(t28)*9.70458984375E-2-p[4]*cos(t32)*(3.0/8.0)-p[4]*cos(t34)*3.955078125E-1-p[4]*cos(t36)*2.838134765625E-1-p[4]*cos(t38)*1.2740478515625)*(1.0/1.2E1)
        grad[1] = t0

        t2 = p[0]*(1.1E1/2.0)
        t3 = p[1]*9.0
        t4 = p[2]*(9.0/2.0)
        t5 = p[0]*(9.0/2.0)
        t6 = p[1]*(2.7E1/2.0)
        t7 = p[2]*(2.7E1/2.0)
        t8 = p[3]*(9.0/2.0)
        t9 = t5-t6+t7-t8
        t10 = p[0]*9.0
        t11 = p[1]*(4.5E1/2.0)
        t12 = p[2]*1.8E1
        t13 = t8-t10+t11-t12
        t14 = p[3]-t2+t3-t4
        t15 = p[0]*p[4]
        t16 = p[0]*p[4]*(1.0/2.0)
        t17 = p[0]*p[4]*(3.0/4.0)
        t18 = p[0]*p[4]*(1.0/4.0)
        t19 = p[0]*p[4]*(3.0/8.0)
        t20 = p[0]*p[4]*(1.0/8.0)
        t21 = p[0]*p[4]*(5.0/8.0)
        t22 = p[0]*p[4]*(7.0/8.0)
        t23 = p[4]*(p[3]-t2+t3-t4)*(1.0/2.0)
        t39 = p[4]*t9*(1.0/4.0)
        t40 = p[4]*t13*(1.0/3.0)
        t24 = t15+t23-t39-t40
        t25 = p[4]*(p[3]-t2+t3-t4)*(1.0/8.0)
        t41 = p[4]*t9*(1.0/6.4E1)
        t42 = p[4]*t13*(1.0/2.4E1)
        t26 = t16+t25-t41-t42
        t27 = p[4]*(p[3]-t2+t3-t4)*(1.0/3.2E1)
        t45 = p[4]*t9*9.765625E-4
        t46 = p[4]*t13*(1.0/1.92E2)
        t28 = t18+t27-t45-t46
        t29 = p[4]*(p[3]-t2+t3-t4)*(9.0/3.2E1)
        t43 = p[4]*t9*7.91015625E-2
        t44 = p[4]*t13*(9.0/6.4E1)
        t30 = t17+t29-t43-t44
        t31 = p[4]*(p[3]-t2+t3-t4)*(1.0/1.28E2)
        t49 = p[4]*t9*6.103515625E-5
        t50 = p[4]*t13*6.510416666666667E-4
        t32 = t20+t31-t49-t50
        t33 = p[4]*(p[3]-t2+t3-t4)*(9.0/1.28E2)
        t47 = p[4]*t9*4.94384765625E-3
        t48 = p[4]*t13*(9.0/5.12E2)
        t34 = t19+t33-t47-t48
        t35 = p[4]*(p[3]-t2+t3-t4)*(2.5E1/1.28E2)
        t51 = p[4]*t9*3.814697265625E-2
        t52 = p[4]*t13*8.138020833333333E-2
        t36 = t21+t35-t51-t52
        t37 = p[4]*(p[3]-t2+t3-t4)*(4.9E1/1.28E2)
        t53 = p[4]*t9*1.4654541015625E-1
        t54 = p[4]*t13*2.233072916666667E-1
        t38 = t22+t37-t53-t54
        t0 = (self._yf-p[4]*(sin(t15-t39-t40+p[4]*t14*(1.0/2.0))+sin(t16-t41-t42+p[4]*t14*(1.0/8.0))*2.0+sin(t18-t45-t46+p[4]*t14*(1.0/3.2E1))*2.0+sin(t17-t43-t44+p[4]*t14*(9.0/3.2E1))*2.0+sin(t20-t49-t50+p[4]*t14*(1.0/1.28E2))*4.0+sin(t19-t47-t48+p[4]*t14*(9.0/1.28E2))*4.0+sin(t21-t51-t52+p[4]*t14*(2.5E1/1.28E2))*4.0+sin(t22-t53-t54+p[4]*t14*(4.9E1/1.28E2))*4.0)*(1.0/2.4E1))*(sin(t24)*(1.0/2.4E1)+sin(t26)*(1.0/1.2E1)+sin(t28)*(1.0/1.2E1)+sin(t30)*(1.0/1.2E1)+sin(t32)*(1.0/6.0)+sin(t34)*(1.0/6.0)+sin(t36)*(1.0/6.0)+sin(t38)*(1.0/6.0)+p[4]*(cos(t24)*(p[0]*(1.0/8.0)+p[1]*(3.0/8.0)+p[2]*(3.0/8.0)+p[3]*(1.0/8.0))+cos(t26)*(p[0]*(1.5E1/1.28E2)+p[1]*(5.1E1/1.28E2)-p[2]*(3.0/1.28E2)+p[3]*(1.0/1.28E2))*2.0+cos(t28)*(p[0]*1.2060546875E-1+p[1]*1.7724609375E-1-p[2]*6.005859375E-2+p[3]*1.220703125E-2)*2.0+cos(t30)*(p[0]*1.1279296875E-1+p[1]*4.3505859375E-1+p[2]*1.9775390625E-1+p[3]*4.39453125E-3)*2.0+cos(t32)*(p[0]*8.7615966796875E-2+p[1]*5.6488037109375E-2-p[2]*2.4261474609375E-2+p[3]*5.157470703125E-3)*4.0+cos(t34)*(p[0]*1.24237060546875E-1+p[1]*3.04046630859375E-1-p[2]*6.6741943359375E-2+p[3]*1.3458251953125E-2)*4.0+cos(t36)*(p[0]*1.11541748046875E-1+p[1]*4.41741943359375E-1+p[2]*7.0953369140625E-2+p[3]*7.62939453125E-4)*4.0+cos(t38)*(p[0]*1.19842529296875E-1+p[1]*3.99261474609375E-1+p[2]*3.18511962890625E-1+p[3]*3.7384033203125E-2)*4.0)*(1.0/2.4E1))*-2.0
        grad[2] = t0

        return grad

    def ftf(self, p):
      t2 = self._tf-p[0]*p[4]+p[4]*(p[0]*(1.1E1/2.0)-p[1]*9.0+p[2]*(9.0/2.0)-p[3])*(1.0/2.0)+p[4]*(p[0]*(9.0/2.0)-p[1]*(2.7E1/2.0)+p[2]*(2.7E1/2.0)-p[3]*(9.0/2.0))*(1.0/4.0)-p[4]*(p[0]*9.0-p[1]*(4.5E1/2.0)+p[2]*1.8E1-p[3]*(9.0/2.0))*(1.0/3.0)
      t0 = t2*t2
      return t0

    def ftf_grad(self, p):
        grad = [0.0, 0.0, 0.0]

        t0 = p[4]*(self._tf-p[0]*p[4]+p[4]*(p[0]*(1.1E1/2.0)-p[1]*9.0+p[2]*(9.0/2.0)-p[3])*(1.0/2.0)+p[4]*(p[0]*(9.0/2.0)-p[1]*(2.7E1/2.0)+p[2]*(2.7E1/2.0)-p[3]*(9.0/2.0))*(1.0/4.0)-p[4]*(p[0]*9.0-p[1]*(4.5E1/2.0)+p[2]*1.8E1-p[3]*(9.0/2.0))*(1.0/3.0))*(-3.0/4.0)
        grad[0] = t0

        t0 = p[4]*(self._tf-p[0]*p[4]+p[4]*(p[0]*(1.1E1/2.0)-p[1]*9.0+p[2]*(9.0/2.0)-p[3])*(1.0/2.0)+p[4]*(p[0]*(9.0/2.0)-p[1]*(2.7E1/2.0)+p[2]*(2.7E1/2.0)-p[3]*(9.0/2.0))*(1.0/4.0)-p[4]*(p[0]*9.0-p[1]*(4.5E1/2.0)+p[2]*1.8E1-p[3]*(9.0/2.0))*(1.0/3.0))*(-3.0/4.0)
        grad[1] = t0

        t0 = (p[0]*(1.0/8.0)+p[1]*(3.0/8.0)+p[2]*(3.0/8.0)+p[3]*(1.0/8.0))*(self._tf-p[0]*p[4]+p[4]*(p[0]*(1.1E1/2.0)-p[1]*9.0+p[2]*(9.0/2.0)-p[3])*(1.0/2.0)+p[4]*(p[0]*(9.0/2.0)-p[1]*(2.7E1/2.0)+p[2]*(2.7E1/2.0)-p[3]*(9.0/2.0))*(1.0/4.0)-p[4]*(p[0]*9.0-p[1]*(4.5E1/2.0)+p[2]*1.8E1-p[3]*(9.0/2.0))*(1.0/3.0))*-2.0
        grad[2] = t0

        return grad

    def fbe(self, p):
        t0 = p[4]*(p[0]*p[1]*9.9E1-p[0]*p[2]*3.6E1+p[0]*p[3]*1.9E1-p[1]*p[2]*8.1E1-p[1]*p[3]*3.6E1+p[2]*p[3]*9.9E1+(p[0]*p[0])*6.4E1+(p[1]*p[1])*3.24E2+(p[2]*p[2])*3.24E2+(p[3]*p[3])*6.4E1)*(1.0/8.4E2)
        return t0

    def fbe_grad(self, p):
        grad = [0.0, 0.0, 0.0]

        t0 = p[4]*(p[0]*9.9E1+p[1]*6.48E2-p[2]*8.1E1-p[3]*3.6E1)*(1.0/8.4E2)
        grad[0] = t0

        t0 = p[4]*(p[0]*3.6E1+p[1]*8.1E1-p[2]*6.48E2-p[3]*9.9E1)*(-1.0/8.4E2)
        grad[1] = t0

        t0 = p[0]*p[1]*(3.3E1/2.8E2)-p[0]*p[2]*(3.0/7.0E1)+p[0]*p[3]*(1.9E1/8.4E2)-p[1]*p[2]*(2.7E1/2.8E2)-p[1]*p[3]*(3.0/7.0E1)+p[2]*p[3]*(3.3E1/2.8E2)+(p[0]*p[0])*(8.0/1.05E2)+(p[1]*p[1])*(2.7E1/7.0E1)+(p[2]*p[2])*(2.7E1/7.0E1)+(p[3]*p[3])*(8.0/1.05E2)
        grad[2] = t0

        return grad



# VelocityPlanner
class VelocityPlanner:
    def __init__(self, time_gap, a_max, slow_speed, stop_line_buffer):
        self._time_gap = time_gap
        self._a_max = a_max
        self._slow_speed = slow_speed
        self._stop_line_buffer = stop_line_buffer
        self._prev_trajectory = [[0.0, 0.0, 0.0]]

    # Computes an open loop speed estimate based on the previously planned
    # trajectory, and the timestep since the last planning cycle.
    # Input: timestep is in seconds
    def get_open_loop_speed(self, timestep):
        if len(self._prev_trajectory) == 1:
            return self._prev_trajectory[0][2]

            # If simulation time step is zero, give the start of the trajectory as the
        # open loop estimate.
        if timestep < 1e-4:
            return self._prev_trajectory[0][2]

        for i in range(len(self._prev_trajectory) - 1):
            distance_step = np.linalg.norm(np.subtract(self._prev_trajectory[i + 1][0:2],
                                                       self._prev_trajectory[i][0:2]))
            velocity = self._prev_trajectory[i][2]
            time_delta = distance_step / velocity

            # If time_delta exceeds the remaining time in our simulation timestep,
            # interpolate between the velocity of the current step and the velocity
            # of the next step to estimate the open loop velocity.
            if time_delta > timestep:
                v1 = self._prev_trajectory[i][2]
                v2 = self._prev_trajectory[i + 1][2]
                v_delta = v2 - v1
                interpolation_ratio = timestep / time_delta
                return v1 + interpolation_ratio * v_delta

            # Otherwise, keep checking.
            else:
                timestep -= time_delta

        # Simulation time step exceeded the length of the path, which means we have likely
        # stopped. Return the end velocity of the trajectory.
        return self._prev_trajectory[-1][2]

    ######################################################
    ######################################################
    # MODULE 7: COMPUTE VELOCITY PROFILE
    #   Read over the function comments to familiarize yourself with the
    ######################################################
    ######################################################
    # Takes a path, and computes a velocity profile to our desired speed.
    # - decelerate_to_stop denotes whether or not we need to decelerate to a
    #   stop line
    # - follow_lead_vehicle denotes whether or not we need to follow a lead
    #   vehicle, with state given by lead_car_state.
    # The order of precedence for handling these cases is stop sign handling,
    # lead vehicle handling, then nominal lane maintenance. In a real velocity
    # planner you would need to handle the coupling between these states, but
    # for simplicity this project can be implemented by isolating each case.
    # For all profiles, the required acceleration is given by self._a_max.
    # Recall that the path is of the form [x_points, y_points, t_points].
    def compute_velocity_profile(self, path, desired_speed, ego_state,
                                 closed_loop_speed, decelerate_to_stop,
                                 lead_car_state, follow_lead_vehicle):
        """Computes the velocity profile for the local planner path.

        args:
            path: Path (global frame) that the vehicle will follow.
                Format: [x_points, y_points, t_points]
                        x_points: List of x values (m)
                        y_points: List of y values (m)
                        t_points: List of yaw values (rad)
                    Example of accessing the ith point's y value:
                        paths[1][i]
                It is assumed that the stop line is at the end of the path.
            desired_speed: speed which the vehicle should reach (m/s)
            ego_state: ego state vector for the vehicle, in the global frame.
                format: [ego_x, ego_y, ego_yaw, ego_open_loop_speed]
                    ego_x and ego_y     : position (m)
                    ego_yaw             : top-down orientation [-pi to pi]
                    ego_open_loop_speed : open loop speed (m/s)
            closed_loop_speed: current (closed-loop) speed for vehicle (m/s)
            decelerate_to_stop: Flag where if true, should decelerate to stop
            lead_car_state: the lead vehicle current state.
                Format: [lead_car_x, lead_car_y, lead_car_speed]
                    lead_car_x and lead_car_y   : position (m)
                    lead_car_speed              : lead car speed (m/s)
            follow_lead_vehicle: If true, the ego car should perform lead
                vehicle handling, as the lead vehicle is close enough to
                influence the speed profile of the local path.
        internal parameters of interest:
            self._slow_speed: coasting speed (m/s) of the vehicle before it
                comes to a stop
            self._stop_line_buffer: buffer distance to stop line (m) for vehicle
                to stop at
            self._a_max: maximum acceleration/deceleration of the vehicle (m/s^2)
            self._time_gap: Amount of time taken to reach the lead vehicle from
                the current position
        returns:
            profile: Updated profile which contains the local path as well as
                the speed to be tracked by the controller (global frame).
                Length and speed in m and m/s.
                Format: [[x0, y0, v0],
                         [x1, y1, v1],
                         ...,
                         [xm, ym, vm]]
                example:
                    profile[2][1]:
                    returns the 3rd point's y position in the local path

                    profile[5]:
                    returns [x5, y5, v5] (6th point in the local path)
        """
        profile = []
        # For our profile, use the open loop speed as our initial speed.
        start_speed = ego_state[3]
        # Generate a trapezoidal profile to decelerate to stop.
        if decelerate_to_stop:
            profile = self.decelerate_profile(path, start_speed)

        # If we need to follow the lead vehicle, make sure we decelerate to its
        # speed by the time we reach the time gap point.
        elif follow_lead_vehicle:
            profile = self.follow_profile(path, start_speed, desired_speed,
                                          lead_car_state)

        # Otherwise, compute the profile to reach our desired speed.
        else:
            profile = self.nominal_profile(path, start_speed, desired_speed)

        # Interpolate between the zeroth state and the first state.
        # This prevents the myopic controller from getting stuck at the zeroth
        # state.
        if len(profile) > 1:
            interpolated_state = [(profile[1][0] - profile[0][0]) * 0.1 + profile[0][0],
                                  (profile[1][1] - profile[0][1]) * 0.1 + profile[0][1],
                                  (profile[1][2] - profile[0][2]) * 0.1 + profile[0][2]]
            del profile[0]
            profile.insert(0, interpolated_state)

        # Save the planned profile for open loop speed estimation.
        self._prev_trajectory = profile

        return profile

    # Computes a trapezoidal profile for decelerating to stop.
    def decelerate_profile(self, path, start_speed):
        """Computes the velocity profile for the local path to decelerate to a
        stop.

        args:
            path: Path (global frame) that the vehicle will follow.
                Format: [x_points, y_points, t_points]
                        x_points: List of x values (m)
                        y_points: List of y values (m)
                        t_points: List of yaw values (rad)
                    Example of accessing the ith point's y value:
                        paths[1][i]
                It is assumed that the stop line is at the end of the path.
            start_speed: speed which the vehicle starts with (m/s)
        internal parameters of interest:
            self._slow_speed: coasting speed (m/s) of the vehicle before it
                comes to a stop
            self._stop_line_buffer: buffer distance to stop line (m) for vehicle
                to stop at
            self._a_max: maximum acceleration/deceleration of the vehicle (m/s^2)
        returns:
            profile: deceleration profile which contains the local path as well
                as the speed to be tracked by the controller (global frame).
                Length and speed in m and m/s.
                Format: [[x0, y0, v0],
                         [x1, y1, v1],
                         ...,
                         [xm, ym, vm]]
                example:
                    profile[2][1]:
                    returns the 3rd point's y position in the local path

                    profile[5]:
                    returns [x5, y5, v5] (6th point in the local path)
        """
        profile = []
        slow_speed = self._slow_speed
        stop_line_buffer = self._stop_line_buffer

        # Using d = (v_f^2 - v_i^2) / (2 * a), compute the two distances
        # used in the trapezoidal stop behaviour. decel_distance goes from
        #  start_speed to some coasting speed (slow_speed), then brake_distance
        #  goes from slow_speed to 0, both at a constant deceleration.
        decel_distance = calc_distance(start_speed, slow_speed, -self._a_max)
        brake_distance = calc_distance(slow_speed, 0, -self._a_max)

        # compute total path length
        path_length = 0.0
        for i in range(len(path[0]) - 1):
            path_length += np.linalg.norm([path[0][i + 1] - path[0][i],
                                           path[1][i + 1] - path[1][i]])

        stop_index = len(path[0]) - 1
        temp_dist = 0.0
        # Compute the index at which we should stop.
        while (stop_index > 0) and (temp_dist < stop_line_buffer):
            temp_dist += np.linalg.norm([path[0][stop_index] - path[0][stop_index - 1],
                                         path[1][stop_index] - path[1][stop_index - 1]])
            stop_index -= 1

        # If the brake distance exceeds the length of the path, then we cannot
        # perform a smooth deceleration and require a harder deceleration. Build
        # the path up in reverse to ensure we reach zero speed at the required
        # time.
        if brake_distance + decel_distance + stop_line_buffer > path_length:
            speeds = []
            vf = 0.0
            # The speeds past the stop line buffer should be zero.
            for i in reversed(range(stop_index, len(path[0]))):
                speeds.insert(0, 0.0)
            # The rest of the speeds should be a linear ramp from zero,
            # decelerating at -self._a_max.
            for i in reversed(range(stop_index)):
                dist = np.linalg.norm([path[0][i + 1] - path[0][i],
                                       path[1][i + 1] - path[1][i]])
                vi = calc_final_speed(vf, self._a_max, dist)  # 修改
                # We don't want to have points above the starting speed
                # along our profile, so clamp to start_speed.
                if vi > start_speed:
                    vi = start_speed

                speeds.insert(0, vi)
                vf = vi

            # Generate the profile, given the computed speeds.
            for i in range(len(speeds)):
                profile.append([path[0][i], path[1][i], speeds[i]])

        # Otherwise, we will perform a full trapezoidal profile. The
        # brake_index will be the index of the path at which we start
        # braking, and the decel_index will be the index at which we stop
        # decelerating to our slow_speed. These two indices denote the
        # endpoints of the ramps in our trapezoidal profile.
        else:
            brake_index = stop_index
            temp_dist = 0.0
            # Compute the index at which to start braking down to zero.
            while (brake_index > 0) and (temp_dist < brake_distance):
                temp_dist += np.linalg.norm([path[0][brake_index] - path[0][brake_index - 1],
                                             path[1][brake_index] - path[1][brake_index - 1]])
                brake_index -= 1

            # Compute the index to stop decelerating to the slow speed.  This is
            # done by stepping through the points until accumulating
            # decel_distance of distance to said index, starting from the the
            # start of the path.
            decel_index = 0
            temp_dist = 0.0
            while (decel_index < brake_index) and (temp_dist < decel_distance):
                temp_dist += np.linalg.norm([path[0][decel_index + 1] - path[0][decel_index],
                                             path[1][decel_index + 1] - path[1][decel_index]])
                decel_index += 1

            # The speeds from the start to decel_index should be a linear ramp
            # from the current speed down to the slow_speed, decelerating at
            # -self._a_max.
            vi = start_speed
            for i in range(decel_index):
                dist = np.linalg.norm([path[0][i + 1] - path[0][i],
                                       path[1][i + 1] - path[1][i]])
                vf = calc_final_speed(vi, -self._a_max, dist)
                # We don't want to overshoot our slow_speed, so clamp it to that.
                if vf < slow_speed:
                    vf = slow_speed

                profile.append([path[0][i], path[1][i], vi])
                vi = vf

            # In this portion of the profile, we are maintaining our slow_speed.
            for i in range(decel_index, brake_index):
                profile.append([path[0][i], path[1][i], vi])

            # The speeds from the brake_index to stop_index should be a
            # linear ramp from the slow_speed down to the 0, decelerating at
            # -self._a_max.
            for i in range(brake_index, stop_index):
                dist = np.linalg.norm([path[0][i + 1] - path[0][i],
                                       path[1][i + 1] - path[1][i]])
                vf = calc_final_speed(vi, -self._a_max, dist)
                profile.append([path[0][i], path[1][i], vi])
                vi = vf

            # The rest of the profile consists of our stop_line_buffer, so
            # it contains zero speed for all points.
            for i in range(stop_index, len(path[0])):
                profile.append([path[0][i], path[1][i], 0.0])

        return profile

    # Computes a profile for following a lead vehicle..
    def follow_profile(self, path, start_speed, desired_speed, lead_car_state):
        """Computes the velocity profile for following a lead vehicle.

        args:
            path: Path (global frame) that the vehicle will follow.
                Format: [x_points, y_points, t_points]
                        x_points: List of x values (m)
                        y_points: List of y values (m)
                        t_points: List of yaw values (rad)
                    Example of accessing the ith point's y value:
                        paths[1][i]
                It is assumed that the stop line is at the end of the path.
            start_speed: speed which the vehicle starts with (m/s)
            desired_speed: speed which the vehicle should reach (m/s)
            lead_car_state: the lead vehicle current state.
                Format: [lead_car_x, lead_car_y, lead_car_speed]
                    lead_car_x and lead_car_y   : position (m)
                    lead_car_speed              : lead car speed (m/s)
        internal parameters of interest:
            self._a_max: maximum acceleration/deceleration of the vehicle (m/s^2)
            self._time_gap: Amount of time taken to reach the lead vehicle from
                the current position
        returns:
            profile: Updated follow vehicle profile which contains the local
                path as well as the speed to be tracked by the controller
                (global frame).
                Length and speed in m and m/s.
                Format: [[x0, y0, v0],
                         [x1, y1, v1],
                         ...,
                         [xm, ym, vm]]
                example:
                    profile[2][1]:
                    returns the 3rd point's y position in the local path

                    profile[5]:
                    returns [x5, y5, v5] (6th point in the local path)
        """
        profile = []
        # Find the closest point to the lead vehicle on our planned path.
        min_index = len(path[0]) - 1
        min_dist = float('Inf')
        for i in range(len(path)):
            dist = np.linalg.norm([path[0][i] - lead_car_state[0],
                                   path[1][i] - lead_car_state[1]])
            if dist < min_dist:
                min_dist = dist
                min_index = i

        # Compute the time gap point, assuming our velocity is held constant at
        # the minimum of the desired speed and the ego vehicle's velocity, from
        # the closest point to the lead vehicle on our planned path.
        desired_speed = min(lead_car_state[2], desired_speed)
        ramp_end_index = min_index
        distance = min_dist
        distance_gap = desired_speed * self._time_gap
        while (ramp_end_index > 0) and (distance > distance_gap):
            distance += np.linalg.norm([path[0][ramp_end_index] - path[0][ramp_end_index - 1],
                                        path[1][ramp_end_index] - path[1][ramp_end_index - 1]])
            ramp_end_index -= 1

        # We now need to reach the ego vehicle's speed by the time we reach the
        # time gap point, ramp_end_index, which therefore is the end of our ramp
        # velocity profile.
        if desired_speed < start_speed:
            decel_distance = calc_distance(start_speed, desired_speed, -self._a_max)
        else:
            decel_distance = calc_distance(start_speed, desired_speed, self._a_max)

        # Here we will compute the speed profile from our initial speed to the
        # end of the ramp.
        vi = start_speed
        for i in range(ramp_end_index + 1):
            dist = np.linalg.norm([path[0][i + 1] - path[0][i],
                                   path[1][i + 1] - path[1][i]])
            if desired_speed < start_speed:
                vf = calc_final_speed(vi, -self._a_max, dist)
            else:
                vf = calc_final_speed(vi, self._a_max, dist)

            profile.append([path[0][i], path[1][i], vi])
            vi = vf

        # Once we hit the time gap point, we need to be at the desired speed.
        # If we can't get there using a_max, do an abrupt change in the profile
        # to use the controller to decelerate more quickly.
        for i in range(ramp_end_index + 1, len(path[0])):
            profile.append([path[0][i], path[1][i], desired_speed])

        return profile

    # Computes a profile for nominal speed tracking.
    def nominal_profile(self, path, start_speed, desired_speed):
        """Computes the velocity profile for the local planner path in a normal
        speed tracking case.

        args:
            path: Path (global frame) that the vehicle will follow.
                Format: [x_points, y_points, t_points]
                        x_points: List of x values (m)
                        y_points: List of y values (m)
                        t_points: List of yaw values (rad)
                    Example of accessing the ith point's y value:
                        paths[1][i]
                It is assumed that the stop line is at the end of the path.
            desired_speed: speed which the vehicle should reach (m/s)
        internal parameters of interest:
            self._a_max: maximum acceleration/deceleration of the vehicle (m/s^2)
        returns:
            profile: Updated nominal speed profile which contains the local path
                as well as the speed to be tracked by the controller (global frame).
                Length and speed in m and m/s.
                Format: [[x0, y0, v0],
                         [x1, y1, v1],
                         ...,
                         [xm, ym, vm]]
                example:
                    profile[2][1]:
                    returns the 3rd point's y position in the local path

                    profile[5]:
                    returns [x5, y5, v5] (6th point in the local path)
        """
        profile = []
        # Compute distance travelled from start speed to desired speed using
        # a constant acceleration.
        if desired_speed < start_speed:
            accel_distance = calc_distance(start_speed, desired_speed, -self._a_max)
        else:
            accel_distance = calc_distance(start_speed, desired_speed, self._a_max)

        # Here we will compute the end of the ramp for our velocity profile.
        # At the end of the ramp, we will maintain our final speed.
        ramp_end_index = 0
        distance = 0.0
        while (ramp_end_index < len(path[0]) - 1) and (distance < accel_distance):
            distance += np.linalg.norm([path[0][ramp_end_index + 1] - path[0][ramp_end_index],
                                        path[1][ramp_end_index + 1] - path[1][ramp_end_index]])
            ramp_end_index += 1

        # Here we will actually compute the velocities along the ramp.
        vi = start_speed
        for i in range(ramp_end_index):
            dist = np.linalg.norm([path[0][i + 1] - path[0][i],
                                   path[1][i + 1] - path[1][i]])
            if desired_speed < start_speed:
                vf = calc_final_speed(vi, -self._a_max, dist)
                # clamp speed to desired speed
                if vf < desired_speed:
                    vf = desired_speed
            else:
                vf = calc_final_speed(vi, self._a_max, dist)
                # clamp speed to desired speed
                if vf > desired_speed:
                    vf = desired_speed

            profile.append([path[0][i], path[1][i], vi])
            vi = vf

        # If the ramp is over, then for the rest of the profile we should
        # track the desired speed.
        for i in range(ramp_end_index + 1, len(path[0])):
            profile.append([path[0][i], path[1][i], desired_speed])

        return profile


######################################################
######################################################
# MODULE 7: COMPUTE TOTAL DISTANCE WITH CONSTANT ACCELERATION
#   Read over the function comments to familiarize yourself with the
#   arguments and necessary variables to return. Then follow the TODOs
#   (top-down) and use the surrounding comments as a guide.
######################################################
######################################################
# Using d = (v_f^2 - v_i^2) / (2 * a), compute the distance
# required for a given acceleration/deceleration.
def calc_distance(v_i, v_f, a):
    """Computes the distance given an initial and final speed, with a constant
    acceleration.

    args:
        v_i: initial speed (m/s)
        v_f: final speed (m/s)
        a: acceleration (m/s^2)
    returns:
        d: the final distance (m)
    """
    pass

    # TODO: INSERT YOUR CODE BETWEEN THE DASHED LINES
    # ------------------------------------------------------------------
    d = (v_f ** 2 - v_i ** 2) / (2 * a)
    return d
    # ------------------------------------------------------------------


######################################################
######################################################
# MODULE 7: COMPUTE FINAL SPEED WITH CONSTANT ACCELERATION
#   Read over the function comments to familiarize yourself with the
#   arguments and necessary variables to return. Then follow the TODOs
#   (top-down) and use the surrounding comments as a guide.
######################################################
######################################################
# Using v_f = sqrt(v_i^2 + 2ad), compute the final speed for a given
# acceleration across a given distance, with initial speed v_i.
# Make sure to check the discriminant of the radical. If it is negative,
# return zero as the final speed.
def calc_final_speed(v_i, a, d):
    """Computes the final speed given an initial speed, distance travelled,
    and a constant acceleration.

    args:
        v_i: initial speed (m/s)
        a: acceleration (m/s^2)
        d: distance to be travelled (m)
    returns:
        v_f: the final speed (m/s)
    """
    pass

    # TODO: INSERT YOUR CODE BETWEEN THE DASHED LINES
    # ------------------------------------------------------------------
    v_f = np.sqrt(v_i ** 2 + 2 * a * d)
    return v_f
    # ------------------------------------------------------------------




# LocalPlanner
class LocalPlanner:
    def __init__(self, num_paths, path_offset, circle_offsets, circle_radii,
                 path_select_weight, time_gap, a_max, slow_speed,
                 stop_line_buffer, prev_best_path):
        self._num_paths = num_paths
        self._path_offset = path_offset
        self._path_optimizer = PathOptimizer()
        self._collision_checker = \
            CollisionChecker(circle_offsets,
                                               circle_radii,
                                               path_select_weight)
        self._velocity_planner = \
            VelocityPlanner(time_gap, a_max, slow_speed,
                                             stop_line_buffer)
        self.prev_best_path = []

    ######################################################
    ######################################################
    # MODULE 7: GOAL STATE COMPUTATION
    #   Read over the function comments to familiarize yourself with the
    #   arguments and necessary variables to return. Then follow the TODOs
    #   (top-down) and use the surrounding comments as a guide.
    ######################################################
    ######################################################
    # Computes the goal state set from a given goal position. This is done by
    # laterally sampling offsets from the goal location along the direction
    # perpendicular to the goal yaw of the ego vehicle.
    def get_goal_state_set(self, goal_index, goal_state, waypoints, ego_state):
        """Gets the goal states given a goal position.

        Gets the goal states given a goal position. The states

        args:
            goal_index: Goal index for the vehicle to reach
                i.e. waypoints[goal_index] gives the goal waypoint
            goal_state: Goal state for the vehicle to reach (global frame)
                format: [x_goal, y_goal, v_goal], in units [m, m, m/s]
            waypoints: current waypoints to track. length and speed in m and m/s.
                (includes speed to track at each x,y location.) (global frame)
                format: [[x0, y0, v0],
                         [x1, y1, v1],
                         ...
                         [xn, yn, vn]]
                example:
                    waypoints[2][1]:
                    returns the 3rd waypoint's y position

                    waypoints[5]:
                    returns [x5, y5, v5] (6th waypoint)
            ego_state: ego state vector for the vehicle, in the global frame.
                format: [ego_x, ego_y, ego_yaw, ego_open_loop_speed]
                    ego_x and ego_y     : position (m)
                    ego_yaw             : top-down orientation [-pi to pi]
                    ego_open_loop_speed : open loop speed (m/s)
        returns:
            goal_state_set: Set of goal states (offsetted laterally from one
                another) to be used by the local planner to plan multiple
                proposal paths. This goal state set is in the vehicle frame.
                format: [[x0, y0, t0, v0],
                         [x1, y1, t1, v1],
                         ...
                         [xm, ym, tm, vm]]
                , where m is the total number of goal states
                  [x, y, t] are the position and yaw values at each goal
                  v is the goal speed at the goal point.
                  all units are in m, m/s and radians
        """
        # Compute the final heading based on the next index.
        # If the goal index is the last in the set of waypoints, use
        # the previous index instead.
        # To do this, compute the delta_x and delta_y values between
        # consecutive waypoints, then use the np.arctan2() function.
        # TODO: INSERT YOUR CODE BETWEEN THE DASHED LINES
        # ------------------------------------------------------------------
        if goal_index != len(waypoints) - 1:
            delta_x = waypoints[goal_index + 1][0] - waypoints[goal_index][0]
            delta_y = waypoints[goal_index + 1][1] - waypoints[goal_index][1]
            heading = np.arctan2(delta_y, delta_x)
        else:
            delta_x = waypoints[goal_index][0] - waypoints[goal_index - 1][0]
            delta_y = waypoints[goal_index][1] - waypoints[goal_index - 1][1]
            heading = np.arctan2(delta_y, delta_x)
        # ------------------------------------------------------------------
        # Compute the center goal state in the local frame using
        # the ego state. The following code will transform the input
        # goal state to the ego vehicle's local frame.
        # The goal state will be of the form (x, y, t, v).
        goal_state_local = copy.copy(goal_state)

        # Translate so the ego state is at the origin in the new frame.
        # This is done by subtracting the ego_state from the goal_state_local.
        # TODO: INSERT YOUR CODE BETWEEN THE DASHED LINES
        # ------------------------------------------------------------------
        goal_state_local[0] -= ego_state[0]
        goal_state_local[1] -= ego_state[1]
        # ------------------------------------------------------------------

        # Rotate such that the ego state has zero heading in the new frame.
        # Recall that the general rotation matrix is [cos(theta) -sin(theta)
        #                                             sin(theta)  cos(theta)]
        # and that we are rotating by -ego_state[2] to ensure the ego vehicle's
        # current yaw corresponds to theta = 0 in the new local frame.
        # TODO: INSERT YOUR CODE BETWEEN THE DASHED LINES
        # ------------------------------------------------------------------
        goal_x = goal_state_local[0] * np.cos(ego_state[2]) + goal_state_local[1] * np.sin(ego_state[2])
        goal_y = goal_state_local[0] * -np.sin(ego_state[2]) + goal_state_local[1] * np.cos(ego_state[2])
        # ------------------------------------------------------------------

        # Compute the goal yaw in the local frame by subtracting off the
        # current ego yaw from the heading variable.
        # TODO: INSERT YOUR CODE BETWEEN THE DASHED LINES
        # ------------------------------------------------------------------
        goal_t = heading - ego_state[2]
        # ------------------------------------------------------------------

        # Velocity is preserved after the transformation.
        goal_v = goal_state[2]

        # Keep the goal heading within [-pi, pi] so the optimizer behaves well.
        if goal_t > pi:
            goal_t -= 2 * pi
        elif goal_t < -pi:
            goal_t += 2 * pi

        # Compute and apply the offset for each path such that
        # all of the paths have the same heading of the goal state,
        # but are laterally offset with respect to the goal heading.
        goal_state_set = []
        for i in range(self._num_paths):
            # Compute offsets that span the number of paths set for the local
            # planner. Each offset goal will be used to generate a potential
            # path to be considered by the local planner.
            offset = (i - self._num_paths // 2) * self._path_offset

            # Compute the projection of the lateral offset along the x
            # and y axis. To do this, multiply the offset by cos(goal_theta + pi/2)
            # and sin(goal_theta + pi/2), respectively.
            # TODO: INSERT YOUR CODE BETWEEN THE DASHED LINES
            # ------------------------------------------------------------------
            x_offset = offset * np.cos(goal_t + pi / 2)
            y_offset = offset * np.sin(goal_t + pi / 2)
            # ------------------------------------------------------------------

            goal_state_set.append([goal_x + x_offset,
                                   goal_y + y_offset,
                                   goal_t,
                                   goal_v])

        return goal_state_set

        # Plans the path set using polynomial spiral optimization to

    # each of the goal states.
    def plan_paths(self, goal_state_set):
        """Plans the path set using the polynomial spiral optimization.

        Plans the path set using polynomial spiral optimization to each of the
        goal states.

        args:
            goal_state_set: Set of goal states (offsetted laterally from one
                another) to be used by the local planner to plan multiple
                proposal paths. These goals are with respect to the vehicle
                frame.
                format: [[x0, y0, t0, v0],
                         [x1, y1, t1, v1],
                         ...
                         [xm, ym, tm, vm]]
                , where m is the total number of goal states
                  [x, y, t] are the position and yaw values at each goal
                  v is the goal speed at the goal point.
                  all units are in m, m/s and radians
        returns:
            paths: A list of optimized spiral paths which satisfies the set of
                goal states. A path is a list of points of the following format:
                    [x_points, y_points, t_points]:
                        x_points: List of x values (m) along the spiral
                        y_points: List of y values (m) along the spiral
                        t_points: List of yaw values (rad) along the spiral
                    Example of accessing the ith path, jth point's t value:
                        paths[i][2][j]
                Note that this path is in the vehicle frame, since the
                optimize_spiral function assumes this to be the case.
            path_validity: List of booleans classifying whether a path is valid
                (true) or not (false) for the local planner to traverse. Each ith
                path_validity corresponds to the ith path in the path list.
        """
        paths = []
        path_validity = []
        for goal_state in goal_state_set:
            path = self._path_optimizer.optimize_spiral(goal_state[0],
                                                        goal_state[1],
                                                        goal_state[2])
            if np.linalg.norm([path[0][-1] - goal_state[0],
                               path[1][-1] - goal_state[1],
                               path[2][-1] - goal_state[2]]) > 0.1:
                path_validity.append(False)
            else:
                paths.append(path)
                path_validity.append(True)
        return paths, path_validity

def transform_paths(paths, ego_state):
    """ Converts the to the global coordinate frame.

    Converts the paths from the local (vehicle) coordinate frame to the
    global coordinate frame.

    args:
        paths: A list of paths in the local (vehicle) frame.
            A path is a list of points of the following format:
                [x_points, y_points, t_points]:
                    , x_points: List of x values (m)
                    , y_points: List of y values (m)
                    , t_points: List of yaw values (rad)
                Example of accessing the ith path, jth point's t value:
                    paths[i][2][j]
        ego_state: ego state vector for the vehicle, in the global frame.
            format: [ego_x, ego_y, ego_yaw, ego_open_loop_speed]
                ego_x and ego_y     : position (m)
                ego_yaw             : top-down orientation [-pi to pi]
                ego_open_loop_speed : open loop speed (m/s)
    returns:
        transformed_paths: A list of transformed paths in the global frame.
            A path is a list of points of the following format:
                [x_points, y_points, t_points]:
                    , x_points: List of x values (m)
                    , y_points: List of y values (m)
                    , t_points: List of yaw values (rad)
                Example of accessing the ith transformed path, jth point's
                y value:
                    paths[i][1][j]
    """
    transformed_paths = []
    for path in paths:
        x_transformed = []
        y_transformed = []
        t_transformed = []

        for i in range(len(path[0])):
            x_transformed.append(ego_state[0] + path[0][i] * cos(ego_state[2]) - \
                                 path[1][i] * sin(ego_state[2]))
            y_transformed.append(ego_state[1] + path[0][i] * sin(ego_state[2]) + \
                                 path[1][i] * cos(ego_state[2]))
            t_transformed.append(path[2][i] + ego_state[2])

        transformed_paths.append([x_transformed, y_transformed, t_transformed])

    return transformed_paths


# State machine states
FOLLOW_LANE = 0
DECELERATE_TO_STOP = 1
STAY_STOPPED = 2
# Stop speed threshold
STOP_THRESHOLD = 0.02
# Number of cycles before moving from stop sign.
STOP_COUNTS = 10



# BehaviouralPlanner
class BehaviouralPlanner:
    def __init__(self, lookahead, stopsign_fences, lead_vehicle_lookahead):
        self._lookahead = lookahead
        self._stopsign_fences = stopsign_fences
        self._follow_lead_vehicle_lookahead = lead_vehicle_lookahead
        self._state = FOLLOW_LANE
        self._follow_lead_vehicle = False
        self._goal_state = [0.0, 0.0, 0.0]
        self._goal_index = 0
        self._stop_count = 0

    def set_lookahead(self, lookahead):
        self._lookahead = lookahead

    ######################################################
    ######################################################
    # MODULE 7: TRANSITION STATE FUNCTION
    #   Read over the function comments to familiarize yourself with the
    #   arguments and necessary internal variables to set. Then follow the TODOs
    #   and use the surrounding comments as a guide.
    ######################################################
    ######################################################
    # Handles state transitions and computes the goal state.
    def transition_state(self, waypoints, ego_state, closed_loop_speed):
        """Handles state transitions and computes the goal state.

        args:
            waypoints: current waypoints to track (global frame).
                length and speed in m and m/s.
                (includes speed to track at each x,y location.)
                format: [[x0, y0, v0],
                         [x1, y1, v1],
                         ...
                         [xn, yn, vn]]
                example:
                    waypoints[2][1]:
                    returns the 3rd waypoint's y position

                    waypoints[5]:
                    returns [x5, y5, v5] (6th waypoint)
            ego_state: ego state vector for the vehicle. (global frame)
                format: [ego_x, ego_y, ego_yaw, ego_open_loop_speed]
                    ego_x and ego_y     : position (m)
                    ego_yaw             : top-down orientation [-pi to pi]
                    ego_open_loop_speed : open loop speed (m/s)
            closed_loop_speed: current (closed-loop) speed for vehicle (m/s)
        variables to set:
            self._goal_index: Goal index for the vehicle to reach
                i.e. waypoints[self._goal_index] gives the goal waypoint
            self._goal_state: Goal state for the vehicle to reach (global frame)
                format: [x_goal, y_goal, v_goal]
            self._state: The current state of the vehicle.
                available states:
                    FOLLOW_LANE         : Follow the global waypoints (lane).
                    DECELERATE_TO_STOP  : Decelerate to stop.
                    STAY_STOPPED        : Stay stopped.
            self._stop_count: Counter used to count the number of cycles which
                the vehicle was in the STAY_STOPPED state so far.
        useful_constants:
            STOP_THRESHOLD  : Stop speed threshold (m). The vehicle should fully
                              stop when its speed falls within this threshold.
            STOP_COUNTS     : Number of cycles (simulation iterations)
                              before moving from stop sign.
        """
        # In this state, continue tracking the lane by finding the
        # goal index in the waypoint list that is within the lookahead
        # distance. Then, check to see if the waypoint path intersects
        # with any stop lines. If it does, then ensure that the goal
        # state enforces the car to be stopped before the stop line.
        # You should use the get_closest_index(), get_goal_index(), and
        # check_for_stop_signs() helper functions.
        # Make sure that get_closest_index() and get_goal_index() functions are
        # complete, and examine the check_for_stop_signs() function to
        # understand it.
        if self._state == FOLLOW_LANE:
            print("State following the lane")
            # First, find the closest index to the ego vehicle.
            # TODO: INSERT YOUR CODE BETWEEN THE DASHED LINES
            # ------------------------------------------------------------------
            closest_len, closest_index = get_closest_index(waypoints, ego_state)
            # ------------------------------------------------------------------
            # Next, find the goal index that lies within the lookahead distance
            # along the waypoints.
            # TODO: INSERT YOUR CODE BETWEEN THE DASHED LINES
            # ------------------------------------------------------------------
            goal_index = self.get_goal_index(waypoints, ego_state, closest_len, closest_index)
            # ------------------------------------------------------------------
            # Finally, check the index set between closest_index and goal_index
            # for stop signs, and compute the goal state accordingly.
            # TODO: INSERT YOUR CODE BETWEEN THE DASHED LINES
            # ------------------------------------------------------------------
            goal_index, stop_sign_found = self.check_for_stop_signs(waypoints, closest_index, goal_index)
            self._goal_index = goal_index
            self._goal_state = waypoints[self._goal_index]
            # ------------------------------------------------------------------

            # If stop sign found, set the goal to zero speed, then transition to
            # the deceleration state.
            # TODO: INSERT YOUR CODE BETWEEN THE DASHED LINES
            # ------------------------------------------------------------------
            if stop_sign_found:
                self._goal_state[2] = 0
                self._state = DECELERATE_TO_STOP
            # ------------------------------------------------------------------
            print("stop_sign_found = %d" % stop_sign_found)
            pass

        # In this state, check if we have reached a complete stop. Use the
        # closed loop speed to do so, to ensure we are actually at a complete
        # stop, and compare to STOP_THRESHOLD.  If so, transition to the next
        # state.
        elif self._state == DECELERATE_TO_STOP:
            print("State Decelerating to stop")
            # TODO: INSERT YOUR CODE BETWEEN THE DASHED LINES
            # ------------------------------------------------------------------
            if closed_loop_speed > STOP_THRESHOLD:
                self._state = DECELERATE_TO_STOP
                print("closed_loop_speed = %f" % closed_loop_speed)
                print("The state of car: ")
                print(ego_state)
            else:
                self._state = STAY_STOPPED
                print(
                    "Begin to stop ======================================================================================")
            # ------------------------------------------------------------------
            pass

        # In this state, check to see if we have stayed stopped for at
        # least STOP_COUNTS number of cycles. If so, we can now leave
        # the stop sign and transition to the next state.
        elif self._state == STAY_STOPPED:
            print("Staying stopped")
            # We have stayed stopped for the required number of cycles.
            # Allow the ego vehicle to leave the stop sign. Once it has
            # passed the stop sign, return to lane following.
            # You should use the get_closest_index(), get_goal_index(), and
            # check_for_stop_signs() helper functions.
            if self._stop_count == STOP_COUNTS:
                print("Try to follow the lane")
                # TODO: INSERT YOUR CODE BETWEEN THE DASHED LINES
                # --------------------------------------------------------------
                closest_len, closest_index = get_closest_index(waypoints, ego_state)
                goal_index = self.get_goal_index(waypoints, ego_state, closest_len, closest_index)
                # --------------------------------------------------------------

                # We've stopped for the required amount of time, so the new goal
                # index for the stop line is not relevant. Use the goal index
                # that is the lookahead distance away.
                # TODO: INSERT YOUR CODE BETWEEN THE DASHED LINES
                # --------------------------------------------------------------
                stop_sign_found = self.check_for_stop_signs(waypoints, closest_index, goal_index)[1]
                self._goal_index = goal_index
                self._goal_state = waypoints[self._goal_index]
                # --------------------------------------------------------------

                # If the stop sign is no longer along our path, we can now
                # transition back to our lane following state.
                # TODO: INSERT YOUR CODE BETWEEN THE DASHED LINES
                # --------------------------------------------------------------
                if not stop_sign_found:
                    self._state = FOLLOW_LANE
                # --------------------------------------------------------------

                pass

            # Otherwise, continue counting.
            else:
                # TODO: INSERT YOUR CODE BETWEEN THE DASHED LINES
                # --------------------------------------------------------------
                self._stop_count += 1
                # --------------------------------------------------------------

                pass
        else:
            raise ValueError('Invalid state value.')

    ######################################################
    ######################################################
    # MODULE 7: GET GOAL INDEX FOR VEHICLE
    #   Read over the function comments to familiarize yourself with the
    #   arguments and necessary variables to return. Then follow the TODOs
    #   and use the surrounding comments as a guide.
    ######################################################
    ######################################################
    # Gets the goal index in the list of waypoints, based on the lookahead and
    # the current ego state. In particular, find the earliest waypoint that has accumulated
    # arc length (including closest_len) that is greater than or equal to self._lookahead.
    def get_goal_index(self, waypoints, ego_state, closest_len, closest_index):
        """Gets the goal index for the vehicle.

        Set to be the earliest waypoint that has accumulated arc length
        accumulated arc length (including closest_len) that is greater than or
        equal to self._lookahead.

        args:
            waypoints: current waypoints to track. (global frame)
                length and speed in m and m/s.
                (includes speed to track at each x,y location.)
                format: [[x0, y0, v0],
                         [x1, y1, v1],
                         ...
                         [xn, yn, vn]]
                example:
                    waypoints[2][1]:
                    returns the 3rd waypoint's y position

                    waypoints[5]:
                    returns [x5, y5, v5] (6th waypoint)
            ego_state: ego state vector for the vehicle. (global frame)
                format: [ego_x, ego_y, ego_yaw, ego_open_loop_speed]
                    ego_x and ego_y     : position (m)
                    ego_yaw             : top-down orientation [-pi to pi]
                    ego_open_loop_speed : open loop speed (m/s)
            closest_len: length (m) to the closest waypoint from the vehicle.
            closest_index: index of the waypoint which is closest to the vehicle.
                i.e. waypoints[closest_index] gives the waypoint closest to the vehicle.
        returns:
            wp_index: Goal index for the vehicle to reach
                i.e. waypoints[wp_index] gives the goal waypoint
        """
        # Find the farthest point along the path that is within the
        # lookahead distance of the ego vehicle.
        # Take the distance from the ego vehicle to the closest waypoint into
        # consideration.
        arc_length = closest_len
        wp_index = closest_index

        # In this case, reaching the closest waypoint is already far enough for
        # the planner.  No need to check additional waypoints.
        if arc_length > self._lookahead:
            return wp_index

        # We are already at the end of the path.
        if wp_index == len(waypoints) - 1:
            return wp_index

        # Otherwise, find our next waypoint.
        # TODO: INSERT YOUR CODE BETWEEN THE DASHED LINES
        # ------------------------------------------------------------------
        while wp_index < len(waypoints) - 1:
            # arc_length += math.sqrt(1 + ((waypoints[wp_index+1][1] - waypoints[wp_index][1])
            #                     / (waypoints[wp_index+1][0] - waypoints[wp_index][0])) ** 2)
            arc_length += math.sqrt((waypoints[wp_index + 1][0] - waypoints[wp_index][0]) ** 2 + (
                        waypoints[wp_index + 1][1] - waypoints[wp_index][1]) ** 2)
            if arc_length >= self._lookahead:
                wp_index += 1
                break
            else:
                wp_index += 1
        # ------------------------------------------------------------------
        return wp_index

    # Checks the given segment of the waypoint list to see if it
    # intersects with a stop line. If any index does, return the
    # new goal state accordingly.
    def check_for_stop_signs(self, waypoints, closest_index, goal_index):
        """Checks for a stop sign that is intervening the goal path.

        Checks for a stop sign that is intervening the goal path. Returns a new
        goal index (the current goal index is obstructed by a stop line), and a
        boolean flag indicating if a stop sign obstruction was found.

        args:
            waypoints: current waypoints to track. (global frame)
                length and speed in m and m/s.
                (includes speed to track at each x,y location.)
                format: [[x0, y0, v0],
                         [x1, y1, v1],
                         ...
                         [xn, yn, vn]]
                example:
                    waypoints[2][1]:
                    returns the 3rd waypoint's y position

                    waypoints[5]:
                    returns [x5, y5, v5] (6th waypoint)
                closest_index: index of the waypoint which is closest to the vehicle.
                    i.e. waypoints[closest_index] gives the waypoint closest to the vehicle.
                goal_index (current): Current goal index for the vehicle to reach
                    i.e. waypoints[goal_index] gives the goal waypoint
        variables to set:
            [goal_index (updated), stop_sign_found]:
                goal_index (updated): Updated goal index for the vehicle to reach
                    i.e. waypoints[goal_index] gives the goal waypoint
                stop_sign_found: Boolean flag for whether a stop sign was found or not
        """
        for i in range(closest_index, goal_index):
            # Check to see if path segment crosses any of the stop lines.
            intersect_flag = False
            for stopsign_fence in self._stopsign_fences:
                wp_1 = np.array(waypoints[i][0:2])
                wp_2 = np.array(waypoints[i + 1][0:2])
                s_1 = np.array(stopsign_fence[0:2])
                s_2 = np.array(stopsign_fence[2:4])

                v1 = np.subtract(wp_2, wp_1)
                v2 = np.subtract(s_1, wp_2)
                sign_1 = np.sign(np.cross(v1, v2))
                v2 = np.subtract(s_2, wp_2)
                sign_2 = np.sign(np.cross(v1, v2))

                v1 = np.subtract(s_2, s_1)
                v2 = np.subtract(wp_1, s_2)
                sign_3 = np.sign(np.cross(v1, v2))
                v2 = np.subtract(wp_2, s_2)
                sign_4 = np.sign(np.cross(v1, v2))

                # Check if the line segments intersect.
                if (sign_1 != sign_2) and (sign_3 != sign_4):
                    intersect_flag = True

                # Check if the collinearity cases hold.
                if (sign_1 == 0) and pointOnSegment(wp_1, s_1, wp_2):
                    intersect_flag = True
                if (sign_2 == 0) and pointOnSegment(wp_1, s_2, wp_2):
                    intersect_flag = True
                if (sign_3 == 0) and pointOnSegment(s_1, wp_1, s_2):
                    intersect_flag = True
                if (sign_3 == 0) and pointOnSegment(s_1, wp_2, s_2):
                    intersect_flag = True

                # If there is an intersection with a stop line, update
                # the goal state to stop before the goal line.
                if intersect_flag:
                    goal_index = i
                    return goal_index, True

        return goal_index, False

    # Checks to see if we need to modify our velocity profile to accomodate the
    # lead vehicle.
    def check_for_lead_vehicle(self, ego_state, lead_car_position):
        """Checks for lead vehicle within the proximity of the ego car, such
        that the ego car should begin to follow the lead vehicle.

        args:
            ego_state: ego state vector for the vehicle. (global frame)
                format: [ego_x, ego_y, ego_yaw, ego_open_loop_speed]
                    ego_x and ego_y     : position (m)
                    ego_yaw             : top-down orientation [-pi to pi]
                    ego_open_loop_speed : open loop speed (m/s)
            lead_car_position: The [x, y] position of the lead vehicle.
                Lengths are in meters, and it is in the global frame.
        sets:
            self._follow_lead_vehicle: Boolean flag on whether the ego vehicle
                should follow (true) the lead car or not (false).
        """
        # Check lead car position delta vector relative to heading, as well as
        # distance, to determine if car should be followed.
        # Check to see if lead vehicle is within range, and is ahead of us.
        if not self._follow_lead_vehicle:
            # Compute the angle between the normalized vector between the lead vehicle
            # and ego vehicle position with the ego vehicle's heading vector.
            lead_car_delta_vector = [lead_car_position[0] - ego_state[0],
                                     lead_car_position[1] - ego_state[1]]
            lead_car_distance = np.linalg.norm(lead_car_delta_vector)
            # In this case, the car is too far away.
            if lead_car_distance > self._follow_lead_vehicle_lookahead:
                return

            lead_car_delta_vector = np.divide(lead_car_delta_vector,
                                              lead_car_distance)
            ego_heading_vector = [math.cos(ego_state[2]),
                                  math.sin(ego_state[2])]
            # Check to see if the relative angle between the lead vehicle and the ego
            # vehicle lies within +/- 45 degrees of the ego vehicle's heading.
            if np.dot(lead_car_delta_vector,
                      ego_heading_vector) < (1 / math.sqrt(2)):
                return

            self._follow_lead_vehicle = True

        else:
            lead_car_delta_vector = [lead_car_position[0] - ego_state[0],
                                     lead_car_position[1] - ego_state[1]]
            lead_car_distance = np.linalg.norm(lead_car_delta_vector)

            # Add a 15m buffer to prevent oscillations for the distance check.
            if lead_car_distance < self._follow_lead_vehicle_lookahead + 15:
                return
            # Check to see if the lead vehicle is still within the ego vehicle's
            # frame of view.
            lead_car_delta_vector = np.divide(lead_car_delta_vector, lead_car_distance)
            ego_heading_vector = [math.cos(ego_state[2]), math.sin(ego_state[2])]
            if np.dot(lead_car_delta_vector, ego_heading_vector) > (1 / math.sqrt(2)):
                return

            self._follow_lead_vehicle = False


######################################################
######################################################
# MODULE 7: CLOSEST WAYPOINT INDEX TO VEHICLE
#   Read over the function comments to familiarize yourself with the
#   arguments and necessary variables to return. Then follow the TODOs
#   and use the surrounding comments as a guide.
######################################################
######################################################
# Compute the waypoint index that is closest to the ego vehicle, and return
# it as well as the distance from the ego vehicle to that waypoint.
def get_closest_index(waypoints, ego_state):
    """Gets closest index a given list of waypoints to the vehicle position.

    args:
        waypoints: current waypoints to track. (global frame)
            length and speed in m and m/s.
            (includes speed to track at each x,y location.)
            format: [[x0, y0, v0],
                     [x1, y1, v1],
                     ...
                     [xn, yn, vn]]
            example:
                waypoints[2][1]:
                returns the 3rd waypoint's y position

                waypoints[5]:
                returns [x5, y5, v5] (6th waypoint)
        ego_state: ego state vector for the vehicle. (global frame)
            format: [ego_x, ego_y, ego_yaw, ego_open_loop_speed]
                ego_x and ego_y     : position (m)
                ego_yaw             : top-down orientation [-pi to pi]
                ego_open_loop_speed : open loop speed (m/s)

    returns:
        [closest_len, closest_index]:
            closest_len: length (m) to the closest waypoint from the vehicle.
            closest_index: index of the waypoint which is closest to the vehicle.
                i.e. waypoints[closest_index] gives the waypoint closest to the vehicle.
    """
    closest_len = float('Inf')
    closest_index = 0
    # TODO: INSERT YOUR CODE BETWEEN THE DASHED LINES
    # ------------------------------------------------------------------
    for i in range(len(waypoints)):
        distence = math.sqrt((ego_state[0] - waypoints[i][0]) ** 2 + (ego_state[1] - waypoints[i][1]) ** 2)
        if distence <= closest_len:
            closest_len = distence
            closest_index = i
    # ------------------------------------------------------------------

    return closest_len, closest_index


# Checks if p2 lies on segment p1-p3, if p1, p2, p3 are collinear.
def pointOnSegment(p1, p2, p3):
    if (p2[0] <= max(p1[0], p3[0]) and (p2[0] >= min(p1[0], p3[0])) and \
            (p2[1] <= max(p1[1], p3[1])) and (p2[1] >= min(p1[1], p3[1]))):
        return True
    else:
        return False


class BasicPlanner(object):
    def __init__(self,perception):
        self.lp = LocalPlanner(NUM_PATHS,
                          PATH_OFFSET,
                          CIRCLE_OFFSETS,
                          CIRCLE_RADII,
                          PATH_SELECT_WEIGHT,
                          TIME_GAP,
                          A_MAX,
                          SLOW_SPEED,
                          STOP_LINE_BUFFER,
                          PREV_BEST_PATH)
        self.bp=BehaviouralPlanner(BP_LOOKAHEAD_BASE,
                                perception.stopsign_fences,
                                LEAD_VEHICLE_LOOKAHEAD)

        self.local_waypoints=None
        self.path_validity = np.zeros((NUM_PATHS, 1), dtype=bool)


        self.wp_interp =None
        self.best_index =None
        self.paths =None
        self.ego_state =None
        self.path_validity =None
        self.collision_check_array =None

        self.print_verbose=False

    def run_step(self,world_representation, prev_timestamp):
        lp=self.lp
        bp=self.bp
        current_speed = world_representation.ego_vehicle.forward_speed
        current_timestamp = world_representation.current_timestamp
        current_x, current_y, current_yaw = world_representation.ego_vehicle.localization
        lead_car_pos = world_representation.lead_car_pos
        lead_car_speed = world_representation.lead_car_speed
        waypoints= world_representation.waypoints
        parkedcar_box_pts=world_representation.parkedcar_box_pts
        # TODO: Uncomment each code block between the dashed lines to run the planner.
        # --------------------------------------------------------------
        # Compute open loop speed estimate.
        print("================= start ======================")
        open_loop_speed = lp._velocity_planner.get_open_loop_speed(current_timestamp - prev_timestamp)

        # Calculate the goal state set in the local frame for the local planner.
        # Current speed should be open loop for the velocity profile generation.
        ego_state = [current_x, current_y, current_yaw, open_loop_speed]

        # Set lookahead based on current speed.
        bp.set_lookahead(BP_LOOKAHEAD_BASE + BP_LOOKAHEAD_TIME * open_loop_speed)

        # Perform a state transition in the behavioural planner.
        bp.transition_state(waypoints, ego_state, current_speed)
        print("The current speed = %f" % current_speed)
        # Check to see if we need to follow the lead vehicle.
        bp.check_for_lead_vehicle(ego_state, lead_car_pos[1])

        # Compute the goal state set from the behavioural planner's computed goal state.
        goal_state_set = lp.get_goal_state_set(bp._goal_index, bp._goal_state, waypoints, ego_state)

        # Calculate planned paths in the local frame.
        paths, path_validity = lp.plan_paths(goal_state_set)

        # Transform those paths back to the global frame.
        paths = transform_paths(paths, ego_state)

        # Perform collision checking.
        collision_check_array = lp._collision_checker.collision_check(paths, [parkedcar_box_pts])

        # Compute the best local path.
        best_index = lp._collision_checker.select_best_path_index(paths, collision_check_array, bp._goal_state)

        if self.print_verbose:
            print("The best_index = %d" % best_index)

        # If no path was feasible, continue to follow the previous best path.
        if best_index == None:
            best_path = lp.prev_best_path
        else:
            best_path = paths[best_index]
            lp.prev_best_path = best_path

        # Compute the velocity profile for the path, and compute the waypoints.
        # Use the lead vehicle to inform the velocity profile's dynamic obstacle handling.
        # In this scenario, the only dynamic obstacle is the lead vehicle at index 1.
        desired_speed = bp._goal_state[2]
        print("The desired_speed = %f" % desired_speed)
        lead_car_state = [lead_car_pos[1][0], lead_car_pos[1][1], lead_car_speed[1]]
        decelerate_to_stop = bp._state == DECELERATE_TO_STOP
        local_waypoints = lp._velocity_planner.compute_velocity_profile(best_path, desired_speed, ego_state,
                                                                        current_speed, decelerate_to_stop,
                                                                        lead_car_state, bp._follow_lead_vehicle)
        print("================= end ======================")
        # --------------------------------------------------------------

        if local_waypoints != None:
            # Update the controller waypoint path with the best local path.
            # This controller is similar to that developed in Course 1 of this
            # specialization.  Linear interpolation computation on the waypoints
            # is also used to ensure a fine resolution between points.
            wp_distance = []  # distance array
            local_waypoints_np = np.array(local_waypoints)
            for i in range(1, local_waypoints_np.shape[0]):
                wp_distance.append(
                    np.sqrt((local_waypoints_np[i, 0] - local_waypoints_np[i - 1, 0]) ** 2 +
                            (local_waypoints_np[i, 1] - local_waypoints_np[i - 1, 1]) ** 2))
            wp_distance.append(0)  # last distance is 0 because it is the distance
            # from the last waypoint to the last waypoint

            # Linearly interpolate between waypoints and store in a list
            wp_interp = []  # interpolated values
            # (rows = waypoints, columns = [x, y, v])
            for i in range(local_waypoints_np.shape[0] - 1):
                # Add original waypoint to interpolated waypoints list (and append
                # it to the hash table)
                wp_interp.append(list(local_waypoints_np[i]))

                # Interpolate to the next waypoint. First compute the number of
                # points to interpolate based on the desired resolution and
                # incrementally add interpolated points until the next waypoint
                # is about to be reached.
                num_pts_to_interp = int(np.floor(wp_distance[i] / \
                                                 float(INTERP_DISTANCE_RES)) - 1)
                wp_vector = local_waypoints_np[i + 1] - local_waypoints_np[i]
                wp_uvector = wp_vector / np.linalg.norm(wp_vector[0:2])

                for j in range(num_pts_to_interp):
                    next_wp_vector = INTERP_DISTANCE_RES * float(j + 1) * wp_uvector
                    wp_interp.append(list(local_waypoints_np[i] + next_wp_vector))
            # add last waypoint at the end
            wp_interp.append(list(local_waypoints_np[-1]))


            pass

        self.local_waypoints=local_waypoints
        self.wp_interp=wp_interp
        self.best_index=best_index
        self.paths=paths
        self.ego_state=ego_state
        self.path_validity=path_validity
        self.collision_check_array=collision_check_array

        return local_waypoints, wp_interp
def run_plannerv2(perception,prev_timestamp):
    #############################################
    # Local Planner Variables
    #############################################
    local_waypoints = None
    path_validity = np.zeros((NUM_PATHS, 1), dtype=bool)

    lp = LocalPlanner(NUM_PATHS, PATH_OFFSET, CIRCLE_OFFSETS, CIRCLE_RADII, PATH_SELECT_WEIGHT, TIME_GAP,
                                    A_MAX, SLOW_SPEED, STOP_LINE_BUFFER, PREV_BEST_PATH)
    bp = BehaviouralPlanner(BP_LOOKAHEAD_BASE, perception.stopsign_fences, LEAD_VEHICLE_LOOKAHEAD)


    # Initialize the current timestamp.
    current_timestamp = perception.current_timestamp
    current_speed = perception.ego_vehicle.forward_speed
    current_x, current_y, current_yaw = perception.ego_vehicle.localization
    waypoints=perception.waypoints
    lead_car_pos=perception.lead_car_pos
    parkedcar_box_pts=perception.parkedcar_box_pts
    lead_car_speed=perception.lead_car_speed

    # Initialize collision history
    # prev_collision_vehicles = 0
    # prev_collision_pedestrians = 0
    # prev_collision_other = 0

    perception.prev_collision_vehicles=0
    perception.prev_collision_pedestrians=0
    perception.prev_collision_other=0
    # TODO: Uncomment each code block between the dashed lines to run the planner.
    # --------------------------------------------------------------
    # Compute open loop speed estimate.
    print("================= start ======================")
    open_loop_speed = lp._velocity_planner.get_open_loop_speed(current_timestamp - prev_timestamp)

    # Calculate the goal state set in the local frame for the local planner.
    # Current speed should be open loop for the velocity profile generation.
    ego_state = [current_x, current_y, current_yaw, open_loop_speed]

    # Set lookahead based on current speed.
    bp.set_lookahead(BP_LOOKAHEAD_BASE + BP_LOOKAHEAD_TIME * open_loop_speed)

    # Perform a state transition in the behavioural planner.
    bp.transition_state(waypoints, ego_state, current_speed)
    print("The current speed = %f" % current_speed)
    # Check to see if we need to follow the lead vehicle.
    bp.check_for_lead_vehicle(ego_state, lead_car_pos[1])

    # Compute the goal state set from the behavioural planner's computed goal state.
    goal_state_set = lp.get_goal_state_set(bp._goal_index, bp._goal_state, waypoints, ego_state)

    # Calculate planned paths in the local frame.
    paths, path_validity = lp.plan_paths(goal_state_set)

    # Transform those paths back to the global frame.
    paths = transform_paths(paths, ego_state)

    # Perform collision checking.
    collision_check_array = lp._collision_checker.collision_check(paths, [parkedcar_box_pts])

    # Compute the best local path.
    best_index = lp._collision_checker.select_best_path_index(paths, collision_check_array, bp._goal_state)

    print("The best_index = %d" % best_index)

    # If no path was feasible, continue to follow the previous best path.
    if best_index == None:
        best_path = lp.prev_best_path
    else:
        best_path = paths[best_index]
        lp.prev_best_path = best_path

    # Compute the velocity profile for the path, and compute the waypoints.
    # Use the lead vehicle to inform the velocity profile's dynamic obstacle handling.
    # In this scenario, the only dynamic obstacle is the lead vehicle at index 1.
    desired_speed = bp._goal_state[2]
    print("The desired_speed = %f" % desired_speed)
    lead_car_state = [lead_car_pos[1][0], lead_car_pos[1][1], lead_car_speed[1]]
    decelerate_to_stop = bp._state == DECELERATE_TO_STOP
    local_waypoints = lp._velocity_planner.compute_velocity_profile(best_path, desired_speed, ego_state, current_speed,
                                                                    decelerate_to_stop, lead_car_state,
                                                                    bp._follow_lead_vehicle)
    print("================= end ======================")
    # --------------------------------------------------------------
    wp_interp = []  # interpolated values
    if local_waypoints != None:
        # Update the controller waypoint path with the best local path.
        # This controller is similar to that developed in Course 1 of this
        # specialization.  Linear interpolation computation on the waypoints
        # is also used to ensure a fine resolution between points.
        wp_distance = []  # distance array
        local_waypoints_np = np.array(local_waypoints)
        for i in range(1, local_waypoints_np.shape[0]):
            wp_distance.append(
                np.sqrt((local_waypoints_np[i, 0] - local_waypoints_np[i - 1, 0]) ** 2 +
                        (local_waypoints_np[i, 1] - local_waypoints_np[i - 1, 1]) ** 2))
        wp_distance.append(0)  # last distance is 0 because it is the distance
        # from the last waypoint to the last waypoint

        # Linearly interpolate between waypoints and store in a list

        # (rows = waypoints, columns = [x, y, v])
        for i in range(local_waypoints_np.shape[0] - 1):
            # Add original waypoint to interpolated waypoints list (and append
            # it to the hash table)
            wp_interp.append(list(local_waypoints_np[i]))

            # Interpolate to the next waypoint. First compute the number of
            # points to interpolate based on the desired resolution and
            # incrementally add interpolated points until the next waypoint
            # is about to be reached.
            num_pts_to_interp = int(np.floor(wp_distance[i] / \
                                             float(INTERP_DISTANCE_RES)) - 1)
            wp_vector = local_waypoints_np[i + 1] - local_waypoints_np[i]
            wp_uvector = wp_vector / np.linalg.norm(wp_vector[0:2])

            for j in range(num_pts_to_interp):
                next_wp_vector = INTERP_DISTANCE_RES * float(j + 1) * wp_uvector
                wp_interp.append(list(local_waypoints_np[i] + next_wp_vector))
        # add last waypoint at the end
        wp_interp.append(list(local_waypoints_np[-1]))

        # Update the other controller values and controls
        return local_waypoints,wp_interp,best_index,paths,ego_state,path_validity,collision_check_array

        pass

    return local_waypoints,wp_interp,best_index,paths,ego_state,path_validity,collision_check_array
def run_planner(perception,controller,prev_timestamp,lp,bp):

    current_speed = perception.ego_vehicle.forward_speed
    current_timestamp = perception.current_timestamp
    current_x, current_y, current_yaw = perception.ego_vehicle.localization
    lead_car_pos = perception.lead_car_pos
    lead_car_speed = perception.lead_car_speed




    # TODO: Uncomment each code block between the dashed lines to run the planner.
    # --------------------------------------------------------------
    # Compute open loop speed estimate.
    print("================= start ======================")
    open_loop_speed = lp._velocity_planner.get_open_loop_speed(current_timestamp - prev_timestamp)

    # Calculate the goal state set in the local frame for the local planner.
    # Current speed should be open loop for the velocity profile generation.
    ego_state = [current_x, current_y, current_yaw, open_loop_speed]

    # Set lookahead based on current speed.
    bp.set_lookahead(BP_LOOKAHEAD_BASE + BP_LOOKAHEAD_TIME * open_loop_speed)

    # Perform a state transition in the behavioural planner.
    bp.transition_state(perception.waypoints, ego_state, current_speed)
    print("The current speed = %f" % current_speed)
    # Check to see if we need to follow the lead vehicle.
    bp.check_for_lead_vehicle(ego_state, lead_car_pos[1])

    # Compute the goal state set from the behavioural planner's computed goal state.
    goal_state_set = lp.get_goal_state_set(bp._goal_index, bp._goal_state, perception.waypoints, ego_state)

    # Calculate planned paths in the local frame.
    paths, path_validity = lp.plan_paths(goal_state_set)

    # Transform those paths back to the global frame.
    paths = transform_paths(paths, ego_state)

    # Perform collision checking.
    collision_check_array = lp._collision_checker.collision_check(paths, [perception.parkedcar_box_pts])

    # Compute the best local path.
    best_index = lp._collision_checker.select_best_path_index(paths, collision_check_array, bp._goal_state)

    print("The best_index = %d" % best_index)

    # If no path was feasible, continue to follow the previous best path.
    if best_index == None:
        best_path = lp.prev_best_path
    else:
        best_path = paths[best_index]
        lp.prev_best_path = best_path

    # Compute the velocity profile for the path, and compute the waypoints.
    # Use the lead vehicle to inform the velocity profile's dynamic obstacle handling.
    # In this scenario, the only dynamic obstacle is the lead vehicle at index 1.
    desired_speed = bp._goal_state[2]
    print("The desired_speed = %f" % desired_speed)
    lead_car_state = [lead_car_pos[1][0], lead_car_pos[1][1], lead_car_speed[1]]
    decelerate_to_stop = bp._state == DECELERATE_TO_STOP
    local_waypoints = lp._velocity_planner.compute_velocity_profile(best_path, desired_speed, ego_state,
                                                                    current_speed, decelerate_to_stop,
                                                                    lead_car_state, bp._follow_lead_vehicle)
    print("================= end ======================")
    # --------------------------------------------------------------

    if local_waypoints != None:
        # Update the controller waypoint path with the best local path.
        # This controller is similar to that developed in Course 1 of this
        # specialization.  Linear interpolation computation on the waypoints
        # is also used to ensure a fine resolution between points.
        wp_distance = []  # distance array
        local_waypoints_np = np.array(local_waypoints)
        for i in range(1, local_waypoints_np.shape[0]):
            wp_distance.append(
                np.sqrt((local_waypoints_np[i, 0] - local_waypoints_np[i - 1, 0]) ** 2 +
                        (local_waypoints_np[i, 1] - local_waypoints_np[i - 1, 1]) ** 2))
        wp_distance.append(0)  # last distance is 0 because it is the distance
        # from the last waypoint to the last waypoint

        # Linearly interpolate between waypoints and store in a list
        wp_interp = []  # interpolated values
        # (rows = waypoints, columns = [x, y, v])
        for i in range(local_waypoints_np.shape[0] - 1):
            # Add original waypoint to interpolated waypoints list (and append
            # it to the hash table)
            wp_interp.append(list(local_waypoints_np[i]))

            # Interpolate to the next waypoint. First compute the number of
            # points to interpolate based on the desired resolution and
            # incrementally add interpolated points until the next waypoint
            # is about to be reached.
            num_pts_to_interp = int(np.floor(wp_distance[i] / \
                                             float(INTERP_DISTANCE_RES)) - 1)
            wp_vector = local_waypoints_np[i + 1] - local_waypoints_np[i]
            wp_uvector = wp_vector / np.linalg.norm(wp_vector[0:2])

            for j in range(num_pts_to_interp):
                next_wp_vector = INTERP_DISTANCE_RES * float(j + 1) * wp_uvector
                wp_interp.append(list(local_waypoints_np[i] + next_wp_vector))
        # add last waypoint at the end
        wp_interp.append(list(local_waypoints_np[-1]))

        # Update the other controller values and controls
        controller.update_waypoints(wp_interp)
        pass

    return local_waypoints,wp_interp,best_index,paths,ego_state,path_validity,collision_check_array
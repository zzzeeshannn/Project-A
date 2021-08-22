# Base code for Randomized Rapidly-Exploring Random Trees (RRT) inspired from AtsushiSakai (@Atsushi_twi)

import math
import random
from copy import deepcopy

import matplotlib.pyplot as plt
import numpy as np

from scipy.linalg import expm

# Flag for animation on or off
show_animation = True

# Define a basic zonotope here
# Some of the functions are added directly to the RRT class for simplicity
class Zonotope:
    def __init__(self, box, a_mat=None):

        self.box = np.array(box, dtype=float)
        self.a_mat = a_mat if a_mat is not None else np.identity(self.box.shape[0])

        self.dims = self.a_mat.shape[0]
        self.gens = self.a_mat.shape[1]

        self.b_vec = np.zeros((self.dims, 1))

    def max(self, direction):
        direction = self.a_mat.transpose().dot(direction)

        box = self.box
        rv = []

        for dim, (lb, ub) in enumerate(box):
            if direction[dim] > 0:
                rv.append([ub])
            else:
                rv.append([lb])

        pt = np.array(rv)

        return self.a_mat.dot(pt) + self.b_vec


class RRT:
    # Main class for RRT
    class Node:
        # Defines a single node in the tree
        def __init__(self, x, y):
            self.x = x
            self.y = y
            self.path_x = []
            self.path_y = []
            self.parent = None

    def __init__(self, start, goal, obstacle_list, rand_area, expand_dis=3.0, path_resolution=0.5, goal_sample_rate=5, max_iter=500):
        """
        Setting Parameter

        start:Start Position [x,y]
        goal:Goal Position [x,y]
        obstacleList:obstacle Positions [[x,y,size],...]
        randArea:Random Sampling Area [min,max]

        """
        # Defines the starting point
        self.start = self.Node(start[0], start[1])
        # Defines the end point
        self.end = self.Node(goal[0], goal[1])
        # Minimum and Maximum random sampling area
        self.min_rand = rand_area[0]
        self.max_rand = rand_area[1]

        self.expand_dis = expand_dis
        self.path_resolution = path_resolution
        # Used in random sampling, decides how frequently the end point is sampled to check
        self.goal_sample_rate = goal_sample_rate
        # Maximum number of iteration before terminating
        self.max_iter = max_iter
        # List of obstacles
        self.obstacle_list = obstacle_list
        # List of nodes in the tree
        self.node_list = []

    def verts(self, zono, xdim=0, ydim=1):
        # Used to get the vertices in 2D
        verts = []

        for angle in np.linspace(0, 2 * math.pi, 32):
            direction = np.zeros((zono.dims,))
            direction[xdim] = math.cos(angle)
            direction[ydim] = math.sin(angle)

            pt = zono.max(direction)
            xy_pt = (pt[xdim][0], pt[ydim][0])

            if verts and np.allclose(xy_pt, verts[-1]):
                continue
            verts.append(xy_pt)

        return verts

    def planning(self, animation=True):
        # Initialize variable requirements here
        center_list = []
        area_list = []
        time_step = math.pi/8

        # Initialize the node list with the start point
        self.node_list = [self.start]

        # Define the initial set of states here for the dynamic obstacles
        # First Dynamic Obstacle States
        first_init_box = [[1.0, 2.0], [5.0, 6.0]]
        # Second Dynamic Obstacle States
        second_init_box = [[15.0, 16.0], [15.0, 16.0]]

        # Define the dynamics matrix for dynamic obstacle 1
        first_dynamics_mat = np.array([[0.0, 1.0], [-1.0, 0.0]], dtype = float)
        # Define the dynamics matrix for dynamic obstacle 2
        second_dynamics_mat = np.array([[0.0], [1.0]], dtype=float)

        # Initialize the zonotopes and define the solution matrices (X = e^(A*t) * x)
        first_init_zono = Zonotope(first_init_box)
        second_init_zono = Zonotope(second_init_box)
        sol_mat = expm(first_dynamics_mat * time_step)

        # Define the parameters here
        time_step = math.pi / 8

        # Get the center and area of the 2D projection of the zonotopes
        # This is used later in the case where we want to avoid our path crossing with the particular obstacle
        z = deepcopy(first_init_zono)

        z2 = deepcopy(second_init_zono)
        z_verts = self.verts(second_init_zono)
        z_xs = [v[0] for v in z_verts]
        z_ys = [v[1] for v in z_verts]
        min_x = np.min(z_xs)
        min_y = np.min(z_ys)
        max_x = np.max(z_xs)
        max_y = np.max(z_ys)
        z_center = ((min_x + max_x) / 2, (min_y + max_y) / 2)
        z_area = (max_x - min_x) * (max_y - min_y)

        center_list.append(z_center)
        area_list.append(z_area)

        # Initialize process
        for i in range(self.max_iter):
            # Sample a random node
            # Possible to sample more points using RK4 method?
            rnd_node = self.get_random_node()
            # Get the index of the node in the existing tree that is closest to this node
            nearest_ind = self.get_nearest_node_index(self.node_list, rnd_node)
            # Get the node corresponding to this index
            nearest_node = self.node_list[nearest_ind]

            # Define this randomly sampled point as the new node of the tree
            # Create a path from the closest node to the new node
            new_node = self.steer(nearest_node, rnd_node, self.expand_dis)

            # Propagate the dynamic obstacles here
            if (i % 4 == 0):
                z.a_mat = sol_mat.dot(z.a_mat)
                z.b_vec = sol_mat.dot(z.b_vec)
                z_verts = self.verts(z)
                z_xs = [v[0] for v in z_verts]
                z_ys = [v[1] for v in z_verts]
                min_x = np.min(z_xs)
                min_y = np.min(z_ys)
                max_x = np.max(z_xs)
                max_y = np.max(z_ys)
                z_center = ((min_x + max_x) / 2, (min_y + max_y) / 2)
                z_area = (max_x - min_x) * (max_y - min_y)

                temp_zono = deepcopy(z)
                temp_zono.a_mat = sol_mat.dot(z.a_mat)
                temp_zono.b_vec = sol_mat.dot(z.b_vec)
                z_verts2 = self.verts(temp_zono)
                z_xs2 = [v[0] for v in z_verts2]
                z_ys2 = [v[1] for v in z_verts2]
                min_x = np.min(z_xs2)
                min_y = np.min(z_ys2)
                max_x = np.max(z_xs2)
                max_y = np.max(z_ys2)
                z_center2 = ((min_x + max_x) / 2, (min_y + max_y) / 2)
                z_area2 = (max_x - min_x) * (max_y - min_y)

                if (i < 40):
                    # Second Dynamic Model
                    z2.b_vec -= np.array([[0.0], [1.0]])
                    z_verts3 = self.verts(z2)
                    z_xs3 = [vt[0] for vt in z_verts3]
                    z_ys3 = [vt[1] for vt in z_verts3]

                    min_x3 = np.min(z_xs3)
                    min_y3 = np.min(z_ys3)
                    max_x3 = np.max(z_xs3)
                    max_y3 = np.max(z_ys3)

                    z_center3 = ((min_x3 + max_x3) / 2, (min_y3 + max_y3) / 2)
                    z_area3 = (max_x3 - min_x3) * (max_y3 - min_y3)

                    center_list.append(z_center3)
                    area_list.append(z_area3)

            # Collision Check
            if self.check_collision(new_node, self.obstacle_list, z_center, z_area, z_center2, z_area2, center_list,
                                    area_list):
                self.node_list.append(new_node)

            if animation and i % 2 == 0:
                self.draw_graph(rnd_node, z, z2)

            # Check if the distance from the latest node to the goal state is within the set area
            if self.calc_dist_to_goal(self.node_list[-1].x, self.node_list[-1].y) <= self.expand_dis:
                final_node = self.steer(self.node_list[-1], self.end, self.expand_dis)
                # Check for collision
                if self.check_collision(final_node, self.obstacle_list, z_center, z_area, z_center2, z_area2,
                                        center_list, area_list):
                    return self.generate_final_course(len(self.node_list) - 1)

            if animation and i % 2:
                done = True
                self.draw_graph(rnd_node, z, z2)

        return None  # cannot find path

    def steer(self, from_node, to_node, extend_length=float("inf")):

        new_node = self.Node(from_node.x, from_node.y)
        d, theta = self.calc_distance_and_angle(new_node, to_node)

        new_node.path_x = [new_node.x]
        new_node.path_y = [new_node.y]

        if extend_length > d:
            extend_length = d

        n_expand = math.floor(extend_length / self.path_resolution)

        for _ in range(n_expand):
            new_node.x += self.path_resolution * math.cos(theta)
            new_node.y += self.path_resolution * math.sin(theta)
            new_node.path_x.append(new_node.x)
            new_node.path_y.append(new_node.y)

        d, _ = self.calc_distance_and_angle(new_node, to_node)
        if d <= self.path_resolution:
            new_node.path_x.append(to_node.x)
            new_node.path_y.append(to_node.y)
            new_node.x = to_node.x
            new_node.y = to_node.y

        new_node.parent = from_node

        return new_node

    def generate_final_course(self, goal_ind):
        path = [[self.end.x, self.end.y]]
        node = self.node_list[goal_ind]
        while node.parent is not None:
            path.append([node.x, node.y])
            node = node.parent
        path.append([node.x, node.y])

        return path

    def calc_dist_to_goal(self, x, y):
        dx = x - self.end.x
        dy = y - self.end.y
        return math.hypot(dx, dy)

    def get_random_node(self):
        if random.randint(0, 100) > self.goal_sample_rate:
            rnd = self.Node(
                random.uniform(self.min_rand, self.max_rand),
                random.uniform(self.min_rand, self.max_rand))
        else:  # goal point sampling
            rnd = self.Node(self.end.x, self.end.y)
        return rnd

    def draw_graph(self, rnd=None, zono=None, zono2=None):

        plt.clf()
        if rnd is not None:
            plt.plot(rnd.x, rnd.y, "^k")
        for node in self.node_list:
            if node.parent:
                plt.plot(node.path_x, node.path_y, "-g")

        for (ox, oy, size) in self.obstacle_list:
            self.plot_circle(ox, oy, size)

        if zono is not None:
            self.plot_circle1(zono)

        if zono2 is not None:
            self.plot_circle1(zono2)

        plt.plot(self.start.x, self.start.y, "xr")
        plt.plot(self.end.x, self.end.y, "xr")
        plt.axis("equal")
        plt.axis([-7, 20, -7, 20])
        plt.grid(True)
        plt.pause(0.01)

    def plot_circle1(self, zono, xdim = 0, ydim = 1, color="-b"):  # pragma: no cover
        v_list = self.verts(zono, xdim=xdim, ydim=ydim)
        xs = [v[xdim] for v in v_list]
        xs.append(v_list[0][xdim])

        ys = [v[ydim] for v in v_list]
        ys.append(v_list[0][ydim])

        plt.plot(xs, ys, color)

    @staticmethod
    def plot_circle(x, y, size, color="-b"):  # pragma: no cover
        deg = list(range(0, 360, 5))
        deg.append(0)
        xl = [x + size * math.cos(np.deg2rad(d)) for d in deg]
        yl = [y + size * math.sin(np.deg2rad(d)) for d in deg]
        plt.plot(xl, yl, color)

    @staticmethod
    def get_nearest_node_index(node_list, rnd_node):
        dlist = [(node.x - rnd_node.x)**2 + (node.y - rnd_node.y)**2
                 for node in node_list]
        minind = dlist.index(min(dlist))

        return minind

    @staticmethod
    def check_collision(node, obstacleList, center, area, center2, area2, center_list, area_list):

        if node is None:
            return False

        dx_list2 = [center2[0] - x for x in node.path_x]
        dy_list2 = [center2[1] - y for y in node.path_y]
        d_list2 = [dx * dx + dy * dy for (dx, dy) in zip(dx_list2, dy_list2)]

        if min(d_list2) <= area2:
            print(f"Future Dynamic collision expected at: ", (node.x, node.y))
            return False  # collision

        for center3, area3 in zip(center_list, area_list):
            # print(f"Center and Area: ", center3, area3)
            dx_list3 = [center3[0] - x for x in node.path_x]
            dy_list3 = [center3[1] - y for y in node.path_y]
            d_list3 = [dx * dx + dy * dy for (dx, dy) in zip(dx_list3, dy_list3)]

            if min(d_list3) <= area3:
                print("Avoiding path of second Dynamic Obstacle")
                return False  # collision

        dx_list1 = [center[0] - x for x in node.path_x]
        dy_list1 = [center[1] - y for y in node.path_y]
        d_list1 = [dx * dx + dy * dy for (dx, dy) in zip(dx_list1, dy_list1)]

        if min(d_list1) <= area:
            print(f"Dynamic collision detected at: ", (node.x, node.y))
            return False  # collision

        for (ox, oy, size) in obstacleList:
            dx_list = [ox - x for x in node.path_x]
            dy_list = [oy - y for y in node.path_y]
            d_list = [dx * dx + dy * dy for (dx, dy) in zip(dx_list, dy_list)]

            if min(d_list) <= size ** 2:
                return False  # collision

        return True  # safe

    @staticmethod
    def calc_distance_and_angle(from_node, to_node):
        dx = to_node.x - from_node.x
        dy = to_node.y - from_node.y
        d = math.hypot(dx, dy)
        theta = math.atan2(dy, dx)
        return d, theta

# ------------------------------- PID Controller here ---------------------------------------------------------

acc = 1.5
dec = -3.0
min_speed = 10
max_speed = 60

# Defining the Car object here
class Car(object):
    def __init__(self, length = 1.0):
        # Initial co-ordinates of the car
        # Add functionality to spawn the car at the start point entered by the user
        self.x = 0.0
        self.y = 0.0
        # See if possible to detect the next waypoint and define the car orientation accordingly
        self.orientation = 0.0
        self.max_steering_angle = np.pi/3.0
        self.length = length

        self.radius = 0
        # Add the noise factors here
        self.steering_noise = 0.0
        self.velocity_noise = 0.0
        self.steering_drift = 0.0

    def set(self, x, y, direction):
        # This function is used to change the position and orientation of the car

        self.x = x
        self.y = y
        self.orientation = direction % (2 * np.pi)

    def set_noise(self, steering_noise, velocity_noise):
        # This function is used to set the noise

        self.steering_noise = steering_noise
        self.velocity_noise = velocity_noise

    def set_steering_commands(self, drift, max_angle = np.pi/4):
        # Self explanatory name
        self.max_steering_angle = max_angle
        self.steering_drift = drift

    def move_vehicle(self, steering, velocity, tolerance = 0.05):
        """
        :param steering: Takes in the steering angle for the movement of the car, it is limited to self.max_steering_angle
        :param velocity: Takes in the current velocity of the car
        Both these parameters will be updated at the end of this function after accomodating the changes.

        :param tolerance:
        :return:
        """
        max_steering_angle = self.max_steering_angle
        distance = velocity

        if steering > max_steering_angle:
            steering = max_steering_angle
        if steering < -max_steering_angle:
            steering = -max_steering_angle
        if distance < 0.0:
            distance = 0.0

        steering2 = steering
        distance2 = distance
        # apply noise
        #steering2 = random.gauss(steering, self.steering_noise)
        #distance2 = random.gauss(distance, self.velocity_noise)

        # apply steering drift
        #steering2 += self.steering_drift

        # Execute motion

        turn = np.tan(steering2) * distance2 / self.length
        if abs(turn) < tolerance:
            # approximate by straight line motion
            #if distance2 < max_speed:
            #    distance2 += acc

            self.x += distance2 * np.cos(self.orientation)
            self.y += distance2 * np.sin(self.orientation)
            self.orientation = (self.orientation + turn) % (2.0 * np.pi)
        else:
            # approximate bicycle model for motion
            radius = distance2 / turn
            #delta_radius = radius - self.radius

            #if(delta_radius >= 0):
            #    distance2 -= dec
            #elif(delta_radius < 0):
            #    distance2 += acc

            cx = self.x - (np.sin(self.orientation) * radius)
            cy = self.y + (np.cos(self.orientation) * radius)
            self.orientation = (self.orientation + turn) % (2.0 * np.pi)
            self.x = cx + (np.sin(self.orientation) * radius)
            self.y = cy - (np.cos(self.orientation) * radius)

        return turn

def get_distance(x1, y1, x2, y2):
    distance = np.sqrt((x1-x2)**2 + (y1-y2)**2)
    return distance

def calculate_slope(x, y, waypoint_x, waypoint_y):
    # Calculates the slope from current point to the goal waypoint
    delta_w = (waypoint_x - x)
    if delta_w == 0:
        return 1
    else:
        return (waypoint_y - y)/delta_w


# Defining the PID Controller here
def pidController(car, goal, tau_p, tau_d, tau_i, n = 10000, velocity = 10.0):
    # List to hold the x and y points travelled
    x_trajectory = []
    y_trajectory = []

    prev_cte = car.y
    prev_x = car.x
    sum_cte = 0
    i = 1
    waypoint_tolerance = 4.0

    waypoint = goal[i]
    end_goal = goal[-1]
    goal_distance = get_distance(car.x, car.y, end_goal[0], end_goal[1])
    time = 0
    turn_values = []

    slope = calculate_slope(goal[i-1][0], goal[i-1][1], waypoint[0], waypoint[1])
    orientation = car.orientation
    integral = 0
    prev_orientation = np.abs(orientation - slope)
    orientation_tolerance = 0.05
    slope_total = []
    orientation_total = []
    tolerance2 = 10.0

    #while goal_distance > waypoint_tolerance:
    for _ in range(n):
        car_distance = get_distance(car.x, car.y, waypoint[0], waypoint[1])
        goal_distance = get_distance(car.x, car.y, end_goal[0], end_goal[1])
        if (car_distance < waypoint_tolerance or (get_distance(waypoint[0], waypoint[1], goal[i - 1][0], goal[i - 1][1]) + tolerance2 < car_distance)):
            i += 1
            if i >= len(goal):
                break
            else:
                waypoint = goal[i]
                slope = calculate_slope(goal[i - 1][0], goal[i - 1][1], waypoint[0], waypoint[1])

        y_error = car.y - waypoint[1]
        x_error = car.x - waypoint[0]
        orientation = car.orientation

        delta_orientation = orientation - slope
        time += 1

        integral += delta_orientation
        derivative = (delta_orientation - prev_orientation)
        prev_orientation = delta_orientation

        if np.abs(delta_orientation) >= orientation_tolerance:
            sum_cte += y_error
            dev = y_error - prev_cte
            prev_cte = y_error
            steer = - tau_p * y_error - tau_d * dev - tau_i * sum_cte
        else:
            steer = 0

        turn = car.move_vehicle(steer, velocity)
        orientation = car.orientation
        turn_values.append(delta_orientation)
        x_trajectory.append(car.x)
        y_trajectory.append(car.y)
        slope_total.append(slope)
        orientation_total.append(orientation)

    fig, ax1 = plt.subplots(1, 1, figsize=(8, 4))
    ax1.plot(slope_total, 'g')
    ax1.plot(orientation_total, 'r')
    ax1.plot(turn_values, 'b')
    plt.legend()
    plt.show()

    return x_trajectory, y_trajectory

def find_orientation(goal):
    init_x, init_y = goal[0][0], goal[0][1]
    end_x, end_y = goal[1][0], goal[1][1]

    slope = (end_y - init_y)/(end_x - init_x)

    return np.tan(slope)

def main(gx=6.0, gy=10.0):
    print("start " + __file__)
    # Declare required variables here
    follow = []

    # ====Search Path with RRT====
    obstacleList = [(5, 5, 1), (3, 6, 2), (3, 8, 2), (3, 10, 2), (7, 5, 2),
                    (9, 5, 2), (8, 10, 1)]  # [x, y, radius]
    # Set Initial parameters
    rrt = RRT( start=[0, 0],
        goal=[gx, gy],
        rand_area=[-2, 15],
        obstacle_list=obstacleList)
    path = rrt.planning(animation=show_animation)

    # The 'path' is in reversed order i.e the starting point is at the end of the list "path"
    # We reverse the order here
    list_length = len(path)
    high = list_length - 1
    while(high >= 0):
        follow.append(path[high])
        high -= 1


    if path is None:
        print("Cannot find path")
    else:
        print("Found path")

        # Draw final path
        if show_animation:
            rrt.draw_graph()
            plt.plot([x for (x, y) in path], [y for (x, y) in path], '-r')
            plt.grid(True)
            plt.pause(0.01)  # Need for Mac
            plt.show()


    # Motion planning starts here

    goal = follow
    goal_x = []
    goal_y = []

    for i in range(len(goal)):
        goal_x.append(goal[i][0])
        goal_y.append(goal[i][1])

    init_orientation = find_orientation(goal)

    car = Car()
    car.set(0, 0, init_orientation)
    car.set_steering_commands(10.0 / 180.0 * np.pi)
    x_trajectory, y_trajectory = pidController(car, goal, 0.01 ,0.5, 0.00005, velocity=0.01)
    n = len(x_trajectory)

    fig, ax1 = plt.subplots(1, 1, figsize=(8, 4))
    ax1.plot(x_trajectory, y_trajectory, 'g')
    ax1.plot(goal_x, goal_y, 'r')
    plt.legend()
    plt.show()

if __name__ == '__main__':
    main()

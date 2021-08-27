# Base code for Randomized Rapidly-Exploring Random Trees (RRT) and Pure Pursuit inspired from AtsushiSakai (@Atsushi_twi)

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

    def __init__(self, start, goal, obstacle_list, rand_area, expand_dis=3.0, path_resolution=0.5, goal_sample_rate=5, max_iter=500, path = None):
        """
        Setting Parameter

        start:Start Position [x,y]
        goal:Goal Position [x,y]
        obstacleList:obstacle Positions [[x,y,size],...]
        randArea:Random Sampling Area [min,max]

        """
        # Path for graph
        self.temp_path = path

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
                self.draw_graph(rnd_node)

            # Check if the distance from the latest node to the goal state is within the set area
            if self.calc_dist_to_goal(self.node_list[-1].x, self.node_list[-1].y) <= self.expand_dis:
                final_node = self.steer(self.node_list[-1], self.end, self.expand_dis)
                # Check for collision
                if self.check_collision(final_node, self.obstacle_list, z_center, z_area, z_center2, z_area2,
                                        center_list, area_list):
                    return self.generate_final_course(len(self.node_list) - 1)

            if animation and i % 2:
                done = True
                self.draw_graph(rnd_node, z)

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

    def draw_graph(self, rnd=None,  zono=None, zono2=None):

        plt.clf()
        if self.temp_path is not None:
            plt.plot([x for (x, y) in self.temp_path], [y for (x, y) in self.temp_path], 'r-')

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
        """
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
        """
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

def node_length(a, b):
    return np.sqrt((a[0] - b[0])**2 + (a[1] - b[1])**2)

def find_distance(path):
    pathlen = len(path)
    i = 0
    dist = 0

    while (i < pathlen - 1):
        dist += node_length(path[i],path[i+1])
        i += 1

    return dist

# ------------------------------- Pure Pursuit Controller here ---------------------------------------------------------

# Parameters
Kp = 1.0  # speed proportional gain
dt = 0.1  # [s] time tick
WB = 2.9  # [m] wheel base of vehicle

show_animation = True


class State:

    def __init__(self, x=0.0, y=0.0, yaw=0.0, v=0.0):
        self.x = x
        self.y = y
        self.yaw = yaw
        self.v = v
        self.rear_x = self.x - ((WB / 2) * math.cos(self.yaw))
        self.rear_y = self.y - ((WB / 2) * math.sin(self.yaw))

    def update(self, a, delta):
        self.x += self.v * math.cos(self.yaw) * dt
        self.y += self.v * math.sin(self.yaw) * dt
        self.yaw += self.v / WB * math.tan(delta) * dt
        self.v += a * dt
        self.rear_x = self.x - ((WB / 2) * math.cos(self.yaw))
        self.rear_y = self.y - ((WB / 2) * math.sin(self.yaw))

    def calc_distance(self, point_x, point_y):
        dx = self.rear_x - point_x
        dy = self.rear_y - point_y
        return math.hypot(dx, dy)


class States:

    def __init__(self):
        self.x = []
        self.y = []
        self.yaw = []
        self.v = []
        self.t = []

    def append(self, t, state):
        self.x.append(state.x)
        self.y.append(state.y)
        self.yaw.append(state.yaw)
        self.v.append(state.v)
        self.t.append(t)


def proportional_control(target, current, flag=1):
    # Accelerate
    if flag == 1:
        Kp = 0.25
    # Decelerate
    else:
        Kp = 2.0

    a = Kp * (target - current)

    return a


class TargetCourse:

    def __init__(self, cx, cy):
        self.cx = cx
        self.cy = cy
        self.old_nearest_point_index = None

    def search_target_index(self, state, k = 0.01, Lfc=2.5):

        # To speed up nearest point search, doing it at only first time.
        if self.old_nearest_point_index is None:
            # search nearest point index
            dx = [state.rear_x - icx for icx in self.cx]
            dy = [state.rear_y - icy for icy in self.cy]
            d = np.hypot(dx, dy)
            ind = np.argmin(d)
            self.old_nearest_point_index = ind
        else:
            ind = self.old_nearest_point_index
            distance_this_index = state.calc_distance(self.cx[ind],
                                                      self.cy[ind])
            while True:
                distance_next_index = state.calc_distance(self.cx[ind + 1],
                                                          self.cy[ind + 1])
                if distance_this_index < distance_next_index:
                    break
                ind = ind + 1 if (ind + 1) < len(self.cx) else ind
                distance_this_index = distance_next_index
            self.old_nearest_point_index = ind

        Lf = k * state.v + Lfc  # update look ahead distance

        print(ind)
        # search look ahead target point index
        while Lf > state.calc_distance(self.cx[ind], self.cy[ind]):
            if (ind) > len(self.cx):
                break  # not exceed goal
            ind += 1

        return ind, Lf

def find_distance_alternate(ax, ay, bx, by):
    return np.sqrt((ax - bx)**2 + (ay - by)**2)

def pure_pursuit_steer_control(state, trajectory, pind, k = 0.01, Lfc=2.5):
    # Define the required parameters here
    waypoint_tolerance = 1.5
    ca = 0.30
    max_steering_angle = np.pi / 4

    ind, Lf = trajectory.search_target_index(state)

    if pind >= ind:
        ind = pind

    if ind < len(trajectory.cx):
        tx = trajectory.cx[ind]
        ty = trajectory.cy[ind]
    else:  # toward goal
        tx = trajectory.cx[-1]
        ty = trajectory.cy[-1]
        ind = len(trajectory.cx) - 1

    if ind < len(trajectory.cx) - 2:
        next_tx = trajectory.cx[ind + 1]
        next_ty = trajectory.cy[ind + 1]
    else:
        next_tx = tx
        next_ty = ty
        ca = 0

    # If the robot is nearing its waypoint, start adding the bearing for the next waypoint
    # This leads to smoother turning for this particular project
    curr_distance = find_distance_alternate(state.x, state.y, tx, ty)
    if curr_distance < waypoint_tolerance:
        temp_alpha = math.atan2(next_ty - state.rear_y, next_tx - state.rear_x) - state.yaw
        alpha = math.atan2(ty - state.rear_y, tx - state.rear_x) - state.yaw + ca * temp_alpha
    else:
        alpha = math.atan2(ty - state.rear_y, tx - state.rear_x) - state.yaw

    # Modified delta by increasing the weightage of Lookahead distance
    # ------------------------- ADD A CHECK FOR MAX STEERING ANGLE ------------------------------------------
    delta = math.atan2(2.0 * WB * math.sin(alpha) / 2.0 * Lf, 1.0)

    if delta > max_steering_angle:
        delta = max_steering_angle

    return delta, ind


def draw_graph2(self, rnd=None, zono=None, zono2=None):
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


def plot_arrow(x, y, yaw, length=1.0, width=0.5, fc="r", ec="k"):
    """
    Plot arrow
    """

    if not isinstance(x, float):
        for ix, iy, iyaw in zip(x, y, yaw):
            plot_arrow(ix, iy, iyaw)
    else:
        plt.arrow(x, y, length * math.cos(yaw), length * math.sin(yaw),
                  fc=fc, ec=ec, head_width=width, head_length=width)
        plt.plot(x, y)

def find_orientation(goal):
    # This function is priarily used to find the inital orientation for the car

    init_x, init_y = goal[0][0], goal[0][1]
    end_x, end_y = goal[1][0], goal[1][1]

    slope = end_y - init_y/end_x - init_x

    return np.tan(slope)

def check_collision_alternate(x, y, obstacleList):
    for (ox, oy, size) in obstacleList:
        dx_list = [ox - x]
        dy_list = [oy - y]
        d_list = [np.sqrt(dx * dx + dy * dy) for (dx, dy) in zip(dx_list, dy_list)]

        if min(d_list) <= size - 2:
            return False  # collision

        return True  # safe

def main(gx=6.0, gy=10.0):
    # Variable declarations here
    follow = []
    all_paths = []
    collision_time = []
    counter = 0
    k = 0.01
    Lfc = 2.5
    flag = 1
    max_steering_angle = np.pi/4


    print("start " + __file__)

    # ====Search Path with RRT====
    obstacleList = [(5, 5, 1), (3, 6, 2), (3, 8, 2), (3, 10, 2), (7, 5, 2),
                    (9, 5, 2), (8, 10, 1)]  # [x, y, radius]
    # Set Initial parameters
    rrt = RRT(start=[0, 0],
              goal=[gx, gy],
              rand_area=[-2, 15],
              obstacle_list=obstacleList)
    path = rrt.planning(animation=show_animation)
    path_len = len(path)
    distance = find_distance(path)

    if path is None:
        print("Cannot find path")
    else:
        print("found path!!")

    for every_node in path:
        counter += 1
        new_path = []
        z = counter
        temp_x = every_node[0]
        temp_y = every_node[1]

        # Call RRT from the selected node
        rrt = RRT(start=[temp_x, temp_y],
                  goal=[gx, gy],
                  rand_area=[-2, 15],
                  obstacle_list=obstacleList,
                  path = path)
        new_path = rrt.planning(animation=show_animation)

        while (z < path_len):
            new_path.append(path[z])
            z += 1

        all_paths.append(new_path)

    for every_path in all_paths:
        new_distance = find_distance(every_path)
        if new_distance < distance:
            path = every_path

    # Using this for observation purpose
    temp_holder = path

    # Repeat process
    for every_node in path:
        counter += 1
        new_path = []
        z = counter
        temp_x = every_node[0]
        temp_y = every_node[1]

        # Call RRT from the selected node
        rrt = RRT(start=[temp_x, temp_y],
                  goal=[gx, gy],
                  rand_area=[-2, 15],
                  obstacle_list=obstacleList,
                  path = path)
        new_path = rrt.planning(animation=show_animation)

        while (z < path_len):
            new_path.append(path[z])
            z += 1

        all_paths.append(new_path)

    for every_path in all_paths:
        new_distance = find_distance(every_path)
        if new_distance < distance:
            path = every_path

    # Draw final paths
    if show_animation:
        rrt.draw_graph()
        for every_path in all_paths:
            plt.plot([x for (x, y) in every_path], [y for (x, y) in every_path], '-g')
        plt.plot([x for (x, y) in path], [y for (x, y) in path], 'black')
        plt.plot([x for (x, y) in temp_holder], [y for (x, y) in temp_holder], 'orange')
        plt.grid(True)
        plt.pause(0.01)  # Need for Mac
        plt.show()


    # Motion planning starts here

    # The 'path' is in reversed order i.e the starting point is at the end of the list "path"
    # We reverse the order here
    list_length = len(path)
    high = list_length - 1
    while (high >= 0):
        follow.append(path[high])
        high -= 1
    goal = follow
    goal_x = []
    goal_y = []

    for i in range(len(goal)):
        goal_x.append(goal[i][0])
        goal_y.append(goal[i][1])

    #  target course
    cx = goal_x
    cy = goal_y

    target_speed = 10.0 / 10.0  # [m/s]

    T = 100.0  # max simulation time

    init_orientation = find_orientation(goal)

    # initial state
    state = State(x=goal_x[0], y=goal_y[0], yaw=-init_orientation, v=0.0)
    lastIndex = len(cx) - 1
    time = 0.0
    states = States()
    states.append(time, state)
    target_course = TargetCourse(cx, cy)
    target_ind, _ = target_course.search_target_index(state)

    # Simulation variables
    temp_state = State(x=goal_x[0], y=goal_y[0], yaw=-init_orientation, v=0.0)
    temp_lastIndex = len(cx) - 1
    temp_time = 0.0
    temp_states = States()
    temp_states.append(temp_time, temp_state)
    temp_target_course = TargetCourse(cx, cy)
    temp_target_ind, _ = temp_target_course.search_target_index(temp_state)

    while T >= temp_time and temp_lastIndex > temp_target_ind:
        # Calc control input
        temp_time += dt

        temp_ai = proportional_control(target_speed, temp_state.v)
        temp_di, temp_target_ind = pure_pursuit_steer_control(
            temp_state, temp_target_course, temp_target_ind, k, Lfc)

        temp_state.update(temp_ai, temp_di)  # Control vehicle
        if check_collision_alternate(temp_state.x, temp_state.y, obstacleList):
            collision_time.append(temp_time)

    while T >= time and lastIndex > target_ind:

        new_speed = target_speed
        # Calc control input
        time += dt
        for every_time in collision_time:
            if (time == every_time - 2 or time == every_time - 3 or time == every_time - 4):
                k = 0.05
                Lfc = 1.0
                new_speed = 0.1 * target_speed
                flag = 0
            else:
                k = 0.01
                Lfc = 2.5

        di, target_ind = pure_pursuit_steer_control(
            state, target_course, target_ind, k, Lfc)
        if di >= 0.6 * max_steering_angle:
            new_speed = 0.4 * target_speed
            flag = 0
        ai = proportional_control(new_speed, state.v, flag)

        state.update(ai, di)  # Control vehicle
        states.append(time, state)

        if show_animation:  # pragma: no cover
            plt.cla()
            for (ox, oy, size) in obstacleList:
                deg = list(range(0, 360, 5))
                deg.append(0)
                xl = [ox + size * math.cos(np.deg2rad(d)) for d in deg]
                yl = [oy + size * math.sin(np.deg2rad(d)) for d in deg]
                plt.plot(xl, yl, "b-")

            plot_arrow(state.x, state.y, state.yaw)
            plt.plot(cx, cy, "-r", label="course")
            plt.plot(states.x, states.y, "-g", label="trajectory")
            plt.plot(cx[target_ind], cy[target_ind], "xg", label="target")
            plt.axis("equal")
            plt.grid(True)
            plt.title("Speed[km/h]:" + str(state.v * 3.6)[:4])
            plt.pause(0.001)


    # Test
    assert lastIndex >= target_ind, "Cannot goal"
    plt.plot(cx, cy, ".r", label="course")
    plt.plot(states.x, states.y, "-g", label="trajectory")
    plt.grid(True)
    plt.pause(0.01)
    plt.show()


    plt.plot(states.t, [iv * 3.6 for iv in states.v], "-r")
    plt.xlabel("Time[s]")
    plt.ylabel("Speed[km/h]")
    plt.grid(True)
    plt.show()


if __name__ == '__main__':
    main()

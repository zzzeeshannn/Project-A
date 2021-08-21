"""

Path planning Sample Code with Randomized Rapidly-Exploring Random Trees (RRT)

author: AtsushiSakai(@Atsushi_twi)

"""

import math
import random
from scipy.linalg import expm
import matplotlib.pyplot as plt
import numpy as np
from copy import deepcopy

import matplotlib.animation as animation

show_animation = True


class RungeKutta(object):

    def __init__(self,pt,ul,ur):
        self.point = pt
        self.ul = ul
        self.ur = ur
        self.h = 0.1
        self.epsilon = 0.5
        self.radius = 2
        self.length = 5

    # Der function
    def function(self,left_vel,right_vel,q_theta):
        #print()
        x_dot = (self.radius/2) * (right_vel+left_vel) * math.cos(q_theta[2])
        y_dot = (self.radius/2) * (right_vel+left_vel) * math.sin(q_theta[2])
        theta_dot = (self.radius/self.length) * (right_vel - left_vel)
        return np.array([x_dot,y_dot,theta_dot])

    # RK4 sum
    def calculate(self,left_vel,right_vel):
        q = np.array(self.point)
        #print(f"Points is: ", q)
        h_temp = self.h
        i = 1
        while h_temp <= self.epsilon:
            k_1 = self.function(left_vel,right_vel,q)
            k_2 = self.function(left_vel,right_vel,(q+0.5*self.h*k_1))
            k_3 = self.function(left_vel,right_vel,(q+0.5*self.h*k_2))
            k_4 = self.function(left_vel,right_vel,(q+self.h*k_3))

            q = q + ((1/6) * (k_1+2*k_2+2*k_3+k_4) * self.h)
            h_temp += self.h
            i += 1
        #return np.array(q)
        return tuple(q)

    def runge_kutta(self):
        point_list = []
        # go straight
        straight_pt = self.calculate(self.ul,self.ur)
        point_list.append(straight_pt)
        # go left
        left_pt = self.calculate(-self.ul,self.ur)
        point_list.append(left_pt)
        # go right
        right_pt = self.calculate(self.ul,-self.ur)
        point_list.append(right_pt)
        return point_list

class Zonotope:
    'zonotope class'

    def __init__(self, box, a_mat=None):

        self.box = np.array(box, dtype=float)
        self.a_mat = a_mat if a_mat is not None else np.identity(self.box.shape[0])

        self.dims = self.a_mat.shape[0]
        self.gens = self.a_mat.shape[1]

        self.b_vec = np.zeros((self.dims, 1))

    def max(self, direction):
        '''returns the point in the box that is the maximum in the passed in direction

        if x is the point and c is the direction, this should be the maximum dot of x and c
        '''

        direction = self.a_mat.transpose().dot(direction)

        # box has two columns and n rows
        # direction is a vector (one column and n rows)

        # returns a point (one column with n rows)

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
    """
    Class for RRT planning
    """
    class Node:
        """
        RRT Node
        """

        def __init__(self, x, y, theta=math.pi/18):
            self.x = x
            self.y = y
            self.theta = theta
            self.path_x = []
            self.path_y = []
            self.parent = None

    def __init__(self,
                 start,
                 goal,
                 obstacle_list,
                 rand_area,
                 expand_dis=3.0,
                 path_resolution=0.5,
                 goal_sample_rate=5,
                 max_iter=500):
        """
        Setting Parameter

        start:Start Position [x,y]
        goal:Goal Position [x,y]
        obstacleList:obstacle Positions [[x,y,size],...]
        randArea:Random Sampling Area [min,max]

        """
        self.start = self.Node(start[0], start[1])
        self.end = self.Node(goal[0], goal[1])
        self.min_rand = rand_area[0]
        self.max_rand = rand_area[1]
        self.expand_dis = expand_dis
        self.path_resolution = path_resolution
        self.goal_sample_rate = goal_sample_rate
        self.max_iter = max_iter
        self.obstacle_list = obstacle_list
        self.node_list = []

    @staticmethod
    def calc_distance(from_node, to_node):
        dx = to_node.x - from_node.x
        dy = to_node.y - from_node.y
        d = np.sqrt((dx*dx + dy*dy))

        return d

    def verts(self, zono, xdim=0, ydim=1):
        'get verticies for plotting 2d projections'

        verts = []
        #print(f"Inside verts: ", zono.dims)
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

    def get_nodes(self, possible_points):
        point1 = possible_points[0]
        point2 = possible_points[1]
        point3 = possible_points[2]

        node1 = self.Node(point1[0], point1[1], point1[2])
        node2 = self.Node(point2[0], point2[1], point2[2])
        node3 = self.Node(point3[0], point3[1], point3[2])

        return node1, node2, node3

    def planning(self, animation=True):
        """
        rrt path planning

        animation: flag for animation on or off
        """

        ul = 5
        ur = 5
        self.node_list = [self.start]
        # Zonotope
        init_box = [[1.0, 2.0], [5.0, 6.0]]
        init_box2 = [[15.0, 16.0], [15.0, 16.0]]
        dynamics_mat = np.array([[0.0, 1.0], [-1.0, 0.0]], dtype=float)  # mode 1: x' = y, y' = -x
        time_step = math.pi / 8
        num_steps = 10
        mode_boundary = 3.9
        sol_mat = expm(dynamics_mat * time_step)
        init_zono = Zonotope(init_box)
        init_zono2 = Zonotope(init_box2)
        center_list = []
        area_list = []

        z = deepcopy(init_zono)

        z_verts = self.verts(init_zono)
        z_xs = [v[0] for v in z_verts]
        z_ys = [v[1] for v in z_verts]
        min_x = np.min(z_xs)
        min_y = np.min(z_ys)
        max_x = np.max(z_xs)
        max_y = np.max(z_ys)
        z_center = ((min_x + max_x) / 2, (min_y + max_y) / 2)
        z_area = (max_x - min_x) * (max_y - min_y)

        z3 = deepcopy(init_zono2)

        z_verts3 = self.verts(init_zono2)
        z_xs3 = [v[0] for v in z_verts3]
        z_ys3 = [v[1] for v in z_verts3]
        min_x3 = np.min(z_xs3)
        min_y3 = np.min(z_ys3)
        max_x3 = np.max(z_xs3)
        max_y3 = np.max(z_ys3)
        z_center3 = ((min_x3 + max_x3) / 2, (min_y3 + max_y3) / 2)
        z_area3 = (max_x3 - min_x3) * (max_y3 - min_y3)

        center_list.append(z_center3)
        area_list.append(z_area3)

        for i in range(self.max_iter):
            rnd_node = self.get_random_node()
            nearest_ind = self.get_nearest_node_index(self.node_list, rnd_node)
            nearest_node = self.node_list[nearest_ind]

            #print(f"Node is: ", nearest_node.x, nearest_node.y)

            node_point = (nearest_node.x, nearest_node.y, nearest_node.theta)
            #print(f"Node point: ", node_point)
            runge_kutta = RungeKutta(node_point, ul, ur)
            possible_pts = runge_kutta.runge_kutta()

            node1, node2, node3 = self.get_nodes(possible_pts)
            dict = {}

            #dict['rnd'] = self.calc_distance(rnd_node, self.end)
            #dict['node1'] = self.calc_distance(node1, self.end)
            #dict['node2'] = self.calc_distance(node2, self.end)
            #dict['node3'] = self.calc_distance(node3, self.end)

            #if i%2 == 0:
            #    dict_list = list(dict.items())
            #    random_sample = random.choice(dict_list)
            #    final_node = random_sample[0]
            #else:
            #    final_node = max(dict, key=dict.get)

            #if final_node == 'rnd':
            #    final_node2 = rnd_node
            #elif final_node == 'node1':
            #    final_node2 = node1
            #elif final_node == 'node2':
            #    final_node2 = node2
            #else:
            #    final_node2 = node3

            final_node = random.randint(1,4)
            if final_node == 1:
                final_node2 = rnd_node
            elif final_node == 2:
                final_node2 = node1
            elif final_node == 3:
                final_node2 = node2
            else:
                final_node2 = node3

            nearest_ind2 = self.get_nearest_node_index(self.node_list, final_node2)
            nearest_node2 = self.node_list[nearest_ind2]

            new_node = self.steer(nearest_node2, final_node2, self.expand_dis)

            #z.a_mat = sol_mat.dot(z.a_mat)
            #z.b_vec = sol_mat.dot(z.b_vec)
            #print(z.dims)

            if(i%4 == 0):
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

                if(i<40):
                    # Second Dynamic Model
                    z3.b_vec -= np.array([[0.0], [1.0]])
                    z_verts3 = self.verts(z3)
                    z_xs3 = [vt[0] for vt in z_verts3]
                    z_ys3 = [vt[1] for vt in z_verts3]
                    # print(f"x and y of vertices: ", z_xs3, z_ys3)
                    min_x3 = np.min(z_xs3)
                    min_y3 = np.min(z_ys3)
                    max_x3 = np.max(z_xs3)
                    max_y3 = np.max(z_ys3)
                    # print(f"Min and Max y: ", min_y3, max_y3)
                    z_center3 = ((min_x3 + max_x3) / 2, (min_y3 + max_y3) / 2)
                    z_area3 = (max_x3 - min_x3) * (max_y3 - min_y3)

                    center_list.append(z_center3)
                    area_list.append(z_area3)


                # Future states of second dynamic model
                """"
                temp_zono2 = deepcopy(z3)
                temp_zono2.b_vec -= np.array([[0.0], [1.0]])
                z_verts4 = self.verts(temp_zono2)
                z_xs4 = [v[0] for v in z_verts4]
                z_ys4 = [v[1] for v in z_verts4]
                min_x4 = np.min(z_xs4)
                min_y4 = np.min(z_ys4)
                max_x4 = np.max(z_xs4)
                max_y4 = np.max(z_ys4)
                z_center4 = ((min_x + max_x) / 2, (min_y + max_y) / 2)
                z_area4 = (max_x - min_x) * (max_y - min_y)

                center_list.append(z_center4)
                area_list.append(z_area4)
                """""

            if self.check_collision(new_node, self.obstacle_list, z_center, z_area, z_center2, z_area2, center_list, area_list):
                self.node_list.append(new_node)

            if animation and i % 2 == 0:
                self.draw_graph(rnd_node, z, z3)
                #z.plot()

            if self.calc_dist_to_goal(self.node_list[-1].x, self.node_list[-1].y) <= self.expand_dis:
                final_node = self.steer(self.node_list[-1], self.end, self.expand_dis)
                #return self.generate_final_course(len(self.node_list) - 1)
                if self.check_collision(final_node, self.obstacle_list, z_center, z_area, z_center2, z_area2,center_list, area_list):
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
            rnd = self.Node(random.uniform(self.min_rand, self.max_rand), random.uniform(self.min_rand, self.max_rand))
        else:  # goal point sampling
            rnd = self.Node(self.end.x, self.end.y)
        return rnd

    def draw_graph(self, rnd=None, zono=None, zono2=None):

        #print(f"Inside draw_graph: ", zono.dims)
        plt.clf()
        # for stopping simulation with the esc key.
        plt.gcf().canvas.mpl_connect(
            'key_release_event',
            lambda event: [exit(0) if event.key == 'escape' else None])
        if rnd is not None:
            plt.plot(rnd.x, rnd.y, "^k")
        for node in self.node_list:
            if node.parent:
                plt.plot(node.path_x, node.path_y, "-g")

        #for (ox, oy, size) in self.obstacle_list:
        for (ox, oy, size) in self.obstacle_list:
            self.plot_circle(ox, oy, size)

        if zono is not None:
            self.plot_circle1(zono)

        if zono2 is not None:
            self.plot_circle1(zono2)

        #if zono is not None:
        #    xs, ys = self.plot_zono(zono)
        #    plt.plot(xs, ys, "^k")

        plt.plot(self.start.x, self.start.y, "xr")
        plt.plot(self.end.x, self.end.y, "xr")
        plt.axis("equal")
        plt.axis([-7, 20, -7, 20])
        plt.grid(True)
        plt.pause(0.01)



    def plot_zono(self, color='-b', xdim=0, ydim=1):
        'plot 2d projections'

        v_list = self.verts(xdim=xdim, ydim=ydim)
        #print(f"List of vertices: ", v_list)
        #print(f"PLOT CALLED")
        xs = [v[xdim] for v in v_list]
        #print(f"xs:", xs)
        xs.append(v_list[0][xdim])

        ys = [v[ydim] for v in v_list]
        ys.append(v_list[0][ydim])

        plt.plot(xs, ys, color)

    def plot_circle(self, x, y, size, color="-b"):  # pragma: no cover
        deg = list(range(0, 360, 5))
        deg.append(0)
        xl = [x + size * math.cos(np.deg2rad(d)) for d in deg]
        yl = [y + size * math.sin(np.deg2rad(d)) for d in deg]
        plt.plot(xl, yl, color)

    def plot_circle1(self, zono, xdim = 0, ydim = 1, color="-b"):  # pragma: no cover
        #print(f"Inside plot_circle: ", zono.dims)
        v_list = self.verts(zono, xdim=xdim, ydim=ydim)
        #print(f"List of vertices: ", v_list)
        #print(f"PLOT CALLED")
        xs = [v[xdim] for v in v_list]
        #print(f"xs:", xs)
        xs.append(v_list[0][xdim])

        ys = [v[ydim] for v in v_list]
        ys.append(v_list[0][ydim])

        plt.plot(xs, ys, color)

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
            #print(f"Center and Area: ", center3, area3)
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

            if min(d_list) <= size**2:
                return False  # collision

        return True  # safe

    @staticmethod
    def calc_distance_and_angle(from_node, to_node):
        dx = to_node.x - from_node.x
        dy = to_node.y - from_node.y
        d = math.hypot(dx, dy)
        theta = math.atan2(dy, dx)
        return d, theta


def main(gx=6.0, gy=9.0):
    print("start " + __file__)

    # ====Search Path with RRT====
    obstacleList = [(5, 5, 1), (3, 6, 2), (3, 8, 2), (3, 10, 2), (7, 5, 2),
                    (9, 5, 2), (8, 10, 1)]  # [x, y, radius]

    #obstacleList = [(8, 10, 1)]

    # Set Initial parameters
    rrt = RRT(
        start=[0, 0],
        goal=[gx, gy],
        rand_area=[-2, 30],
        obstacle_list= obstacleList )
    path = rrt.planning(animation=show_animation)

    if path is None:
        print("Cannot find path")
    else:
        print("found path!!")

    ax = []
    ay = []

    for waypoint in path:
        ax.append(round(waypoint[0], 1))
        ay.append(round(waypoint[1], 1))

    print(ax)
    print(ay)

    # Draw final path
    if show_animation:
            rrt.draw_graph()
            plt.plot([x for (x, y) in path], [y for (x, y) in path], '-r')
            plt.grid(True)
            plt.pause(0.01)  # Need for Mac
            plt.show()


if __name__ == '__main__':
    main()

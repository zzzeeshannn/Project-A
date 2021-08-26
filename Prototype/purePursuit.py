"""

Path tracking simulation with pure pursuit steering and PID speed control highly influenced from authors:
Atsushi Sakai (@Atsushi_twi)
Guillaume Jacquenot (@Gjacquenot)

Additional features and changes added by:
Zeeshan Shaikh
"""
import numpy as np
import math
import matplotlib.pyplot as plt

# Parameters
k = 0.02  # look forward gain
Lfc = 2.5  # [m] look-ahead distance
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


def proportional_control(target, current):
    a = Kp * (target - current)

    return a


class TargetCourse:

    def __init__(self, cx, cy):
        self.cx = cx
        self.cy = cy
        self.old_nearest_point_index = None

    def search_target_index(self, state):

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
                ind = ind + 1 if (ind) < len(self.cx) else ind
                distance_this_index = distance_next_index
            self.old_nearest_point_index = ind

        Lf = k * state.v + Lfc  # update look ahead distance

        # search look ahead target point index
        while Lf > state.calc_distance(self.cx[ind], self.cy[ind]):
            if (ind + 1) >= len(self.cx):
                break  # not exceed goal
            ind += 1

        return ind, Lf

def find_distance(ax, ay, bx, by):
    return np.sqrt((ax - bx)**2 + (ay - by)**2)

def pure_pursuit_steer_control(state, trajectory, pind):
    # Define the required parameters here
    waypoint_tolerance = 2.5
    ca = 0.15
    max_steering_angle = np.pi/4

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

    if ind < len(trajectory.cx) - 1:
        next_tx = trajectory.cx[ind + 1]
        next_ty = trajectory.cy[ind + 1]
    else:
        next_tx = tx
        next_ty = ty
        ca = 0

    # If the robot is nearing its waypoint, start adding the bearing for the next waypoint
    # This leads to smoother turning for this particular project
    curr_distance = find_distance(state.x, state.y, tx, ty)
    if curr_distance < waypoint_tolerance:
        temp_alpha = math.atan2(next_ty - state.rear_y, next_tx - state.rear_x) - state.yaw
        alpha = math.atan2(ty - state.rear_y, tx - state.rear_x) - state.yaw + ca * temp_alpha
    else:
        alpha = math.atan2(ty - state.rear_y, tx - state.rear_x) - state.yaw

    # Modified delta by increasing the weightage of Lookahead distance
    # ------------------------- ADD A CHECK FOR MAX STEERING ANGLE ------------------------------------------
    delta = math.atan2(2.0 * WB * math.sin(alpha) / 2.0*Lf, 1.0)

    if delta > max_steering_angle:
        delta = max_steering_angle

    return delta, ind


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


def main():
    follow = [[0, 0], [2.2538388513413974, 1.979952128760716], [4.156393043599898, 4.299496813719297],
              [6.6273304268671005, 2.598183071938368], [8.897970556529515, 0.6375216015241987],
              [11.802417234603617, -0.11360388348102193], [11.796886419228192, 2.6444432314235433],
              [12.589082696744692, 5.537957540682961], [9.964121865069755, 7.5660352990221575],
              [7.685400076812032, 8.35045010840275], [6.0, 10.0]]

    goal = follow
    goal_x = []
    goal_y = []

    for i in range(len(goal)):
        goal_x.append(goal[i][0])
        goal_y.append(goal[i][1])

    #  target course
    cx = goal_x
    cy = goal_y

    target_speed = 10.0 / 20.0  # [m/s]

    T = 100.0  # max simulation time

    # initial state
    state = State(x=goal_x[0], y=goal_y[0], yaw=0.0, v=0.0)

    lastIndex = len(cx) - 1
    time = 0.0
    states = States()
    states.append(time, state)
    target_course = TargetCourse(cx, cy)
    target_ind, _ = target_course.search_target_index(state)

    while T >= time and lastIndex > target_ind:

        # Calc control input
        ai = proportional_control(target_speed, state.v)
        di, target_ind = pure_pursuit_steer_control(
            state, target_course, target_ind)

        state.update(ai, di)  # Control vehicle

        time += dt
        states.append(time, state)

        if show_animation:  # pragma: no cover
            plt.cla()
            # for stopping simulation with the esc key.
            plt.gcf().canvas.mpl_connect(
                'key_release_event',
                lambda event: [exit(0) if event.key == 'escape' else None])
            plot_arrow(state.x, state.y, state.yaw)
            plt.plot(cx, cy, "-r", label="course")
            plt.plot(states.x, states.y, "-b", label="trajectory")
            plt.plot(cx[target_ind], cy[target_ind], "xg", label="target")
            plt.axis("equal")
            plt.grid(True)
            plt.title("Speed[km/h]:" + str(state.v * 3.6)[:4])
            plt.pause(0.001)

    # Test
    assert lastIndex >= target_ind, "Cannot goal"

    if show_animation:  # pragma: no cover
        plt.cla()
        plt.plot(cx, cy, ".r", label="course")
        plt.plot(states.x, states.y, "-b", label="trajectory")
        plt.legend()
        plt.xlabel("x[m]")
        plt.ylabel("y[m]")
        plt.axis("equal")
        plt.grid(True)

        plt.subplots(1)
        plt.plot(states.t, [iv * 3.6 for iv in states.v], "-r")
        plt.xlabel("Time[s]")
        plt.ylabel("Speed[km/h]")
        plt.grid(True)
        plt.show()


if __name__ == '__main__':
    print("Pure pursuit path tracking simulation start")
    main()

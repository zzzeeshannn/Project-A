# Import files here
import numpy as np
import matplotlib.pyplot as plt
import random

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
            #self.orientation = (self.orientation + turn) % (2.0 * np.pi)
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
            #steer = - tau_p * delta_orientation - tau_d * derivative
            #steer = - tau_p * (delta_orientation) - tau_d * derivative - tau_i * integral
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

def main():
    #goal = [[0.0, 0.0], [3.0, 0.5], [5.9, 1.0], [8.9, 1.6], [11.8, 2.1], [14.8, 2.6],
    #       [17.7, 3.1], [20.7, 3.6], [20.7, 6.6], [19.5, 9.5]]
    goal = [[3, 2], [70, 15], [120, 4], [160, 4], [200, 25], [270, 25]]
    goal_x = []
    goal_y = []

    for i in range(len(goal)):
        goal_x.append(goal[i][0])
        goal_y.append(goal[i][1])

    init_orientation = find_orientation(goal)

    car = Car()
    car.set(goal_x[0], goal_y[0], init_orientation)
    car.set_steering_commands(10.0/180.0*np.pi)
    x_trajectory, y_trajectory = pidController(car, goal, 0.009 ,.50, 0.00002, velocity=1.0)
    n = len(x_trajectory)

    fig, ax1 = plt.subplots(1, 1, figsize=(8,4))
    ax1.plot(x_trajectory, y_trajectory, 'g')
    ax1.plot(goal_x, goal_y, 'r')
    plt.legend()
    plt.show()

if __name__ == '__main__':
    main()

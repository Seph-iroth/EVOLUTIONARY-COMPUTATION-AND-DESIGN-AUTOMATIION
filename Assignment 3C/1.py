import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import math
import random
from multiprocessing import Pool, Process
import time
# Class for Mass
from multiprocessing import Pool
import os
class Mass:
    def __init__(self, m, p, v=None, a=None,f=None,omega=None):
        """
        m: mass [kg]
        p: 3D position vector [meter]
        v: 3D velocity vector [meter/s], initialized to zero if not provided
        a: 3D acceleration vector [meters/s^2], initialized to zero if not provided
        """
        self.m = m
        self.p = np.array(p)

        if v is None:
            self.v = [0,0,0]
        else:
            self.v = np.array(v)

        if a is None:
            self.a = [0,0,0]
        else:
            self.a = np.array(a)
        if f is None:
            self.f = [0, 0, 0]
        else:
            self.f = np.array(f)
        if omega is None:
            self.omega = [0, 0, 0]
        else:
            self.omega = np.array(omega)


    def __str__(self):
        return f"Mass(m={self.m}, p={self.p}, v={self.v}, a={self.a})"

# Class for Spring
class Spring:
    def __init__(self, k, L0, m1, m2,leg=None,amplitude=None,angular_f = None,phase_shift = None,max_L0=None):
        """
        k: spring constant [N/m]
        L0: original rest length [meters]
        m1, m2: indices of the two masses it connects
        A*sin(wt+B)
        """
        self.k = k
        self.L0 = L0
        self.m1 = m1
        self.m2 = m2
        if max_L0 is None:
            self.max_L0 = None
        else:
            self.max_L0 = L0*1.25

        if leg is None:
            self.leg = None
        else:
            self.leg = leg

        if amplitude is None:
            self.amplitude = None
        else:
            self.amplitude = amplitude

        if angular_f is None:
            self.angular_f = None
        else:
            self.angular_f = angular_f

        if phase_shift is None:
            self.phase_shift = None
        else:
            self.phase_shift = phase_shift



    def __str__(self):
        return f"Mass(m={self.k}, p={self.L0}, v={self.m1}, a={self.m2})"

def compute_torque(force, position):
    return np.cross(position, force)

def distance(point1, point2):
    """Compute the Euclidean distance between two 3D points."""
    return math.sqrt((point1[0] - point2[0])**2 +
                     (point1[1] - point2[1])**2 +
                     (point1[2] - point2[2])**2)

def normalize(vector):
    """Normalize a 3D vector."""
    magnitude = math.sqrt(vector[0]**2 + vector[1]**2 + vector[2]**2)
    if magnitude == 0:  # to prevent division by zero
        return [0, 0, 0]
    return [vector[0]/magnitude, vector[1]/magnitude, vector[2]/magnitude]

def subtract_vectors(v1, v2):
    """Subtract v2 from v1."""
    return [v1[0] - v2[0], v1[1] - v2[1], v1[2] - v2[2]]

def multiply_vector_by_scalar(vector, scalar):
    """Multiply a 3D vector by a scalar."""
    return [vector[0] * scalar, vector[1] * scalar, vector[2] * scalar]

def add_vectors(v1, v2):
    """Add two 3D vectors."""
    return [v1[0] + v2[0], v1[1] + v2[1], v1[2] + v2[2]]

def mutiple_vectors(vector1,vector2):

    return [a * b for a, b in zip(vector1, vector2)]

def divide_vector_by_scalar(vector, scalar):
    """Divide a 3D vector by a scalar."""
    if scalar == 0:  # to prevent division by zero
        raise ValueError("Cannot divide by zero!")
    return [vector[0]/scalar, vector[1]/scalar, vector[2]/scalar]

def compute_gravitational_PE(mass):
    """Compute the gravitational potential energy of a mass."""
    h = mass.p[2]  # height is the z-coordinate
    return mass.m * g * h

def compute_spring_PE(spring):
    """Compute the elastic potential energy stored in a spring."""
    L = distance(spring.m1.p, spring.m2.p)
    return 0.5 * spring.k * (L - spring.L0)**2

def compute_kinetic_energy(mass):
    """Compute the kinetic energy of a mass."""
    v_magnitude = np.linalg.norm(mass.v)
    return 0.5 * mass.m * v_magnitude**2

def spring_gen(k,mass_list):
    n = len(mass_list)
    spring = []
    num = 0
    for j in range(n):
        for i in range(j, n-1):
            s = Spring(k=k, L0 = np.linalg.norm(mass_list[i].p - mass_list[i + 1].p), m1=mass_list[i], m2 = mass_list[i + 1])
            spring.append(s)
    return spring

def create_cube_springs(masses, k):
    n = len(masses)
    spring = []
    for j in range(n):
        for i in range(j-1, n ):
            s = Spring(k=k, L0=np.linalg.norm(masses[i].p - masses[j].p), m1=masses[i], m2=masses[j])
            spring.append(s)
    return spring




def dot_product(v1, v2):
    """Calculate the dot product of two vectors represented as lists."""
    return sum(x*y for x, y in zip(v1, v2))

def draw_vertices(a):
    points = []
    for i in a:
        points.append(i.p)
    x = [row[0] for row in points]
    y = [row[1] for row in points]
    z = [row[2] for row in points]
    ax.scatter3D(x, y, z, color='red', s=5)
    return points

def draw_springs(vertices,shadow = False):

    xs, ys, zs = zip(*vertices)
    # for i in range(8):
    #     for j in range(i + 1, 8):
    for i in range(len(vertices)):
        for j in range(i + 1, len(vertices)):  # start from i+1 to avoid connecting a point to itself and duplicate lines
            x_values = [vertices[i][0], vertices[j][0]]
            y_values = [vertices[i][1], vertices[j][1]]
            z_values = [vertices[i][2], vertices[j][2]]
            if shadow == True:
                ax.plot(x_values, y_values, 0, color='gray', lw=1)
            ax.plot(x_values, y_values, z_values, color='blue',lw=1)


gravitational_PEs = []
spring_PEs = []
KEs = []
total_energies = []

m = 1.0
v0 = [1,0,0]
a = [0,0,0]
x0 = 3
y0 = 1
z0 = 5
p0 = [x0,y0,z0]
length = 1
fig = plt.figure(figsize=(10, 10))

ax = fig.add_subplot(111, projection='3d')



# 1. Create lists for masses and springs


def rotation_matrix_z(theta):
    """Return the rotation matrix for a rotation around the Z axis."""
    return np.array([[np.cos(theta), -np.sin(theta), 0],
                     [np.sin(theta), np.cos(theta),  0],
                     [0,             0,              1]])
def rotation_matrix_x(theta):
    """Return the rotation matrix for a rotation around the X axis."""
    return np.array([
        [1, 0, 0],
        [0, np.cos(theta), -np.sin(theta)],
        [0, np.sin(theta), np.cos(theta)]
    ])
def rotation_matrix_y(theta):
    """Return the rotation matrix for a rotation around the Y axis."""
    return np.array([
        [np.cos(theta), 0, np.sin(theta)],
        [0, 1, 0],
        [-np.sin(theta), 0, np.cos(theta)]
    ])
def transform_positions(mass_objs, transformation_matrix):
    for mass_obj in mass_objs:
        mass_obj.p = np.dot(transformation_matrix, mass_obj.p)


def output_springVar(Amp_range,frequency_range,k_range):

    #Asin(wt+B)

    A = np.random.uniform(Amp_range,-Amp_range)
    w = np.random.randint(0, frequency_range)
    B = np.random.uniform(-2*math.pi,2*math.pi)

    return A,w,B

def create_X_cube(initial_loc,mass = m, k = 10000,Amp_range=None,frequency_range=None,k_range=None):
    amass = []
    spring0 = []
    x0,y0 ,z0 = initial_loc[0],initial_loc[1],initial_loc[2]
    amass.append(Mass(m=m, p=[x0 + length / 2, y0 + length / 2, z0 + length / 2], v=v0, a=a))
    amass.append(Mass(m=m, p=[x0 + length / 2, y0 - length / 2, z0 + length / 2], v=v0, a=a))
    amass.append(Mass(m=m, p=[x0 - length / 2, y0 + length / 2, z0 + length / 2], v=v0, a=a))
    amass.append(Mass(m=m, p=[x0 - length / 2, y0 - length / 2, z0 + length / 2], v=v0, a=a))

    amass.append(Mass(m=m, p=[x0 + length / 2, y0 + length / 2, z0 - length / 2], v=v0, a=a))
    amass.append(Mass(m=m, p=[x0 + length / 2, y0 - length / 2, z0 - length / 2], v=v0, a=a))
    amass.append(Mass(m=m, p=[x0 - length / 2, y0 + length / 2, z0 - length / 2], v=v0, a=a))
    amass.append(Mass(m=m, p=[x0 - length / 2, y0 - length / 2, z0 - length / 2], v=v0, a=a))
    # amass.extend([m1, m2, m3, m4, m5, m6, m7, m8])
    for q in create_cube_springs(amass,k):
        spring0.append(q)
    # amass.append(Mass(m=m, p=[x0 + 2.5*length , y0 , z0 - 2], v=v0, a=a))
    # amass.append(Mass(m=m, p=[x0 - 2.5 * length, y0, z0 - 2], v=v0, a=a))
    # amass.append(Mass(m=m, p=[x0 , y0- 2.5 * length, z0 - 2], v=v0, a=a))
    # amass.append(Mass(m=m, p=[x0 , y0+ 2.5 * length, z0 - 2], v=v0, a=a))
    amass.append(Mass(m=m, p=[x0 + 2.5*length , y0 , z0 - length / 2], v=v0, a=a))
    amass.append(Mass(m=m, p=[x0 - 2.5 * length, y0, z0 - length / 2], v=v0, a=a))
    amass.append(Mass(m=m, p=[x0 , y0- 2.5 * length, z0 - length / 2], v=v0, a=a))
    amass.append(Mass(m=m, p=[x0 , y0+ 2.5 * length, z0 - length / 2], v=v0, a=a))
    # print(len(amass))
    g = [8,9,10,11]
    y = [4,5,6,7]
    leg_number = 1
    inputForEachLeg = []
    for i in g:
        leg_number += 1
        A, w, B = output_springVar(Amp_range, frequency_range, k_range)
        # print(A,w,B)
        for j in y:
            spring0.append(Spring(k=k, L0=np.linalg.norm(amass[i].p - amass[j].p), m1=amass[i], m2=amass[j],leg = leg_number, amplitude=A, angular_f = w, phase_shift = B))


    # Spring(k=k, L0=np.linalg.norm(masses[i].p - masses[j].p), m1=masses[i], m2=masses[j])

    return amass, spring0

def addingOnemass(mass, mass_to_be_added,springrate,leg_number=None):
    for i in mass_to_be_added:
        Spring(k=springrate, L0=np.linalg.norm(mass.p - i.p), m1=mass, m2=i, leg=leg_number)







def center_of_mass(masses):
    total_mass = sum(mass.m for mass in masses)
    if total_mass == 0:
        return np.array([0.0, 0.0, 0.0])  # Ensure this is float

    weighted_positions = np.array([0.0, 0.0, 0.0])  # Initialize as float
    for mass in masses:
        weighted_positions += np.array(mass.p, dtype=np.float64) * mass.m

    center_of_mass = weighted_positions / total_mass
    return center_of_mass

# print(center_of_mass(masses))

def append_to_file(filename, data):
    with open(filename, 'a') as file:
        file.write(data + "\n")



def simulation_one_mass(masses, spring,spring_sin):
    # draw_vertices(kaka)
    for mass in masses:
        F = [0, 0, mass.m * g]  # starting with gravity force
        # print
        memoryForSpring = []
        for spr in spring:
            if spr.m1 == mass or spr.m2 == mass:
                # print(spr.L0)
                other_mass = spr.m2 if spr.m1 == mass else spr.m1

                L = distance(mass.p, other_mass.p) #+ 3*math.sin(math.pi*dt+0.4)
                # L = spr.L0

                if spr.leg == 1:
                    amplitude,angular_f,phase_shift = spring_sin[0]
                    L = distance(mass.p, other_mass.p) + amplitude*math.sin(angular_f*dt+phase_shift)
                if spr.leg == 2:
                    amplitude,angular_f,phase_shift = spring_sin[1]
                    L = distance(mass.p, other_mass.p) + amplitude*math.sin(angular_f*dt+phase_shift)
                if spr.leg == 3:
                    amplitude,angular_f,phase_shift = spring_sin[2]
                    L = distance(mass.p, other_mass.p) + amplitude*math.sin(angular_f*dt+phase_shift)
                if spr.leg == 4:
                    amplitude,angular_f,phase_shift = spring_sin[3]
                    L = distance(mass.p, other_mass.p) + amplitude*math.sin(angular_f*dt+phase_shift)


                spring_force_magnitude = spr.k * (L - spr.L0)
                direction = normalize(subtract_vectors(other_mass.p, mass.p))

                # Damping
                damping_coefficient = 0.01  # Adjust as needed
                relative_velocity = subtract_vectors(other_mass.v, mass.v)
                damping_force = multiply_vector_by_scalar(direction, -damping_coefficient * dot_product(relative_velocity,direction))
                spring_force = multiply_vector_by_scalar(direction, spring_force_magnitude)
                total_force = add_vectors(spring_force, damping_force)
                F = add_vectors(F, total_force)

        if mass.p[2] <= 0:  # checking the z-axis
            Fc = [0, 0, -kc*mass.p[2]]
            # upward restoration force
            F = add_vectors(F, Fc)
            friction_coefficient = 50  # Adjust as needed
            normal_force_magnitude = abs(mass.m * g)
            friction_force_magnitude = friction_coefficient * normal_force_magnitude
            friction_force = [-friction_force_magnitude * v / abs(v) if v != 0 else 0 for v in mass.v[:2]]
            friction_force.append(0)  # No friction force in the z-direction
            F = add_vectors(F, friction_force)
            # mass.p[2] = 0

        mass.total_force = F
    for mass in masses:
        # print(mass.m)
        acceleration = divide_vector_by_scalar(mass.total_force, mass.m)  # calculate the acceleration.
        # mass.v = multiply_vector_by_scalar(mass.v, 0.9)
        mass.v = add_vectors(mass.v, multiply_vector_by_scalar(acceleration, dt)) #get velocity. v =at
        mass.p = add_vectors(mass.p, multiply_vector_by_scalar(mass.v, dt) ) #get position. p =v*t+
    # for i in masses:
    #     print(i.m)
    # print(center_of_mass(masses))
    # draw_springs(draw_vertices(masses))
    return center_of_mass(masses)

scale = 10
running = 0
def update(frame):
    ax.cla()
    # First, compute all forces (springs, gravity, external forces, ground collision, etc.)
    ax.set_xlim3d([-1, scale])
    ax.set_xlabel('X')

    ax.set_ylim3d([-1, scale])
    ax.set_ylabel('Y')

    ax.set_zlim3d([0, scale])
    ax.set_zlabel('Z')

    simulation_one_mass(masses, spring,True)
    return ax,


# ani = FuncAnimation(fig, update,frames=2000,interval=0, blit=True)
# dt = 0.002
# plt.show()
filename = "delet.txt"
duration_per_trial = 2 #seconds
# with open('yourfile.txt', 'w'):
#     pass  # The file is now cleared
# for i in range(1000):
#     append_to_file(filename,str(i))
# print(time.localtime().tm_mon,time.localtime().tm_mday,time.localtime().tm_hour,time.localtime().tm_min,time.localtime().tm_sec)
A_range = 1.5    #range of amplitude
w_range = 10     #range of wave frequency
B_range = 2*math.pi # range of time shift

duration = 100000  #duration per trial
iter = 10   # how many iterations
dt = 0.002  #time step
g = -9.81*10    #gravity
kc = 10000  #spring constant
inital_loc = [3,1,0.5] #initial postion
A = 1.7
w = 60
range_K = 8000
masses = []
masses,spring = create_X_cube(inital_loc,mass = m,k=kc,Amp_range=A,frequency_range=w,k_range=range_K)
masses1,spring = create_X_cube(inital_loc,mass = m,k=kc,Amp_range=A,frequency_range=w,k_range=range_K)
masses2,spring = create_X_cube(inital_loc,mass = m,k=kc,Amp_range=A,frequency_range=w,k_range=range_K)
masses3,spring = create_X_cube(inital_loc,mass = m,k=kc,Amp_range=A,frequency_range=w,k_range=range_K)
masses4,spring = create_X_cube(inital_loc,mass = m,k=kc,Amp_range=A,frequency_range=w,k_range=range_K)

theta = 15 * np.pi / 180
def BUNCHofMassAndSpring(num):
    mass = []
    spring = []
    for i in range(num):
        mass.append(create_X_cube([3,1,2],mass = m,k=kc,Amp_range=A,frequency_range=w,k_range=range_K)[0])
        spring.append(create_X_cube([3, 1, 2], mass=m, k=kc, Amp_range=A, frequency_range=w, k_range=range_K)[1])
    return mass ,spring
# Rotate by 45 degrees for this example
# transformation = rotation_matrix_x(theta)
# transform_positions(masses, transformation)
# theta = 30 * np.pi / 180  # Rotate by 45 degrees for this example
# transformation = rotation_matrix_z(theta)
# transform_positions(masses2, transformation)
# theta = 45 * np.pi / 180  # Rotate by 45 degrees for this example
# transformation = rotation_matrix_z(theta)
# transform_positions(masses3, transformation)

# masses.extend([m1,m2,m3,m4,m5])
# spring = create_cube_springs(masses, kc)
#

def simulation_mutiple_mass(num):
    mass, spring = BUNCHofMassAndSpring(num)


full_path = __file__
curr_file_name = os.path.splitext(os.path.basename(full_path))[0]

file_name = curr_file_name +"_"+"speed List.txt"
file_name2 = curr_file_name +"_"+ "parameter list.txt"
staringPosition = center_of_mass(masses)
if __name__ == "__main__":
    bestSpeed = 0.0
    for k in range(iter):
        input_sp_list = []
        for j in range(4):
            A, w, B = output_springVar(A_range, w_range, B_range)
            input_sp_list.append([A, w, B])
        masses, spring = create_X_cube(inital_loc, mass=m, k=kc, Amp_range=A, frequency_range=w, k_range=range_K)

        for i in range(duration):
            #pass in sin wave into for on mass.
            simulation_one_mass(masses, spring, input_sp_list)
            # print(center_of_mass(masses))
            if i == (duration - 1):
                speed = abs(distance(staringPosition,center_of_mass(masses))) / (duration * dt)
                append_to_file(file_name,str(speed))
                print(center_of_mass(masses))
            if i == (duration - 1):
                append_to_file(file_name2,str(input_sp_list))



    print("done")

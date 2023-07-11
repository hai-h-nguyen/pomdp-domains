import pygame
import numpy as np
from pdomains.utils import rotation_matrix

class Cmd(object):
  def __init__(self):
    self.reset()

  def reset(self):
    self.axis0 = 0
    self.axis1 = 0
    self.axis2 = 0 
    self.axis3 = 0
    self.axis4 = 0
    self.axis5 = 0 
    self.btn0 = 0
    self.btn1 = 0
    self.btn2 = 0
    self.btn3 = 0
    self.btn4 = 0
    self.btn5 = 0
    self.btn6 = 0
    self.btn7 = 0
    self.btn8 = 0
    self.btn9 = 0

class Joystick(object):
  def __init__(self, pos_xy_scale=0.1, pos_z_scale=0.1, rot_scale=0.1):
    # max velocity and acceleration to be send to robot
    self.pos_xy_scale = pos_xy_scale
    self.pos_z_scale = pos_z_scale
    self.rot_scale = rot_scale

    self.cmd = Cmd()
    self.joystick = None

  def start_control(self):
    pygame.init()
    self.joystick = pygame.joystick.Joystick(0)
    self.joystick.init()
    print('Initialized Joystick : %s' % self.joystick.get_name())

    self.rotation = np.array([[-1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, -1.0]])
    self.raw_drotation = np.zeros(3)  # immediate roll, pitch, yaw delta values from keyboard hits
    self.last_drotation = np.zeros(3)

  def update_joystick_state(self):
    self.cmd.reset()
    pygame.event.pump() # Seems we need polling in pygame...

    # get joystick state
    for i in range(0, self.joystick.get_numaxes()):
      val = self.joystick.get_axis(i)
      if abs(val) < 0.1:
        val = 0
      tmp = "self.cmd.axis" + str(i) + " = " + str(val)
      exec(tmp)

    # get button state
    for i in range(0, self.joystick.get_numbuttons()):
      if self.joystick.get_button(i) != 0:
        tmp = "self.cmd.btn" + str(i) + " = 1"
        #print(tmp)
        exec(tmp)

  def calc_speeds(self):
    speeds = [0, 0, 0, 0]
    speeds[0] =  self.cmd.axis0 * self.pos_xy_scale  # left/right
    speeds[1] =  self.cmd.axis1 * self.pos_xy_scale  # front/back
    speeds[2] =  -self.cmd.axis4 * self.pos_z_scale  # up/down
    speeds[3] =  self.cmd.axis3 * self.rot_scale  # yaw left/right

    reset = self.cmd.btn5 > 0

    dict_return = {}
    dict_return["left_right"] = speeds[0]
    dict_return["front_back"] = speeds[1]
    dict_return["up_down"] = speeds[2]
    dict_return["reset"] = reset

    drot = rotation_matrix(angle=speeds[3], direction=[1.0, 0.0, 0.0])[:3, :3]
    self.rotation = self.rotation.dot(drot)  # rotates z
    self.raw_drotation[2] -= speeds[3]

    raw_drotation = (
        self.raw_drotation - self.last_drotation
    )  # create local variable to return, then reset internal drotation
    self.last_drotation = np.array(self.raw_drotation)

    dict_return["rot_left_right"] = raw_drotation

    return dict_return

  def get_controller_state(self):
    self.update_joystick_state()
    return self.calc_speeds()

  def close(self):
    if self.joystick:
      self.joystick.quit()
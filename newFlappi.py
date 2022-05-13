import pygame as pg
import random
from random import *
import numpy as np

win = pg.display.set_mode((400, 600))
night = pg.image.load("night.png")
night = pg.transform.scale(night, (600, 600))
count = 0

###BIRD###
class Bird:
    def __init__(self):
        self.surface = pg.image.load("flappi.png")
        self.surface = pg.transform.scale(self.surface, (50, 50))
        self.pos = [100, 300]
        self.speed = 0
        self.is_alive = True
        self.distX = 0
        self.distYB = 0
        self.distYA = 0
        self.score = 0
        self.angle = 0
        self.passed = False
        self.trains = 0

    # draw bird
    def draw(self, win):
        win.blit(self.surface, self.pos)
        pg.display.flip()

    # bird colisions
    def cols(self, pipe, pype):
        if self.pos[0] + 40 >= pipe.pos[0]:
            if self.pos[0] <= pipe.pos[0] + 80:
                if self.pos[1] <= 200:
                    self.is_alive = False
                elif self.pos[1] + 25 > pype.pos[1]:
                    self.is_alive = False
                else:
                    self.is_alive = True
        if self.pos[1] >= 600:
            self.is_alive = False
        if pype.psd > 1:
            if self.pos[1] < pipe.h:
                self.passed = True
            elif self.pos[1] > pype.pos[1]:
                self.passed = True

    # return inputs
    def get_data(self, pipe, pype):
        self.distX = pipe.pos[0] - self.pos[0] + 50
        self.distYA = self.pos[1] - pipe.h
        self.distYB = pype.pos[1] - self.pos[1] + 50
        ip = [self.distX, self.distYA, self.distYB]
        return ip

    # rewards?
    def get_reward(self):
        return self.distance

    # return bird status (alive or not)
    def get_alive(self):
        return self.is_alive

    # bird gravity
    def move(self):
        if self.pos[1] < 600:
            self.pos[1] += 2

    # bird jump
    def jump(self):
        self.pos[1] -= 4
        self.isjump = True

    # update bird distance travelled
    def get_distance(self):
        while self.is_alive:
            self.distance += 1

    # reset bird
    def reset(self):
        self.pos = [100, 300]


###PIPE1###
class Pipe:
    def __init__(self):
        self.surface = pg.image.load("pipE.png")
        self.w = 80
        self.h = 250
        self.dist = 60
        self.surface = pg.transform.scale(self.surface, (self.w, self.h))
        self.pos = [200, 0]

    def draw(self, win):
        win.blit(self.surface, self.pos)

    def move(self):
        if self.pos[0] > -100:
            self.pos[0] -= 2
            pg.display.flip()
        else:
            self.pos[0] = 500
            # self.h = randint(100, 500)
            # self.surface = pg.transform.scale(self.surface, (80, self.h))
            pg.display.flip()

    def reset(self):
        self.pos = [500, 0]
        self.h = randint(100, 200)


###PIPE2###
class Pype:
    def __init__(self):
        self.surface = pg.image.load("pype.png")
        self.surface = pg.transform.scale(self.surface, (80, 300))
        self.pos = [200, 300]
        self.psd = 0

    def draw(self, win):
        win.blit(self.surface, self.pos)

    def move(self, pipe):
        if self.pos[0] > -100:
            self.pos[0] -= 2
            pg.display.flip()
        else:
            self.pos[0] = 500
            self.psd += 1
            # self.pos[1] = pipe.h + 60
            # self.h = 600 - self.pos[1]
            pg.display.flip()
            # self.surface = pg.transform.scale(self.surface, (80, self.h))

    def reset(self, pipe):
        self.pos[0] = 500
        self.pos[1] = pipe.h + 200


###NEURAL NET (that i kinda understand now)###
class NN(object):
    def __init__(self):
        self.inputSize = 3
        self.outputSize = 1
        self.hiddenSize = 3
        self.W1 = np.random.randn(self.inputSize, self.hiddenSize)
        self.W2 = np.random.randn(self.hiddenSize, self.outputSize)

    # decides whether a neuron should be activated or not by calculating weighted sum and further adding bias with it.
    def sigmoid(self, s):
        return 1 / (1 + np.exp(-s))

    def forward(self, X):
        #returns the product of two 2d arrays
        self.z = np.dot(X, self.W1)
        self.z2 = self.sigmoid(self.z)
        self.z3 = np.dot(self.z2, self.W2)
        o = self.sigmoid(self.z3)
        return o

    def sigmoidPrime(self, s):
        return s * (1 - s)
    #how much of the loss every node is responsible for, and subsequently updating the weights in such a way that minimizes the loss by giving the nodes with higher error rates lower weights and vice versa
    def backward(self, X, y, o):
        self.o_error = y - o
        self.o_delta = self.o_error * self.sigmoidPrime(o)
        self.z2_error = self.o_delta.dot(self.W2.T)
        self.z2_delta = self.z2_error * self.sigmoidPrime(self.z2)
        self.W1 += X.T.dot(self.z2_delta)
        #self.W2 += self.z2.T.dot(self.o_delta)

    def train(self, X, y):
        o = self.forward(X)
        self.backward(X, y, o)
##RESET SCREEN##
def reset(bird, pipe, pype):
        bird.reset()
        pipe.reset()
        pype.reset(pipe)


##MOVES BASED ON OUTPUT##
def moves(bird, out):
    if out > 0.5:
        bird.jump()
        pg.display.flip()
    bird.move()
    pg.display.flip()


##DRAW STUFF##
def draw(win, bird, pipe, pype, night):
    # draw visuals
    win.blit(night, (0, 0))
    pipe.draw(win)
    pype.draw(win)
    bird.draw(win)
    pg.font.init()
    font = pg.font.SysFont('Comic Sans MS', 50)
    scoreTXT = font.render(str(bird.score), False, (250, 200, 20))
    win.blit(scoreTXT, (200, 0))
    pg.display.update()
def scores(bird, win, pipe, pype):
    pg.display.update()
    if bird.pos[0] > pipe.pos[0]:
        if bird.pos[0] + 20 == pipe.pos[0] + 50:
            if bird.pos[1] > pipe.h:
                if bird.pos[1] < pype.pos[1]:
                    bird.score += 1


##DEFINE GOAL OUTPUTS##
def nn_inputs_outputs(bird, pipe, pype, count):
    # define goal outputs
    nn = NN()
    if count < 1:
        out = False
        outputs = np.array(([out]), dtype=float)
        count += 1
        prevYB = bird.distYB
        prevYA = bird.distYA
    else:
        outputs = np.array(([out]), dtype=float)
        out = ouputs
    # inputs
    inputs = np.array((bird.get_data(pipe, pype)), dtype=float)
    inputs = inputs / np.max(inputs, axis=0)
    X = np.split(inputs, [3])[0]
    o = nn.forward(X)
    if bird.passed:
        if bird.pos[1] < pipe.h:
            out = False
        else:
            out = True
        nn.train(X, out)
        bird.trains += 1
    # define actual outputs
    return outputs, o, bird.get_data(pipe, pype)


##INIT GAME##
def games():
    global win
    # define visuals
    pipe = Pipe()
    pype = Pype()
    bird = Bird()
    eps = 100
    done = False
    win.blit(night, (0, 0))
    while not done:
        #win.blit(night, (0, 0))
        pg.display.flip()
        for event in pg.event.get():
            if event.type == pg.QUIT:
                pg.quit()
        bird.cols(pipe, pype)
        if pipe.pos[0] < -50:
            if bird.passed:
                pipe.reset()
                pype.reset(pipe)
        if not bird.is_alive:
            reset(bird, pipe, pype)
            bird.is_alive = True
            print("After " + str(bird.trains) + " iterations of training")
            print("Score = " + str(bird.score))
            print("******************************************************************")
            bird.score = 0
        scores(bird, win, pipe, pype)
        draw(win, bird, pipe, pype, night)
        outputs, out, inputs = nn_inputs_outputs(bird, pipe, pype, count)
        moves(bird, out)
        # begin movements
        pipe.move()
        pype.move(pipe)
        #print(inputs)
        #print(outputs)
        #print(out
        #win.blit(night, (0, 0))
        pg.display.flip()

games()

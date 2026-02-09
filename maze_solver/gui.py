"""GUI module."""
import pygame
class MazeGUI:
    def __init__(self, env, cell=40):
        self.env=env
        self.cell=cell
        pygame.init()
        self.screen=pygame.display.set_mode((env.size*cell, env.size*cell))
        pygame.display.set_caption("Maze Solver")
        self.clock=pygame.time.Clock()
    def close(self):
        pygame.quit()

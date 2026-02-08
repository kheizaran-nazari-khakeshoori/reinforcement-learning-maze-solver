"""GUI module."""
import pygame
class MazeGUI:
    def __init__(self, env):
        self.env=env
    def close(self):
        pygame.quit()

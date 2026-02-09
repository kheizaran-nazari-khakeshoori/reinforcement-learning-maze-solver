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
    def draw_grid(self):
        for r in range(self.env.size):
            for c in range(self.env.size):
                rect=pygame.Rect(c*self.cell, r*self.cell, self.cell, self.cell)
                pygame.draw.rect(self.screen, (255,255,255), rect)
                pygame.draw.rect(self.screen, (0,0,0), rect, 1)
    def close(self):
        pygame.quit()

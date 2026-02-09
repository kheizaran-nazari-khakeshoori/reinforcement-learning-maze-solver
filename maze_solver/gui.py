"""GUI module."""
import pygame
COLORS={"wall":(80,80,80),"trap":(220,50,50),"goal":(255,215,0),"agent":(50,100,255),"start":(150,255,150)}
class MazeGUI:
    def __init__(self, env, cell=40):
        self.env=env
        self.cell=cell
        pygame.init()
        self.screen=pygame.display.set_mode((env.size*cell, env.size*cell))
        pygame.display.set_caption("Maze Solver")
        self.clock=pygame.time.Clock()
    def _color(self, pos):
        if pos in self.env.walls: return COLORS["wall"]
        if pos in getattr(self.env, "traps", set()): return COLORS["trap"]
        if pos==self.env.goal: return COLORS["goal"]
        if pos==self.env.start: return COLORS["start"]
        return (255,255,255)
    def draw_grid(self):
        for r in range(self.env.size):
            for c in range(self.env.size):
                rect=pygame.Rect(c*self.cell, r*self.cell, self.cell, self.cell)
                pygame.draw.rect(self.screen, self._color((r,c)), rect)
                pygame.draw.rect(self.screen, (0,0,0), rect, 1)
    def close(self):
        pygame.quit()

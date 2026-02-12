"""GUI module."""
import pygame
COLORS={"wall":(80,80,80),"trap":(220,50,50),"goal":(255,215,0),"agent":(50,100,255),"start":(150,255,150)}
class MazeGUI:
    def __init__(self, env, cell=None):
        self.env=env
        if cell is None:
            cell=max(20, 600//env.size)
        self.cell=cell
        pygame.init()
        w=env.size*cell; h=env.size*cell
        self.screen=pygame.display.set_mode((w, h))
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
    def draw_agent(self):
        r,c=self.env.agent_pos
        x=c*self.cell+self.cell//2; y=r*self.cell+self.cell//2
        pygame.draw.circle(self.screen, COLORS["agent"], (x,y), self.cell//3)
    def handle_events(self):
        for e in pygame.event.get():
            if e.type==pygame.QUIT:
                return False
            if e.type==pygame.KEYDOWN:
                if e.key==pygame.K_r:
                    self.env.reset()
                if e.key==pygame.K_SPACE:
                    self.env.step(self.env.action_space.sample())
        return True
    def loop_once(self):
        self.screen.fill((255,255,255))
        self.draw_grid(); self.draw_agent()
        pygame.display.flip()
        self.clock.tick(60)
        return self.handle_events()
    def animate_step(self, old, new, steps=6):
        for i in range(steps):
            t=(i+1)/steps
            r=int(old[0]+(new[0]-old[0])*t); c=int(old[1]+(new[1]-old[1])*t)
            self.env.agent_pos=(r,c)
            self.screen.fill((255,255,255))
            self.draw_grid(); self.draw_agent()
            pygame.display.flip()
            self.clock.tick(60)
            if not self.handle_events():
                break
    def close(self):
        pygame.quit()

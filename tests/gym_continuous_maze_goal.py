from typing import Dict, Optional, Tuple

import gymnasium as gym
import numpy as np
from gymnasium import spaces
from pygame import gfxdraw
import os

import pygame
from pygame.math import Vector2
from pygame.image import load
from pygame.transform import rotozoom

from gymnasium_robotics.core import GoalEnv

BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
RED = (255, 0, 0)


ex_walls = np.array(
    [[[-12.0, -12.0], [-12.0, 12.0]], [[-10.0, 8.0], [-10.0, 10.0]], [[-10.0, 0.0], [-10.0, 6.0]],
        [[-10.0, -4.0], [-10.0, -2.0]],[[-10.0, -10.0], [-10.0, -6.0]],[[-8.0, 4.0], [-8.0, 8.0]],
        [[-8.0, -4.0], [-8.0, 0.0]],[[-8.0, -8.0], [-8.0, -6.0]], [[-6.0, 8.0], [-6.0, 10.0]],
        [[-6.0, 4.0], [-6.0, 6.0]],[[-6.0, 0.0], [-6.0, 2.0]], [[-6.0, -6.0], [-6.0, -4.0]],
        [[-4.0, 2.0], [-4.0, 8.0]],[[-4.0, -2.0], [-4.0, 0.0]], [[-4.0, -10.0], [-4.0, -6.0]],
        [[-2.0, 8.0], [-2.0, 12.0]],[[-2.0, 2.0], [-2.0, 6.0]],[[-2.0, -4.0], [-2.0, -2.0]],
        [[0.0, 6.0], [0.0, 12.0]],[[0.0, 2.0], [0.0, 4.0]], [[0.0, -8.0], [0.0, -6.0]],
        [[2.0, 8.0], [2.0, 10.0]],[[2.0, -8.0], [2.0, 6.0]], [[4.0, 10.0], [4.0, 12.0]],[[4.0, 4.0], [4.0, 6.0]],
        [[4.0, 0.0], [4.0, 2.0]], [[4.0, -6.0], [4.0, -2.0]], [[4.0, -10.0], [4.0, -8.0]],[[6.0, 10.0], [6.0, 12.0]],
        [[6.0, 6.0], [6.0, 8.0]],[[6.0, 0.0], [6.0, 2.0]], [[6.0, -8.0], [6.0, -6.0]],[[8.0, 10.0], [8.0, 12.0]],
        [[8.0, 4.0], [8.0, 6.0]],[[8.0, -4.0], [8.0, 2.0]], [[8.0, -10.0], [8.0, -8.0]], [[10.0, 10.0], [10.0, 12.0]],
        [[10.0, 4.0], [10.0, 8.0]],[[10.0, -2.0], [10.0, 0.0]], [[12.0, -12.0], [12.0, 12.0]], [[-12.0, 12.0], [12.0, 12.0]],
        [[-12.0, 10.0], [-10.0, 10.0]], [[-8.0, 10.0], [-6.0, 10.0]],[[-4.0, 10.0], [-2.0, 10.0]],[[2.0, 10.0], [4.0, 10.0]],
        [[-8.0, 8.0], [-2.0, 8.0]],[[2.0, 8.0], [8.0, 8.0]],[[-10.0, 6.0], [-8.0, 6.0]], [[-6.0, 6.0], [-2.0, 6.0]],
        [[6.0, 6.0], [8.0, 6.0]], [[0.0, 4.0], [6.0, 4.0]], [[-10.0, 2.0], [-6.0, 2.0]],[[-2.0, 2.0], [0.0, 2.0]],
        [[8.0, 2.0], [10.0, 2.0]], [[-4.0, 0.0], [-2.0, 0.0]],[[2.0, 0.0], [4.0, 0.0]],[[6.0, 0.0], [8.0, 0.0]],
        [[-6.0, -2.0], [2.0, -2.0]], [[4.0, -2.0], [10.0, -2.0]],[[-12.0, -4.0], [-8.0, -4.0]],
        [[-4.0, -4.0], [-2.0, -4.0]],[[0.0, -4.0], [6.0, -4.0]],[[8.0, -4.0], [10.0, -4.0]], [[-8.0, -6.0], [-6.0, -6.0]],[[-2.0, -6.0], [0.0, -6.0]],
        [[6.0, -6.0], [10.0, -6.0]], [[-12.0, -8.0], [-6.0, -8.0]], [[-2.0, -8.0], [2.0, -8.0]],[[4.0, -8.0], [6.0, -8.0]],
        [[8.0, -8.0], [10.0, -8.0]], [[-10.0, -10.0], [-8.0, -10.0]], [[-4.0, -10.0], [4.0, -10.0]], [[-12.0, -12.0], [12.0, -12.0]]]) + 12
ex_wall_size = 24


class Wall():
    '''
    Stationary w x h pixel obstacle in the environment, through which the agent cannot pass
    '''
    def __init__(self, segament, scale, thickness=0.1, name='wall', degrees=0):
        self.rotated = True
        self.direction = degrees
        
        x1 = segament[0,0]
        x2 = segament[0,1]
        y1 = segament[1,0]
        y2 = segament[1,1]
        min_dist = np.min([np.abs(x2-x1), np.abs(y2-y1)])
        if min_dist==0:
            thickness=thickness
        else:
            thickness = min_dist
        if x1 < x2: # horiz
            self.h = scale * thickness
            self.w = scale * (x2 - x1)
            self.position =  np.array([scale * x1, scale*y1]) #left, top
        elif x1 > x2: # horiz
            self.h = scale * thickness
            self.w = scale * (x1 - x2)
            self.position =  np.array([scale*x2, scale*y1])
        elif y1 < y2:
            self.w = scale * thickness
            self.h = scale * (y2 - y1)
            self.position = np.array([scale*x1, scale*y1])
        elif y1 > y2:
            self.w = scale * thickness
            self.h = scale * (y1 - y2)
            self.position = scale * np.array([scale*x1, scale*y2])
        self.image=None # set to None to distinguish from Goal and Agent
        self.color = np.array([100, 100, 100])
        self.rectangle = True

        self.can_overlap = False

        self.name = name

        ## Create the wall object
        self.image = pygame.Surface([self.w, self.h])
        self.image.fill(self.color)
        
        self.rect = pygame.Rect(self.position[0], self.position[1], self.w, self.h)

    def draw(self, surface):
        pygame.draw.rect(surface, self.color, self.rect)
        
class World:
    ''' 
    Create the world the agent will move around in
    '''

    def __init__(self, width, height):
        ## set minimum size of box (in pixels)
        assert width >= 100
        assert height >= 100

        self.width = width
        self.height = height

        self.contents = {}

    def add_obj(self, object):
        #####
        self.contents[object.name] = object  

class Agent():
    '''
    agent for checking collisions
    '''   
    def __init__(self, position, scale, radius=0.1, name='agent'):
        
        ## Create an agent sprite for collision checking 
        #agent_sprite = load_sprite(agent.name)
        ## Turn the agent sprite by the selected angle
        #rotated_agent = pygame.transform.rotate(agent_sprite, new_dir)
        self.rect = pygame.Rect(scale * position[0], scale * position[1], scale* radius, scale*radius)
        self.collision = False 
        self.name = name
        self.pos = scale * position
        self.rectangle = False
        ball = pygame.Surface((5,5))
        self.color=np.array([255, 0, 0])
        ball.fill(self.color)

        self.rect = ball.get_rect(center = self.pos)
        self.radius = radius * scale
        
    def draw(self, surface):
        # Draw the player
        pygame.draw.circle(surface, self.color, self.pos, self.radius)


class Dummy_Agent():
    '''
    Dummy agent for checking collisions
    '''   
    def __init__(self, position, scale, radius=0.1):
        self.rect = pygame.Rect(scale * position[0], scale * position[1], scale*radius, scale*radius)
        


class GoalContinuousMaze(GoalEnv):
    """Continuous maze goal environment."""
    metadata = {
        "render_modes": ["human", "rgb_array"],
        "render_fps": 10,
    }

    def __init__(self, maze_file=None, max_steps=1e4, res=500,
                 render_mode='human',reward_type='sparse', **kwargs) -> None:
        self.screen = None
        self.isopen = True
        self.window=None
        self.render_mode = render_mode
        self.reward_type = reward_type
        self.all_pos = []
        self.max_steps = max_steps
        self.action_space = spaces.Box(-1.0, 1.0, (2,), dtype="float32")
        if maze_file is None:
            self.walls = ex_walls
            size = ex_wall_size
        else:
            size = float(maze_file.split('_')[1].split('x')[0])
            if not os.path.exists(maze_file):
                dir_path = os.path.dirname(os.path.abspath(__file__))
                rel_path = os.path.join(dir_path, "maze_samples", maze_file)
                if os.path.exists(rel_path):
                    maze_file = rel_path
                else:
                    raise FileExistsError("Cannot find %s." % maze_file)
                self.walls = np.load(maze_file, allow_pickle=False, fix_imports=True)
                
        self.res = res
        self.scale = res/(size + 0.1)
        self.width = res # right now assuming a square area
        self.height = res
        self.goal_dist = 0.45
        
        self.goal = np.array([size-1.0,size-1.0])
        self.observation_space = spaces.Dict(
            dict(
                desired_goal=spaces.Box(0, size, (2,), dtype="float64"),
                achieved_goal=spaces.Box(0, size, (2,), dtype="float64"),
                observation=spaces.Box(0, size, (2,), dtype="float64"),
            )
        )
        
            
    def _gen_world(self, width, height):
        # Create an empty grid
        self.world = World(width, height)

        ## Place agent in the world
        self.agent = Agent(self.pos, self.scale)
        self.world.add_obj(self.agent)
        
        for i in range(self.walls.shape[0]):
            self.world.add_obj(Wall(self.walls[i], scale=self.scale, name=f'wall_{i}', degrees=0))

    def reset(self, seed=0, **kwargs):
        self.seed = seed
        self.pos = 0.5 * np.ones(2)
        # self.all_pos = []
        
        self._gen_world(self.width, self.height)
        
        self.num_steps = 0
        obs_dict = {"desired_goal": self.goal.copy(),
                    "achieved_goal": self.pos.copy(),
                    "observation": self.pos.copy()}
        return obs_dict, {}
    
    def _check_collision(self, agent_pos, action):
        '''Check the agent's path for collisions. '''
        
        ## Get list of obstacles names
        object_names = list(self.world.contents.keys())[2:]
                
        ## If there are obstacles in the world
        if len(object_names) != 0:
            ## Add them to a list of rect objects
            wall_list = []
            
            for name in object_names:
                if self.world.contents[str(name)].rectangle == True:
                    wall_list.append(self.world.contents[str(name)].rect)
               
            
            locations = np.linspace(agent_pos, agent_pos + action, 100)
            for i in range(locations.shape[0]):
                agent_sprite = Dummy_Agent(locations[i], self.scale)
                index = agent_sprite.rect.collidelist(wall_list) 
            
                if index != -1:
                    self.agent.collision = True
                    break    
                
                agent_pos = locations[i]
        
        # pos_in_bounds = self._check_bounds(agent_pos)

        return agent_pos
    
    def _check_bounds(self, agent_pos):
        '''
        Check whether the agent's path takes it outside world bounds. 
        If so, stop the agent at the world bound.
        '''
        width = self.world.width 
        height = self.world.height 
        agent_pos = self.scale * agent_pos
        ## Check whether new position is within world bounds.
        ## If agent is outside bounds, move back and set collision to True
        if agent_pos[0] > width:
            agent_pos[0] = width
            self.agent.collision = True
        if agent_pos[0] < 0:
            agent_pos[0] = 0
            self.agent.collision = True
        if agent_pos[1] > height: 
            agent_pos[1] = height
            self.agent.collision = True
        if agent_pos[1] < 0:
            agent_pos[1] = 0
            self.agent.collision = True

        return agent_pos/self.scale
    
    def _get_game_objects(self):
        '''Fetch a list of objects in the environment'''
        game_object=[]
        for obj in self.world.contents:
            game_object.append(self.world.contents[obj])

        return game_object
    
    def _update_agent(self, position):
        self.agent.pos = self.scale * position
        self.agent.rect = pygame.Rect(self.scale * position[0], self.scale * position[1],
                                      self.scale * self.agent.radius, self.scale * self.agent.radius)

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, Dict]:
        self.num_steps += 1
        self.pos = self._check_collision(self.pos.copy(), action)
        self._update_agent(self.pos)
        self.agent.collision = False
        # new_pos = self.pos + action
        # for wall in self.walls:
        #     intersection = get_intersect(wall[0], wall[1], self.pos, new_pos)
        #     if intersection is not None:
        #         new_pos = self.pos
        # self.all_pos.append(self.pos.copy())
        dist = np.linalg.norm(self.pos - self.goal) 
        
        if dist <= self.goal_dist:
            reward = 1.0 - 0.9*(self.num_steps/self.max_steps)
            terminated = True
        else:
            reward = 0
            terminated = False
            
        if self.reward_type == "dense":
            reward = np.exp(-dist + np.log(1 - 0.9*(self.num_steps/self.max_steps)))
        
        if self.num_steps >= self.max_steps:
            truncated = True
        else:
            truncated = False
        obs_dict = {"desired_goal": self.goal.copy(),
                    "achieved_goal": self.pos.copy(),
                    "observation": self.pos.copy()}
        return obs_dict, reward, terminated, truncated,  {}
    
    def compute_reward(self, achieved_goal, desired_goal, info):
        dist = np.linalg.norm(self.pos - self.goal) 
        
        if self.reward_type == "sparse":
            if dist <= self.goal_dist:
                reward = 1.0 - 0.9*(self.num_steps/self.max_steps)
            else:
                reward = 0
        elif self.reward_type == "dense":
            reward = np.exp(-dist + np.log(1 - 0.9*(self.num_steps/self.max_steps)))
            
        return reward
        
            
    def compute_terminated(self, achieved_goal, desired_goal, info):
        if np.linalg.norm(self.pos - self.goal) <= self.goal_dist:
            terminated = True
        else:
            terminated = False
        return terminated
            
    def compute_truncated(self, achieved_goal, desired_goal, info):
        if self.num_steps >= self.max_steps:
            truncated = True
        else:
            truncated = False
        return truncated

    def render(self, **kwargs):
        ## call pygame
        pygame.init()

        ## Screen dimensions in pixels
        self.screen_width = self.width
        self.screen_height = self.height

        ## for "human" mode, show the render window on screen
        if self.render_mode == "human":
            pygame.display.init()
            self.window = pygame.display.set_mode(
                (self.screen_width, self.screen_height), flags = pygame.SHOWN
            )
        elif self.render_mode == "rgb_array":
            self.window = pygame.display.set_mode(
                (self.screen_width, self.screen_height), flags = pygame.HIDDEN
            )

        self.clock = pygame.time.Clock() ## use clock

        ## Set screen background 
        self.background = pygame.Surface((self.screen_width, self.screen_height))
        self.background.fill(np.array([255, 255, 255]))
        self.window.blit(self.background, (0, 0))

        ## fetch game objects
        game_object = self._get_game_objects()
        
    
        for i in range(len(game_object)):
            obj = game_object[i]
            obj.draw(self.window)


        if self.render_mode == "human":
            pygame.event.pump()
            self.clock.tick(self.metadata["render_fps"])
            pygame.display.flip()

        elif self.render_mode == "rgb_array":
            img = np.transpose(
                np.array(pygame.surfarray.pixels3d(self.window)), axes=(1, 0, 2)
                )

            return img 

    # def render(self, mode: str = "human"):
    #     screen_dim = 500
    #     bound = 13
    #     scale = screen_dim / (bound * 2)
    #     offset = screen_dim // 2

    #     if self.screen is None:
    #         pygame.init()
    #         try:
    #             pygame.display.list_modes()
    #         except:
    #             import os

    #             os.environ["SDL_VIDEODRIVER"] = "dummy"

    #         self.screen = pygame.display.set_mode((screen_dim, screen_dim))
    #     self.surf = pygame.Surface((screen_dim, screen_dim))
    #     self.surf.fill(BLACK)
    #     for pos in self.all_pos:
    #         x, y = pos * scale + offset
    #         gfxdraw.filled_circle(self.surf, int(x), int(y), 1, RED)

    #     for wall in self.walls:
    #         x1, y1 = wall[0] * scale + offset
    #         x2, y2 = wall[1] * scale + offset
    #         gfxdraw.line(self.surf, int(x1), int(y1), int(x2), int(y2), WHITE)

    #     self.surf = pygame.transform.flip(self.surf, flip_x=False, flip_y=True)
    #     self.screen.blit(self.surf, (0, 0))
    #     if mode == "human":
    #         pygame.display.flip()
    #     elif mode == "rgb_array":
    #         return np.transpose(np.array(pygame.surfarray.pixels3d(self.screen)), axes=(1, 0, 2))
    #     else:
    #         return self.isopen

    def close(self):
        if self.window is not None:
            pygame.quit()
            self.isopen = False



class GoalContinuousMaze5x5(GoalContinuousMaze):
    def __init__(self, enable_render=False,render_mode=None):
        super(GoalContinuousMaze5x5, self).__init__(maze_file="maze2d_5x5.npy",render_mode=render_mode)


class GoalContinuousMaze6x6(GoalContinuousMaze):
    def __init__(self, enable_render=False,render_mode=None):
        super(GoalContinuousMaze6x6, self).__init__(maze_file="maze2d_6x6.npy",render_mode=render_mode)

class GoalContinuousMaze7x7(GoalContinuousMaze):
    def __init__(self, enable_render=False,render_mode=None):
        super(GoalContinuousMaze7x7, self).__init__(maze_file="maze2d_7x7.npy",render_mode=render_mode)

class GoalContinuousMaze8x8(GoalContinuousMaze):
    def __init__(self, enable_render=False,render_mode=None):
        super(GoalContinuousMaze8x8, self).__init__(maze_file="maze2d_8x8.npy",render_mode=render_mode)

class GoalContinuousMaze9x9(GoalContinuousMaze):
    def __init__(self, enable_render=False,render_mode=None):
        super(GoalContinuousMaze9x9, self).__init__(maze_file="maze2d_9x9.npy",render_mode=render_mode)

class GoalContinuousMaze10x10(GoalContinuousMaze):
    def __init__(self, enable_render=False,render_mode=None):
        super(GoalContinuousMaze10x10, self).__init__(maze_file="maze2d_10x10.npy",render_mode=render_mode)


class GoalContinuousMaze11x11(GoalContinuousMaze):
    def __init__(self, enable_render=False,render_mode=None):
        super(GoalContinuousMaze11x11, self).__init__(maze_file="maze2d_11x11.npy",render_mode=render_mode)

class GoalContinuousMaze12x12(GoalContinuousMaze):
    def __init__(self, enable_render=False,render_mode=None):
        super(GoalContinuousMaze12x12, self).__init__(maze_file="maze2d_12x12.npy",render_mode=render_mode)


class GoalContinuousMaze13x13(GoalContinuousMaze):
    def __init__(self, enable_render=False,render_mode=None):
        super(GoalContinuousMaze13x13, self).__init__(maze_file="maze2d_13x13.npy",render_mode=render_mode)

class GoalContinuousMaze14x14(GoalContinuousMaze):
    def __init__(self, enable_render=False,render_mode=None):
        super(GoalContinuousMaze14x14, self).__init__(maze_file="maze2d_14x14.npy",render_mode=render_mode)

class GoalContinuousMaze15x15(GoalContinuousMaze):
    def __init__(self, enable_render=False,render_mode=None):
        super(GoalContinuousMaze15x15, self).__init__(maze_file="maze2d_15x15.npy",render_mode=render_mode)

class GoalContinuousMaze20x20(GoalContinuousMaze):
    def __init__(self, enable_render=False,render_mode=None):
        super(GoalContinuousMaze20x20, self).__init__(maze_file="maze2d_20x20.npy",render_mode=render_mode)
        
        
class GoalContinuousMazeBlocks5x5(GoalContinuousMaze):
	def __init__(self, enable_render=False,render_mode=None):
		super(GoalContinuousMazeBlocks5x5, self).__init__(maze_file='block2d_5x5.npy',render_mode=render_mode)

class GoalContinuousMazeBlocks6x6(GoalContinuousMaze):
	def __init__(self, enable_render=False,render_mode=None):
		super(GoalContinuousMazeBlocks6x6, self).__init__(maze_file='block2d_6x6.npy',render_mode=render_mode)

class GoalContinuousMazeBlocks7x7(GoalContinuousMaze):
	def __init__(self, enable_render=False,render_mode=None):
		super(GoalContinuousMazeBlocks7x7, self).__init__(maze_file='block2d_7x7.npy',render_mode=render_mode)

class GoalContinuousMazeBlocks8x8(GoalContinuousMaze):
	def __init__(self, enable_render=False,render_mode=None):
		super(GoalContinuousMazeBlocks8x8, self).__init__(maze_file='block2d_8x8.npy',render_mode=render_mode)

class GoalContinuousMazeBlocks9x9(GoalContinuousMaze):
	def __init__(self, enable_render=False,render_mode=None):
		super(GoalContinuousMazeBlocks9x9, self).__init__(maze_file='block2d_9x9.npy',render_mode=render_mode)

class GoalContinuousMazeBlocks10x10(GoalContinuousMaze):
	def __init__(self, enable_render=False,render_mode=None):
		super(GoalContinuousMazeBlocks10x10, self).__init__(maze_file='block2d_10x10.npy',render_mode=render_mode)

class GoalContinuousMazeBlocks11x11(GoalContinuousMaze):
	def __init__(self, enable_render=False,render_mode=None):
		super(GoalContinuousMazeBlocks11x11, self).__init__(maze_file='block2d_11x11.npy',render_mode=render_mode)

class GoalContinuousMazeBlocks12x12(GoalContinuousMaze):
	def __init__(self, enable_render=False,render_mode=None):
		super(GoalContinuousMazeBlocks12x12, self).__init__(maze_file='block2d_12x12.npy',render_mode=render_mode)

class GoalContinuousMazeBlocks13x13(GoalContinuousMaze):
	def __init__(self, enable_render=False,render_mode=None):
		super(GoalContinuousMazeBlocks13x13, self).__init__(maze_file='block2d_13x13.npy',render_mode=render_mode)

class GoalContinuousMazeBlocks14x14(GoalContinuousMaze):
	def __init__(self, enable_render=False,render_mode=None):
		super(GoalContinuousMazeBlocks14x14, self).__init__(maze_file='block2d_14x14.npy',render_mode=render_mode)

class GoalContinuousMazeBlocks15x15(GoalContinuousMaze):
	def __init__(self, enable_render=False,render_mode=None):
		super(GoalContinuousMazeBlocks15x15, self).__init__(maze_file='block2d_15x15.npy',render_mode=render_mode)

class GoalContinuousMazeBlocks16x16(GoalContinuousMaze):
	def __init__(self, enable_render=False,render_mode=None):
		super(GoalContinuousMazeBlocks16x16, self).__init__(maze_file='block2d_16x16.npy',render_mode=render_mode)

class GoalContinuousMazeBlocks17x17(GoalContinuousMaze):
	def __init__(self, enable_render=False,render_mode=None):
		super(GoalContinuousMazeBlocks17x17, self).__init__(maze_file='block2d_17x17.npy',render_mode=render_mode)

class GoalContinuousMazeBlocks18x18(GoalContinuousMaze):
	def __init__(self, enable_render=False,render_mode=None):
		super(GoalContinuousMazeBlocks18x18, self).__init__(maze_file='block2d_18x18.npy',render_mode=render_mode)

class GoalContinuousMazeBlocks19x19(GoalContinuousMaze):
	def __init__(self, enable_render=False,render_mode=None):
		super(GoalContinuousMazeBlocks19x19, self).__init__(maze_file='block2d_19x19.npy',render_mode=render_mode)


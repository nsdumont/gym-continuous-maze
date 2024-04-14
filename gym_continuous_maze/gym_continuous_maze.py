from typing import Dict, Optional, Tuple

import gymnasium as gym
import numpy as np
import pygame
from gymnasium import spaces
from pygame import gfxdraw
import os

BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
RED = (255, 0, 0)


def get_intersect(A: np.ndarray, B: np.ndarray, C: np.ndarray, D: np.ndarray) -> Optional[np.ndarray]:
    """
    Get the intersection of [A, B] and [C, D]. Return False if segment don't cross.

    :param A: Point of the first segment
    :param B: Point of the first segment
    :param C: Point of the second segment
    :param D: Point of the second segment
    :return: The intersection if any, otherwise None.
    """
    det = (B[0] - A[0]) * (C[1] - D[1]) - (C[0] - D[0]) * (B[1] - A[1])
    if det == 0:
        # Parallel
        return None
    else:
        t1 = ((C[0] - A[0]) * (C[1] - D[1]) - (C[0] - D[0]) * (C[1] - A[1])) / det
        t2 = ((B[0] - A[0]) * (C[1] - A[1]) - (C[0] - A[0]) * (B[1] - A[1])) / det
        if t1 > 1 or t1 < 0 or t2 > 1 or t2 < 0:
            # not intersect
            return None
        else:
            xi = A[0] + t1 * (B[0] - A[0])
            yi = A[1] + t1 * (B[1] - A[1])
            return np.array([xi, yi])


class ContinuousMaze(gym.Env):
    """Continuous maze environment."""

    def __init__(self, maze_file=None, max_steps=1e4, **kwargs) -> None:
        self.screen = None
        self.isopen = True
        self.all_pos = []
        self.max_steps = max_steps
        self.action_space = spaces.Box(-1, 1, (2,))
        if maze_file is None:
            self.observation_space = spaces.Box(-12, 12, (2,))
            self.goal = np.arry([11,11])
            self.walls = np.array(
                [
                    [[-12.0, -12.0], [-12.0, 12.0]],
                    [[-10.0, 8.0], [-10.0, 10.0]],
                    [[-10.0, 0.0], [-10.0, 6.0]],
                    [[-10.0, -4.0], [-10.0, -2.0]],
                    [[-10.0, -10.0], [-10.0, -6.0]],
                    [[-8.0, 4.0], [-8.0, 8.0]],
                    [[-8.0, -4.0], [-8.0, 0.0]],
                    [[-8.0, -8.0], [-8.0, -6.0]],
                    [[-6.0, 8.0], [-6.0, 10.0]],
                    [[-6.0, 4.0], [-6.0, 6.0]],
                    [[-6.0, 0.0], [-6.0, 2.0]],
                    [[-6.0, -6.0], [-6.0, -4.0]],
                    [[-4.0, 2.0], [-4.0, 8.0]],
                    [[-4.0, -2.0], [-4.0, 0.0]],
                    [[-4.0, -10.0], [-4.0, -6.0]],
                    [[-2.0, 8.0], [-2.0, 12.0]],
                    [[-2.0, 2.0], [-2.0, 6.0]],
                    [[-2.0, -4.0], [-2.0, -2.0]],
                    [[0.0, 6.0], [0.0, 12.0]],
                    [[0.0, 2.0], [0.0, 4.0]],
                    [[0.0, -8.0], [0.0, -6.0]],
                    [[2.0, 8.0], [2.0, 10.0]],
                    [[2.0, -8.0], [2.0, 6.0]],
                    [[4.0, 10.0], [4.0, 12.0]],
                    [[4.0, 4.0], [4.0, 6.0]],
                    [[4.0, 0.0], [4.0, 2.0]],
                    [[4.0, -6.0], [4.0, -2.0]],
                    [[4.0, -10.0], [4.0, -8.0]],
                    [[6.0, 10.0], [6.0, 12.0]],
                    [[6.0, 6.0], [6.0, 8.0]],
                    [[6.0, 0.0], [6.0, 2.0]],
                    [[6.0, -8.0], [6.0, -6.0]],
                    [[8.0, 10.0], [8.0, 12.0]],
                    [[8.0, 4.0], [8.0, 6.0]],
                    [[8.0, -4.0], [8.0, 2.0]],
                    [[8.0, -10.0], [8.0, -8.0]],
                    [[10.0, 10.0], [10.0, 12.0]],
                    [[10.0, 4.0], [10.0, 8.0]],
                    [[10.0, -2.0], [10.0, 0.0]],
                    [[12.0, -12.0], [12.0, 12.0]],
                    [[-12.0, 12.0], [12.0, 12.0]],
                    [[-12.0, 10.0], [-10.0, 10.0]],
                    [[-8.0, 10.0], [-6.0, 10.0]],
                    [[-4.0, 10.0], [-2.0, 10.0]],
                    [[2.0, 10.0], [4.0, 10.0]],
                    [[-8.0, 8.0], [-2.0, 8.0]],
                    [[2.0, 8.0], [8.0, 8.0]],
                    [[-10.0, 6.0], [-8.0, 6.0]],
                    [[-6.0, 6.0], [-2.0, 6.0]],
                    [[6.0, 6.0], [8.0, 6.0]],
                    [[0.0, 4.0], [6.0, 4.0]],
                    [[-10.0, 2.0], [-6.0, 2.0]],
                    [[-2.0, 2.0], [0.0, 2.0]],
                    [[8.0, 2.0], [10.0, 2.0]],
                    [[-4.0, 0.0], [-2.0, 0.0]],
                    [[2.0, 0.0], [4.0, 0.0]],
                    [[6.0, 0.0], [8.0, 0.0]],
                    [[-6.0, -2.0], [2.0, -2.0]],
                    [[4.0, -2.0], [10.0, -2.0]],
                    [[-12.0, -4.0], [-8.0, -4.0]],
                    [[-4.0, -4.0], [-2.0, -4.0]],
                    [[0.0, -4.0], [6.0, -4.0]],
                    [[8.0, -4.0], [10.0, -4.0]],
                    [[-8.0, -6.0], [-6.0, -6.0]],
                    [[-2.0, -6.0], [0.0, -6.0]],
                    [[6.0, -6.0], [10.0, -6.0]],
                    [[-12.0, -8.0], [-6.0, -8.0]],
                    [[-2.0, -8.0], [2.0, -8.0]],
                    [[4.0, -8.0], [6.0, -8.0]],
                    [[8.0, -8.0], [10.0, -8.0]],
                    [[-10.0, -10.0], [-8.0, -10.0]],
                    [[-4.0, -10.0], [4.0, -10.0]],
                    [[-12.0, -12.0], [12.0, -12.0]],
                ]
            )
        else:
            size = float(maze_file.split('_')[1].split('x')[0])
            if not os.path.exists(maze_file):
                dir_path = os.path.dirname(os.path.abspath(__file__))
                rel_path = os.path.join(dir_path, "maze_samples", maze_file)
                if os.path.exists(rel_path):
                    maze_file = rel_path
                else:
                    raise FileExistsError("Cannot find %s." % maze_file)
            self.observation_space = spaces.Box(-0.5, size - 0.5, (2,))
            self.goal = np.array([size-1,size-1])
            self.walls = np.load(maze_file, allow_pickle=False, fix_imports=True)

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, Dict]:
        self.num_steps += 1
        new_pos = self.pos + action
        for wall in self.walls:
            intersection = get_intersect(wall[0], wall[1], self.pos, new_pos)
            if intersection is not None:
                new_pos = self.pos
        self.pos = new_pos
        self.all_pos.append(self.pos.copy())
        if np.linalg.norm(self.pos - self.goal) < 0.5:
            reward = 1 - 0.9*(self.num_steps/self.max_steps)
            terminated = True
        else:
            reward = 0
            terminated = False
        if self.num_steps >= self.max_steps:
            truncated = True
        else:
            truncated = False
        return self.pos.copy(), reward, terminated, truncated,  {}

    def reset(self, seed=0):
        self.seed = seed
        self.pos = np.zeros(2)
        self.all_pos.append(self.pos.copy())
        self.num_steps = 0
        return self.pos.copy(), {}

    def render(self, mode: str = "human"):
        screen_dim = 500
        bound = 13
        scale = screen_dim / (bound * 2)
        offset = screen_dim // 2

        if self.screen is None:
            pygame.init()
            try:
                pygame.display.list_modes()
            except:
                import os

                os.environ["SDL_VIDEODRIVER"] = "dummy"

            self.screen = pygame.display.set_mode((screen_dim, screen_dim))
        self.surf = pygame.Surface((screen_dim, screen_dim))
        self.surf.fill(BLACK)
        for pos in self.all_pos:
            x, y = pos * scale + offset
            gfxdraw.filled_circle(self.surf, int(x), int(y), 1, RED)

        for wall in self.walls:
            x1, y1 = wall[0] * scale + offset
            x2, y2 = wall[1] * scale + offset
            gfxdraw.line(self.surf, int(x1), int(y1), int(x2), int(y2), WHITE)

        self.surf = pygame.transform.flip(self.surf, flip_x=False, flip_y=True)
        self.screen.blit(self.surf, (0, 0))
        if mode == "human":
            pygame.display.flip()
        elif mode == "rgb_array":
            return np.transpose(np.array(pygame.surfarray.pixels3d(self.screen)), axes=(1, 0, 2))
        else:
            return self.isopen

    def close(self):
        if self.screen is not None:
            pygame.quit()
            self.isopen = False



class ContinuousMaze5x5(ContinuousMaze):
    def __init__(self, enable_render=False,render_mode=None):
        super(ContinuousMaze5x5, self).__init__(maze_file="maze2d_5x5.npy")


class ContinuousMaze6x6(ContinuousMaze):
    def __init__(self, enable_render=False,render_mode=None):
        super(ContinuousMaze6x6, self).__init__(maze_file="maze2d_6x6.npy")

class ContinuousMaze7x7(ContinuousMaze):
    def __init__(self, enable_render=False,render_mode=None):
        super(ContinuousMaze7x7, self).__init__(maze_file="maze2d_7x7.npy")

class ContinuousMaze8x8(ContinuousMaze):
    def __init__(self, enable_render=False,render_mode=None):
        super(ContinuousMaze8x8, self).__init__(maze_file="maze2d_8x8.npy")

class ContinuousMaze9x9(ContinuousMaze):
    def __init__(self, enable_render=False,render_mode=None):
        super(ContinuousMaze9x9, self).__init__(maze_file="maze2d_9x9.npy")

class ContinuousMaze10x10(ContinuousMaze):
    def __init__(self, enable_render=False,render_mode=None):
        super(ContinuousMaze10x10, self).__init__(maze_file="maze2d_10x10.npy")


class ContinuousMaze11x11(ContinuousMaze):
    def __init__(self, enable_render=False,render_mode=None):
        super(ContinuousMaze11x11, self).__init__(maze_file="maze2d_11x11.npy")

class ContinuousMaze12x12(ContinuousMaze):
    def __init__(self, enable_render=False,render_mode=None):
        super(ContinuousMaze12x12, self).__init__(maze_file="maze2d_12x12.npy")


class ContinuousMaze13x13(ContinuousMaze):
    def __init__(self, enable_render=False,render_mode=None):
        super(ContinuousMaze13x13, self).__init__(maze_file="maze2d_13x13.npy")

class ContinuousMaze14x14(ContinuousMaze):
    def __init__(self, enable_render=False,render_mode=None):
        super(ContinuousMaze14x14, self).__init__(maze_file="maze2d_14x14.npy")

class ContinuousMaze15x15(ContinuousMaze):
    def __init__(self, enable_render=False,render_mode=None):
        super(ContinuousMaze15x15, self).__init__(maze_file="maze2d_15x15.npy")

class ContinuousMaze20x20(ContinuousMaze):
    def __init__(self, enable_render=False,render_mode=None):
        super(ContinuousMaze20x20, self).__init__(maze_file="maze2d_20x20.npy")

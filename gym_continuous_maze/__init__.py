from gymnasium.envs.registration import register

register(
    id="ContinuousMaze-v0",
    entry_point="gym_continuous_maze.gym_continuous_maze:ContinuousMaze",
    max_episode_steps=100,
)

register(
    id="ContinuousMaze-5x5-v0",
    entry_point="gym_continuous_maze.gym_continuous_maze:ContinuousMaze5x5",
    max_episode_steps=500,
)

register(
    id="ContinuousMaze-6x6-v0",
    entry_point="gym_continuous_maze.gym_continuous_maze:ContinuousMaze6x6",
    max_episode_steps=600,
)

register(
    id="ContinuousMaze-7x7-v0",
    entry_point="gym_continuous_maze.gym_continuous_maze:ContinuousMaze7x7",
    max_episode_steps=700,
)

register(
    id="ContinuousMaze-8x8-v0",
    entry_point="gym_continuous_maze.gym_continuous_maze:ContinuousMaze8x8",
    max_episode_steps=800,
)

register(
    id="ContinuousMaze-9x9-v0",
    entry_point="gym_continuous_maze.gym_continuous_maze:ContinuousMaze9x9",
    max_episode_steps=900,
)

register(
    id="ContinuousMaze-10x10-v0",
    entry_point="gym_continuous_maze.gym_continuous_maze:ContinuousMaze10x10",
    max_episode_steps=1000,
)

register(
    id="ContinuousMaze-11x11-v0",
    entry_point="gym_continuous_maze.gym_continuous_maze:ContinuousMaze11x11",
    max_episode_steps=1000,
)

register(
    id="ContinuousMaze-12x12-v0",
    entry_point="gym_continuous_maze.gym_continuous_maze:ContinuousMaze12x12",
    max_episode_steps=1000,
)

register(
    id="ContinuousMaze-13x13-v0",
    entry_point="gym_continuous_maze.gym_continuous_maze:ContinuousMaze13x13",
    max_episode_steps=1000,
)

register(
    id="ContinuousMaze-14x14-v0",
    entry_point="gym_continuous_maze.gym_continuous_maze:ContinuousMaze14x14",
    max_episode_steps=1000,
)

register(
    id="ContinuousMaze-15x15-v0",
    entry_point="gym_continuous_maze.gym_continuous_maze:ContinuousMaze15x15",
    max_episode_steps=1000,
)

register(
    id="ContinuousMaze-20x20-v0",
    entry_point="gym_continuous_maze.gym_continuous_maze:ContinuousMaze20x20",
    max_episode_steps=1000,
)

######

register(
	id='ContinuousMazeBlocks-5x5-v0',
	entry_point='gym_continuous_maze.gym_continuous_maze:ContinuousMazeBlocks5x5',
	max_episode_steps=500,
)

register(
	id='ContinuousMazeBlocks-6x6-v0',
	entry_point='gym_continuous_maze.gym_continuous_maze:ContinuousMazeBlocks6x6',
	max_episode_steps=720,
)

register(
	id='ContinuousMazeBlocks-7x7-v0',
	entry_point='gym_continuous_maze.gym_continuous_maze:ContinuousMazeBlocks7x7',
	max_episode_steps=980,
)

register(
	id='ContinuousMazeBlocks-8x8-v0',
	entry_point='gym_continuous_maze.gym_continuous_maze:ContinuousMazeBlocks8x8',
	max_episode_steps=1280,
)

register(
	id='ContinuousMazeBlocks-9x9-v0',
	entry_point='gym_continuous_maze.gym_continuous_maze:ContinuousMazeBlocks9x9',
	max_episode_steps=1620,
)

register(
	id='ContinuousMazeBlocks-10x10-v0',
	entry_point='gym_continuous_maze.gym_continuous_maze:ContinuousMazeBlocks10x10',
	max_episode_steps=2000,
)

register(
	id='ContinuousMazeBlocks-11x11-v0',
	entry_point='gym_continuous_maze.gym_continuous_maze:ContinuousMazeBlocks11x11',
	max_episode_steps=2420,
)

register(
	id='ContinuousMazeBlocks-12x12-v0',
	entry_point='gym_continuous_maze.gym_continuous_maze:ContinuousMazeBlocks12x12',
	max_episode_steps=2880,
)

register(
	id='ContinuousMazeBlocks-13x13-v0',
	entry_point='gym_continuous_maze.gym_continuous_maze:ContinuousMazeBlocks13x13',
	max_episode_steps=3380,
)

register(
	id='ContinuousMazeBlocks-14x14-v0',
	entry_point='gym_continuous_maze.gym_continuous_maze:ContinuousMazeBlocks14x14',
	max_episode_steps=3920,
)

register(
	id='ContinuousMazeBlocks-15x15-v0',
	entry_point='gym_continuous_maze.gym_continuous_maze:ContinuousMazeBlocks15x15',
	max_episode_steps=4500,
)

register(
	id='ContinuousMazeBlocks-16x16-v0',
	entry_point='gym_continuous_maze.gym_continuous_maze:ContinuousMazeBlocks16x16',
	max_episode_steps=5120,
)

register(
	id='ContinuousMazeBlocks-17x17-v0',
	entry_point='gym_continuous_maze.gym_continuous_maze:ContinuousMazeBlocks17x17',
	max_episode_steps=5780,
)

register(
	id='ContinuousMazeBlocks-18x18-v0',
	entry_point='gym_continuous_maze.gym_continuous_maze:ContinuousMazeBlocks18x18',
	max_episode_steps=6480,
)

register(
	id='ContinuousMazeBlocks-19x19-v0',
	entry_point='gym_continuous_maze.gym_continuous_maze:ContinuousMazeBlocks19x19',
	max_episode_steps=7220,
)

############### goalenvs
register(
    id="GoalContinuousMaze-v0",
    entry_point="gym_continuous_maze.gym_continuous_maze_goal:GoalContinuousMaze",
    max_episode_steps=100,
)

register(
    id="GoalContinuousMaze-5x5-v0",
    entry_point="gym_continuous_maze.gym_continuous_maze_goal:GoalContinuousMaze5x5",
    max_episode_steps=500,
)

register(
    id="GoalContinuousMaze-6x6-v0",
    entry_point="gym_continuous_maze.gym_continuous_maze_goal:GoalContinuousMaze6x6",
    max_episode_steps=600,
)

register(
    id="GoalContinuousMaze-7x7-v0",
    entry_point="gym_continuous_maze.gym_continuous_maze_goal:GoalContinuousMaze7x7",
    max_episode_steps=700,
)

register(
    id="GoalContinuousMaze-8x8-v0",
    entry_point="gym_continuous_maze.gym_continuous_maze_goal:GoalContinuousMaze8x8",
    max_episode_steps=800,
)

register(
    id="GoalContinuousMaze-9x9-v0",
    entry_point="gym_continuous_maze.gym_continuous_maze_goal:GoalContinuousMaze9x9",
    max_episode_steps=900,
)

register(
    id="GoalContinuousMaze-10x10-v0",
    entry_point="gym_continuous_maze.gym_continuous_maze_goal:GoalContinuousMaze10x10",
    max_episode_steps=1000,
)

register(
    id="GoalContinuousMaze-11x11-v0",
    entry_point="gym_continuous_maze.gym_continuous_maze_goal:GoalContinuousMaze11x11",
    max_episode_steps=1000,
)

register(
    id="GoalContinuousMaze-12x12-v0",
    entry_point="gym_continuous_maze.gym_continuous_maze_goal:GoalContinuousMaze12x12",
    max_episode_steps=1000,
)

register(
    id="GoalContinuousMaze-13x13-v0",
    entry_point="gym_continuous_maze.gym_continuous_maze_goal:GoalContinuousMaze13x13",
    max_episode_steps=1000,
)

register(
    id="GoalContinuousMaze-14x14-v0",
    entry_point="gym_continuous_maze.gym_continuous_maze_goal:GoalContinuousMaze14x14",
    max_episode_steps=1000,
)

register(
    id="GoalContinuousMaze-15x15-v0",
    entry_point="gym_continuous_maze.gym_continuous_maze_goal:GoalContinuousMaze15x15",
    max_episode_steps=1000,
)

register(
    id="GoalContinuousMaze-20x20-v0",
    entry_point="gym_continuous_maze.gym_continuous_maze_goal:GoalContinuousMaze20x20",
    max_episode_steps=1000,
)

######

# register(
# 	id='GoalContinuousMazeBlocks-5x5-v0',
# 	entry_point='gym_continuous_maze.gym_continuous_maze_goal:GoalContinuousMazeBlocks5x5',
# 	max_episode_steps=500,
# )

# register(
# 	id='GoalContinuousMazeBlocks-6x6-v0',
# 	entry_point='gym_continuous_maze.gym_continuous_maze_goal:GoalContinuousMazeBlocks6x6',
# 	max_episode_steps=720,
# )

# register(
# 	id='GoalContinuousMazeBlocks-7x7-v0',
# 	entry_point='gym_continuous_maze.gym_continuous_maze_goal:GoalContinuousMazeBlocks7x7',
# 	max_episode_steps=980,
# )

# register(
# 	id='GoalContinuousMazeBlocks-8x8-v0',
# 	entry_point='gym_continuous_maze.gym_continuous_maze_goal:GoalContinuousMazeBlocks8x8',
# 	max_episode_steps=1280,
# )

# register(
# 	id='GoalContinuousMazeBlocks-9x9-v0',
# 	entry_point='gym_continuous_maze.gym_continuous_maze_goal:GoalContinuousMazeBlocks9x9',
# 	max_episode_steps=1620,
# )

# register(
# 	id='GoalContinuousMazeBlocks-10x10-v0',
# 	entry_point='gym_continuous_maze.gym_continuous_maze_goal:GoalContinuousMazeBlocks10x10',
# 	max_episode_steps=2000,
# )

# register(
# 	id='GoalContinuousMazeBlocks-11x11-v0',
# 	entry_point='gym_continuous_maze.gym_continuous_maze_goal:GoalContinuousMazeBlocks11x11',
# 	max_episode_steps=2420,
# )

# register(
# 	id='GoalContinuousMazeBlocks-12x12-v0',
# 	entry_point='gym_continuous_maze.gym_continuous_maze_goal:GoalContinuousMazeBlocks12x12',
# 	max_episode_steps=2880,
# )

# register(
# 	id='GoalContinuousMazeBlocks-13x13-v0',
# 	entry_point='gym_continuous_maze.gym_continuous_maze_goal:GoalContinuousMazeBlocks13x13',
# 	max_episode_steps=3380,
# )

# register(
# 	id='GoalContinuousMazeBlocks-14x14-v0',
# 	entry_point='gym_continuous_maze.gym_continuous_maze_goal:GoalContinuousMazeBlocks14x14',
# 	max_episode_steps=3920,
# )

# register(
# 	id='GoalContinuousMazeBlocks-15x15-v0',
# 	entry_point='gym_continuous_maze.gym_continuous_maze_goal:GoalContinuousMazeBlocks15x15',
# 	max_episode_steps=4500,
# )

# register(
# 	id='GoalContinuousMazeBlocks-16x16-v0',
# 	entry_point='gym_continuous_maze.gym_continuous_maze_goal:GoalContinuousMazeBlocks16x16',
# 	max_episode_steps=5120,
# )

# register(
# 	id='GoalContinuousMazeBlocks-17x17-v0',
# 	entry_point='gym_continuous_maze.gym_continuous_maze_goal:GoalContinuousMazeBlocks17x17',
# 	max_episode_steps=5780,
# )

# register(
# 	id='GoalContinuousMazeBlocks-18x18-v0',
# 	entry_point='gym_continuous_maze.gym_continuous_maze_goal:GoalContinuousMazeBlocks18x18',
# 	max_episode_steps=6480,
# )

# register(
# 	id='GoalContinuousMazeBlocks-19x19-v0',
# 	entry_point='gym_continuous_maze.gym_continuous_maze_goal:GoalContinuousMazeBlocks19x19',
# 	max_episode_steps=7220,
# )
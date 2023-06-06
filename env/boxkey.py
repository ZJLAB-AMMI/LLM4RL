import gymnasium as gym
from minigrid.core.constants import COLOR_NAMES
from minigrid.core.grid import Grid
from minigrid.core.mission import MissionSpace
from minigrid.core.world_object import Door, Key, Wall, Box
from .historicalobs import HistoricalObsEnv

    
class BoxKeyEnv(HistoricalObsEnv):
    def __init__(
        self,
        minRoomSize: int = 5,
        maxRoomSize: int = 10,
        agent_view_size: int = 7,
        max_steps: int | None = None,
        **kwargs,
    ):

        self.minRoomSize = minRoomSize
        self.maxRoomSize = maxRoomSize

        mission_space = MissionSpace(mission_func=self._gen_mission)

        if max_steps is None:
            max_steps = maxRoomSize ** 2

        super().__init__(
            mission_space=mission_space,
            width=maxRoomSize,
            height=maxRoomSize,
            agent_view_size=agent_view_size,
            max_steps=max_steps,
            **kwargs,
        )

    @staticmethod
    def _gen_mission():
        return "open the door"

    def _gen_grid(self, width, height):

        # Create the grid
        self.grid = Grid(width, height)

        # Choose the room size randomly
        sizeX = self._rand_int(self.minRoomSize, self.maxRoomSize + 1)
        sizeY = self._rand_int(self.minRoomSize, self.maxRoomSize + 1)
        topX, topY = 0, 0

        # Draw the top and bottom walls
        wall = Wall()
        for i in range(0, sizeX):
            self.grid.set(topX + i, topY, wall)
            self.grid.set(topX + i, topY + sizeY - 1, wall)

        # Draw the left and right walls
        for j in range(0, sizeY):
            self.grid.set(topX, topY + j, wall)
            self.grid.set(topX + sizeX - 1, topY + j, wall)

        # Pick which wall to place the out door on
        wallSet = {0, 1, 2, 3}
        exitDoorWall = self._rand_elem(sorted(wallSet))

        # Pick the exit door position
        # Exit on right wall
        if exitDoorWall == 0:
            exitDoorPos = (topX + sizeX - 1, topY + self._rand_int(1, sizeY - 1))
        # Exit on south wall
        elif exitDoorWall == 1:
            exitDoorPos = (topX + self._rand_int(1, sizeX - 1), topY + sizeY - 1)
        # Exit on left wall
        elif exitDoorWall == 2:
            exitDoorPos = (topX, topY + self._rand_int(1, sizeY - 1))
        # Exit on north wall
        elif exitDoorWall == 3:
            exitDoorPos = (topX + self._rand_int(1, sizeX - 1), topY)
        else:
            assert False
          
        # Place the door
        doorColor = self._rand_elem(sorted(set(COLOR_NAMES)))
        exitDoor = Door(doorColor, is_locked=True)
        self.door = exitDoor
        self.grid.set(exitDoorPos[0], exitDoorPos[1], exitDoor)

        # Randomize the starting agent position and direction
        self.place_agent((topX, topY), (sizeX, sizeY), rand_dir=True)

        # Randomize the box and key position
        key = Key(doorColor)
        box = Box(doorColor, contains=key)
        self.place_obj(box, (topX, topY), (sizeX, sizeY))

        self.mission = "open the door"
          
    def step(self, action):
        obs, reward, terminated, truncated, info = super().step(action)

        if action == self.actions.toggle:
            if self.door.is_open:
                reward = self._reward()
                terminated = True

        return obs, reward, terminated, truncated, info
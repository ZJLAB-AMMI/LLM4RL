import numpy as np
from gymnasium import spaces
from minigrid.core.grid import Grid
from minigrid.core.world_object import Wall, WorldObj
from minigrid.minigrid_env import MiniGridEnv
from minigrid.core.mission import MissionSpace
from minigrid.core.constants import TILE_PIXELS
from minigrid.utils.rendering import fill_coords, point_in_rect, point_in_circle

class Unseen(WorldObj):
    def __init__(self):
        super().__init__("unseen", "grey")

    def render(self, img):
        fill_coords(img, point_in_rect(0, 1, 0, 1), np.array([200, 200, 200]))
        # fill_coords(img, point_in_rect(0, 1, 0.45, 0.55), np.array([200, 200, 200]))
        
class Footprint(WorldObj):
    def __init__(self):
        super().__init__("empty", "red")

    def render(self, img):
        fill_coords(img, point_in_circle(0.5, 0.5, 0.15), np.array([225, 0, 0]))    

class FlexibleGrid(Grid):
    def __init__(self, width: int, height: int):
        assert width >= 1
        assert height >= 1

        self.width: int = width
        self.height: int = height

        self.grid: list[WorldObj | None] = [None] * (width * height)
        self.mask: np.ndarray = np.zeros(shape=(self.width, self.height), dtype=bool)

    def looking_vertical(self, i, direction, out_of_bound):
        for j in range(0, self.height - 1): # north to south
            if not self.mask[i, j]:
                continue

            cell = self.get(i, j)
            if cell and not cell.see_behind():
                continue

            # print(i, j)
            self.mask[i, j + 1] = True
            if not out_of_bound:
                self.mask[i + direction, j] = True
                self.mask[i + direction, j + 1] = True
            # print(self.mask.T)

        for j in reversed(range(1, self.height)): # south to north
            if not self.mask[i, j]:
                continue

            cell = self.get(i, j)
            if cell and not cell.see_behind():
                continue

            # print(i, j)
            self.mask[i, j - 1] = True
            if not out_of_bound:
                self.mask[i + direction, j] = True
                self.mask[i + direction, j - 1] = True
            # print(self.mask.T)
        
    def looking_horizontal(self, j, direction, out_of_bound):
        for i in range(0, self.width - 1): # west to east
            if not self.mask[i, j]:
                continue

            cell = self.get(i, j)
            if cell and not cell.see_behind():
                continue

            # print(i, j)
            self.mask[i + 1, j] = True
            if not out_of_bound:
                self.mask[i, j + direction] = True
                self.mask[i + 1, j + direction] = True
            # print(self.mask.T)

        for i in reversed(range(1, self.width)): # east to west
            if not self.mask[i, j]:
                continue

            cell = self.get(i, j)
            if cell and not cell.see_behind():
                continue

            # print(i, j)
            self.mask[i - 1, j] = True
            if not out_of_bound:
                self.mask[i, j + direction] = True
                self.mask[i - 1, j + direction] = True
            # print(self.mask.T)
            
    def process_vis(self, agent_pos, agent_dir):
        self.mask[agent_pos[0], agent_pos[1]] = True

        if agent_dir == 0: # east
            for i in range(0, self.width):
                out_of_bound = i == self.width - 1
                self.looking_vertical(i, 1, out_of_bound)
                    
        elif agent_dir == 2: # west
            for i in reversed(range(0, self.width)):
                out_of_bound = i == 0
                self.looking_vertical(i, -1, out_of_bound)
                    
        elif agent_dir == 1: # south
            for j in range(0, self.height):
                out_of_bound = j == self.height - 1
                self.looking_horizontal(j, 1, out_of_bound)
                    
        elif agent_dir == 3: # north
            for j in reversed(range(0, self.height)):
                out_of_bound = j == 0
                self.looking_horizontal(j, -1, out_of_bound)
        return self.mask
    
    
class HistoricalObsEnv(MiniGridEnv):
    def __init__(
        self,
        mission_space: MissionSpace,
        width: int | None = None,
        height: int | None = None,
        max_steps: int = 100,
        **kwargs,
    ):
        super().__init__(
                mission_space=mission_space,
                width=width,
                height=height,
                max_steps=max_steps,
                **kwargs,
            )
        
        image_observation_space = spaces.Box(
            low=0,
            high=255,
            shape=(self.width, self.height, 4),
            dtype="uint8",
        )
        mission_space = self.observation_space["mission"]
        self.observation_space = spaces.Dict(
            {
                "image": image_observation_space,
                "mission": mission_space,
            }
        )
        
        self.mask = np.zeros(shape=(self.width, self.height), dtype=bool)
        
    def reset(self, seed=None):
        obs, info = super().reset(seed=seed)
        self.mask = np.zeros(shape=(self.width, self.height), dtype=bool)
        topX, topY, botX, botY = self.get_view_exts(agent_view_size=None, clip=True)
        vis_grid = self.slice_grid(topX, topY, botX - topX, botY - topY)
        if not self.see_through_walls:
            vis_mask = vis_grid.process_vis((self.agent_pos[0] - topX, self.agent_pos[1] - topY), self.agent_dir)
        else:
            vis_mask = np.ones(shape=(botX - topX, botY - topY), dtype=bool)
            
        self.mask[topX:botX, topY:botY] += vis_mask
        return obs, info
                
    def slice_grid(self, topX, topY, width, height) -> FlexibleGrid:
        """
        Get a subset of the grid
        """

        vis_grid = FlexibleGrid(width, height)

        for j in range(0, height):
            for i in range(0, width):
                x = topX + i
                y = topY + j

                if 0 <= x < self.grid.width and 0 <= y < self.grid.height:
                    v = self.grid.get(x, y)
                else:
                    v = Wall()

                vis_grid.set(i, j, v)

        return vis_grid
                
    def get_view_exts(self, agent_view_size=None, clip=False):
        """
        Get the extents of the square set of tiles visible to the agent
        if agent_view_size is None, use self.agent_view_size
        """
        
        topX, topY, botX, botY = super().get_view_exts(agent_view_size)
        if clip:
            topX = max(0, topX)
            topY = max(0, topY)
            botX = min(botX, self.width)
            botY = min(botY, self.height)
        return topX, topY, botX, botY

    def gen_hist_obs_grid(self, agent_view_size=None):
        topX, topY, botX, botY = self.get_view_exts(agent_view_size, clip=True)
        grid = self.grid.copy()
        vis_grid = self.slice_grid(topX, topY, botX - topX, botY - topY)
        if not self.see_through_walls:
            vis_mask = vis_grid.process_vis((self.agent_pos[0] - topX, self.agent_pos[1] - topY), self.agent_dir)
        else:
            vis_mask = np.ones(shape=(botX - topX, botY - topY), dtype=bool)
            
        self.mask[topX:botX, topY:botY] += vis_mask

        # Make it so the agent sees what it's carrying
        if self.carrying:
            grid.set(*self.agent_pos, self.carrying)
        else:
            grid.set(*self.agent_pos, None)

        return grid

    def gen_obs(self):
        grid = self.gen_hist_obs_grid()

        image = grid.encode(self.mask)
        
        agent_pos_dir = np.zeros((self.width, self.height), dtype="uint8") + 4
        agent_pos_dir[self.agent_pos] = self.agent_dir

        obs = {"image": np.concatenate((image, agent_pos_dir[:,:,None]), axis=2), "mission": self.mission}
        return obs

    def get_full_render(self, highlight, tile_size):
        grid = self.gen_hist_obs_grid()
        img = grid.render(
            tile_size,
            agent_pos=self.agent_pos,
            agent_dir=self.agent_dir,
            highlight_mask=self.mask,
        )
        return img
    
    def get_mask_render(self, path_mask=None, tile_size=TILE_PIXELS):
        grid = self.gen_hist_obs_grid()
        unseen_mask = np.ones(shape=(self.width, self.height), dtype=bool) ^ self.mask
        
        if path_mask is None:
            path_mask = np.zeros(shape=(self.width, self.height), dtype=bool)

        # Compute the total grid size
        width_px = self.width * tile_size
        height_px = self.height * tile_size

        img = np.zeros(shape=(height_px, width_px, 3), dtype=np.uint8)

        # Render the grid
        for j in range(0, self.height):
            for i in range(0, self.width):     
                if unseen_mask[i, j]:
                    cell = Unseen()
                elif path_mask[i, j]:
                    cell = Footprint()
                else:
                    cell = grid.get(i, j)

                agent_here = np.array_equal(self.agent_pos, (i, j))
                tile_img = Grid.render_tile(
                    cell,
                    agent_dir=self.agent_dir if agent_here else None,
                    highlight=False,
                    tile_size=tile_size,
                )

                ymin = j * tile_size
                ymax = (j + 1) * tile_size
                xmin = i * tile_size
                xmax = (i + 1) * tile_size
                img[ymin:ymax, xmin:xmax, :] = tile_img

        return img
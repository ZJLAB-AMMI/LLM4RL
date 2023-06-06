from .historicalobs import *
from .singleroom import *
from .boxkey import *
from .monster import *
from .randomboxkey import *
from .keydistraction import *
from .Game import *
from .Game_RL import *
from .Game_multi_heads import *
gym.envs.register(
    id='MiniGrid-SimpleDoorKey-Min5-Max10-View3',
    entry_point='env.singleroom:SingleRoomEnv',
    kwargs={'minRoomSize' : 5, \
            'maxRoomSize' : 10, \
            'agent_view_size' : 3},
)

gym.envs.register(
    id='MiniGrid-KeyInBox-Min5-Max10-View3',
    entry_point='env.boxkey:BoxKeyEnv',
    kwargs={'minRoomSize' : 5, \
            'maxRoomSize' : 10, \
            'agent_view_size' : 3},
)

gym.envs.register(
    id='MiniGrid-RandomBoxKey-Min5-Max10-View3',
    entry_point='env.randomboxkey:RandomBoxKeyEnv',
    kwargs={'minRoomSize' : 5, \
            'maxRoomSize' : 10, \
            'agent_view_size' : 3},
)

gym.envs.register(
    id='MiniGrid-ColoredDoorKey-Min5-Max10-View3',
    entry_point='env.keydistraction:KeyDistractionEnv',
    kwargs={'minRoomSize' : 5, \
            'maxRoomSize' : 10, \
            'minNumKeys' : 2, \
            'maxNumKeys' : 2, \
            'agent_view_size' : 3},
)
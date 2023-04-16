# available versions

from gym.envs.registration import register

register(
        id='binpacking_posco-v0',
        entry_point='binpacking_posco.envs:binpacking_posco_v0',
)

register(
        id='binpacking_poscoppo-v1',
        entry_point='binpacking_posco.envs:binpacking_poscoppo_v1',
)

register(
        id='binpacking_myposco-v1',
        entry_point='binpacking_posco.envs:binpacking_myposco_v1',
)

register(
        id='binpacking_poscoppo-v2',
        entry_point='binpacking_posco.envs:binpacking_poscoppo_v2',
)

register(
        id='binpacking_posco-v00',
        entry_point='binpacking_posco.envs:binpacking_posco_v00',
)

register(
        id='binpacking_poscopredict-v00',
        entry_point='binpacking_posco.envs:binpacking_poscopredict_v00',
)

register(
        id='binpacking_posco-v1',
        entry_point='binpacking_posco.envs:binpacking_posco_v1',
)

register(
        id='binpacking_poscopred-v1',
        entry_point='binpacking_posco.envs:binpacking_poscopred_v1',
)

register(
        id='binpacking_posco-v2',
        entry_point='binpacking_posco.envs:binpacking_posco_v2',
)

register(
        id='binpacking_myposco-v2',
        entry_point='binpacking_posco.envs:binpacking_myposco_v2',
)

register(
        id='binpacking_posco-v3',
        entry_point='binpacking_posco.envs:binpacking_posco_v3',
)

register(
        id='binpacking_posco-v4',
        entry_point='binpacking_posco.envs:binpacking_posco_v4',
)

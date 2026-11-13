"""Explicit ablation contracts. The released Revision 4 remains the default."""
from copy import deepcopy
import json
from pathlib import Path

MODES = ('vanilla_s1', 's1_astar_v4', 's1_guided_flow_2d', 's1_guided_flow_25d_full')


def resolve_revision5_config(config=None, mode=None):
    base = json.loads((Path(__file__).resolve().parents[1] / 'configs/go2_terrain_research.json').read_text())
    for key, value in (config or {}).items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            base[key].update(deepcopy(value))
        else:
            base[key] = deepcopy(value)
    selected = mode or base.get('mode', 's1_astar_v4')
    if selected not in MODES:
        raise ValueError(f'Unknown ablation mode: {selected}')
    base['mode'] = selected
    base['use_guidance'] = selected.startswith('s1_guided_flow_')
    base['use_elevation'] = selected == 's1_guided_flow_25d_full'
    base['use_odom_gate'] = base['use_guidance']
    base.setdefault('tracking', dict(min_frames=4, direction_cosine=.80, min_net_m=.25,
                                     min_speed_mps=.20, max_speed_mps=3.5, max_gap_s=.35))
    base.setdefault('control', dict(stop_edge_gap_m=.45, max_retreat_m=.19, ttc_s=4.,
                                    grass_margin_m=.35, stair_speed_mps=.35,
                                    allow_stair_proxy=False, support_follow_rate_mps=.4))
    base.setdefault('cost_weights', dict(terrain=18., grass=24., dyn_ttc=24., retreat=100.,
                                         pullover=3., goal=1., smooth=.2))
    if not 0 < base['control']['max_retreat_m'] <= .25:
        raise ValueError('Retreat budget must be in (0, .25] metres')
    if not .40 <= base['control']['stop_edge_gap_m'] <= .50:
        raise ValueError('Edge stopping clearance must be in [.40, .50] metres')
    if base['sampling']['candidates'] < 1 or base['sampling']['euler_steps'] < 1:
        raise ValueError('Sampling counts must be positive')
    return base

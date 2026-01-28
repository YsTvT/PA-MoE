import json
import numpy as np
from alfworld.gen.graph import graph_obj
from alfworld.gen.utils.game_util import get_objects_with_name_and_prop
from alfworld.env.reward import get_action

class BaseTask(object):

    def __init__(self, traj, env, args, reward_type='sparse', max_episode_length=2000):
        self.traj = traj
        self.env = env
        self.args = args
        self.task_type = self.traj['task_type']
        self.max_episode_length = max_episode_length
        self.reward_type = reward_type
        self.step_num = 0
        self.num_subgoals = len(self.traj['plan']['high_pddl']) - 1  # ignore end noop
        self.goal_finished = False

        self.goal_idx = 0
        self.finished = -1

        self.gt_graph = None
        self.load_nav_graph()

        self.reward_config = None
        self.load_reward_config(args.reward_config)
        self.strict = 'strict' in reward_type

        self.prev_state = self.env.last_event

    def load_reward_config(self, config_file):
        
        with open(config_file, 'r') as rc:
            reward_config = json.load(rc)
        self.reward_config = reward_config

    def load_nav_graph(self):
        
        floor_plan = self.traj['scene']['floor_plan']
        scene_num = self.traj['scene']['scene_num']
        self.gt_graph = graph_obj.Graph(use_gt=True, construct_graph=True, scene_id=scene_num)

    def goal_satisfied(self, state):
        
        raise NotImplementedError

    def transition_reward(self, state):
        
        reward = 0

        if self.goal_finished:
            done = True
            return reward, done

        expert_plan = self.traj['plan']['high_pddl']
        action_type = expert_plan[self.goal_idx]['planner_action']['action']

        if "dense" in self.reward_type:
            action = get_action(action_type, self.gt_graph, self.env, self.reward_config, self.strict)
            sg_reward, sg_done = action.get_reward(state, self.prev_state, expert_plan, self.goal_idx)
            reward += sg_reward
            if sg_done:
                self.finished += 1
                if self.goal_idx + 1 < self.num_subgoals:
                    self.goal_idx += 1

        goal_finished = self.goal_satisfied(state)
        if goal_finished:
            reward += self.reward_config['Generic']['goal_reward']
            self.goal_finished = True

        if "success" in self.reward_type and self.env.last_event.metadata['lastActionSuccess']:
            reward += self.reward_config['Generic']['success']

        if "failure" in self.reward_type and not self.env.last_event.metadata['lastActionSuccess']:
            reward += self.reward_config['Generic']['failure']

        if self.step_num > len(self.traj['plan']['low_actions']):
            reward += self.reward_config['Generic']['step_penalty']

        self.prev_state = self.env.last_event

        self.step_num += 1
        done = self.goal_idx >= self.num_subgoals or self.step_num >= self.max_episode_length
        return reward, done

    def reset(self):
        
        self.goal_idx = 0
        self.finished = -1
        self.step_num = 0
        self.goal_finished = False

    def get_subgoal_idx(self):
        return self.finished

    def get_target(self, var):
        
        return self.traj['pddl_params'][var] if self.traj['pddl_params'][var] is not None else None

    def get_targets(self):
        
        targets = {
            'object': self.get_target('object_target'),
            'parent': self.get_target('parent_target'),
            'toggle': self.get_target('toggle_target'),
            'mrecep': self.get_target('mrecep_target'),
        }

        if 'object_sliced' in self.traj['pddl_params'] and self.traj['pddl_params']['object_sliced']:
            targets['object'] += 'Sliced'  # Change, e.g., "Apple" -> "AppleSliced" as pickup target.

        return targets

class PickAndPlaceSimpleTask(BaseTask):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def goal_satisfied(self, state):
        pcs = self.goal_conditions_met(state)
        return pcs[0] == pcs[1]

    def goal_conditions_met(self, state):
        ts = 1
        s = 0

        targets = self.get_targets()
        receptacles = get_objects_with_name_and_prop(targets['parent'], 'receptacle', state.metadata)
        pickupables = get_objects_with_name_and_prop(targets['object'], 'pickupable', state.metadata)

        if 'Sliced' in targets['object']:
            ts += 1
            if len([p for p in pickupables if 'Sliced' in p['objectId']]) >= 1:
                s += 1

        if np.any([np.any([p['objectId'] in r['receptacleObjectIds']
                           for r in receptacles if r['receptacleObjectIds'] is not None])
                   for p in pickupables]):
            s += 1

        return s, ts

    def reset(self):
        super().reset()

class PickTwoObjAndPlaceTask(BaseTask):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def goal_satisfied(self, state):
        pcs = self.goal_conditions_met(state)
        return pcs[0] == pcs[1]

    def goal_conditions_met(self, state):
        ts = 2
        s = 0

        targets = self.get_targets()
        receptacles = get_objects_with_name_and_prop(targets['parent'], 'receptacle', state.metadata)
        pickupables = get_objects_with_name_and_prop(targets['object'], 'pickupable', state.metadata)

        if 'Sliced' in targets['object']:
            ts += 2
            s += min(len([p for p in pickupables if 'Sliced' in p['objectId']]), 2)

        s += min(np.max([sum([1 if r['receptacleObjectIds'] is not None
                                   and p['objectId'] in r['receptacleObjectIds'] else 0
                              for p in pickupables])
                         for r in receptacles]), 2)
        return s, ts

    def reset(self):
        super().reset()

class LookAtObjInLightTask(BaseTask):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def goal_satisfied(self, state):
        pcs = self.goal_conditions_met(state)
        return pcs[0] == pcs[1]

    def goal_conditions_met(self, state):
        ts = 2
        s = 0

        targets = self.get_targets()
        toggleables = get_objects_with_name_and_prop(targets['toggle'], 'toggleable', state.metadata)
        pickupables = get_objects_with_name_and_prop(targets['object'], 'pickupable', state.metadata)
        inventory_objects = state.metadata['inventoryObjects']

        if 'Sliced' in targets['object']:
            ts += 1
            if len([p for p in pickupables if 'Sliced' in p['objectId']]) >= 1:
                s += 1

        if len(inventory_objects) > 0 and inventory_objects[0]['objectId'] in [p['objectId'] for p in pickupables]:
            s += 1
        if np.any([t['isToggled'] and t['visible'] for t in toggleables]):
            s += 1

        return s, ts

    def reset(self):
        super().reset()

class PickHeatThenPlaceInRecepTask(BaseTask):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def goal_satisfied(self, state):
        pcs = self.goal_conditions_met(state)
        return pcs[0] == pcs[1]

    def goal_conditions_met(self, state):
        ts = 3
        s = 0

        targets = self.get_targets()
        receptacles = get_objects_with_name_and_prop(targets['parent'], 'receptacle', state.metadata)
        pickupables = get_objects_with_name_and_prop(targets['object'], 'pickupable', state.metadata)

        if 'Sliced' in targets['object']:
            ts += 1
            if len([p for p in pickupables if 'Sliced' in p['objectId']]) >= 1:
                s += 1

        objs_in_place = [p['objectId'] for p in pickupables for r in receptacles
                         if r['receptacleObjectIds'] is not None and p['objectId'] in r['receptacleObjectIds']]
        objs_heated = [p['objectId'] for p in pickupables if p['objectId'] in self.env.heated_objects]

        if len(objs_in_place) > 0:
            s += 1
        if len(objs_heated) > 0:
            s += 1
        if np.any([obj_id in objs_heated for obj_id in objs_in_place]):
            s += 1

        return s, ts

    def reset(self):
        super().reset()

class PickCoolThenPlaceInRecepTask(BaseTask):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def goal_satisfied(self, state):
        pcs = self.goal_conditions_met(state)
        return pcs[0] == pcs[1]

    def goal_conditions_met(self, state):
        ts = 3
        s = 0

        targets = self.get_targets()
        receptacles = get_objects_with_name_and_prop(targets['parent'], 'receptacle', state.metadata)
        pickupables = get_objects_with_name_and_prop(targets['object'], 'pickupable', state.metadata)

        if 'Sliced' in targets['object']:
            ts += 1
            if len([p for p in pickupables if 'Sliced' in p['objectId']]) >= 1:
                s += 1

        objs_in_place = [p['objectId'] for p in pickupables for r in receptacles
                         if r['receptacleObjectIds'] is not None and p['objectId'] in r['receptacleObjectIds']]
        objs_cooled = [p['objectId'] for p in pickupables if p['objectId'] in self.env.cooled_objects]

        if len(objs_in_place) > 0:
            s += 1
        if len(objs_cooled) > 0:
            s += 1
        if np.any([obj_id in objs_cooled for obj_id in objs_in_place]):
            s += 1

        return s, ts

    def reset(self):
        super().reset()

class PickCleanThenPlaceInRecepTask(BaseTask):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def goal_satisfied(self, state):
        pcs = self.goal_conditions_met(state)
        return pcs[0] == pcs[1]

    def goal_conditions_met(self, state):
        ts = 3
        s = 0

        targets = self.get_targets()
        receptacles = get_objects_with_name_and_prop(targets['parent'], 'receptacle', state.metadata)
        pickupables = get_objects_with_name_and_prop(targets['object'], 'pickupable', state.metadata)

        if 'Sliced' in targets['object']:
            ts += 1
            if len([p for p in pickupables if 'Sliced' in p['objectId']]) >= 1:
                s += 1

        objs_in_place = [p['objectId'] for p in pickupables for r in receptacles
                         if r['receptacleObjectIds'] is not None and p['objectId'] in r['receptacleObjectIds']]
        objs_cleaned = [p['objectId'] for p in pickupables if p['objectId'] in self.env.cleaned_objects]

        if len(objs_in_place) > 0:
            s += 1
        if len(objs_cleaned) > 0:
            s += 1
        if np.any([obj_id in objs_cleaned for obj_id in objs_in_place]):
            s += 1

        return s, ts

    def reset(self):
        super().reset()

class PickAndPlaceWithMovableRecepTask(BaseTask):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def goal_satisfied(self, state):
        pcs = self.goal_conditions_met(state)
        return pcs[0] == pcs[1]

    def goal_conditions_met(self, state):
        ts = 3
        s = 0

        targets = self.get_targets()
        receptacles = get_objects_with_name_and_prop(targets['parent'], 'receptacle', state.metadata)
        pickupables = get_objects_with_name_and_prop(targets['object'], 'pickupable', state.metadata)
        movables = get_objects_with_name_and_prop(targets['mrecep'], 'pickupable', state.metadata)

        if 'Sliced' in targets['object']:
            ts += 1
            if len([p for p in pickupables if 'Sliced' in p['objectId']]) >= 1:
                s += 1

        pickup_in_place = [p for p in pickupables for m in movables
                           if 'receptacleObjectIds' in p and m['receptacleObjectIds'] is not None
                           and p['objectId'] in m['receptacleObjectIds']]
        movable_in_place = [m for m in movables for r in receptacles
                            if 'receptacleObjectIds' in r and r['receptacleObjectIds'] is not None
                            and m['objectId'] in r['receptacleObjectIds']]
        if len(pickup_in_place) > 0:
            s += 1
        if len(movable_in_place) > 0:
            s += 1
        if np.any([np.any([p['objectId'] in m['receptacleObjectIds'] for p in pickupables]) and
                   np.any([r['objectId'] in m['parentReceptacles'] for r in receptacles]) for m in movables
                   if m['parentReceptacles'] is not None and m['receptacleObjectIds'] is not None]):
            s += 1

        return s, ts

    def reset(self):
        super().reset()

def get_task(task_type, traj, env, args, reward_type='sparse', max_episode_length=2000):
    task_class_str = task_type.replace('_', ' ').title().replace(' ', '') + "Task"

    if task_class_str in globals():
        task = globals()[task_class_str]
        return task(traj, env, args, reward_type=reward_type, max_episode_length=max_episode_length)
    else:
        raise Exception("Invalid task_type %s" % task_class_str)
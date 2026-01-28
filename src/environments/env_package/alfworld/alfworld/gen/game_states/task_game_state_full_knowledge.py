import copy
import os
import sys

import numpy as np

import alfworld.gen.constants as constants
from alfworld.gen.game_states.game_state_base import GameStateBase
from alfworld.gen.game_states.planned_game_state import PlannedGameState
from alfworld.gen.game_states.task_game_state import TaskGameState
from alfworld.gen.utils import bb_util
from alfworld.gen.utils import game_util

class TaskGameStateFullKnowledge(TaskGameState):
    def __init__(self, env, seed=None, action_space=None):
        super(TaskGameStateFullKnowledge, self).__init__(env, seed, action_space)

    def update_receptacle_nearest_points(self):
        if self.receptacle_to_point is None:
            object_dict = game_util.get_object_dict(self.env.last_event.metadata)
            object_to_point_reliable_point = self.openable_object_to_point

            points = self.gt_graph.points
            self.receptacle_to_point = {}
            self.point_to_receptacle = {}
            self.object_to_point = {}
            self.point_to_object = {}
            self.in_receptacle_ids = {}
            receptacle_types = constants.RECEPTACLES - constants.MOVABLE_RECEPTACLES_SET
            hold_size = sys.maxsize
            for _ in range(4):
                event = self.env.step({'action': 'RotateRight'})

            if constants.FULL_OBSERVABLE_STATE:
                objects = []
                receptacles = []
                for obj in self.env.last_event.metadata['objects']:
                    cls = obj['objectType']
                    if cls not in constants.OBJECTS_SET:
                        continue
                    if cls in constants.MOVABLE_RECEPTACLES_SET:
                        objects.append(obj)
                        receptacles.append(obj)
                    elif cls in receptacle_types:
                        receptacles.append(obj)
                    else:
                        objects.append(obj)

                for obj in receptacles:
                    cls = obj['objectType']
                    obj_id = obj['objectId']
                    obj_name_s = obj['objectId']

                    box = np.array([[obj['position']['x'], obj['position']['x']],
                                    [obj['position']['z'], obj['position']['z']],
                                    [obj['position']['y'], obj['position']['y']]]) / constants.AGENT_STEP_SIZE

                    known_point = None
                    if obj_name_s in object_to_point_reliable_point:
                        known_point = np.asarray(object_to_point_reliable_point[obj_name_s][:2]) / constants.AGENT_STEP_SIZE

                    coord = self.get_obj_coords(box, cls, obj_id, points, known_point=known_point,
                                                object_type=cls, current_scene=self.scene_num)

                    if (obj['openable'] and not obj['pickupable'] and
                            known_point is None and constants.PRUNE_UNREACHABLE_POINTS):
                        print("WARNING: no precomputed, good opening point for '%s'; will drop openability from planner"
                              % obj_name_s)

                    self.receptacle_to_point[obj_id] = np.array(coord)
                    if coord not in self.point_to_receptacle:
                        self.point_to_receptacle[coord] = []
                    self.point_to_receptacle[coord].append(obj_id)
                    if obj_id not in self.in_receptacle_ids:
                        self.in_receptacle_ids[obj_id] = set()
                    if obj_id not in self.was_in_receptacle_ids:
                        self.was_in_receptacle_ids[obj_id] = set()

                for obj in objects:
                    cls = obj['objectType']
                    obj_id = obj['objectId']

                    box = np.array([[obj['position']['x'], obj['position']['x']],
                                    [obj['position']['z'], obj['position']['z']],
                                    [obj['position']['y'], obj['position']['y']]]) / constants.AGENT_STEP_SIZE

                    coord = self.get_obj_coords(box, cls, obj_id, points,
                                                object_type=cls, current_scene=self.scene_num)
                    if not isinstance(obj['parentReceptacles'], list):
                        obj['parentReceptacles'] = [obj['parentReceptacles']]

                    parent_obj = None
                    for parent in obj['parentReceptacles']:
                        if parent is None:
                            break

                        parent_obj = object_dict[parent]
                        if parent_obj['objectType'] not in constants.RECEPTACLES:
                            continue
                        self.in_receptacle_ids[parent].add(obj_id)
                        self.was_in_receptacle_ids[parent].add(obj_id)

                    self.object_to_point[obj_id] = np.array(coord)
                    self.point_to_object[tuple(self.object_to_point[obj_id].tolist())] = obj_id

                    if obj['toggleable'] and obj['objectType'] in constants.VAL_ACTION_OBJECTS['Toggleable']:
                        if not obj_id in self.toggleable_object_ids:
                            self.toggleable_object_ids.add(obj_id)

                        if obj['isToggled']:
                            if not obj_id in self.on_object_ids:
                                self.on_object_ids.add(obj_id)
                        else:
                            if obj_id in self.on_object_ids:
                                self.on_object_ids.remove(obj_id)

    def get_extra_facts(self):
        object_dict = game_util.get_object_dict(self.env.last_event.metadata)
        object_nearest_point_strs = []
        objects = self.env.last_event.metadata['objects']
        for obj in objects:
            cls = obj['objectType']
            obj_id = obj['objectId']

            if obj_id not in self.object_to_point:
                print(obj_id)
                continue

            nearest_point = self.object_to_point[obj_id]
            object_nearest_point_strs.append('(objectAtLocation %s loc|%d|%d|%d|%d)' % (
               obj_id,
               nearest_point[0], nearest_point[1], nearest_point[2], nearest_point[3]))

        object_at_location_str = '\n        '.join(object_nearest_point_strs)

        holds_str = ''
        if len(self.inventory_ids) > 0:
            holds_str = ('\n        (holdsAny agent1)\n        (holds agent1 %s)' %
                         self.inventory_ids.get_any()[1])

        fillable_receptacles = copy.deepcopy(constants.RECEPTACLES)
        if self.task_target[1] is not None:
            fillable_receptacles.remove(constants.OBJECTS[self.task_target[1]])
        fillable_receptacles.add('Cabinet')
        return object_at_location_str + holds_str

    def get_obj_coords(self, box, cls, obj_name, points, known_point=None,
                       object_type=None, current_scene=None):
        center = ((box[:, 0] + box[:, 1]) / 2)
        point_dists = center[:2][np.newaxis, :] - points
        point_dists_mag = np.sum(np.abs(point_dists), axis=1)
        best_dist = None
        best_point = None
        if known_point is not None:
            best_point = (known_point[0], known_point[1])
            best_dist = center[:2] - known_point
        if best_point is None:
            best_loc = np.argmin(point_dists_mag)
            best_dist = point_dists[best_loc, :]
            best_point = (points[best_loc, 0], points[best_loc, 1])
        best_point = np.array(best_point)
        dist_to_obj = np.sqrt(np.sum(np.square(
            np.array([
                best_point[0],
                best_point[1],
                self.camera_height / constants.AGENT_STEP_SIZE
            ]) - center))) * constants.AGENT_STEP_SIZE
        if abs(best_dist)[0] > abs(best_dist[1]):
            if best_dist[0] > 0:
                rotation = 1
            else:
                rotation = 3
        else:
            if best_dist[1] > 0:
                rotation = 0
            else:
                rotation = 2
        if dist_to_obj < 0.5:
            new_best_point = best_point.copy()
            if rotation == 0:
                new_best_point[1] -= 1
            elif rotation == 1:
                new_best_point[0] -= 1
            elif rotation == 2:
                new_best_point[1] += 1
            else:
                new_best_point[0] += 1
            new_point_dists = (new_best_point[np.newaxis, :] - points).astype(np.int32)
            new_point_dists_mag = np.sum(np.abs(new_point_dists), axis=1)
            new_best_loc = np.argmin(new_point_dists_mag)
            if new_point_dists_mag[new_best_loc] == 0:
                best_loc = new_best_loc
                best_dist = point_dists[best_loc, :]
                best_point = np.array([points[best_loc, 0], points[best_loc, 1]])
        horizontal_dist_to_obj = np.max(np.abs(best_dist)) * constants.AGENT_STEP_SIZE
        obj_height = self.camera_height - box[2, 0] * constants.AGENT_STEP_SIZE
        camera_angle = int(np.clip(np.round(np.arctan2(obj_height, horizontal_dist_to_obj) * (
                180 / np.pi / constants.HORIZON_GRANULARITY)) * constants.HORIZON_GRANULARITY, -30, 60))

        if object_type is not None and current_scene is not None:
            if (object_type, current_scene) in constants.FORCED_HORIZON_OBJS:
                camera_angle = constants.FORCED_HORIZON_OBJS[(object_type, current_scene)]
            elif (object_type, None) in constants.FORCED_HORIZON_OBJS:
                camera_angle = constants.FORCED_HORIZON_OBJS[(object_type, None)]

        coord = (int(best_point[0]),
                 int(best_point[1]),
                 int(rotation),
                 int(camera_angle))
        return coord

    def get_action(self, action_or_ind):
        action = super(PlannedGameState, self).get_action(action_or_ind)[0]
        should_fail = False
        if 'forceVisible' in action:
            forceVisible = action['forceVisible']
        else:
            forceVisible = True
        if action['action'] == 'TeleportLocal':
            point_dists = np.sum(np.abs(self.gt_graph.points - np.array([action['x'], action['z']])), axis=1)
            dist_min = np.argmin(point_dists)
            if point_dists[dist_min] < 0.0001:
                point_x = action['x']
                point_z = action['z']
            else:
                point_x = self.gt_graph.points[dist_min][0]
                point_z = self.gt_graph.points[dist_min][1]
                should_fail = True

            action = {
                'action': 'Teleport',
                'x': point_x * constants.AGENT_STEP_SIZE,
                'y': self.agent_height,
                'z': point_z * constants.AGENT_STEP_SIZE,
                'rotateOnTeleport': True,
                'rotation': action['rotation'],
            }

        elif ((action['action'] == 'OpenObject' or action['action'] == 'CloseObject') and
              ('objectId' not in action)):
            openable = [obj for obj in self.env.last_event.metadata['objects']
                        if (obj['visible'] and obj['openable'] and
                            (obj['isOpen'] == (action['action'] == 'CloseObject')) and
                            obj['objectId'] in self.event.instance_detections2D)]
            if len(openable) > 0:
                boxes = np.array([self.event.instance_detections2D[obj['objectId']]
                                  for obj in openable]) * constants.SCREEN_WIDTH / constants.DETECTION_SCREEN_WIDTH
                boxes_xywh = bb_util.xyxy_to_xywh(boxes.T).T
                mids = boxes_xywh[:, :2]
                dists = np.sqrt(np.sum(np.square(
                    (mids - np.array([constants.SCREEN_WIDTH / 2, constants.SCREEN_HEIGHT / 2]))), axis=1))
                obj_ind = int(np.argmin(dists))
                action['objectId'] = openable[obj_ind]['objectId']
            else:
                should_fail = True
        elif (action['action'] == 'OpenObject' and 'objectId' in action):
            action['forceVisible'] = forceVisible
            should_fail = False
        elif action['action'] == 'CloseObject':
            if len(self.currently_opened_object_ids) > 0:
                action['objectId'] = self.currently_opened_object_ids.get_any()
                action['forceVisible'] = forceVisible
            else:
                should_fail = True
        elif (action['action'] == 'ToggleObject' and 'objectId' in action):
            action['forceVisible'] = forceVisible
            should_fail = False
        elif (action['action'] == 'SliceObject' and 'objectId' in action):
            action['forceVisible'] = forceVisible
            should_fail = False
        elif action['action'] == 'PickupObject':
            should_fail = False
            action['forceVisible'] = forceVisible
        elif action['action'] == 'PutObject':
            if len(self.inventory_ids) == 0:
                should_fail = True
            else:
                action['objectId'] = self.inventory_ids.get_any()
                action['forceVisible'] = forceVisible
                should_fail = False
        elif action['action'] == 'CleanObject':
            action['objectId'] = action['receptacleObjectId']
            action['cleanObjectId'] = action['objectId']
            action['forceVisible'] = forceVisible
            should_fail = False
        elif action['action'] in {'HeatObject', 'CoolObject'}:
            action['objectId'] = action['receptacleObjectId']
            action['forceVisible'] = forceVisible
            should_fail = False
        return action, should_fail

    def process_frame(self, changed_object_id=None):
        self.event = self.env.last_event
        self.pose = game_util.get_pose(self.event)

        self.s_t_orig = self.event.frame
        self.s_t = game_util.imresize(self.event.frame,
                                      (constants.SCREEN_HEIGHT, constants.SCREEN_WIDTH), rescale=False)

        self.s_t_depth = game_util.imresize(self.event.depth_frame,
                                            (constants.SCREEN_HEIGHT, constants.SCREEN_WIDTH), rescale=False)

    def step(self, action_or_ind):

        self.update_receptacle_nearest_points()
        action, should_fail = self.get_action(action_or_ind)

        if 'objectId' in action:
            assert isinstance(action['objectId'], str)
        if 'receptacleObjectId' in action:
            assert isinstance(action['receptacleObjectId'], str)

        if action['action'] == 'PutObject' and self.env.last_event.metadata['lastActionSuccess']:
            object_cls = constants.OBJECT_CLASS_TO_ID[action['objectId'].split('|')[0]]
            receptacle_cls = constants.OBJECT_CLASS_TO_ID[action['receptacleObjectId'].split('|')[0]]
            if object_cls == self.object_target and receptacle_cls == self.parent_target:
                pass

        GameStateBase.step(self, action_or_ind)

        if action['action'] == 'PickupObject':
            if 'receptacleObjectId' in action:
                if action['objectId'] in self.in_receptacle_ids[action['receptacleObjectId']]:
                    self.in_receptacle_ids[action['receptacleObjectId']].remove(action['objectId'])

        elif action['action'] == 'PutObject':
            key = action['receptacleObjectId']
            assert isinstance(key, str)
            if key not in self.in_receptacle_ids:
                self.in_receptacle_ids[key] = set()
            self.in_receptacle_ids[key].add(action['objectId'])

        elif action['action'] == 'CleanObject':
            if self.env.last_event.metadata['lastActionSuccess']:
                self.cleaned_object_ids.add(action['objectId'])

        elif action['action'] == 'HeatObject':
            pass

        elif action['action'] == "ToggleObject":
            pass

        elif action['action'] == 'CoolObject':
            pass

        elif action['action'] == 'SliceObject':
            pass

        visible_objects = self.event.instance_detections2D.keys() if self.event.instance_detections2D != None else []
        for obj in visible_objects:
            obj = game_util.get_object(obj, self.env.last_event.metadata)
            if obj is None:
                continue
            cls = obj['objectType']
            obj_id = obj['objectId']
            if cls not in constants.OBJECTS_SET:
                continue

            if type(obj['parentReceptacles']) is list:
                if len(obj['parentReceptacles']) == 1:
                    parent = obj['parentReceptacles'][0]
                    if len(obj['parentReceptacles']) > 1:
                        print("Warning: selecting first parent of " + str(obj_id) +
                              " from list " + str(obj['parentReceptacles']))
                else:
                    parent = None
            else:
                parent = obj['parentReceptacles']
            if parent is not None and len(parent) > 0:
                fix_basin = False
                if parent.startswith('Sink') and not parent.endswith('Basin'):
                    parent = parent + "|SinkBasin"
                    fix_basin = True
                elif parent.startswith('Bathtub') and not parent.endswith('Basin'):
                    parent = parent + "|BathtubBasin"
                    fix_basin = True

                if fix_basin:
                    try:
                        parent = game_util.get_object(parent, self.env.last_event.metadata)
                    except KeyError:
                        raise Exception('No object named %s in scene %s' % (parent, self.scene_name))
                else:
                    parent = game_util.get_object(parent, self.env.last_event.metadata)

                if not parent['openable'] or parent['isOpen']:
                    parent_receptacle = parent['objectId']
                    self.in_receptacle_ids[parent_receptacle].add(obj_id)
                    self.object_to_point[obj_id] = self.receptacle_to_point[parent_receptacle]
                    self.point_to_object[tuple(self.receptacle_to_point[parent_receptacle].tolist())] = obj_id

        self.need_plan_update = True

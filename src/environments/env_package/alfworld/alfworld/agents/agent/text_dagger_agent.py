import copy
import operator
import logging
from queue import PriorityQueue

import numpy as np
try:
    import torch
except ImportError:
    raise ImportError("torch not found. Please install them via `pip install alfworld[full]`.")
logging.getLogger("transformers.tokenization_utils").setLevel(logging.ERROR)

from alfworld.agents.agent import BaseAgent
import alfworld.agents.modules.memory as memory
from alfworld.agents.modules.generic import to_np, to_pt, _words_to_ids, pad_sequences, preproc, max_len, ez_gather_dim_1, LinearSchedule, BeamSearchNode
from alfworld.agents.modules.layers import NegativeLogLoss, masked_mean, compute_mask, GetGenerationQValue

class TextDAggerAgent(BaseAgent):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        assert self.training_method == "dagger"

    def choose_softmax_action(self, action_rank, action_mask=None):
        action_rank = action_rank - torch.min(action_rank, -1, keepdim=True)[0] + 1e-2
        if action_mask is not None:
            assert action_mask.size() == action_rank.size(), (action_mask.size().shape, action_rank.size())
            action_rank = action_rank * action_mask
        pred_softmax = torch.log_softmax(action_rank, dim=1)
        action_indices = torch.argmax(pred_softmax, -1)
        return pred_softmax, to_np(action_indices)

    def update_dagger(self):
        if self.recurrent:
            return self.train_dagger_recurrent()
        else:
            return self.train_dagger()

    def train_dagger(self):

        if len(self.dagger_memory) < self.dagger_replay_batch_size:
            return None
        transitions = self.dagger_memory.sample(self.dagger_replay_batch_size)
        if transitions is None:
            return None
        batch = memory.dagger_transition(*zip(*transitions))

        if self.action_space == "generation":
            return self.command_generation_teacher_force(batch.observation_list, batch.task_list, batch.target_list)
        elif self.action_space in ["admissible", "exhaustive"]:
            return self.admissible_commands_teacher_force(batch.observation_list, batch.task_list, batch.action_candidate_list, batch.target_indices)
        else:
            raise NotImplementedError()

    def admissible_commands_teacher_force(self, observation_strings, task_desc_strings, action_candidate_list, target_indices):
        expert_indicies = to_pt(np.array(target_indices), enable_cuda=self.use_cuda, type='long')

        h_obs, obs_mask = self.encode(observation_strings, use_model="online")
        h_td, td_mask = self.encode(task_desc_strings, use_model="online")

        action_scores, _, _ = self.action_scoring(action_candidate_list,
                                                  h_obs, obs_mask,
                                                  h_td, td_mask,
                                                  None,
                                                  use_model="online")

        loss = self.cross_entropy_loss(action_scores, expert_indicies)
        self.online_net.zero_grad()
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.online_net.parameters(), self.clip_grad_norm)
        self.optimizer.step()
        return to_np(loss)

    def command_generation_teacher_force(self, observation_strings, task_desc_strings, target_strings):
        input_target_strings = [" ".join(["[CLS]"] + item.split()) for item in target_strings]
        output_target_strings = [" ".join(item.split() + ["[SEP]"]) for item in target_strings]

        input_obs = self.get_word_input(observation_strings)
        h_obs, obs_mask = self.encode(observation_strings, use_model="online")
        h_td, td_mask = self.encode(task_desc_strings, use_model="online")

        aggregated_obs_representation = self.online_net.aggretate_information(h_obs, obs_mask, h_td, td_mask)

        input_target = self.get_word_input(input_target_strings)
        ground_truth = self.get_word_input(output_target_strings)
        target_mask = compute_mask(input_target)
        pred = self.online_net.decode(input_target, target_mask, aggregated_obs_representation, obs_mask, None, input_obs)

        batch_loss = NegativeLogLoss(pred * target_mask.unsqueeze(-1), ground_truth, target_mask, smoothing_eps=self.smoothing_eps)
        loss = torch.mean(batch_loss)

        if loss is None:
            return None
        self.online_net.zero_grad()
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.online_net.parameters(), self.clip_grad_norm)
        self.optimizer.step()
        return to_np(loss)

    def train_dagger_recurrent(self):

        if len(self.dagger_memory) < self.dagger_replay_batch_size:
            return None
        sequence_of_transitions, contains_first_step = self.dagger_memory.sample_sequence(self.dagger_replay_batch_size, self.dagger_replay_sample_history_length)
        if sequence_of_transitions is None:
            return None

        batches = []
        for transitions in sequence_of_transitions:
            batch = memory.dagger_transition(*zip(*transitions))
            batches.append(batch)

        if self.action_space == "generation":
            return self.command_generation_recurrent_teacher_force([batch.observation_list for batch in batches], [batch.task_list for batch in batches], [batch.target_list for batch in batches], contains_first_step)
        elif self.action_space in ["admissible", "exhaustive"]:
            return self.admissible_commands_recurrent_teacher_force([batch.observation_list for batch in batches], [batch.task_list for batch in batches], [batch.action_candidate_list for batch in batches], [batch.target_indices for batch in batches], contains_first_step)
        else:
            raise NotImplementedError()

    def admissible_commands_recurrent_teacher_force(self, seq_observation_strings, seq_task_desc_strings, seq_action_candidate_list, seq_target_indices, contains_first_step=False):
        loss_list = []
        previous_dynamics = None
        h_td, td_mask = self.encode(seq_task_desc_strings[0], use_model="online")
        for step_no in range(self.dagger_replay_sample_history_length):
            expert_indicies = to_pt(np.array(seq_target_indices[step_no]), enable_cuda=self.use_cuda, type='long')
            h_obs, obs_mask = self.encode(seq_observation_strings[step_no], use_model="online")
            action_scores, _, current_dynamics = self.action_scoring(seq_action_candidate_list[step_no], h_obs, obs_mask, h_td, td_mask,
                                                                     previous_dynamics, use_model="online")
            previous_dynamics = current_dynamics
            if (not contains_first_step) and step_no < self.dagger_replay_sample_update_from:
                previous_dynamics = previous_dynamics.detach()
                continue

            loss = self.cross_entropy_loss(action_scores, expert_indicies)
            loss_list.append(loss)
        loss = torch.stack(loss_list).mean()
        self.online_net.zero_grad()
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.online_net.parameters(), self.clip_grad_norm)
        self.optimizer.step()
        return to_np(loss)

    def command_generation_recurrent_teacher_force(self, seq_observation_strings, seq_task_desc_strings, seq_target_strings, contains_first_step=False):
        loss_list = []
        previous_dynamics = None
        h_td, td_mask = self.encode(seq_task_desc_strings[0], use_model="online")
        for step_no in range(self.dagger_replay_sample_history_length):
            input_target_strings = [" ".join(["[CLS]"] + item.split()) for item in seq_target_strings[step_no]]
            output_target_strings = [" ".join(item.split() + ["[SEP]"]) for item in seq_target_strings[step_no]]

            input_obs = self.get_word_input(seq_observation_strings[step_no])
            h_obs, obs_mask = self.encode(seq_observation_strings[step_no], use_model="online")
            aggregated_obs_representation = self.online_net.aggretate_information(h_obs, obs_mask, h_td, td_mask)

            averaged_representation = self.online_net.masked_mean(aggregated_obs_representation, obs_mask)
            current_dynamics = self.online_net.rnncell(averaged_representation, previous_dynamics) if previous_dynamics is not None else self.online_net.rnncell(averaged_representation)

            input_target = self.get_word_input(input_target_strings)
            ground_truth = self.get_word_input(output_target_strings)
            target_mask = compute_mask(input_target)
            pred = self.online_net.decode(input_target, target_mask, aggregated_obs_representation, obs_mask, current_dynamics, input_obs)

            previous_dynamics = current_dynamics
            if (not contains_first_step) and step_no < self.dagger_replay_sample_update_from:
                previous_dynamics = previous_dynamics.detach()
                continue

            batch_loss = NegativeLogLoss(pred * target_mask.unsqueeze(-1), ground_truth, target_mask, smoothing_eps=self.smoothing_eps)
            loss = torch.mean(batch_loss)
            loss_list.append(loss)

        loss = torch.stack(loss_list).mean()
        if loss is None:
            return None
        self.online_net.zero_grad()
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.online_net.parameters(), self.clip_grad_norm)
        self.optimizer.step()
        return to_np(loss)

    def admissible_commands_greedy_generation(self, observation_strings, task_desc_strings, action_candidate_list, previous_dynamics):

        with torch.no_grad():
            h_obs, obs_mask = self.encode(observation_strings, use_model="online")
            h_td, td_mask = self.encode(task_desc_strings, use_model="online")
            action_scores, action_masks, current_dynamics = self.action_scoring(action_candidate_list,
                                                                                h_obs, obs_mask,
                                                                                h_td, td_mask,
                                                                                previous_dynamics,
                                                                                use_model="online")

            _, chosen_indices = self.choose_softmax_action(action_scores, action_masks)
            chosen_indices = chosen_indices.astype(int)
            chosen_actions = [item[idx] for item, idx in zip(action_candidate_list, chosen_indices)]
            return chosen_actions, chosen_indices, current_dynamics

    def command_generation_greedy_generation(self, observation_strings, task_desc_strings, previous_dynamics):
        with torch.no_grad():
            batch_size = len(observation_strings)

            input_obs = self.get_word_input(observation_strings)
            h_obs, obs_mask = self.encode(observation_strings, use_model="online")
            h_td, td_mask = self.encode(task_desc_strings, use_model="online")
            aggregated_obs_representation = self.online_net.aggretate_information(h_obs, obs_mask, h_td, td_mask)

            if self.recurrent:
                averaged_representation = self.online_net.masked_mean(aggregated_obs_representation, obs_mask)
                current_dynamics = self.online_net.rnncell(averaged_representation, previous_dynamics) if previous_dynamics is not None else self.online_net.rnncell(averaged_representation)
            else:
                current_dynamics = None

            input_target_list = [[self.word2id["[CLS]"]] for i in range(batch_size)]
            eos = np.zeros(batch_size)
            for _ in range(self.max_target_length):

                input_target = copy.deepcopy(input_target_list)
                input_target = pad_sequences(input_target, maxlen=max_len(input_target)).astype('int32')
                input_target = to_pt(input_target, self.use_cuda)
                target_mask = compute_mask(input_target)
                pred = self.online_net.decode(input_target, target_mask, aggregated_obs_representation, obs_mask, current_dynamics, input_obs)
                pred = to_np(pred[:, -1])
                pred = np.argmax(pred, -1)
                for b in range(batch_size):
                    new_stuff = [pred[b]] if eos[b] == 0 else []
                    input_target_list[b] = input_target_list[b] + new_stuff
                    if pred[b] == self.word2id["[SEP]"]:
                        eos[b] = 1
                if np.sum(eos) == batch_size:
                    break
            res = [self.tokenizer.decode(item) for item in input_target_list]
            res = [item.replace("[CLS]", "").replace("[SEP]", "").strip() for item in res]
            return res, current_dynamics

    def command_generation_beam_search_generation(self, observation_strings, task_desc_strings, previous_dynamics):
        with torch.no_grad():

            batch_size = len(observation_strings)
            beam_width = self.beam_width
            if beam_width == 1:
                res, current_dynamics = self.command_generation_greedy_generation(observation_strings, task_desc_strings, previous_dynamics)
                res = [[item] for item in res]
                return res, current_dynamics
            generate_top_k = self.generate_top_k
            res = []

            input_obs = self.get_word_input(observation_strings)
            h_obs, obs_mask = self.encode(observation_strings, use_model="online")
            h_td, td_mask = self.encode(task_desc_strings, use_model="online")
            aggregated_obs_representation = self.online_net.aggretate_information(h_obs, obs_mask, h_td, td_mask)

            if self.recurrent:
                averaged_representation = self.online_net.masked_mean(aggregated_obs_representation, obs_mask)
                current_dynamics = self.online_net.rnncell(averaged_representation, previous_dynamics) if previous_dynamics is not None else self.online_net.rnncell(averaged_representation)
            else:
                current_dynamics = None

            for b in range(batch_size):

                __input_target_list = [self.word2id["[CLS]"]]
                __input_obs = input_obs[b: b + 1]
                __obs_mask = obs_mask[b: b + 1]
                __aggregated_obs_representation = aggregated_obs_representation[b: b + 1]
                if current_dynamics is not None:
                    __current_dynamics = current_dynamics[b: b + 1]
                else:
                    __current_dynamics = None
                ended_nodes = []

                node = BeamSearchNode(None, __input_target_list, 0, 1)
                nodes_queue = PriorityQueue()
                nodes_queue.put((node.val, node))
                queue_size = 1

                while(True):
                    if queue_size > 2000:
                        break

                    score, n = nodes_queue.get()
                    __input_target_list = n.input_target

                    if (n.input_target[-1] == self.word2id["[SEP]"] or n.length >= self.max_target_length) and n.previous_node != None:
                        ended_nodes.append((score, n))
                        if len(ended_nodes) >= generate_top_k:
                            break
                        else:
                            continue

                    input_target = pad_sequences([__input_target_list], dtype='int32')
                    input_target = to_pt(input_target, self.use_cuda)
                    target_mask = compute_mask(input_target)
                    pred = self.online_net.decode(input_target, target_mask, __aggregated_obs_representation, __obs_mask, __current_dynamics, __input_obs)
                    pred = pred[0][-1].cpu()
                    gt_zero = torch.gt(pred, 0.0).float()
                    epsilon = torch.le(pred, 0.0).float() * 1e-8
                    log_pred = torch.log(pred + epsilon) * gt_zero

                    top_beam_width_log_probs, top_beam_width_indicies = torch.topk(log_pred, beam_width)
                    next_nodes = []

                    for new_k in range(beam_width):
                        pos = top_beam_width_indicies[new_k]
                        log_p = top_beam_width_log_probs[new_k].item()
                        node = BeamSearchNode(n, __input_target_list + [pos], n.log_prob + log_p, n.length + 1)
                        next_nodes.append((node.val, node))

                    for i in range(len(next_nodes)):
                        score, nn = next_nodes[i]
                        nodes_queue.put((score, nn))
                    queue_size += len(next_nodes) - 1

                if len(ended_nodes) == 0:
                    ended_nodes = [nodes_queue.get() for _ in range(generate_top_k)]

                utterances = []
                for score, n in sorted(ended_nodes, key=operator.itemgetter(0)):
                    utte = n.input_target
                    utte_string = self.tokenizer.decode(utte)
                    utterances.append(utte_string)
                utterances = [item.replace("[CLS]", "").replace("[SEP]", "").strip() for item in utterances]
                res.append(utterances)
            return res, current_dynamics

    def admissible_commands_random_generation(self, action_candidate_list):

        chosen_actions, chosen_indicies = [], []
        for i in range(len(action_candidate_list)):
            _action_idx = np.random.choice(len(action_candidate_list[i]))
            chosen_indicies.append(_action_idx)
            chosen_actions.append(action_candidate_list[i][_action_idx])
        return chosen_actions, chosen_indicies

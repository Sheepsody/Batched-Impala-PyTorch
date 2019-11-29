from threading import Thread
import torch
from src.Statistics import SummaryType
import time
from queue import Empty
from src.Trajectory import Trajectory
from src.VTrace import VTrace

# We don't need any locks because of the GIL
# We can't use the same thing two times

device = torch.device('cuda')

class Trainer(Thread):

    def __init__(self, id_, training_queue, batch_size, model, optimizer, statistics_queue, learning_step):
        
        super(Trainer, self).__init__()

        # We set this thread as child (daemon thread)
        self.setDaemon(True)

        self.id = id_
        self.training_queue = training_queue

        # Add value for exit flag
        self.exit = False
        self.batch_size = batch_size
        self.model = model
        self.optimizer = optimizer

        # Prevent to have train/untrainable or others at the same time
        self.stats_queue = statistics_queue
        self.lr_step = learning_step

    def run(self):

        super(Trainer, self).run()

        # FIXME move this
        discount_factor = torch.tensor(0.9).to("cuda")
        self.vtrace = VTrace(discount_factor=0.9, rho=1.0, cis=1.0).to("cuda")

        while not self.exit:

            batch_start = time.time()

            batch_size = 0

            # Collect a batch of trajectories
            trajectories = [] # Only stores the addresses (shared tensors)
            while batch_size < self.batch_size:
                try:
                    trajectories.append(self.training_queue.get(timeout=1))
                    batch_size += 1
                except Empty:
                    if self.exit :
                        break
                    continue

            # TODO put on the right device
            # FIXME define lambda functions ?
            # FIXME replace x and target/... prefixes
            # FIXME add detach
            # FIXME check for the sizes that are needed !!!!!

            seq_len = trajectories[0].length
        
            #----------------------------------
            # 1. EFFICIENT COMPUTATION ON CNNs (time is folded with batch size)
            #----------------------------------

            observations = torch.cat(tensors=[traj.observations for traj in trajectories], dim=0) # (batch*seq_len, ...)
            x = self.model.body(observations)
            x = x.chunk(batch_size, dim=0)

            #--------------
            # 2. LSTM LOOP (same length events but can be resetted)
            #--------------

            lstm_hxs = (
                torch.cat(tensors=[traj.lstm_initial_hidden for traj in trajectories], dim=1), # (1, 1, hidden)
                torch.cat(tensors=[traj.lstm_initial_cell for traj in trajectories], dim=1) # (1, 1, hidden)
            ) # (1, batch, hidden)
            x = torch.stack(tensors=x, dim=1).unsqueeze_(1) # (seq_len, 1, batch, input)
            done_mask = torch.stack(tensors=[traj.done for traj in trajectories], dim=0).unsqueeze_(0) # (1, batch, length)
            
            # TODO execute on CPU

            # We need to invert the done_mask
            done_mask = 1 - done_mask

            # Custom LSTM loop because states can be resetted 
            # I found this was the less-computationnaly expensive method
            x_out = []

            for i in range(seq_len):
                # Don't do the operation inplace ! It breaks the autograd functionnality
                # Hidden size and input size are the same in the model
                result, lstm_hxs = self.model.lstm(x[i], lstm_hxs)
                # Applying the mask to zero the hidden states of resetted games
                lstm_hxs = [(done_mask[:, :, i]*state) for state in lstm_hxs]
                x_out.append(result)

            x = torch.stack(tensors=x_results, dim=0)

            x.squeeze_(1) # (seq_len, batch, input)

            #-----------------------------------------
            # 3. EFFICIENT COMPUTATION OF LAST LAYERS (time is folded with batch size)
            #-----------------------------------------
            
            x = torch.cat(tensors=[x[:, i] for i in range(x.size(1))], dim=0)
            assert x.size(0) == batch_size*seq_len, f"Issue last layers"
            target_value, dist = self.model.tail(x)

            # Selecting probabilities and co. for agents actions (behaviour policy)
            behaviour_actions = torch.cat(tensors=[traj.actions for traj in trajectories], dim=0) # (batch*seq_len, ...)
            target_log_probs = dist.log_prob(behaviour_actions.squeeze_(-1)).unsqueeze_(-1) # batch* (seq_len, ...)
            target_entropy = dist.entropy() # batch* (seq_len, ...)

            # Reshaping for v-trace
            target_log_probs = torch.stack(tensors=target_log_probs.chunk(batch_size, dim=0), dim=1) # (seq_len, batch, ...)
            target_entropy = torch.stack(tensors=target_entropy.chunk(batch_size, dim=0), dim=1) # (seq_len, batch, ...)
            target_value = torch.stack(tensors=target_value.chunk(batch_size, dim=0), dim=1) # (seq_len, batch, ...)

            # TODO assert size equals

            #------------
            # 4. V-TRACE
            #------------

            # Additional tensors for v-trace
            rewards = torch.stack(tensors=[traj.rewards for traj in trajectories], dim=1) # (seq_len, batch, ...)
            behaviour_log_probs = torch.stack(tensors=[traj.log_probs for traj in trajectories], dim=1) # (seq_len, batch, ...)

            # v-trace computation
            v_targets, rhos = self.vtrace(target_value=target_value,
                                          rewards=rewards,
                                          target_log_policy=target_log_probs, 
                                          behaviour_log_policy=behaviour_log_probs)

            #-----------
            # 5. LOSSES
            #-----------

            # Value loss = l2 target loss -> (v_s - V_w(x_s))**2
            loss_value = (v_targets[:-1] - target_value[:-1]).pow_(2)  # Remove bootstrapping, l2 loss
            loss_value = loss_value.sum()

            # Policy loss -> rho * advantage * log_policy & entropy bonus sum(policy*log_policy)
            # We detach the advantage because we don't compute
            advantage = rewards[:-1] + discount_factor * v_targets[1:] - target_value[:-1]
            loss_policy = rhos[:-1] * target_log_probs[:-1] * advantage.detach_()
            loss_policy = loss_policy.sum()
            # Adding the entropy bonus (much like A3C for instance)
            entropy_bonus = - target_entropy[:-1]
            entropy_bonus = entropy_bonus.sum()

            # Summing all the losses together
            loss = loss_policy + 0.5 * loss_value - 0.01 * entropy_bonus

            #------------------
            # 8. BACKWARD PASS
            #------------------

            torch.autograd.set_detect_anomaly(True)

            # Reset the gradients
            self.optimizer.zero_grad()

            # Gradient update
            loss.backward()

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 40.0)

            # Not necessary to clip the gradient because of epsilon
            self.optimizer.step()

            # Increasing the learning step
            self.lr_step.value += 1
            
            # Loss statistics
            # Losses
            self.stats_queue.put((SummaryType.SCALAR, "loss/loss", loss.item()))
            self.stats_queue.put((SummaryType.SCALAR, "loss/policy", loss_policy.item()))
            self.stats_queue.put((SummaryType.SCALAR, "loss/value", loss_value.item()))
            self.stats_queue.put((SummaryType.SCALAR, "loss/entropy", entropy_bonus.item()))
            # Times
            updates_per_sec = batch_size/(time.time()-batch_start)
            self.stats_queue.put((SummaryType.SCALAR, "rate/training", updates_per_sec))
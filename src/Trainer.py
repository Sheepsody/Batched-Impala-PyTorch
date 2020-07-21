import time
from queue import Empty
from threading import Thread
import torch

from src.Statistics import SummaryType


class Trainer(Thread):
    """Trainer thread that batches trajectories and performs gradient descent"""

    def __init__(
        self,
        id_,
        training_queue,
        batch_size,
        model,
        optimizer,
        statistics_queue,
        learning_step,
        sequence_length,
    ):

        super(Trainer, self).__init__()

        # We set this thread as child (daemon thread)
        self.setDaemon(True)

        self.id = id_
        self.training_queue = training_queue

        # Add value for exit flag
        self.exit = False

        # Traning step sizes
        self.sequence_length = sequence_length
        self.batch_size = batch_size

        # Model
        self.model = model
        self.optimizer = optimizer

        # Prevent to have train/untrainable or others at the same time
        self.stats_queue = statistics_queue
        self.lr_step = learning_step

    def run(self):

        super(Trainer, self).run()

        while not self.exit:

            batch_start = time.time()

            batch_size = 0

            # Collect a batch of trajectories
            trajectories = []  # Only stores the addresses (shared tensors)
            while batch_size < self.batch_size:
                try:
                    trajectories.append(self.training_queue.get(timeout=1))
                    batch_size += 1
                except Empty:
                    if self.exit:
                        break
                    continue

            # Create the batch, and put on GPU
            (
                obs,
                lstm_hxs,
                mask,
                behaviour_actions,
                behaviour_log_probs,
                rewards,
            ) = Trainer._batch_trajectories(trajectories)

            # ----------------
            # BACKWARD PASS
            # ----------------

            loss, detached_losses = self.model.loss(
                obs=obs,
                behaviour_actions=behaviour_actions,
                reset_mask=mask,
                lstm_hxs=lstm_hxs,
                rewards=rewards,
                behaviour_log_probs=behaviour_log_probs,
            )

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
            self.stats_queue.put(
                (SummaryType.SCALAR, "loss/policy", detached_losses["policy"].item())
            )
            self.stats_queue.put(
                (SummaryType.SCALAR, "loss/value", detached_losses["value"].item())
            )
            self.stats_queue.put(
                (SummaryType.SCALAR, "loss/entropy", detached_losses["entropy"].item())
            )
            # Times
            updates_per_sec = (
                self.sequence_length * batch_size / (time.time() - batch_start)
            )
            self.stats_queue.put((SummaryType.SCALAR, "rate/training", updates_per_sec))

    @classmethod
    def _batch_trajectories(cls, trajectories):
        """Concatenates or stacks the trajectories"""
        # (seq_len+1, batch, ...)
        obs = torch.stack([traj.observations for traj in trajectories], dim=1)

        # (1, batch, hidden)
        lstm_hxs = (
            torch.cat(
                tensors=[traj.lstm_initial_hidden for traj in trajectories], dim=1
            ),
            torch.cat(tensors=[traj.lstm_initial_cell for traj in trajectories], dim=1),
        )

        # (seq_len+1, batch)
        mask = 1 - torch.stack(tensors=[traj.done for traj in trajectories], dim=1)
        behaviour_actions = torch.stack(
            tensors=[traj.actions for traj in trajectories], dim=1
        )

        # (seq_len, batch)
        rewards = torch.stack(tensors=[traj.rewards for traj in trajectories], dim=1)
        behaviour_log_probs = torch.stack(
            tensors=[traj.log_probs for traj in trajectories], dim=1
        )

        return obs, lstm_hxs, mask, behaviour_actions, behaviour_log_probs, rewards

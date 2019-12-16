from queue import Empty
import time
from threading import Thread
import torch

from src.Statistics import SummaryType


class Predictor(Thread):
    """Thread predictor that sends batches of observations for predictions"""

    def __init__(self, id_, prediction_queue, agents, batch_size, model, statistics_queue, device):

        super(Predictor, self).__init__()

        # We set this thread as child (daemon thread)
        self.setDaemon(True)

        self.id = id_
        self.prediction_queue = prediction_queue
        self.agents = agents

        # Add value for exit flag
        self.exit = False
        self.batch_size = batch_size
        self.model = model
        self.device = device

        # Prevent to have train/untrainable or others at the same time
        self.stats_queue = statistics_queue

    def run(self):

        # Thread's start function
        super(Predictor, self).run()

        while not self.exit:

            batch_start = time.time()

            # Background processes issue fix
            try:
                id_, obs_, lstm_hxs_ = self.prediction_queue.get(timeout=1)
                length = 1
            except Empty:
                if self.exit:
                    break
                continue

            lstm_hxs = [lstm_hxs_]
            observations = [obs_]
            ids = [id_]

            # To handle while removing agents
            while len(observations) < self.batch_size and not self.prediction_queue.empty():
                id_, obs_, lstm_hxs_ = self.prediction_queue.get()
                observations.append(obs_)
                ids.append(id_)
                lstm_hxs.append(lstm_hxs_)
                length += 1

            # Batching the observations
            # (batch, c, h, w)
            observations = torch.stack(observations, dim=0).to(self.device)
            lstm_hxs = (
                torch.cat(tensors=[state[0]
                                   for state in lstm_hxs], dim=1).to(self.device),
                torch.cat(tensors=[state[1]
                                   for state in lstm_hxs], dim=1).to(self.device)
            )

            # Predictions on GPU
            # I tried different means but passing as cpu tensor still seems the best option
            with torch.no_grad():
                actions, log_probs, lstm_hxs = self.model.act(
                    observations, lstm_hxs)

            actions = actions.cpu().chunk(chunks=length, dim=0)
            log_probs = log_probs.cpu().chunk(chunks=length, dim=0)
            lstm_hxs = [
                lstm_hxs[0].cpu().chunk(chunks=length, dim=1),
                lstm_hxs[1].cpu().chunk(chunks=length, dim=1)
            ]

            # Send back all the observations
            for index, id_ in enumerate(ids):
                self.agents[id_].action_queue.put(
                    (actions[index], log_probs[index], (lstm_hxs[0][index], lstm_hxs[1][index])))

            # Statistics
            pred_per_sec = len(observations)/(time.time()-batch_start)
            self.stats_queue.put(
                (SummaryType.SCALAR, "rate/predictions", pred_per_sec))

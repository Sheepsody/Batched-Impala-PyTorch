from threading import Thread
import torch
import time
from src.Statistics import SummaryType
from queue import Empty

# We don't need any locks because of the GIL
# We can't use the same thing two times

# TODO device

device = torch.device('cuda')

class Predictor(Thread):

    def __init__(self, id_, prediction_queue, agents, batch_size, model, statistics_queue):
        
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

        # Prevent to have train/untrainable or others at the same time
        self.stats_queue = statistics_queue
    
    def run(self):

        super(Predictor, self).run()
        
        while not self.exit:

            batch_start = time.time()
            
            # Background processes issue fix
            try:
                id_, obs_ = self.prediction_queue.get(timeout=1)
            except Empty:
                if self.exit :
                    break
                continue
                
            observations = [obs_]
            ids = [id_]

            # To handle while removing agents
            while len(observations) < self.batch_size and not self.prediction_queue.empty():
                
                id_, obs_ = self.prediction_queue.get()
                observations.append(obs_)
                ids.append(id_) 
            
            # List to Tensor
            observations = torch.cat(observations).to(device)

            # Predictions on GPU
            with self.model.lock, torch.no_grad():
                actions, values = self.model.select_action(observations)

            actions = actions.to('cpu')
            values = values.to('cpu')

            # Send back all the observations
            for index, id_ in enumerate(ids):
                self.agents[id_].action_queue.put((actions[index, :], values[index, :]))
            
            # Statistics
            pred_per_sec = len(observations)/(time.time()-batch_start)
            self.stats_queue.put((SummaryType.SCALAR, "rate/predictions", pred_per_sec))
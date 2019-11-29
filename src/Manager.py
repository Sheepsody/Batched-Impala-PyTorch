import configparser
from src.ActorCritic import ActorCritic
import torch.multiprocessing as mp
from torch.multiprocessing import Queue, Value
from src.Statistics import Statistics
from torch.multiprocessing import Process
from src.Trainer import Trainer
from src.Agent import Agent
from src.Callback import StateCallback
import torch.optim as optim
from ctypes import c_bool
from threading import Thread
import time
import torch
import os


# TODO in main set start method as fork server ! Before calling any cuda operation


class Manager(Thread):
    
    def __init__(self, config_file):
        
        super(Manager, self).__init__()
        # TODO Add maps directory
        # Setting it as daemon child
        self.daemon = True

        # Read config file
        self.config = configparser.ConfigParser()
        # Fixing lower-case keys in config files
        self.config.optionxform = lambda option: option
        self.config.read(config_file)

        # Initializing the device
        self.device = torch.device('cuda') if self.config["settings"]["device"] == "cuda" \
            else torch.device('cpu')

        # Test and training sets
        self.train_set, self.test_set = [], []
        for key, value in self.config["levels"].items():
            if value == "train":
                self.train_set.append(key)
            elif value == "test" :
                self.test_set.append(key)
        
        # Building the model and share it (cf torch.multiprocessing best practices)
        self.model = ActorCritic(56, 128, 1, 9).float().to(self.device)
        # Sharing memory between processes
        self.model.share_memory()

        # Building the optimizer
        self.optimizer = optim.RMSprop(self.model.parameters(),
                                lr=float(self.config["optimizer"]["lr"]), 
                                alpha=float(self.config["optimizer"]["alpha"]),
                                eps=float(self.config["optimizer"]["eps"]),
                                momentum=float(self.config["optimizer"]["momentum"]),
                                weight_decay=float(self.config["optimizer"]["weight_decay"]),
                                centered=self.config["optimizer"]["centered"]=="True")

        # Directory to save infos
        self.checkpoint_path = self.config["settings"]["checkpoint_path"]
        self.callbacks = [int(num) for num in self.config["settings"]["callbacks"].split(",")]
        self.records_folder = self.config["settings"]["records_folder"]
        os.makedirs(self.records_folder, exist_ok=True)

        # Building the torch.multiprocessing-queues
        self.training_queue = Queue(maxsize=int(self.config["settings"]["training_queue"]))
        self.statistics_queue = Queue()

        # Building the torch.multiprocessing-values
        self.learning_step = Value('i', 0)
        self.nb_episodes = Value('i', 0)
        self.max_nb_steps = int(self.config["settings"]["max_nb_episodes"])

        # Statistics thread
        self.tensorboard = self.config["settings"]["tensorboard"]
        self.statistics = Statistics(
            writer_dir=self.tensorboard,
            statistics_queue=self.statistics_queue,
            nb_episodes=self.nb_episodes
        )

        # Agents, predictions and learners
        self.training_batch_size = int(self.config["settings"]["training_batch_size"])
        self.trainers = []
        self.agents = []

        # Adding the threads and agents
        self.add_trainers(int(self.config["settings"]["trainers"]))
        self.add_agents(int(self.config["settings"]["agents"]))

    def add_agents(self, nb):
        old_length = len(self.agents)
        for index in range(old_length, old_length+nb):
            # Generating the agent's model
            actor_critic = ActorCritic(56, 128, 1, 9).float().to(self.device)

            self.agents.append(Agent(
                id_=index,
                target_policy=self.model,
                behaviour_policy=actor_critic,
                training_queue=self.training_queue,
                states=self.train_set,
                exit_flag=Value(c_bool, False),
                statistics_queue=self.statistics_queue,
                episode_counter=self.nb_episodes))

    def add_trainers(self, nb):
        old_length = len(self.trainers)
        for index in range(old_length, old_length+nb):
            self.trainers.append(Trainer(
                id_=index,
                training_queue=self.training_queue,
                batch_size=self.training_batch_size,
                model=self.model,
                optimizer=self.optimizer,
                statistics_queue=self.statistics_queue,
                learning_step=self.learning_step))

    def remove_agents(self, nb):
        # Removes the nb last agents
        assert len(self.agents) >= nb, "Too many agents to remove"
        # Stop the agents after their episode 
        for agent in self.agents[-nb:len(self.agents)]:
            agent.exit.value = True
        # Waiting for all these agents to stop
        for agent in self.agents[-nb:len(self.agents)]:
            self.agents[-1].join()
            self.agents.pop()

    def remove_trainer(self):
        self.trainers[-1].exit = True
        self.trainers[-1].join()
        self.trainers.pop()

    def remove_statistics(self):
        self.trainers[-1].exit = True
        self.trainers[-1].join()
        self.trainers.pop()

    def save_model(self):
        torch.save({
            'epoch': self.nb_episodes.value,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict()
        }, self.config["settings"]["checkpoint_path"])

    def run(self):
        
        super(Manager, self).run()

        # Strating the threads and processes
        self.statistics.start()
        [agent.start() for agent in self.agents]
        [trainer.start() for trainer in self.trainers]
        callback_process = []

        # TODO push the model to the tensorboard

        # Loop
        while self.nb_episodes.value < self.max_nb_steps :

            # TODO add param for that
            # time.sleep(2*60)

            self.save_model()

            # Check for callbacks
            if self.callbacks[0] < self.nb_episodes.value:
                # Pop the corresponding step
                self.callbacks.pop(0)

                # Create and run a callback
                # callback = StateCallback(checkpoint_path=self.checkpoint_path, 
                #                          step=self.nb_episodes.value, 
                #                          records_folder=self.records_folder, 
                #                          train_set=self.train_set, 
                #                          test_set=self.test_set,
                #                          statistics_queue=self.statistics_queue)
                # callback.start()
                # callback_process.append(callback)

        # Stopping all the threads        
        for thread in [*self.trainers, self.statistics]:
            thread.exit = True
            thread.join()

        # At last, checking the callbacks (they run on CPU)
        while callback_process:
            callback_process[-1].join()
            callback_process.pop()

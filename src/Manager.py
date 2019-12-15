import configparser
from src.IMPALA import Impala
from src.networks.ActorCritic import ActorCriticLSTM
import torch.multiprocessing as mp
from torch.multiprocessing import Queue, Value
from src.Statistics import Statistics
from torch.multiprocessing import Process
from src.Trainer import Trainer
from src.Predictor import Predictor
from src.Agent import Agent
import torch.optim as optim
from ctypes import c_bool
from threading import Thread
import time
import torch
import os
from src.GymEnv import KartMultiDiscretizer
# TODO : if weights exists reuse thems


class Manager(Thread):
    
    def __init__(self, config_file):
        
        super(Manager, self).__init__()
        # Setting it as daemon child
        self.daemon = True

        # Read config file
        self.config = configparser.ConfigParser()
        # Fixing lower-case keys in config files
        self.config.optionxform = lambda option: option
        self.config.read(config_file)

        # Initializing the device
        if self.config["settings"]["device"] == "cuda":
            assert torch.cuda.is_available()
        self.device = self.config["settings"]["device"]
        self.agent_device = self.config["settings"]["device"]

        # Test and training sets
        self.train_set, self.test_set = [], []
        for key, value in self.config["levels"].items():
            if value == "train":
                self.train_set.append(key)
            elif value == "test" :
                self.test_set.append(key)

        # Dimensions of the view
        self.channels = int(self.config["environnement"]["stacks"])
        self.height = int(self.config["environnement"]["height"])
        self.width = int(self.config["environnement"]["width"])
        
        # Creating the environnement generation function
        self.n_outputs = len(KartMultiDiscretizer.discretized_actions)

        # Impala constants
        self.sequence_length = int(self.config["impala"]["sequence_length"])
        self.rho = float(self.config["impala"]["rho"])
        self.cis = float(self.config["impala"]["cis"])
        self.discount_factor = float(self.config["impala"]["discount_factor"])
        self.entropy_coef = float(self.config["impala"]["entropy_coef"])
        self.value_coef = float(self.config["impala"]["value_coef"])

        # Building the model and share it (cf torch.multiprocessing best practices)
        self.model = torch.jit.script(
            ActorCriticLSTM(
                c = self.channels,
                h = self.height,
                w = self.width,
                n_outputs = self.n_outputs,
                sequence_length=self.sequence_length
            ).float()
        ).to(self.device)

        # To have a multi-machine-case, just place on different devices and sync the models once a while
        self.impala = torch.jit.script(Impala(
            sequence_length=self.sequence_length,
            entropy_coef=self.entropy_coef,
            value_coef=self.value_coef,
            discount_factor=self.discount_factor,
            model=self.model,
            rho=self.rho, 
            cis=self.cis,
            device=self.device
        ))

        # Sharing memory between processes
        self.model.share_memory()
        self.impala.share_memory()

        # Building the optimizer
        self.optimizer = optim.RMSprop(self.model.parameters(),
                                lr=float(self.config["optimizer"]["lr"]), 
                                alpha=float(self.config["optimizer"]["alpha"]),
                                eps=float(self.config["optimizer"]["eps"]),
                                momentum=float(self.config["optimizer"]["momentum"]),
                                weight_decay=float(self.config["optimizer"]["weight_decay"]),
                                centered=self.config["optimizer"]["centered"]=="True")

        # Checkpoints directory
        self.checkpoint_path = self.config["settings"]["checkpoint_path"]

        # Building the torch.multiprocessing-queues
        self.training_queue = Queue(maxsize=int(self.config["settings"]["training_queue"]))
        self.prediction_queue = Queue(maxsize=int(self.config["settings"]["prediction_queue"]))
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
        self.prediction_batch_size = int(self.config["settings"]["prediction_batch_size"])
        self.predictors = []
        self.agents = []

        # Adding the threads and agents
        self.add_trainers(int(self.config["settings"]["trainers"]))
        self.add_agents(int(self.config["settings"]["agents"]))
        self.add_predictors(int(self.config["settings"]["predictors"]))

    def add_agents(self, nb):
        old_length = len(self.agents)
        for index in range(old_length, old_length+nb):
            self.agents.append(Agent(
                id_=index,
                prediction_queue=self.prediction_queue,
                training_queue=self.training_queue,
                states=self.train_set,
                exit_flag=Value(c_bool, False),
                statistics_queue=self.statistics_queue,
                episode_counter=self.nb_episodes,
                observation_shape=(self.channels, self.height, self.width),
                action_space=self.n_outputs,
                device=self.agent_device,
                step_max=self.sequence_length
            ))

    def add_trainers(self, nb):
        old_length = len(self.trainers)
        for index in range(old_length, old_length+nb):
            self.trainers.append(Trainer(
                id_=index,
                training_queue=self.training_queue,
                batch_size=self.training_batch_size,
                model=self.impala,
                optimizer=self.optimizer,
                statistics_queue=self.statistics_queue,
                learning_step=self.learning_step,
                sequence_length=self.sequence_length
            ))
    
    def add_predictors(self, nb):
        old_length = len(self.predictors)
        for index in range(old_length, old_length+nb):
            self.predictors.append(Predictor(
                id_=index,
                prediction_queue=self.prediction_queue,
                agents=self.agents,
                batch_size=self.prediction_batch_size,
                model=self.impala,
                statistics_queue=self.statistics_queue,
                device=self.device
            ))

    def save_model(self):
        torch.save({
            'epoch': self.nb_episodes.value,
            'model_state_dict': self.impala.get_model_state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict()
        }, self.config["settings"]["checkpoint_path"])

    def run(self):
        
        super(Manager, self).run()

        # Strating the threads and processes
        self.statistics.start()

        [agent.start() for agent in self.agents]
        [trainer.start() for trainer in self.trainers]
        [predictor.start() for predictor in self.predictors]

        # Loop
        while self.nb_episodes.value < self.max_nb_steps :
            time.sleep(2*60)
            self.save_model()

        # Marking the agent as stop
        for agent in self.agents:
            agent.exit_flag.value = True

        # Stopping all the threads        
        for thread in [*self.trainers, *self.predictors, self.statistics]:
            thread.exit = True
            thread.join()

        for agent in self.agents:
            agent.join()

FROM nvcr.io/nvidia/pytorch:19.10-py3

RUN pip install gym gym-retro opencv-python gym[atari]

# COPY . /App/KartRL

# Link the files 
RUN ln -s /App/SuperMarioKart-Snes/ /opt/conda/lib/python3.6/site-packages/retro/data/stable/

# Run the trainining and tensorboard
# tensorboard --logdir=/workspace/logs --port=6005 --bind_all
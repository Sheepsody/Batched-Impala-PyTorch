# PyTorch image optimized for GPU
FROM nvcr.io/nvidia/pytorch:19.10-py3

# Install the dependencies
RUN pip install gym gym-retro gym[atari]
RUN pip install opencv-python 

# Copy the source files
COPY . /App/KartRL

# Link the mario kart configuration
RUN ln -s /App/SuperMarioKart-Snes/ /opt/conda/lib/python3.6/site-packages/retro/data/stable/

WORKDIR /App/KartRL

# Exposing the port (for the tensorboard)
EXPOSE 6006
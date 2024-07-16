# Use the pytorch/pytorch:latest image as the base image
FROM pytorch/pytorch:latest

# Set the working directory inside the container
WORKDIR /workspace

# Copy the contents of the source folder to the working directory in the container

COPY mask_rcnn /workspace

# Install Python dependencies using pip
RUN apt-get update -y
RUN pip install --upgrade pip
RUN pip install -r /workspace/requirements.txt

# Set the entry point for the container (optional)

ENTRYPOINT ["/bin/bash"]



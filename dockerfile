# Use the pytorch/pytorch:latest image as the base image
FROM pytorch/pytorch:2.3.1-cuda12.1-cudnn8-devel

# Set the working directory inside the container
WORKDIR /workspace

# Copy the contents of the source folder to the working directory in the container
COPY mask_rcnn /workspace

# Install Python dependencies using pip
RUN apt-get update -y && \
    pip install --upgrade pip && \
    pip install -r /workspace/requirements.txt

# Copy the entrypoint script into the container
COPY run_detection.sh /run_detection.sh

# Give execution permissions to the entrypoint script
RUN chmod +x /run_detection.sh

# Set the entry point for the container
ENTRYPOINT ["/run_detection.sh"]

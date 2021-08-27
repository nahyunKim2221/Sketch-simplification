FROM tiangolo/uvicorn-gunicorn-fastapi:python3.7

RUN pip install dynaconf
RUN pip install numpy
RUN pip install Pillow
RUN pip install natsort
RUN pip install argparse
RUN pip install torch==1.9.0+cu111 torchvision==0.10.0+cu111 torchaudio==0.9.0 -f https://download.pytorch.org/whl/torch_stable.html

RUN apt-get update 
RUN apt-get install ffmpeg libsm6 libxext6  -y
RUN pip install opencv-python

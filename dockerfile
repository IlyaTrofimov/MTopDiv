FROM nvidia/cuda:10.2-cudnn7-devel-ubuntu18.04
MAINTAINER Ilya Trofimov
ENV TZ=Europe/Moscow
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

RUN apt-get -yqq update
RUN apt-get -yqq install git cmake vim wget

#
# Conda
#
RUN wget https://repo.anaconda.com/archive/Anaconda3-2020.11-Linux-x86_64.sh
RUN chmod 777 ./Anaconda3-2020.11-Linux-x86_64.sh
RUN bash ./Anaconda3-2020.11-Linux-x86_64.sh -b -p /home/anaconda
ENV PATH="/home/anaconda/bin:$PATH"
RUN conda create -y -n py37 python=3.7 anaconda
RUN conda init bash

SHELL ["conda", "run", "-n", "py37", "/bin/bash", "-c"]

#
# ripser
#
RUN git clone https://github.com/simonzhang00/ripser-plusplus.git
RUN cd ripser-plusplus && python setup.py install

#
# Geometry Score (patched for multi-threading)
#
RUN git clone https://github.com/IlyaTrofimov/geometry-score.git
RUN conda install -y -c conda-forge gudhi
RUN cd geometry-score && python setup.py install

#
# IMD
#
RUN git clone https://github.com/xgfs/imd.git
RUN cd imd && python setup.py install 

#
# Pytorch-FID
#
RUN pip install pytorch-fid

#
# MTopDiv
#
RUN git clone https://github.com/IlyaTrofimov/MTopDiv.git
RUN cd MTopDiv && python setup.py install

#
# Start jupyter server
#
EXPOSE 8890
RUN echo "(jupyter notebook --ip=0.0.0.0 --port 8890 --allow-root &)" > start_jupyter.sh

CMD ["/bin/bash"]

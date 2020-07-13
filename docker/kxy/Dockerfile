FROM continuumio/anaconda3

RUN apt-get update
RUN apt-get install build-essential -y
RUN apt-get install swig -y

RUN /opt/conda/bin/conda install gxx_linux-64 gcc_linux-64
RUN /opt/conda/bin/conda install jupyter -y --quiet 
RUN pip install emcee
RUN pip install scikit-optimize
RUN pip install pyDOE
RUN pip install kxy
RUN pip install auto-sklearn 
RUN mkdir /opt/notebooks 
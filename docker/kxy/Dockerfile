FROM continuumio/anaconda3

RUN pip install kxy
RUN pip install auto-sklearn --no-cache-dir --force-reinstall -I --no-deps --upgrade
RUN /opt/conda/bin/conda install jupyter -y --quiet 
RUN mkdir /opt/notebooks 
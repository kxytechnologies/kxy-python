FROM continuumio/anaconda3

RUN apt-get update
RUN apt-get install build-essential -y
RUN apt-get install swig -y

RUN /opt/conda/bin/conda install gxx_linux-64 gcc_linux-64
RUN /opt/conda/bin/conda install jupyter -y --quiet 
RUN pip install --upgrade pip
RUN pip install pyarrow
RUN pip install fastparquet
RUN pip install emcee scikit-optimize pyDOE
RUN pip install auto-sklearn

# Install other ML open source librairies
RUN pip install xgboost
RUN pip install lightgbm
RUN pip install tensorflow
RUN pip install tensorflow_probability
RUN pip install botocore
RUN pip install boto3

# Install kxy
RUN pip install kxy==1.4.3

# Copy examples into the Notebooks folder
RUN git clone https://github.com/kxytechnologies/kxy-python.git /opt/kxy-python
RUN mkdir /opt/notebooks
RUN cp -R /opt/kxy-python/docs/latest/applications/case_studies/* /opt/notebooks/
RUN cp -R /opt/kxy-python/docs/latest/applications/illustrations/* /opt/notebooks/
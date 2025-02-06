FROM ubuntu:22.04

RUN apt-get update && apt-get install -y wget git bash libgl1 && \
    rm -rf /var/lib/apt/lists/*
ENV HYDRA_FULL_ERROR=1

WORKDIR /opt
RUN wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh && \
    bash Miniconda3-latest-Linux-x86_64.sh -b -p /opt/miniconda && \
    rm Miniconda3-latest-Linux-x86_64.sh


ENV PATH="/opt/miniconda/bin:${PATH}"



WORKDIR /app
COPY environment_trees_updated_docker.yml /app/environment.yml
RUN conda env create --name trees --file environment.yml
RUN echo "conda activate trees" >> ~/.bashrc
ENV CONDA_DEFAULT_ENV=trees

RUN mkdir -p /app/predictions/ /app/saved_models/

COPY fix_permission.sh /usr/local/bin/fix_permission.sh
RUN chmod +x /usr/local/bin/fix_permission.sh


COPY core2/ /app/core2/
COPY config/hyperps.yaml /app/config/hyperps.yaml
COPY main_docker.py /app/main.py

ENTRYPOINT ["/usr/local/bin/fix_permission.sh"]
CMD ["bash", "-c", "source activate trees && python main.py"]



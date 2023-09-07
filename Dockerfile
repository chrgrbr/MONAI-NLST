FROM pytorch/pytorch:2.0.1-cuda11.7-cudnn8-runtime

RUN groupadd -r user && useradd -m --no-log-init -r -g user user

RUN mkdir -p /opt/app /opt/algorithm /input /output \
    && chown user:user /opt/app /opt/algorithm /input /output

USER user
WORKDIR /opt/app

ENV PATH="/home/user/.local/bin:${PATH}"
    
RUN python -m pip install --user -U pip && python -m pip install --user pip-tools

COPY --chown=user:user model_weights /opt/algorithm/

COPY --chown=user:user requirements.txt /opt/app/
RUN python -m pip install -r requirements.txt

# print pip freeze to stdout
RUN python -m pip freeze


COPY --chown=user:user process.py /opt/app/

ENTRYPOINT [ "python", "-m", "process" ]

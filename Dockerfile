# get slim base image for python
FROM python:3.9.17-slim-bullseye as builder

RUN apt-get -y update && apt-get install -y --no-install-recommends \
    ca-certificates \
    dos2unix \
    g++ \
    && rm -rf /var/lib/apt/lists/*


COPY ./requirements.txt /opt/
RUN pip install --no-cache-dir -r /opt/requirements.txt
COPY src ./opt/src
COPY ./entry_point.sh /opt/
RUN chmod +x /opt/entry_point.sh

WORKDIR /opt/src
RUN chown -R 1000:1000 /opt/src
RUN chmod a+w /opt/

ENV THEANO_FLAGS='base_compiledir=/opt/.theano'
ENV MPLCONFIGDIR="/opt/src/"
ENV PYTHONUNBUFFERED=TRUE
ENV PYTHONDONTWRITEBYTECODE=TRUE
ENV PATH="/opt/src:${PATH}"
# set non-root user
USER 1000

ENTRYPOINT ["/opt/entry_point.sh"]

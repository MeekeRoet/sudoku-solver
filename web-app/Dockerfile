# Pull official base image
FROM python:3.7.6-slim-buster

# Set working directory
WORKDIR /usr/src/app

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

# Install python dependencies
COPY ./requirements.txt /usr/src/app/requirements.txt

RUN buildDeps='gcc musl-dev' \
    && set -x \
    && apt-get update && apt-get install -y $buildDeps libgtk2.0-dev glpk-utils --no-install-recommends \
    && rm -rf /var/lib/apt/lists/* \
    && pip install --upgrade pip \
    && pip install -r requirements.txt \
    && apt-get purge -y --auto-remove $buildDeps

# Copy project files
COPY . /usr/src/app/

ENTRYPOINT ["python", "app.py"]
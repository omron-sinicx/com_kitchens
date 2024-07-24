# Makefile

SHELL := /bin/bash

# Docker
build-docker-images: build-VidChapter-image build-FrozenBiLM-image build-whisperX-image

build-VidChapters-image:
	cd docker/VidChapters && \
	docker build -t com_kitchens/vidchapters . \
	--build-arg USER_ID=`id -u` \
	--build-arg GROUP_ID=`id -g` \
	--build-arg USER_NAME=`whoami`

build-FrozenBiLM-image:
	cd docker/FrozenBiLM && \
	docker build -t com_kitchens/frozenbilm . \
	--build-arg USER_ID=`id -u` \
	--build-arg GROUP_ID=`id -g` \
	--build-arg USER_NAME=`whoami`

build-whisperX-image:
	cd docker/whisperX && \
	docker build -t com_kitchens/whisperx . \
	--build-arg USER_ID=`id -u` \
	--build-arg GROUP_ID=`id -g` \
	--build-arg USER_NAME=`whoami`

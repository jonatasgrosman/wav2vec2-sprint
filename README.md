# Building docker image

Dockerhub available at https://hub.docker.com/r/patilsuraj/hf-wav2vec

to build the docker :

```
$ docker build -t hf-wav2vec-sprint -f Dockerfile .
```

to push it to dockerhub
First create a repository on dockerhub
```
$ docker tag hf-wav2vec-sprint your-dockerhub-user/hf-wav2vec-sprint
```

to push it to dockerhub

```
$ docker push your-dockerhub-user/hf-wav2vec-sprint
```

# Running WandB sweep

Initialize your sweep from any machine...

```
$ export WANDB_API_KEY=YOUR_WANDB_API_KEY
$ export WANDB_ENTITY=YOUR_WANDB_ENTITY
$ export WANDB_PROJECT=YOUR_WANDB_PROJECT

$ wandb sweep sweep.yaml
```
... the execution above will give you a sweep id, save it and on the training machine run:

```
$ export WANDB_API_KEY=YOUR_WANDB_API_KEY
$ export WANDB_ENTITY=YOUR_WANDB_ENTITY
$ export WANDB_PROJECT=YOUR_WANDB_PROJECT

$ wandb agent YOUR_SWEEP_ID
```

# Uploading model to HF

```
$ git config --global user.email "email@example.com"

$ git config --global user.name "Your name"

$ transformers-cli login

$ transformers-cli repo create your-model-name

$ git clone https://username:password_or_token@huggingface.co/username/your-model-name

$ git add .

$ git commit -m "Initial commit"

$ git push

```

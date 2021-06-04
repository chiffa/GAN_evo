# First, we are working on the local machine to make sure everything is working well

# build the docker image for Amir's branch
docker build -t "evoganamir" . | tee log

# mount the volumes and test if everything works as expected on the local docker
docker run --gpus all --rm -v /localhome/kucharav/PycharmProjects/deployments/GAN_evo:/app evoganamir bash -c "mongod --fork --logpath /dev/null && cd /app && python -m src.arena"


# Second, go to the cluster and attempt to build/deploy/run there


# now build it on the cluster
docker build -t "evoganamir" . | tee log
# and push it to the local repository
docker image tag evoganamir:latest master.cluster:5000/andrei.kucharavy/evoganamir

# and now we need to try to run it on slurm
# translate first to something compatible with UNIX (line ends)\
tr -d '\015' <slurm_job.sh >slurm_batch_job.sh
sbatch slurm_batch_job.sh
!/bin/bash
#SBATCH --job-name=ank_evogan
#SBATCH --output=ank_evogan_%x-%j.out
#SBATCH --error=ank_evogan_%x-%j.err
#SBATCH --time=1-00:00:00
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=10240
#SBATCH --partition=nodes
#SBATCH --gres=gpu:a100:1

nvidia-smi

echo $(hostname)
echo $CUDA_VISIBLE_DEVICES
# SLURM provides one or more GPU indices that map to the GPU set it has allocated for
# this job. These do not correspond to the GPU indices known to the system and docker.
# Extract the UUID's for all GPUs allocated for this task.
DEVICES=$(echo $(nvidia-smi -i $CUDA_VISIBLE_DEVICES --query-gpu=uuid --format=csv,noheader) | sed
's/ /,/g')
echo "devices: " $DEVICES

# Generate the command string - has to be done this way since the command expects device
# spec to be defined with single quotes
# Note: In this example, the code to run in the container is passed using an external volume
cmdstr=$(echo docker run --gpus "'\"device=$DEVICES\"'" --shm-size=1g --ulimit memlock=-1 --ulimit
stack=67108864 --rm -v /cluster/raid/home/andrei.kucharavy/deployments/GAN_evo_Amir:/app
andrei.kucharavy/evoganamir
bash -c "mongod --fork --logpath /dev/null && cd /app && python -m src.arena")

echo $cmdstr
# Run the command

eval $cmdstr

echo "Done"
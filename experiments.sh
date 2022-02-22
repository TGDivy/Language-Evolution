############# SIMPLE MADDPG #####################
# python run.py --model "maddpg" --env "simple" --experiment_name "continuous-v0" --total_memory 12 --episode_len 25 --batch_size 4 --n_episodes 10000 --num_layers 3 --num_filters 128 --communicate 0 --n_epochs 3

# python run.py --model "maddpg" --env "simple" --experiment_name "continuous-v1" --total_memory 12 --episode_len 25 --batch_size 4 --n_episodes 10000 --num_layers 3 --num_filters 128 --communicate 0 --n_epochs 3

# python run.py --model "maddpg" --env "simple" --experiment_name "continuous-v2" --total_memory 12 --episode_len 25 --batch_size 4 --n_episodes 10000 --num_layers 3 --num_filters 128 --communicate 0 --n_epochs 3

########################

# python run.py --model "maddpg" --env "spread" --experiment_name "v4-SquareDist" --total_memory 12 --episode_len 25 --batch_size 4 --n_episodes 25000 --num_layers 3 --num_filters 128 --communicate 0 --n_epochs 3

# python run.py --model "maddpg" --env "adversary" --experiment_name "v0" --total_memory 12 --episode_len 25 --batch_size 4 --n_episodes 40000 --num_layers 3 --num_filters 128 --communicate 0 --n_epochs 3

python run.py --model "maddpg" --env "communication" --experiment_name "v4-3commchan" --total_memory 12 --episode_len 50 --batch_size 4 --n_episodes 250 --num_layers 3 --num_filters 128 --communicate 0 --n_epochs 3
############# SIMPLE MADDPG #####################
# python run.py --model "maddpg" --env "simple" --experiment_name "continuous-v0" --total_memory 12 --episode_len 25 --batch_size 4 --n_episodes 10000 --num_layers 3 --num_filters 128 --communicate 0 --n_epochs 3

# python run.py --model "maddpg" --env "simple" --experiment_name "continuous-v1" --total_memory 12 --episode_len 25 --batch_size 4 --n_episodes 10000 --num_layers 3 --num_filters 128 --communicate 0 --n_epochs 3

# python run.py --model "maddpg" --env "simple" --experiment_name "continuous-v2" --total_memory 12 --episode_len 25 --batch_size 4 --n_episodes 10000 --num_layers 3 --num_filters 128 --communicate 0 --n_epochs 3

########################

# python run.py --model "maddpg" --env "spread" --experiment_name "v4-SquareDist" --total_memory 12 --episode_len 25 --batch_size 4 --n_episodes 25000 --num_layers 3 --num_filters 128 --communicate 0 --n_epochs 3

# python run.py --model "maddpg" --env "adversary" --experiment_name "v0" --total_memory 12 --episode_len 25 --batch_size 4 --n_episodes 40000 --num_layers 3 --num_filters 128 --communicate 0 --n_epochs 3

python run.py --model "ppo-rec" --env "simple" --experiment_name "simple-rec4" --total_memory 10 --episode_len 25 --batch_size 5 --n_episodes 3000 --num_layers 3 --num_filters 128 --communicate 3 --n_epochs 3 --lr 0.0001
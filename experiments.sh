############# SIMPLE MADDPG #####################
# python run.py --model "maddpg" --env "simple" --experiment_name "continuous-v0" --total_memory 12 --episode_len 25 --batch_size 4 --n_episodes 10000 --num_layers 3 --num_filters 128 --communicate 0 --n_epochs 3

# python run.py --model "maddpg" --env "simple" --experiment_name "continuous-v1" --total_memory 12 --episode_len 25 --batch_size 4 --n_episodes 10000 --num_layers 3 --num_filters 128 --communicate 0 --n_epochs 3

# python run.py --model "maddpg" --env "simple" --experiment_name "continuous-v2" --total_memory 12 --episode_len 25 --batch_size 4 --n_episodes 10000 --num_layers 3 --num_filters 128 --communicate 0 --n_epochs 3

########################

# python run.py --model "maddpg" --env "spread" --experiment_name "v4-SquareDist" --total_memory 12 --episode_len 25 --batch_size 4 --n_episodes 25000 --num_layers 3 --num_filters 128 --communicate 0 --n_epochs 3

# python run.py --model "maddpg" --env "adversary" --experiment_name "v0" --total_memory 12 --episode_len 25 --batch_size 4 --n_episodes 40000 --num_layers 3 --num_filters 128 --communicate 0 --n_epochs 3

# python run.py --model "ppo_rnn_policy_shared" --env "simple" --experiment_name "ppo-rnn" --episode_len 25 --num-envs 8
python run.py --model "ppo_shared_global_critic" --env "communication" --experiment_name "Big-Run" --num-envs 512 --total-timesteps 50000 --num-minibatches 1 --learning-rate 0.0007 --update-epochs 10 --max-grad-norm 10
# python run.py --model "ppo_policy3" --env "simple" --experiment_name "test" --episode_len 25 --num-envs 8
# python run.py --model "ppo_policy3_shared" --env "communication_full" --experiment_name "reference_shared_info" --episode_len 25 --num-envs 8
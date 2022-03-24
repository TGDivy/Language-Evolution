############# SIMPLE MADDPG #####################
# python run.py --model "maddpg" --env "simple" --experiment_name "continuous-v0" --total_memory 12 --episode_len 25 --batch_size 4 --n_episodes 10000 --num_layers 3 --num_filters 128 --communicate 0 --n_epochs 3

# python run.py --model "maddpg" --env "simple" --experiment_name "continuous-v1" --total_memory 12 --episode_len 25 --batch_size 4 --n_episodes 10000 --num_layers 3 --num_filters 128 --communicate 0 --n_epochs 3

# python run.py --model "maddpg" --env "simple" --experiment_name "continuous-v2" --total_memory 12 --episode_len 25 --batch_size 4 --n_episodes 10000 --num_layers 3 --num_filters 128 --communicate 0 --n_epochs 3

########################

# python run.py --model "maddpg" --env "spread" --experiment_name "v4-SquareDist" --total_memory 12 --episode_len 25 --batch_size 4 --n_episodes 25000 --num_layers 3 --num_filters 128 --communicate 0 --n_epochs 3

# python run.py --model "maddpg" --env "adversary" --experiment_name "v0" --total_memory 12 --episode_len 25 --batch_size 4 --n_episodes 40000 --num_layers 3 --num_filters 128 --communicate 0 --n_epochs 3

# python run.py --model "ppo_rnn_policy_shared" --env "simple" --experiment_name "ppo-rnn" --episode_len 25 --num-envs 8
# python run.py --model "ppo_rec_global_critic" --env "full_communication_3" --experiment_name "main" --num-envs 512 --total-timesteps 75000 --learning-rate 0.0007 --update-epochs 10 --max-grad-norm 10 --episode_len 25 --wandb True #--load_weights_name "/ppo_rec_global_critic-full_communication_2-main"

# python run.py --model "ppo_no_scaling_rec_global_critic" --env "full_communication_2" --experiment_name "main" --num-envs 256 --total-timesteps 25000 --learning-rate 0.0007 --update-epochs 10 --max-grad-norm 10 --episode_len 25 --wandb True #--load_weights_name "/ppo_rec_global_critic-full_communication_2-main"

# python run.py --model "ppo_no_scaling_rec_global_critic" --env "full_communication_3" --experiment_name "no_init_share_representation" --num-envs 512 --total-timesteps 75000 --learning-rate 0.0007 --update-epochs 10 --max-grad-norm 10 --episode_len 25 --wandb True #--load_weights_name "/ppo_no_scaling_rec_global_critic-full_communication_2-main"

# python run.py \
#     --model "ppo_shared_global_critic_rec" \
#     --env "full_communication_3" \
#     --experiment_name "report_env" \
#     --num-envs 1024 \
#     --total-timesteps 50000 \
#     --learning-rate 0.0007 \
#     --update-epochs 10 \
#     --max-grad-norm 10 \
#     --episode_len 25 \
#     --wandb True \
#     --video False

python run.py \
    --model "ppo_shared_use_future" \
    --env "full_communication_2" \
    --experiment_name "test_epcount" \
    --total-episodes 500000 \
    --learning-rate 0.0007 \
    --batch_size 512 \
    --update-epochs 10 \
    --max-grad-norm 10 \
    --episode_len 25 \
    --gru_hidden_size 128 \
    --wandb False \
    --video True
    # --load_weights_name "/ppo_shared_global_critic_rec-full_communication_2-sum_com"

# python run.py \
#     --model "ppo_shared_global_critic_rec" \
#     --env "full_communication_4" \
#     --experiment_name "sum_com" \
#     --num-envs 256 \
#     --total-timesteps 100000 \
#     --learning-rate 0.0007 \
#     --update-epochs 10 \
#     --max-grad-norm 10 \
#     --episode_len 25 \
#     --wandb True \
#     --video True \
#     --load_weights_name "/ppo_shared_global_critic_rec-full_communication_3-sum_com"

# python run.py --model "ppo_policy3" --env "simple" --experiment_name "test" --episode_len 25 --num-envs 8
# python run.py --model "ppo_policy3_shared" --env "communication_full" --experiment_name "reference_shared_info" --episode_len 25 --num-envs 8
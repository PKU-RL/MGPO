# MazeRunner-15
python train.py --env mazerunner --dataset_path 'dataset/mazerunner-d15-g4-4-t64-multigoal-astar.pkl' \
--max_prompt_len 5 --K 64 --max_ep_len 64 --batch_size 64 --test_optimal_prompt \
--subsample_trajectory --subsample_min_len 10

# MazeRunner-30
python train.py --env mazerunner --dataset_path 'dataset/mazerunner-d30-g4-4-t500-multigoal-astar.pkl' \
--max_prompt_len 5 --K 500 --max_ep_len 500 --batch_size 64 --test_optimal_prompt \
--subsample_trajectory --subsample_min_len 10 --test_eval_interval 2000 --max_iters 20000

# GridWorld
python train.py --env kitchen_toy --dataset_path 'dataset/kitchen_toy_t90' \
--max_prompt_len 8 --K 90 --max_ep_len 90 --batch_size 64

# Kitchen
python train.py --env kitchen --dataset_path 'dataset/kitchen_t500' \
--max_prompt_len 6 --K 500 --max_ep_len 500 --batch_size 64 \
--test_eval_interval 2000 --max_iters 20000

# Crafter
python train.py --env crafter --dataset_path 'dataset/crafter_dataset' \
--max_prompt_len 23 --K 500 --max_ep_len 500 --batch_size 64 \
--test_eval_interval 200 --max_iters 5000
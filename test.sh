## MazeRunner-15

# explore
python prompt_tuning_random_search.py --env mazerunner \
--dataset_path 'dataset/mazerunner-d15-g4-4-t64-multigoal-astar.pkl' \
--max_prompt_len 5 --K 64 --max_ep_len 64 \
--load-path 'model_saved/gym-experiment-mazerunner-mazerunner-d15-g4-4-t64-multigoal-astar-20240121180027/prompt_model_mazerunner_iter_4999' \
--task -1 --max_test_episode 100

# ucb
python prompt_tuning_ucb.py --env mazerunner --dataset_path 'dataset/mazerunner-d15-g4-4-t64-multigoal-astar.pkl' \
--max_prompt_len 5 --K 64 --max_ep_len 64 \
--load-path 'model_saved/gym-experiment-mazerunner-mazerunner-d15-g4-4-t64-multigoal-astar-20240121180027/prompt_model_mazerunner_iter_4999' \
--task -1 --max_test_episode 100

# bpe
python prompt_tuning_bpe.py --env mazerunner --dataset_path 'dataset/mazerunner-d15-g4-4-t64-multigoal-astar.pkl' \
--max_prompt_len 5 --K 64 --max_ep_len 64 \
--load-path 'model_saved/gym-experiment-mazerunner-mazerunner-d15-g4-4-t64-multigoal-astar-20240121180027/prompt_model_mazerunner_iter_4999' \
--task -1 --max_test_episode 100


## MazeRunner-30

# explore
python prompt_tuning_random_search.py --env mazerunner \
--dataset_path 'dataset/mazerunner-d30-g4-4-t500-multigoal-astar.pkl' \
--max_prompt_len 5 --K 500 --max_ep_len 500 \
--load-path 'model_saved/gym-experiment-mazerunner-mazerunner-d30-g4-4-t500-multigoal-astar-20240125182845/prompt_model_mazerunner_iter_19999'  \
--task -1 --max_test_episode 100

# ucb
python prompt_tuning_ucb.py --env mazerunner --dataset_path 'dataset/mazerunner-d30-g4-4-t500-multigoal-astar.pkl' \
--max_prompt_len 5 --K 500 --max_ep_len 500 \
--load-path 'model_saved/gym-experiment-mazerunner-mazerunner-d30-g4-4-t500-multigoal-astar-20240125182845/prompt_model_mazerunner_iter_19999'  \
--task -1 --max_test_episode 100

# bpe
python prompt_tuning_bpe.py --env mazerunner --dataset_path 'dataset/mazerunner-d30-g4-4-t500-multigoal-astar.pkl' \
--max_prompt_len 5 --K 500 --max_ep_len 500 \
--load-path 'model_saved/gym-experiment-mazerunner-mazerunner-d30-g4-4-t500-multigoal-astar-20240125182845/prompt_model_mazerunner_iter_19999' \
--task -1 --max_test_episode 100


## GridWorld

# explore
python prompt_tuning_random_search.py --env kitchen_toy --dataset_path 'dataset/kitchen_toy_t90' \
--max_prompt_len 8 --K 90 --max_ep_len 90 \
--load-path 'model_saved/gym-experiment-kitchen_toy-kitchen_toy_t90-20240119181721/prompt_model_kitchen_toy_iter_4999' \
--task -1 --max_test_episode 100

# ucb
python prompt_tuning_ucb.py --env kitchen_toy --dataset_path 'dataset/kitchen_toy_t90' \
--max_prompt_len 8 --K 90 --max_ep_len 90 \
--load-path 'model_saved/gym-experiment-kitchen_toy-kitchen_toy_t90-20240119181721/prompt_model_kitchen_toy_iter_4999' \
--task -1 --max_test_episode 100

# bpe
python prompt_tuning_bpe.py --env kitchen_toy --dataset_path 'dataset/kitchen_toy_t90' \
--max_prompt_len 8 --K 90 --max_ep_len 90 \
--load-path 'model_saved/gym-experiment-kitchen_toy-kitchen_toy_t90-20240119181721/prompt_model_kitchen_toy_iter_4999' \
--task -1 --max_test_episode 100



##### Kitchen

# explore
python prompt_tuning_random_search.py --env kitchen --dataset_path 'dataset/kitchen_t500' \
--max_prompt_len 6 --K 500 --max_ep_len 500 \
--load-path 'model_saved/gym-experiment-kitchen-kitchen_t500-20240125123019/prompt_model_kitchen_iter_16000' \
--task -1 --max_test_episode 100

# ucb
python prompt_tuning_ucb.py --env kitchen --dataset_path 'dataset/kitchen_t500' \
--max_prompt_len 6 --K 500 --max_ep_len 500 \
--load-path 'model_saved/gym-experiment-kitchen-kitchen_t500-20240125123019/prompt_model_kitchen_iter_16000' \
--task -1 --max_test_episode 100

# bpe
python prompt_tuning_bpe.py --env kitchen --dataset_path 'dataset/kitchen_t500' \
--max_prompt_len 6 --K 500 --max_ep_len 500 \
--load-path 'model_saved/gym-experiment-kitchen-kitchen_t500-20240125123019/prompt_model_kitchen_iter_16000' \
--task -1 --max_test_episode 100


##### Crafter
# explore
python prompt_tuning_random_search.py --env crafter --dataset_path 'dataset/crafter_dataset' \
--max_prompt_len 23 --K 500 --max_ep_len 500 \
--load-path 'model_saved/gym-experiment-crafter-crafter_dataset-20240124170136/prompt_model_crafter_iter_4999' \
--task -1 --max_test_episode 100

# ucb
python prompt_tuning_ucb.py --env crafter --env crafter --dataset_path 'dataset/crafter_dataset' \
--max_prompt_len 23 --K 500 --max_ep_len 500 \
--load-path 'model_saved/gym-experiment-crafter-crafter_dataset-20240124170136/prompt_model_crafter_iter_4999' \
--task -1 --max_test_episode 100

# bpe
python prompt_tuning_bpe.py --env crafter --env crafter --dataset_path 'dataset/crafter_dataset' \
--max_prompt_len 23 --K 500 --max_ep_len 500 \
--load-path 'model_saved/gym-experiment-crafter-crafter_dataset-20240124170136/prompt_model_crafter_iter_4999' \
--task -1 --max_test_episode 100
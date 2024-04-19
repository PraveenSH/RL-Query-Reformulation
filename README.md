# Summary
This repo implements a simple Reinforcement learning framework for search query rewriting.
It uses sequence-to-sequence model to generate reformulated query and uses policy gradient algorithm to fine-tune the model.
Reward functions implemented here aims to improve the model’s ability to generate queries with a greater variety of paraphrased keywords. It can be replaced with any other goal specific reward functions or models.

# Framework Details
Framework uses sequence-to-sequence text generation model. It takes a search query as input and generates reformulated queries. Reformulated queries can be further used in retrieval to improve the search results.

## Steps involved in training the model 
1) The sequence-to-sequence model is initialized with [google's t5-base model](https://huggingface.co/google-t5/t5-base).
2) This model is first trained in supervised manner ([code](https://github.com/PraveenSH/RL-Query-Reformulation/blob/master/src/t5_supervised_trainer.py)) using [ms-marco query pairs data](https://github.com/Narabzad/msmarco-query-reformulation/tree/main/datasets/queries)
3) Model is then fine-tuned with an RL framework ([code](https://github.com/PraveenSH/RL-Query-Reformulation/blob/master/src/t5_reward_trainer.py)) to further improve the model's capability generate more diverse but relevant queries.
4) It uses a policy gradient approach to fine-tune the policy (sequence-to-sequence model). For a given input query, a set of trajectories (reformulated queries) are sampled from the model and reward is computed. Policy gradient algorithm is applied to update the model.   
5) Reward is computed heuristically ([code](https://github.com/PraveenSH/RL-Query-Reformulation/blob/master/src/reward_model.py)) to improve the paraphrasing capability. But this can be replaced with any other domain/goal specific reward functions.

<img width="617" alt="image" src="https://github.com/PraveenSH/RL-Query-Reformulation/assets/8490324/ac3639d0-00fd-4e12-9aa1-984a87ddb2c3">

## Reward Function
Reward function used in this implementation increases the model's capability to paraphrase. This can be helpful in sparse retrieval methods such as BM25 based techniques since model generates a greater variety of paraphrased keywords. The reward function can be adjusted to optimize for other retrieval goals, such as enhancing the coverage of product categories in an e-commerce search or improving relevance in academic literature or medical research.

# Model Usage
1) Final trained model is released on hugging-face here.
2) Script to run the model inference - [code](https://github.com/PraveenSH/RL-Query-Reformulation/blob/master/src/t5_inference.py)

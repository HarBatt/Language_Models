import random
import torch
import numpy as np
from BERT import BERTSequence
from datasets import load_datasets_and_vocabs

class Parameters:
    def __init__(self):
        pass 

parameters = Parameters()
parameters.seed = 26

random.seed(parameters.seed)
np.random.seed(parameters.seed)
torch.manual_seed(parameters.seed)
torch.cuda.manual_seed_all(parameters.seed)


# Required parameters
parameters.dataset_name = "rest"
parameters.output_dir = 'data/BERT/'
parameters.num_classes = 3

# Model parameters
parameters.add_non_connect = True
parameters.multi_hop = True
parameters.max_hop = 4
# Training parameters
parameters.per_gpu_train_batch_size = 16
parameters.per_gpu_eval_batch_size = 32
parameters.gradient_accumulation_steps = 2
parameters.learning_rate = 5e-5
parameters.weight_decay = 0.0
parameters.adam_epsilon = 1e-8
parameters.max_grad_norm = 1.0
parameters.num_train_epochs = 30.0
parameters.max_steps = -1
parameters.logging_steps = 100
parameters.bert_model_dir ='bert-base-uncased'



device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
parameters.device = device
print('On Device: {}'.format(parameters.device))


# Load datasets and vocabs
train_dataset, test_dataset, word_vocab, dep_tag_vocab, pos_tag_vocab= load_datasets_and_vocabs(parameters)

# Build Model

model = BERTSequence(parameters)
# Train
_, _,  all_eval_results = model.train(train_dataset, test_dataset)

if len(all_eval_results):
    best_eval_result = max(all_eval_results, key=lambda x: x['acc']) 
    for key in sorted(best_eval_result.keys()):
        print("{} ={}".format(key, str(best_eval_result[key])))



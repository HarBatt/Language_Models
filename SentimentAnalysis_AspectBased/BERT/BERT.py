import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from sklearn.metrics import f1_score

from transformers import BertModel, BertConfig, BertTokenizer
from transformers import AdamW

from tqdm import trange
from datasets import my_collate_pure_bert
from shared.component_logger import component_logger as logger


class BERT(nn.Module):
    def __init__(self, params, hidden_size=256):
        super(BERT, self).__init__()
        config = BertConfig.from_pretrained(params.bert_model_dir)
        self.tokenizer = BertTokenizer.from_pretrained(params.bert_model_dir)
        self.bert = BertModel.from_pretrained(params.bert_model_dir, config=config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

        layers = [
            nn.Linear(config.hidden_size, hidden_size),
            nn.ReLU(), 
            nn.Linear(hidden_size, params.num_classes)
            ]
        self.classifier = nn.Sequential(*layers)

    def forward(self, input_ids, token_type_ids):
        outputs = self.bert(input_ids, token_type_ids=token_type_ids)
        pooled_output = outputs[1]
        # pool output is usually *not* a good summary of the semantic content of the input, you're often better with averaging or poolin the sequence of hidden-states for the whole input sequence.
        # pooled_output = torch.mean(pooled_output, dim = 1)
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)

        return logits


class BERTSequence:
    def __init__(self, params):
        self.params = params
        self.bert = BERT(params).to(params.device)
    
    
    def get_bert_optimizer(self, params):
        # Prepare optimizer and schedule (linear warmup and decay)
        no_decay = ['bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in self.bert.named_parameters() if not any(nd in n for nd in no_decay)],
            'weight_decay': params.weight_decay},
            {'params': [p for n, p in self.bert.named_parameters() if any(
                nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]
        optimizer = AdamW(optimizer_grouped_parameters, lr=params.learning_rate, eps=params.adam_epsilon)

        return optimizer


    def train(self, train_dataset, test_dataset):
        """Train the model"""
        params = self.params
        params.train_batch_size = params.per_gpu_train_batch_size
        train_sampler = RandomSampler(train_dataset)
        collate_fn = my_collate_pure_bert
        train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=params.train_batch_size, collate_fn=collate_fn)

        if params.max_steps > 0:
            t_total = params.max_steps
            params.num_train_epochs = params.max_steps // (
                len(train_dataloader) // params.gradient_accumulation_steps) + 1
        else:
            t_total = len(
                train_dataloader) // params.gradient_accumulation_steps * params.num_train_epochs


        optimizer = self.get_bert_optimizer(params)


        # Train
        logger.log("***** Running training *****")
        logger.log("Num examples = {}".format(len(train_dataset)))
        logger.log("Num Epochs = {}".format(params.num_train_epochs))
        logger.log("Instantaneous batch size per GPU = {}".format(params.per_gpu_train_batch_size))
        logger.log("Gradient Accumulation steps = {}".format(params.gradient_accumulation_steps))
        logger.log("Total optimization steps = {}".format(t_total))


        global_step = 0
        tr_loss, logging_loss = 0.0, 0.0
        all_eval_results = []
        self.bert.zero_grad()
        train_iterator = trange(int(params.num_train_epochs), desc="Epoch")

        for _ in train_iterator:
            # epoch_iterator = tqdm(train_dataloader, desc='Iteration')
            for step, batch in enumerate(train_dataloader):
                self.bert.train()
                batch = tuple(t.to(params.device) for t in batch)
                inputs = {'input_ids': batch[0], 'token_type_ids': batch[1]}
                labels = batch[6]

                logit = self.bert(**inputs)
                loss = F.cross_entropy(logit, labels)

                if params.gradient_accumulation_steps > 1:
                    loss = loss / params.gradient_accumulation_steps

                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.bert.parameters(), params.max_grad_norm)

                tr_loss += loss.item()
                if (step + 1) % params.gradient_accumulation_steps == 0:
                    # scheduler.step()  # Update learning rate schedule
                    optimizer.step()
                    self.bert.zero_grad()
                    global_step += 1

                    # Log metrics
                    if params.logging_steps > 0 and global_step % params.logging_steps == 0:
                        results, eval_loss = self.evaluate(test_dataset)
                        all_eval_results.append(results)
                        for key, value in results.items():
                            logger.log('eval_{} {}'.format(key, value))
                            logger.log('eval_loss {}'.format(eval_loss))
                            logger.log('train_loss {}'.format((tr_loss - logging_loss) / params.logging_steps))
                        logging_loss = tr_loss


        return global_step, tr_loss/global_step, all_eval_results


    def evaluate(self, eval_dataset):
        params = self.params
        results = {}

        params.eval_batch_size = params.per_gpu_eval_batch_size
        eval_sampler = SequentialSampler(eval_dataset)
        collate_fn = my_collate_pure_bert
        eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=params.eval_batch_size, collate_fn=collate_fn)

        # Eval
        logger.log("***** Running evaluation *****")
        logger.log("Num examples = {}".format(len(eval_dataset)))
        logger.log("Batch size = {}".format(params.eval_batch_size))
        eval_loss = 0.0
        nb_eval_steps = 0
        preds = None
        out_label_ids = None
        for batch in eval_dataloader:
        # for batch in tqdm(eval_dataloader, desc='Evaluating'):
            self.bert.eval()
            batch = tuple(t.to(params.device) for t in batch)
            with torch.no_grad():
                inputs = {'input_ids': batch[0], 'token_type_ids': batch[1]}
                labels = batch[6]

                logits = self.bert(**inputs)
                tmp_eval_loss = F.cross_entropy(logits, labels)

                eval_loss += tmp_eval_loss.mean().item()
            nb_eval_steps += 1
            if preds is None:
                preds = logits.detach().cpu().numpy()
                out_label_ids = labels.detach().cpu().numpy()
            else:
                preds = np.append(preds, logits.detach().cpu().numpy(), axis=0)
                out_label_ids = np.append(
                    out_label_ids, labels.detach().cpu().numpy(), axis=0)

        eval_loss = eval_loss / nb_eval_steps
        preds = np.argmax(preds, axis=1)
        # print(preds)
        result = self.compute_metrics(preds, out_label_ids)
        results.update(result)


        logger.log('***** Eval results *****')
        logger.log("eval loss: {}".format(eval_loss))
        for key in sorted(result.keys()):
            logger.log("{} = {}".format(key, str(result[key])))

        return results, eval_loss

    def compute_metrics(self, preds, labels):
        acc = (preds == labels).mean()
        f1 = f1_score(y_true=labels, y_pred=preds, average='macro')
        return {
            "acc": acc,
            "f1": f1
        }


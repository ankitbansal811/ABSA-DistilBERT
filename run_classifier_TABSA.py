# coding=utf-8

"""BERT finetuning runner."""

from __future__ import absolute_import, division, print_function

import argparse
import collections
import logging
import os
import random
import shutil

import numpy as np
import torch 
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data.sampler import RandomSampler, SequentialSampler
from tqdm import tqdm, trange

import tokenization
from modeling import BertConfig, BertForSequenceClassification, BertBinaryClassifier
from optimization import BERTAdam
from processor import Sentihood_QA_M_Processor

logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s', 
                    datefmt = '%m/%d/%Y %H:%M:%S',
                    level = logging.INFO)
logger = logging.getLogger(__name__)

from torch.optim import Adam

class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self, input_ids, input_mask, label_id):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.label_id = label_id


def convert_examples_to_features(examples, label_list, max_seq_length, tokenizer):
    """Loads a data file into a list of `InputBatch`s."""

    label_map = {}
    for (i, label) in enumerate(label_list):
        label_map[label] = i

    features = []
    for (ex_index, example) in enumerate(tqdm(examples)):
        tokens_a = tokenizer.tokenize(example.text_a)

        tokens_b = None
        if example.text_b:
            tokens_b = tokenizer.tokenize(example.text_b)

        if tokens_b:
            # Modifies `tokens_a` and `tokens_b` in place so that the total
            # length is less than the specified length.
            # Account for [CLS], [SEP], [SEP] with "- 3"
            _truncate_seq_pair(tokens_a, tokens_b, max_seq_length - 3)
        else:
            # Account for [CLS] and [SEP] with "- 2"
            if len(tokens_a) > max_seq_length - 2:
                tokens_a = tokens_a[0:(max_seq_length - 2)]

        # The convention in BERT is:
        # (a) For sequence pairs:
        #  tokens:   [CLS] is this jack ##son ##ville ? [SEP] no it is not . [SEP]
        #  type_ids: 0   0  0    0    0     0       0 0    1  1  1  1   1 1
        # (b) For single sequences:
        #  tokens:   [CLS] the dog is hairy . [SEP]
        #  type_ids: 0   0   0   0  0     0 0
        #
        # Where "type_ids" are used to indicate whether this is the first
        # sequence or the second sequence. The embedding vectors for `type=0` and
        # `type=1` were learned during pre-training and are added to the wordpiece
        # embedding vector (and position vector). This is not *strictly* necessary
        # since the [SEP] token unambigiously separates the sequences, but it makes
        # it easier for the model to learn the concept of sequences.
        #
        # For classification tasks, the first vector (corresponding to [CLS]) is
        # used as as the "sentence vector". Note that this only makes sense because
        # the entire model is fine-tuned.
        tokens = []
        tokens.append("[CLS]")
        for token in tokens_a:
            tokens.append(token)
        tokens.append("[SEP]")

        if tokens_b:
            for token in tokens_b:
                tokens.append(token)
            tokens.append("[SEP]")

        input_ids = tokenizer.convert_tokens_to_ids(tokens)

        # The mask has 1 for real tokens and 0 for padding tokens. Only real
        # tokens are attended to.
        input_mask = [1] * len(input_ids)

        # Zero-pad up to the sequence length.
        while len(input_ids) < max_seq_length:
            input_ids.append(0)
            input_mask.append(0)

        assert len(input_ids) == max_seq_length
        assert len(input_mask) == max_seq_length

        label_id = label_map[example.label]

        features.append(
                InputFeatures(
                        input_ids=input_ids,
                        input_mask=input_mask,
                        label_id=label_id))
    return features


def _truncate_seq_pair(tokens_a, tokens_b, max_length):
    """Truncates a sequence pair in place to the maximum length."""

    # This is a simple heuristic which will always truncate the longer sequence
    # one token at a time. This makes more sense than truncating an equal percent
    # of tokens from each, since if one sequence is very short then each token
    # that's truncated likely contains more information than a longer sequence.
    while True:
        total_length = len(tokens_a) + len(tokens_b)
        if total_length <= max_length:
            break
        if len(tokens_a) > len(tokens_b):
            tokens_a.pop()
        else:
            tokens_b.pop()

def arg_parser():
    parser = argparse.ArgumentParser()

    ## Required parameters
    # parser.add_argument("--task_name", default=None, type=str, required=True,
    #                     choices=["sentihood_single", "sentihood_NLI_M", "sentihood_QA_M", \
    #                             "sentihood_NLI_B", "sentihood_QA_B", "semeval_single", \
    #                             "semeval_NLI_M", "semeval_QA_M", "semeval_NLI_B", "semeval_QA_B"],
    #                     help="The name of the task to train.")
    parser.add_argument("--data_dir", default="data/sentihood/bert-pair", type=str, required=False,
                        help="The input data dir. Should contain the .tsv files (or other data files) for the task.")
    parser.add_argument("--vocab_file", default=None, type=str, required=True,
                        help="The vocabulary file that the BERT model was trained on.")
    parser.add_argument("--output_dir", default=None, type=str, required=True,
                        help="The output directory where the model checkpoints will be written.")
    parser.add_argument("--init_checkpoint", default=None, type=str, required=True,
                        help="Initial checkpoint (usually from a pre-trained BERT model).")
    
    ## Other parameters
    parser.add_argument("--eval_test", default=False, action='store_true', help="Whether to run eval on the test set.")                    
    parser.add_argument("--do_lower_case", default=False, action='store_true',
                        help="Whether to lower case the input text. True for uncased models, False for cased models.")
    parser.add_argument("--max_seq_length", default=128, type=int,
                        help="The maximum total input sequence length after WordPiece tokenization. \n"
                             "Sequences longer than this will be truncated, and sequences shorter \n"
                             "than this will be padded.")
    parser.add_argument("--train_batch_size", default=32, type=int, help="Total batch size for training.")
    parser.add_argument("--eval_batch_size", default=8, type=int, help="Total batch size for eval.")
    parser.add_argument("--learning_rate", default=5e-5, type=float, help="The initial learning rate for Adam.")
    parser.add_argument("--num_train_epochs", default=3.0, type=float, help="Total number of training epochs to perform.")
    parser.add_argument("--warmup_proportion", default=0.1, type=float, help="Proportion of training to perform linear learning rate warmup for. "
                             "E.g., 0.1 = 10%% of training.")
    parser.add_argument("--no_cuda", default=False, action='store_true', help="Whether not to use CUDA when available")
    parser.add_argument("--accumulate_gradients", type=int, default=1,
                        help="Number of steps to accumulate gradient on (divide the batch_size and accumulate)")
    parser.add_argument("--local_rank", type=int, default=-1, help="local_rank for distributed training on gpus")
    parser.add_argument('--seed', type=int,  default=42, help="random seed for initialization")
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1,
                        help="Number of updates steps to accumualte before performing a backward/update pass.")                       
    args = parser.parse_args()
    return args


def get_data(processor, label_list, tokenizer, data_dir, set_type):
    examples = processor.get_examples(data_dir, set_type)
    features = convert_examples_to_features(examples, label_list, args.max_seq_length, tokenizer)

    all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
    all_input_mask = torch.tensor([f.input_mask for f in features], dtype=torch.long)
    all_label_ids = torch.tensor([f.label_id for f in features], dtype=torch.long)

    data = TensorDataset(all_input_ids, all_input_mask, all_label_ids)
    return data, len(examples)


def main(args):
    if args.local_rank == -1 or args.no_cuda:
        device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        n_gpu = torch.cuda.device_count()
    else:
        device = torch.device("cuda", args.local_rank)
        n_gpu = 1
        # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.distributed.init_process_group(backend='nccl')
    logger.info("device %s n_gpu %d distributed training %r", device, n_gpu, bool(args.local_rank != -1))

    if args.accumulate_gradients < 1:
        raise ValueError("Invalid accumulate_gradients parameter: {}, should be >= 1".format(
                            args.accumulate_gradients))

    args.train_batch_size = int(args.train_batch_size / args.accumulate_gradients)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)

    bert_config_file = "distilbert_config.json"
    bert_config = BertConfig.from_json_file(bert_config_file)

    if args.max_seq_length > bert_config.max_position_embeddings:
        raise ValueError(
            "Cannot use sequence length {} because the BERT model was only trained up to sequence length {}".format(
            args.max_seq_length, bert_config.max_position_embeddings))

    if os.path.exists(args.output_dir) and os.listdir(args.output_dir):
        shutil.rmtree(args.output_dir)
        # raise ValueError("Output directory ({}) already exists and is not empty.".format(args.output_dir))
    os.makedirs(args.output_dir, exist_ok=True)


    # prepare dataloaders
    processor = Sentihood_QA_M_Processor()
    label_list = processor.get_labels()
    tokenizer = tokenization.FullTokenizer(vocab_file=args.vocab_file, do_lower_case=args.do_lower_case)

    # training set
    num_train_steps = None

    train_data, len_train = get_data(processor, label_list, tokenizer, args.data_dir, 'train')
    if args.local_rank == -1:
        train_sampler = RandomSampler(train_data)
    else:
        train_sampler = DistributedSampler(train_data)
    train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=args.train_batch_size)

    num_train_steps = int(len_train / args.train_batch_size * args.num_train_epochs)
    logger.info("***** Running training *****")
    logger.info("  Num Training examples = %d", len_train)
    logger.info("  Training Batch size = %d", args.train_batch_size)
    logger.info("  Training Num steps = %d", num_train_steps)

    # test set
    if args.eval_test:
        test_data, len_test = get_data(processor, label_list, tokenizer, args.data_dir, 'test')
        test_dataloader = DataLoader(test_data, batch_size=args.eval_batch_size, shuffle=False)
        logger.info("  Num Test examples = %d", len_test)

    # model and optimizer
    #model = BertForSequenceClassification(bert_config, len(label_list))
    #if args.init_checkpoint is not None:
    #    model.bert.load_state_dict(torch.load(args.init_checkpoint, map_location='cpu'))
    
    model = BertBinaryClassifier(len(label_list), dropout = .1)

    model.to(device)

    if args.local_rank != -1:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank],
                                                          output_device=args.local_rank)
    elif n_gpu > 1:
        model = torch.nn.DataParallel(model)
    
    no_decay = ['bias', 'gamma', 'beta']
    optimizer_parameters = [
         {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay_rate': 0.01},
         {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay_rate': 0.0}
         ]
		
    optimizer = BERTAdam(optimizer_parameters,
                         lr=args.learning_rate,
                         warmup=args.warmup_proportion,
                         t_total=num_train_steps)
    #optimizer = Adam(model.parameters(), lr=args.learning_rate)
    # train
    output_log_file = os.path.join(args.output_dir, "log.txt")
    print("output_log_file=",output_log_file)
    with open(output_log_file, "w") as writer:
        if args.eval_test:
            writer.write("epoch\tglobal_step\tloss\ttest_loss\ttest_accuracy\n")
        else:
            writer.write("epoch\tglobal_step\tloss\n")
    
    global_step = 0
    epoch=0
    
    for epoch_num in range(int(args.num_train_epochs)):
        model.train()
        train_loss, nb_train_steps = 0, 0
        for step, batch_data in enumerate(tqdm(train_dataloader, desc="Iteration")):
            #token_ids, masks, labels = tuple(t.to(device) for t in batch_data)
            batch = tuple(t.to(device) for t in batch_data)
            input_ids, input_mask, label_ids = batch
            outputs = model(input_ids, input_mask, label_ids)
           
            #logits is the proba
            loss, logits = outputs[:2]
            train_loss += loss.item()
            loss.backward()
            nb_train_steps += 1
            if (step + 1) % args.gradient_accumulation_steps == 0:
                optimizer.step()    # We have accumulated enought gradients
                model.zero_grad()
                global_step += 1
            
            if step%100==0:
                logger.info("Epoch = %d, Batch = %d", epoch_num, (step+1))
                logger.info("Batch loss = %f, Avg loss = %f", loss.item(), train_loss/(step+1))

            if global_step%1000==0:
                logger.info("Creating a checkpoint.")
                model.eval().cpu()
                ckpt_model_filename = "ckpt_epoch_" + str(epoch_num) + "_steps_" + str(global_step) + ".pth"
                ckpt_model_path = os.path.join(args.output_dir, ckpt_model_filename)
                torch.save(model.state_dict(), ckpt_model_path)
                model.to(device)
        
        logger.info("Training loss after %f epoch = %f", epoch_num, train_loss)

        # eval_test
        if args.eval_test:
            model.eval()
            test_loss, test_accuracy = 0, 0
            nb_test_steps, nb_test_examples = 0, 0
            with open(os.path.join(args.output_dir, "test_ep_"+str(epoch)+".txt"),"w") as f_test:
                for input_ids, input_mask, label_ids in test_dataloader:
                    print(nb_test_steps)
                    input_ids = input_ids.to(device)
                    input_mask = input_mask.to(device)
                    label_ids = label_ids.to(device)

                    with torch.no_grad():
                        outputs = model(input_ids, input_mask, label_ids)

                    tmp_test_loss = outputs[0]
                    logits = outputs[1]
                    logits = logits.detach().cpu().numpy()
                    label_ids = label_ids.to('cpu').numpy()
                    outputs = np.argmax(logits, axis=1)
                    for output_i in range(len(outputs)):
                        f_test.write(str(outputs[output_i]))
                        for ou in logits[output_i]:
                            f_test.write(" "+str(ou))
                        f_test.write("\n")
                    tmp_test_accuracy=np.sum(outputs == label_ids)

                    test_loss += tmp_test_loss.mean().item()
                    test_accuracy += tmp_test_accuracy

                    nb_test_examples += input_ids.size(0)
                    nb_test_steps += 1

            test_loss = test_loss / nb_test_steps
            test_accuracy = test_accuracy / nb_test_examples

        
        result = collections.OrderedDict()
        if args.eval_test:
            result = {'epoch': epoch,
                    'global_step': global_step,
                    'loss': train_loss/nb_train_steps,
                    'test_loss': test_loss,
                    'test_accuracy': test_accuracy}
        else:
            result = {'epoch': epoch,
                    'global_step': global_step,
                    'loss': train_loss/nb_train_steps}

        logger.info("***** Eval results *****")
        with open(output_log_file, "a+") as writer:
            print("Final Classfier Results: ", result)
            for key in result.keys():
                logger.info("  %s = %s\n", key, str(result[key]))
                writer.write("%s\t" % (str(result[key])))
            writer.write("\n")
    

if __name__ == "__main__":
    args = arg_parser()
    main(args)

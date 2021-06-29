#encoding=utf-8
import glob
import logging
import json
import time
import copy
import shutil
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, TensorDataset
from torch.utils.data.distributed import DistributedSampler
from callback.optimizater.adamw import AdamW
from callback.lr_scheduler import get_linear_schedule_with_warmup
from callback.progressbar import ProgressBar
from tools.common import seed_everything,json_to_text
from tools.common import init_logger, logger
from tools.func import loadList, saveList

from models.transformers import WEIGHTS_NAME, BertConfig, AlbertConfig
from models.bertqa_eca import BertForECA
from models.get_cause_spans import get_pre_indexs, get_tru_indexs, get_pre_sti_indexs

from processors_eca.utils_eca import EcaTokenizer
from processors_eca.eca_seq import convert_examples_to_features
from processors_eca.eca_seq import eca_processors as processors, batch_generator, bert_extract_item, extract_multi_item # ner_processors = {"cner": CnerProcessor,'cluener':CluenerProcessor, 'eca':ECAProcessor}
from metrics.eca_metrics import get_prf 
from tools.finetuning_argparse_eca import get_argparse
import numpy as np
import os

args = get_argparse().parse_args()
os.environ["CUDA_VISIBLE_DEVICES"] = str(args.Gpu_num)

MODEL_CLASSES = { 'bert': (BertConfig,BertForECA, EcaTokenizer) }

if args.data_type == 'ch':  
    args.model_name_or_path = os.path.join(args.root_path,'bert_base_ch')
elif args.data_type == 'sti':
    args.model_name_or_path = os.path.join(args.root_path,'bert_base_en')

def train(args, train_features, model, tokenizer, CLS_ID, SEP_ID):
    """ Train the model """
    # Prepare optimizer and schedule (linear warmup and decay)
    no_decay = ["bias", "LayerNorm.weight"]
    bert_param_optimizer = list(model.bert.named_parameters())
    bert_C_param_optimizer = list(model.bert_gate.named_parameters())
    GNN_param_optimizer = list(model.GAN.named_parameters())

    linear_param_optimal_one = list(model.qa_outputs.named_parameters())
    linear_param_optimal_two = list(model.num_outputs.named_parameters())
    linear_param_optimal_three = list(model.clause_out.named_parameters())

    optimizer_grouped_parameters = [
            {'params': [p for n, p in bert_param_optimizer if not any(nd in n for nd in no_decay)],
            'weight_decay': args.weight_decay, 'lr': args.learning_rate},
            {'params': [p for n, p in bert_param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0,
            'lr': args.learning_rate},

            {'params': [p for n, p in bert_C_param_optimizer if not any(nd in n for nd in no_decay)],
            'weight_decay': args.weight_decay, 'lr': args.learning_rate},
            {'params': [p for n, p in bert_C_param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0,
            'lr': args.learning_rate},

            {'params': [p for n, p in GNN_param_optimizer if not any(nd in n for nd in no_decay)],
            'weight_decay': args.weight_decay, 'lr': args.crf_learning_rate},
            {'params': [p for n, p in GNN_param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0,
            'lr': args.crf_learning_rate},

            {'params': [p for n, p in linear_param_optimal_one if not any(nd in n for nd in no_decay)],
            'weight_decay': args.weight_decay, 'lr': args.crf_learning_rate},
            {'params': [p for n, p in linear_param_optimal_one if any(nd in n for nd in no_decay)], 'weight_decay': 0.0,
            'lr': args.crf_learning_rate},

            {'params': [p for n, p in linear_param_optimal_two if not any(nd in n for nd in no_decay)],
            'weight_decay': args.weight_decay, 'lr': args.crf_learning_rate},
            {'params': [p for n, p in linear_param_optimal_two if any(nd in n for nd in no_decay)], 'weight_decay': 0.0,
            'lr': args.crf_learning_rate},

             {'params': [p for n, p in linear_param_optimal_three if not any(nd in n for nd in no_decay)],
            'weight_decay': args.weight_decay, 'lr': args.crf_learning_rate},
            {'params': [p for n, p in linear_param_optimal_three if any(nd in n for nd in no_decay)], 'weight_decay': 0.0,
            'lr': args.crf_learning_rate}
    ]
    
    t_total = len(train_features)//args.batch_size * args.num_train_epochs
    args.warmup_steps = int(t_total * args.warmup_proportion)

    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=args.warmup_steps,
                                                num_training_steps=t_total)
    # Check if saved optimizer or scheduler states exist
    if os.path.isfile(os.path.join(args.model_name_or_path, "optimizer.pt")) and os.path.isfile(
            os.path.join(args.model_name_or_path, "scheduler.pt")):
        # Load in optimizer and scheduler states
        optimizer.load_state_dict(torch.load(os.path.join(args.model_name_or_path, "optimizer.pt")))
        scheduler.load_state_dict(torch.load(os.path.join(args.model_name_or_path, "scheduler.pt")))
    
    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", len(train_features))
    logger.info("  Num Epochs = %d", args.num_train_epochs)
    logger.info("  Instantaneous batch size per GPU = %d", args.batch_size)
    logger.info("  Total train batch size (w. parallel, distributed & accumulation) = %d",
                args.batch_size
                * (torch.distributed.get_world_size() if args.local_rank != -1 else 1),
                )
    # logger.info("  Gradient Accumulation steps = %d", args.gradient_accumulation_steps)
    logger.info("  Total optimization steps = %d", t_total)

    global_step = 0
    tr_loss, logging_loss  = 0.0, 0.0
    pre_result = {}
    model.zero_grad()
    seed_everything(args.seed)  # Added here for reproductibility (even between python 2 and 3)
    total_step = 0
    best_spanf = -1

    test_results = {}
    for ep in range(int(args.num_train_epochs)):
        pbar = ProgressBar(n_total=len(train_features)//args.batch_size, desc='Training')
        # if ep == int(args.num_train_epochs) - 1:
            # eval_features = load_and_cache_examples(args, args.data_type, tokenizer, data_type='dev')
            # train_features.extend(eval_features)
        step= 0
        for batch in batch_generator(features = train_features, CLS_ID = CLS_ID, SEP_ID = SEP_ID, batch_size=args.batch_size):

            batch_input_ids, batch_input_mask, batch_segment_ids, batch_context_mask, combine_input_ids, combine_input_mask, combine_segment_ids, combine_lens, combine_clause_mask, batch_start_position, batch_end_position, clause_lens, clause_num,batch_example = batch
            model.train()
            batch_inputs = tuple(t.to(args.device) for t in batch[0:11])
            inputs = {"input_ids": batch_inputs[0], "attention_mask": batch_inputs[1],  "token_type_ids": batch_inputs[2], 
            "context_mask": batch_inputs[3],  "combine_input_ids": batch_inputs[4], "combine_token_type_ids": batch_inputs[6], 
            "combine_attention_mask": batch_inputs[5],  "clause_mask": batch_inputs[8], "start_positions": batch_inputs[9], 
            "end_positions": batch_inputs[10],  "testing":False}
            
            outputs = model(**inputs)
            loss = outputs[0]  # model outputs are always tuple in pytorch-transformers (see doc)
            if args.n_gpu > 1:
                loss = loss.mean()  # mean() to average on multi-gpu parallel training
            loss.backward()
            if step % 15 == 0:
                pbar(step, {'epoch': ep, 'loss': loss.item()})
            step += 1
            tr_loss += loss.item()
        
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            scheduler.step()  # Update learning rate schedule
            optimizer.step()
            model.zero_grad()
            global_step += 1

            if args.local_rank in [-1, 0] and args.logging_steps > 0 and global_step % args.logging_steps == 0:
                # Log metrics
                print("start evalue")
                if args.local_rank == -1:
                    # Only evaluate when single GPU otherwise metrics may not average well
                    results = evaluate(args = args, model = model, tokenizer = tokenizer,  prefix="dev")
                    span_f = results['span_f'] #在span级别f的值
                    if span_f > best_spanf:
                        output_dir = os.path.join(args.output_dir, "checkpoint-bestf")
                        if os.path.exists(output_dir):
                            shutil.rmtree(output_dir)
                            print('remobe file:',args.output_dir)
                        print('eval results:',results)
                        test_results = evaluate(args = args, model = model, tokenizer = tokenizer,  prefix="test")
                        print('test results', test_results)
                        print('epoch = :', ep)

                        best_spanf = span_f
                        os.makedirs(output_dir)
                        # print('dir = ', output_dir)
                        model_to_save = (
                        model.module if hasattr(model, "module") else model
                        )  # Take care of distributed/parallel training
                        model_to_save.save_pretrained(output_dir)
                        torch.save(args, os.path.join(output_dir, "training_args.bin"))
                        logger.info("Saving model checkpoint to %s", output_dir)
                        tokenizer.save_vocabulary(output_dir)
                        torch.save(optimizer.state_dict(), os.path.join(output_dir, "optimizer.pt"))
                        torch.save(scheduler.state_dict(), os.path.join(output_dir, "scheduler.pt"))
                        logger.info("Saving optimizer and scheduler states to %s", output_dir)

        np.random.seed()
        np.random.shuffle(train_features)
        logger.info("\n")
        # if 'cuda' in str(args.device):
        torch.cuda.empty_cache()
    return global_step, tr_loss / global_step, test_results


def evaluate(args, model, tokenizer, prefix="dev", use_crf = False):
    """
    model
    tokenizer
    prefix="dev"
    use_crf
    """
    eval_output_dir = args.output_dir

    if not os.path.exists(eval_output_dir) and args.local_rank in [-1, 0]:
        os.makedirs(eval_output_dir)
    eval_features, CLS_ID , SEP_ID = load_and_cache_examples(args, args.data_type, tokenizer, data_type=prefix)

    processor = processors[args.data_type]()

    logger.info("***** Running evaluation %s *****", prefix)
    logger.info("  Num examples = %d", len(eval_features))
    logger.info("  Batch size = %d", args.batch_size)
    eval_loss = 0.0

    pbar = ProgressBar(n_total=len(eval_features), desc="Evaluating" + prefix)
    if isinstance(model, nn.DataParallel):
        model = model.module
    
    pre_labels, tru_labels, eval_examples  =[],  [], []
    lens_data = []
    step = 0
    for batch in batch_generator(features = eval_features,CLS_ID = CLS_ID, SEP_ID = SEP_ID, batch_size=args.batch_size):
        batch_input_ids, batch_input_mask, batch_segment_ids, batch_context_mask, combine_input_ids, combine_input_mask, combine_segment_ids, combine_lens, combine_clause_mask, batch_start_position, batch_end_position, clause_lens, clause_num,batch_example = batch
        
        model.eval()
        batch_inputs = tuple(t.to(args.device) for t in batch[0:11])
        inputs = {"input_ids": batch_inputs[0], "attention_mask": batch_inputs[1],  "token_type_ids": batch_inputs[2], 
            "context_mask": batch_inputs[3],  "combine_input_ids": batch_inputs[4], "combine_token_type_ids": batch_inputs[6], 
            "combine_attention_mask": batch_inputs[5],  "clause_mask": batch_inputs[8], "testing":True}
            
        outputs = model(**inputs)
        eval_examples.extend(batch_example)
      
        batch_start_position_t, batch_end_position_t =  batch_start_position.cpu().numpy().tolist(), batch_end_position.cpu().numpy().tolist() 
        batch_context_mask = batch_context_mask.cpu().numpy().tolist()
        start_logits, end_logits, pre_cause_num = outputs #[batch, max_len]

        max_clause_len = np.max(clause_lens) #子句的最大长度
        batch_lens = clause_lens.tolist()
        # batch_emo_lens = batch_emo_lens.tolist()
        s_logits, e_logits = [], []
        s_position, e_position = [], []
        for i in range(len(batch_lens)):
            clause_len = batch_lens[i]
            tagg = 1
            at_s, at_e = batch_start_position_t[i][tagg: tagg + batch_lens[i][0]], batch_end_position_t[i][tagg: tagg + batch_lens[i][0]]
            tagg += max_clause_len
            for jj in range(1, len(clause_len)): 
                tt = clause_len[jj]
                if tt!= 0:
                    at_s.extend(batch_start_position_t[i][tagg: tagg + tt])
                    at_e.extend(batch_end_position_t[i][tagg: tagg + tt])
                    tagg += max_clause_len

            tagg = 1
            al_s, al_e = start_logits[i][tagg: tagg + batch_lens[i][0]], end_logits[i][tagg:tagg + batch_lens[i][0]]
            tagg +=max_clause_len
            for jj in range(1, len(clause_len)):
                tt = clause_len[jj]
                if tt != 0:
                    al_s, al_e = torch.cat((al_s, start_logits[i][tagg:tagg + tt]), 0), torch.cat((al_e, end_logits[i][tagg:tagg + tt]), 0)
                    tagg += max_clause_len
                    # print(al_s.size())

            s_logits.append(al_s)
            e_logits.append(al_e)
            s_position.append(at_s)
            e_position.append(at_e)

        if args.cause_num == 1:
            pre_cause_span_indexs = get_pre_sti_indexs(s_logits, e_logits)
        else:
            pre_cause_span_indexs = get_pre_indexs(s_logits, e_logits, pre_cause_num, args.cause_num)
        true_cause_span_indexs = get_tru_indexs(s_position, e_position)
        
        lens_data.extend(batch_lens)
        pre_labels.extend(pre_cause_span_indexs)
        tru_labels.extend(true_cause_span_indexs)

        step += 1
        if step % 20 == 0:
            pbar(step)

    logger.info("\n")
    assert len(pre_labels) == len(tru_labels)
    results = get_prf(pre_labels, tru_labels, lens_data, eval_examples)
  
    logger.info("***** Eval results %s *****", prefix)
    info = "-".join([f' {key}: {value:.4f} ' for key, value in results.items()])
    logger.info(info)
    return results


def get_content(text, label):
    """
    text: list[str]
    tru: list[int]
    """
    content = []
    for index, item in enumerate(label[1:-1]):
        if item != 0:
            content.append(text[index])
    return content

def load_and_cache_examples(args, task, tokenizer, data_type='train'):
    if args.local_rank not in [-1, 0] and not evaluate:
        torch.distributed.barrier()  # Make sure only the first process in distributed training process the dataset, and the others will use the cache
    processor = processors[task]()

    logger.info("Creating features from dataset file at %s", args.data_dir)
    label_list = processor.get_labels() #[B I O]
    if data_type == 'train':
        examples = processor.get_train_examples(args.data_dir) #
    elif data_type == 'dev':
        examples = processor.get_dev_examples(args.data_dir)
    else:
        examples = processor.get_test_examples(args.data_dir)
   
    features, CLS_ID, SEP_ID = convert_examples_to_features(examples=examples,
                                            tokenizer=tokenizer,
                                            label_list=label_list,
                                            max_seq_length=args.max_seq_length if data_type == 'train' \
                                                else args.max_seq_length, 
                                            pad_token=tokenizer.convert_tokens_to_ids([tokenizer.pad_token])[0],
                                            pad_token_segment_id=4 if args.model_type in ['xlnet'] else 0,
                                            max_cause_num= args.cause_num,
                                            max_clause_len = 70, 
                                            max_clause_num = 20
                                            )
    return features, CLS_ID, SEP_ID


def train_model():
    metrics = {}
    device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
    args.n_gpu = 1
    use_crf = False
    args.device = device
    logger.warning(
        "Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s",
        args.local_rank, device, args.n_gpu, bool(args.local_rank != -1), )
    # Set seed
    seed_everything(args.seed)
    # Prepare NER task
    args.data_type = args.data_type.lower() #
    if args.data_type not in processors:  #
        raise ValueError("Task not found: %s" % (args.data_type))
    processor = processors[args.data_type]()
    label_list = processor.get_labels() #
    args.id2label = {i: label for i, label in enumerate(label_list)} #
    args.label2id = {label: i for i, label in enumerate(label_list)} #
    num_labels = len(label_list) #

    # Load pretrained model and tokenizer
    if args.local_rank not in [-1, 0]:
        torch.distributed.barrier()  # Make sure only the first process in distributed training will download model & vocab
    args.model_type = args.model_type.lower()
    config_class, model_class, tokenizer_class = MODEL_CLASSES[args.model_type] #BertConfig, BertCrfForNer, EcaTokenizer

    #BertConfig.from_pretrained
    config = config_class.from_pretrained(args.config_name if args.config_name else args.model_name_or_path,
                                          num_labels=num_labels, cache_dir=args.cache_dir if args.cache_dir else None, )
    tokenizer = tokenizer_class.from_pretrained(args.tokenizer_name if args.tokenizer_name else args.model_name_or_path,
                                                do_lower_case=args.do_lower_case,
                                                cache_dir=args.cache_dir if args.cache_dir else None,
                                               )
    model = model_class.from_pretrained(args.model_name_or_path, from_tf=bool(".ckpt" in args.model_name_or_path),
                                        config=config, cache_dir=args.cache_dir if args.cache_dir else None) #模型

    if args.local_rank == 0:
        torch.distributed.barrier()  # Make sure only the first process in distributed training will download model & vocab

    model.to(args.device)
    logger.info("Training/evaluation parameters %s", args)

    # Training
    train_features, CLS_ID, SEP_ID = load_and_cache_examples(args, args.data_type, tokenizer, data_type='train')
    global_step, tr_loss, metrics  = train(args, train_features, model, tokenizer, CLS_ID, SEP_ID)
    logger.info(" global_step = %s, average loss = %s", global_step, tr_loss)
    return metrics
    
def main():
    #create the result file
    if not os.path.exists(args.results_dir): #
        os.mkdir(args.results_dir)
    list_result = []
    out_current_path = copy.deepcopy(args.output_dir) 

    if args.data_type == 'ch':
        dev_id_list = loadList(os.path.join(os.getcwd(), 'dataset/eca_ch/split_data_fold/eca_ch_dev_id.pkl'))

    if args.data_type == 'sti': 
        dev_id_list = loadList(os.path.join(os.getcwd(), 'dataset/eca_sti/split_data_fold/eca_sti_dev_ids.pkl'))
    
    if args.data_type == 'en': 
        dev_id_list = loadList(os.path.join(os.getcwd(), 'dataset/eca_en/split_data_fold/eca_en_dev_ids.pkl'))

    for index, item in enumerate(dev_id_list):
        print("*****************************index:{}*******************".format(index))
        args.output_dir = args.output_dir + '{}_{}_{}'.format(args.data_type, args.Gpu_num, args.save_name) #
        if os.path.exists(args.output_dir): #
            shutil.rmtree(args.output_dir)
            print('删除文件夹：',args.output_dir)
        os.mkdir(args.output_dir)
        
        data_current_path = copy.deepcopy(args.data_dir) 
        args.data_dir = args.data_dir  + '{}_{}_{}'.format(args.data_type,  args.Gpu_num, args.save_name)  
        if os.path.exists(args.data_dir):
            shutil.rmtree(args.data_dir)
        os.mkdir(args.data_dir)

       
        train_data, test_data, dev_data = [], [], []
        if args.data_type == 'ch':
            #load the data
            data_set = loadList(os.path.join(os.getcwd(), 'dataset/eca_ch/eca_ch_data.pkl'))
            for i in range(len(data_set)):
                if i in item:
                    test_data.append(data_set[i])
                else:
                    train_data.append(data_set[i])
        
        if args.data_type == 'sti':
            #load the data
            data_set = loadList(os.path.join(os.getcwd(), 'dataset/eca_sti/eca_sti_data.pkl')) 
            for i in range(len(data_set)):
                if i in item:
                    test_data.append(data_set[i])
                else:
                    train_data.append(data_set[i])
            
        #save the data to the path
        num = len(train_data)
        a = train_data[0: int(num * 0.1)]
        b = train_data[int(num * 0.1) : ]
        saveList(b, os.path.join(args.data_dir, 'eca_train.pkl'))
        saveList(test_data, os.path.join(args.data_dir, 'eca_test.pkl'))
        saveList(a, os.path.join(args.data_dir, 'eca_dev.pkl'))

        #get the result after the spit
        metrics = train_model()
        list_result.append(metrics)
        print('\n the results_{} = {}\n'.format(args.data_type, metrics))

        #remove the current data from the path
        if os.path.exists(args.data_dir):
            shutil.rmtree(args.data_dir)
            print('remove files: ', args.data_dir)
        
        if os.path.exists(args.output_dir):
            shutil.rmtree(args.output_dir)
            print('remove files: ', args.output_dir)
            
        args.output_dir = out_current_path
        args.data_dir = data_current_path

    strr = ''
    for each_result in list_result:
        for key, value in each_result.items():
            strr += str(value) + '\t'
        strr += '\n'
    
    print('the result is: ', strr)

if __name__ == "__main__":
    main()

""" Named entity recognition fine-tuning: utilities to work with CLUENER task. """
import torch
import logging
import os
import copy
import json
import numpy as np
from .utils_eca import DataProcessor
logger = logging.getLogger(__name__)

class InputExample(object):
    """A single training/test example for token classification."""
    def __init__(self, guid, text_a, span_label = None, docid = None, data_len_c =None, text_e = None, emotion_len = None):
        """Constructs a InputExample.
        Args:
            guid: Unique id for the example.
            text_a: list. The words of the sequence.
            labels: (Optional) list. The labels for each word of the sequence. This should be
            specified for train and dev examples, but not for test examples.
        """
        self.guid = guid
        self.text_a = text_a
        self.span_label = span_label
      
        self.docid = docid
        self.data_len_c = data_len_c #每个子句的长度
        self.text_e = text_e
        self.emotion_len = emotion_len


    def __repr__(self):
        return str(self.to_json_string())
    def to_dict(self):
        """Serializes this instance to a Python dictionary."""
        output = copy.deepcopy(self.__dict__)
        return output
    def to_json_string(self):
        """Serializes this instance to a JSON string."""
        return json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n"

class InputFeatures(object):
    """A single set of features of data."""
    def __init__(self, input_ids, input_mask, input_len, emotion_len, segment_ids, context_mask = None, start_position = None, end_position = None, start_position_t = None, end_position_t = None, example = None):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
      
        self.input_len = input_len
        self.emotion_len = emotion_len

        self.example = example
        self.context_mask = context_mask
        self.end_position = end_position
        self.start_position = start_position

        self.end_position_t = end_position_t
        self.start_position_t = start_position_t

    def __repr__(self):
        return str(self.to_json_string())

    def to_dict(self):
        """Serializes this instance to a Python dictionary."""
        output = copy.deepcopy(self.__dict__)
        return output

    def to_json_string(self):
        """Serializes this instance to a JSON string."""
        return json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n"


def convert_examples_to_features(examples,label_list,max_seq_length,tokenizer,
                                 pad_token=0, pad_token_segment_id=0,
                                 mask_padding_with_zero=True, max_cause_num = 5):
    
    """ Loads a data file into a list of `InputBatch`s
        `cls_token_at_end` define the location of the CLS token:
            - False (Default, BERT/XLM pattern): [CLS] + A + [SEP] + B + [SEP]
            - True (XLNet/GPT pattern): A + [SEP] + B + [SEP] + [CLS]
        `cls_token_segment_id` define the segment id associated to the CLS token (0 for BERT, 2 for XLNet)
    """
    # print('examples = ', examples[0])
    # label_map = {label: i for i, label in enumerate(label_list)}
    features = []
    for (ex_index, example) in enumerate(examples):
        if ex_index % 10000 == 0:
            logger.info("Writing example %d of %d", ex_index, len(examples))
        tokens = tokenizer.tokenize(example.text_a)
        tokens_e = tokenizer.tokenize(example.text_e) #将情感表达进行分词
        assert len(tokens) == len(example.text_a) #判断是否有的词语在token的时候被划分成为两个词语
        assert len(tokens_e) == len(example.text_e) #判断是否有的词语在token的时候被划分成为两个词语
        # label_ids = [label_map[x] for x in example.labels]

        # Account for [CLS] and [SEP] with "- 2".
        #将数据进行拼接
        combine_token = ['[CLS]']
        combine_token += tokens_e
        emotion_len = len(tokens_e)

        segment_ids = [0] * len(combine_token)
        combine_token += ['[SEP]']
        segment_ids += [1]
        context_mask = [0] * len(combine_token)

        combine_token += tokens
        segment_ids += [1] * len(tokens)
        context_mask += [1] * len(tokens)

        combine_token += ['[SEP]']
        segment_ids += [1] 
        context_mask += [0] 

        input_mask = [1] * len(combine_token)

        #对input进行填充
        input_ids = tokenizer.convert_tokens_to_ids(combine_token)
        input_len = len(input_ids)
        assert len(input_mask) == len(context_mask) ==   len(input_ids) == len(segment_ids)

        # print('len(input_mask) = ', len(input_mask))
        assert len(input_mask) <= max_seq_length
        padding_length = max_seq_length - input_len
        
        input_ids += [pad_token] * padding_length
        input_mask += [0] * padding_length
        segment_ids += [pad_token_segment_id] * padding_length
        # combine_label_ids += [0] * padding_length
        context_mask += [0] * padding_length

        assert len(input_ids) == max_seq_length
        assert len(input_mask) == max_seq_length
        # print('segment_ids = ', len(segment_ids))
        # print('max_seq_length = ', max_seq_length)
        assert len(segment_ids) == max_seq_length
        # assert len(combine_label_ids) == max_seq_length
        assert len(context_mask) == max_seq_length
        
        para_span_label = []
        for item in example.span_label:
            item[0] = item[0] + 2 + emotion_len
            item[1] = item[1] + 2 + emotion_len
            para_span_label.append(item)
        # span_label = example.span_label[0]
        #对情感表达进行填充
        start_position = [0] * len(input_ids) 
        end_position = [0] * len(input_ids) 

        start_position_t = [0] * len(input_ids) 
        end_position_t = [0] * len(input_ids) 

        for indexs, items in enumerate(para_span_label):
            start = items[0]
            end = items[1]
            if indexs < max_cause_num:
                start_position[start] = 1
                end_position[end] = 1
            
            start_position_t[start] = 1
            end_position_t[end] = 1
            
        assert len(start_position) == len(end_position) == len(start_position_t) == len(end_position_t) == max_seq_length

        if ex_index < 1:
            logger.info("*** Example ***")
            # logger.info("guid: %s", example.guid)
            print("\n tokens: %s", " ".join([str(x) for x in tokens]))
            # print("\n input_ids: %s", " ".join([str(x) for x in input_ids]))
            # print("\n input_mask: %s", " ".join([str(x) for x in input_mask]))
            # print("\n segment_ids: %s", " ".join([str(x) for x in segment_ids]))
            # print("\n label_ids: ", " ".join([str(x) for x in combine_label_ids]))
            print("\n para_span_label:", para_span_label)
            print("\nstart_position: ", start_position)
            print("\nend_position: ", end_position)

        assert len(tokens) == np.sum(context_mask)
        features.append(InputFeatures(input_ids=input_ids, input_mask=input_mask, input_len = input_len, emotion_len = emotion_len,
                                      segment_ids=segment_ids,  context_mask = context_mask, start_position = start_position, end_position = end_position, 
                                      start_position_t = start_position_t, end_position_t = end_position_t,  example = example))
    return features



#获取批量数据并打乱
def batch_generator(features, batch_size=128, return_idx=False):
    
    def pad_span(para_span, max_answer_length):
        # print('para_span = ', para_span)
        # print('max_answer_length = ', max_answer_length)
        if len(para_span) > max_answer_length:
            para_span = para_span[0:max_answer_length]
            return np.array(para_span)
        else:
            pad_len = max_answer_length - len(para_span)
            pad_span_data= [[0,0]] * pad_len
            # print('pad_span_data = ', pad_span_data)
            a = para_span.extend(pad_span_data)
            # print('para_span = ', para_span)
            return np.array(para_span)
        
    # Convert to Tensors and build dataset
    all_input_ids = [f.input_ids for f in features]
    all_input_mask = [f.input_mask for f in features]
    all_segment_ids = [f.segment_ids for f in features]
    # all_label_ids = [f.label_ids for f in features]

    all_lens = [f.input_len for f in features]
    all_emo_lens = [f.emotion_len for f in features]

    # dataset = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_lens, all_label_ids)
    all_example = [f.example for f in features]
    all_context_mask = [f.context_mask for f in features]
    # all_span_label = [f.span_label[0] for f in features] #[num, 2]
    # all_multi_span = [pad_span(f.span_label, answer_seq_len) for f in features]

    all_start_position = [f.start_position for f in features]
    all_end_position = [f.end_position for f in features]

    all_start_position_t = [f.start_position_t for f in features]
    all_end_position_t = [f.end_position_t for f in features]

    for offset in range(0, len(features), batch_size):
    # for offset in range(0, 2*batch_size, batch_size):
        #获取x的长度
        input_mask = all_input_mask[offset:offset+batch_size] #为了计算长度
        batch_x_len = np.sum(input_mask, -1)
        max_doc_len =  max(batch_x_len) #文本的最大长度
        batch_idx=batch_x_len.argsort()[::-1]

        input_ids = np.array(all_input_ids[offset:offset+batch_size])[batch_idx]
        input_mask = np.array(all_input_mask[offset:offset+batch_size])[batch_idx]
        segment_ids = np.array(all_segment_ids[offset:offset+batch_size])[batch_idx]
        # label_ids = np.array(all_label_ids[offset:offset+batch_size])[batch_idx]
        # raw_labels = np.array(all_label_ids[offset:offset+batch_size])[batch_idx]
        batch_lens = np.array(all_lens[offset:offset+batch_size])[batch_idx]
        batch_emo_lens = np.array(all_emo_lens[offset:offset+batch_size])[batch_idx]

        batch_context_mask = np.array(all_context_mask[offset:offset+batch_size])[batch_idx]
        # batch_span_label = np.array(all_span_label[offset:offset+batch_size])[batch_idx]
        # batch_multi_span_label = np.array(all_multi_span[offset:offset+batch_size])[batch_idx]
        batch_start_position = np.array(all_start_position[offset:offset+batch_size])[batch_idx]
        batch_end_position = np.array(all_end_position[offset:offset+batch_size])[batch_idx]

        batch_start_position_t = np.array(all_start_position_t[offset:offset+batch_size])[batch_idx]
        batch_end_position_t = np.array(all_end_position_t[offset:offset+batch_size])[batch_idx]

        #情感信息
        batch_example = [all_example[offset:offset+batch_size][i] for i in batch_idx]
        
        #转换为torch类型
        # print('batch_x = ',batch_x)
        batch_input_ids = torch.from_numpy(input_ids[:, 0:max_doc_len]).long().cuda()
        batch_input_mask = torch.from_numpy(input_mask[:, 0:max_doc_len]).long().cuda()
        batch_segment_ids = torch.from_numpy(segment_ids[:, 0:max_doc_len]).long().cuda()
        batch_context_mask = torch.from_numpy(batch_context_mask[:, 0:max_doc_len]).long().cuda()
        # batch_span_label = torch.from_numpy(batch_span_label).long().cuda()
        # batch_multi_span_label = torch.from_numpy(batch_multi_span_label).long().cuda()

        # print('lable_ids = ', label_ids)
        # batch_label_ids = torch.from_numpy(label_ids[:, 0:max_doc_len]).long().cuda()
        # batch_raw_labels = torch.from_numpy(raw_labels[:, 0:max_doc_len]).long().cuda()
        batch_start_position = torch.from_numpy(batch_start_position[:, 0:max_doc_len]).long().cuda()
        batch_end_position = torch.from_numpy(batch_end_position[:, 0:max_doc_len]).long().cuda()

        batch_start_position_t = torch.from_numpy(batch_start_position_t[:, 0:max_doc_len]).long().cuda()
        batch_end_position_t = torch.from_numpy(batch_end_position_t[:, 0:max_doc_len]).long().cuda()
        
        yield (batch_input_ids, batch_input_mask, batch_segment_ids,  batch_context_mask, batch_start_position, batch_end_position, batch_start_position_t, batch_end_position_t,  batch_lens, batch_emo_lens, batch_example)


def bert_extract_item(start_logits, end_logits):
    """
    start_logits: [batch, max_len,3]
    end_logits:[batch, max_len, 3]
    """
    start_pred = torch.argmax(start_logits, -1).cpu().numpy()#[batch, max_len]
    end_pred = torch.argmax(end_logits, -1).cpu().numpy()#[batch, max_len]
    pre_tags = []
    # print('\nstart = ', np.sum(start_pred, -1))
    # print('\nend = ', np.sum(end_pred, -1))
    max_len = start_pred.shape[1]
    for ss, ee in zip(start_pred, end_pred):
        target = [0] * max_len
        for i, s_l in enumerate(ss):
            if s_l == 0:
                continue
            for j, e_l in enumerate(ee[i:]):
                if s_l == e_l:
                    # S.append((s_l, i, i + j))
                    if i == j - 1:
                        target[i] = 1
                    if i < j - 1:
                        target[i] = 1
                        target[i+1:j] = [2] * (j - i - 1)
                    break 
        pre_tags.append(target)
    return pre_tags


def extract_multi_item(start_logits, end_logits, num_logits):
    """
    start_logits: [batch, max_len]
    end_logits:[batch, max_len]
    num_logits:[batch]
    """
    _, s_index= torch.sort(input = start_logits, dim = -1, descending=True) #排序的index
    _, e_index,= torch.sort(input = end_logits, dim = -1, descending=True) #排序的index 从大到小
    num_spans = num_logits.argmax(-1).cpu().numpy() #[batch]

    # print('num_spans = ', num_spans)
    # print('s_index = ', s_index)
    # print('e_index = ', e_index)

    s_index = s_index.detach().cpu().numpy()
    e_index = e_index.detach().cpu().numpy()

    pre_tags = []#预测的标签
    max_len = start_logits.size(1)#子句的最大长度
    batch_size = start_logits.size(0) #batch

    for i in range(batch_size):
        current_tag = [0] * max_len
        nums = num_spans[i]
        ss = s_index[i]
        ee = e_index[i]

        for j in range(nums):
            s = ss[j]
            e = ee[j]
            if s == e - 1:
                current_tag[s] = 1
            if s < e - 1:
                current_tag[int(s)] = 1
                current_tag[int(s)+1:int(e)] = [2] * (int(e - s) - 1)
        pre_tags.append(current_tag)

    return pre_tags

# def bert_extract_item(start_logits, end_logits):
#     S = []
#     start_pred = torch.argmax(start_logits, -1).cpu().numpy()[0][1:-1]
#     end_pred = torch.argmax(end_logits, -1).cpu().numpy()[0][1:-1]
#     for i, s_l in enumerate(start_pred):
#         if s_l == 0:
#             continue
#         for j, e_l in enumerate(end_pred[i:]):
#             if s_l == e_l:
#                 S.append((s_l, i, i + j))
#                 break
#     return S

class ECA_en_Processor(DataProcessor):
    """Processor for the chinese ner data set."""

    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(self._read_en_pkl(data_path = os.path.join(data_dir, "eca_train.pkl"), save_csv_path = os.path.join(data_dir, "ecatext_train.csv")), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(self._read_en_pkl(data_path = os.path.join(data_dir, "eca_dev.pkl"), save_csv_path = os.path.join(data_dir, "ecatext_dev.csv")), "dev")

    def get_test_examples(self, data_dir):
        """See base class."""
        return self._create_examples(self._read_en_pkl(data_path = os.path.join(data_dir, "eca_test.pkl"), save_csv_path = os.path.join(data_dir, "ecatext_test.csv")), "test")

    def get_labels(self):
        """See base class."""
        return ["O", "B", "I"]

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            guid = "%s-%s" % (set_type, i)
            text_a = line['content_data']
            labels = line['target_data'] #BIO 当前文本的标签 list列表
            docid = line['docID']
            emo_tokens = line['emo_data']
            # emotion_index = line['emotion_index']
            data_len_c = line['clause_len']
            emotion_word = line['emotion_word']
            emotion_len = line['emotion_len']
            # ec_index = line['ec_index']
            # BIOS
            span_label = line['span_index'] #[[start, end],[]]

            examples.append(InputExample(guid=guid, text_a=text_a, span_label=span_label,  docid = docid, data_len_c= data_len_c, text_e = emotion_word, emotion_len = emotion_len))
        return examples


class ECA_ch_Processor(DataProcessor):
    """Processor for the chinese ner data set."""

    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(self._read_ch_pkl(data_path = os.path.join(data_dir, "eca_train.pkl"), save_csv_path = os.path.join(data_dir, "ecatext_train.csv")), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(self._read_ch_pkl(data_path = os.path.join(data_dir, "eca_dev.pkl"), save_csv_path = os.path.join(data_dir, "ecatext_dev.csv")), "dev")

    def get_test_examples(self, data_dir):
        """See base class."""
        return self._create_examples(self._read_ch_pkl(data_path = os.path.join(data_dir, "eca_test.pkl"), save_csv_path = os.path.join(data_dir, "ecatext_test.csv")), "test")

    def get_labels(self):
        """See base class."""
        return ["O", "B", "I"]

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            guid = "%s-%s" % (set_type, i)
            text_a = line['content_data']
            labels = line['target_data'] #BIO 当前文本的标签 list列表
            docid = line['docID']
            emo_tokens = line['emo_data']
            # emotion_index = line['emotion_index']
            data_len_c = line['clause_len']
            emotion_len = line['emotion_len']
            span_label = line['span_index'] #[[start, end],[]]
            # ec_index = line['ec_index']
            #BIOS
            examples.append(InputExample(guid=guid, text_a=text_a,  docid = docid, span_label = span_label, data_len_c= data_len_c, text_e = emo_tokens, emotion_len=emotion_len ))
        return examples

class ECA_sti_Processor(DataProcessor):
    """Processor for the chinese ner data set."""

    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(self._read_sti_pkl(data_path = os.path.join(data_dir, "eca_train.pkl"), save_csv_path = os.path.join(data_dir, "ecatext_train.csv")), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(self._read_sti_pkl(data_path = os.path.join(data_dir, "eca_dev.pkl"), save_csv_path = os.path.join(data_dir, "ecatext_dev.csv")), "dev")

    def get_test_examples(self, data_dir):
        """See base class."""
        return self._create_examples(self._read_sti_pkl(data_path = os.path.join(data_dir, "eca_test.pkl"), save_csv_path = os.path.join(data_dir, "ecatext_test.csv")), "test")

    def get_labels(self):
        """See base class."""
        return ["O", "B", "I"]

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            guid = "%s-%s" % (set_type, i)
            text_a = line['content_data']
            labels = line['target_data'] #BIO 当前文本的标签 list列表
            docid = line['docID']
            emo_tokens = line['emotion_word']
            emotion_len = line['emotion_len']
            # emotion_index = line['emotion_index']
            data_len_c = line['clause_len']
            # ec_index = line['ec_index']
            # BIOS
            span_label = line['span_index'] #[[start, end],[]]
            examples.append(InputExample(guid=guid, text_a=text_a,  span_label = span_label, docid = docid, data_len_c= data_len_c, text_e = emo_tokens,  emotion_len=emotion_len))
        return examples
        

eca_processors = {
    'en':ECA_en_Processor,
    'ch':ECA_ch_Processor,
    'sti':ECA_sti_Processor
}

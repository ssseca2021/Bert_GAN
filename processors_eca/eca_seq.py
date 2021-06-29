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
    def __init__(self, guid, text_a, span_label = None, docid = None, data_len_c =None, text_e = None, emotion_len = None, clause_num = None):
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
        self.clause_num = clause_num

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
    def __init__(self, input_ids, input_mask, segment_ids, context_mask = None, start_position = None, end_position = None, clause_len = None,  clause_num= None, combine_input_ids = None, combine_segment_ids = None,combine_mask_ids = None, input_len = None, clause_mask = None, example = None):
        
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.context_mask = context_mask
        self.end_position = end_position
        self.start_position = start_position
        self.clause_len = clause_len
        self.clause_num = clause_num

        self.combine_input_ids = combine_input_ids
        self.combine_segment_ids = combine_segment_ids
        self.combine_mask_ids = combine_mask_ids
        self.input_len = input_len
        self.clause_mask = clause_mask

        self.example = example
        
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
                                 mask_padding_with_zero=True, max_cause_num = 5, max_clause_len = 100, max_clause_num = 50):
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
        doc_data = example.text_a
        clause_num = len(doc_data) #当前文档的子句的个数
        emotion_data = example.text_e #情绪词语
        emo_tokens = tokenizer.tokenize(emotion_data) #将每个子句进行tokensize
        assert len(emo_tokens) == len(emotion_data)

        if clause_num > max_clause_num:
            clause_num = max_clause_num
            print('clause_num > max_clause_num')

        input_ids = [] #扩充后的数据
        input_mask = []
        segment_ids = []

        mask_context_l = [] #只有在是文本内容的地方标记为1，其余的位置标记为0
        para_start_position, para_end_position = [], [] #开始和结束位置list列表

        clause_len = []

        combine_input_ids_l = []
        combine_segment_ids_l = []
        combine_mask_ids_l = []
        input_len_l = []
        clause_mask_l = []

        for clause_index in range(clause_num):
            clause_mask_l.append(1)

            tokens_clause = doc_data[clause_index]
            a = tokenizer.tokenize(tokens_clause) #将每个子句进行tokensize
            assert len(a) == len(tokens_clause) #判断是否有的词语在token的时候被划分成为两个词语

            ##########################获取和情绪拼接的inputids, mask ,segmentsid###################
            combine_input_tokens = ['[CLS]'] 
            combine_input_tokens += emo_tokens
            combine_input_tokens += ['[SEP]']
            combine_segment_ids = [0] * len(combine_input_tokens)
            combine_input_tokens += a
            combine_input_tokens += ['[SEP]']
            combine_segment_ids += [1] * (len(a) + 1)
            combine_mask_ids = [1] * len(combine_input_tokens)
            combine_input_ids = tokenizer.convert_tokens_to_ids(combine_input_tokens)

            combine_input_len = len(combine_mask_ids)
            assert len(combine_input_ids) < max_seq_length
            pad_len = max_seq_length - len(combine_input_ids)
            for _ in range(pad_len):
                combine_input_ids.append(0)
                combine_segment_ids.append(0)
                combine_mask_ids.append(0)
            
            combine_input_ids_l.append(combine_input_ids)
            combine_segment_ids_l.append(combine_segment_ids)
            combine_mask_ids_l.append(combine_mask_ids)
            input_len_l.append(combine_input_len)

            ###################################################################################################
        
            #pad the tokens of each clause
            if len(a) > max_clause_len:
                combine_token = a[0:max_clause_len]
                context_mask = [1] * max_clause_len
                clause_len.append(max_clause_len)
            else:
                combine_token = a + ['[PAD]'] * (max_clause_len - len(a))
                context_mask = [1] * len(a)
                context_mask += [0] *  (max_clause_len - len(a))
                clause_len.append(len(a))

            ids_clause = tokenizer.convert_tokens_to_ids(combine_token)
            assert len(ids_clause) == max_clause_len
            input_ids.append(ids_clause)
            input_mask.append([1] * max_clause_len)
            segment_ids.append([0] * max_clause_len)
            mask_context_l.append(context_mask)

            #获取lable表示的标签
            span_index = example.span_label[clause_index]
            start, end = span_index[0], span_index[1]
            start_position, end_position = [0] * max_clause_len, [0] * max_clause_len
            
            #如果原因被截断，就进行截断处理
            if start != -1 and end != -1:
                if start < max_clause_len - 1 :
                    start_position[start] = 1
                    if end >= max_clause_len:
                        end_position[max_clause_len - 1] = 1
                    else:
                        end_position[end] = 1
                else:
                    print('zzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzz')

            assert len(start_position) ==  len(end_position) == max_clause_len
            para_start_position.append(start_position)
            para_end_position.append(end_position)

        #如果子句的数目小于最多的子句个数
        if clause_num < max_clause_num:
            pad_clause_num = max_clause_num - clause_num
            for _ in range(pad_clause_num):
                a_position = [0] * max_clause_len
                para_start_position.append(a_position)
                para_end_position.append(a_position)

                input_ids.append(tokenizer.convert_tokens_to_ids(['[PAD]'] * max_clause_len))
                input_mask.append([0] * max_clause_len)
                segment_ids.append([0] * max_clause_len)
                mask_context_l.append([0] * max_clause_len)
                clause_len.append(0)

                assert len(start_position) ==  len(end_position) == max_clause_len

                ###########合并的emotion + clause###########
                combine_input_ids_l.append([0] * max_seq_length)
                combine_segment_ids_l.append([0] * max_seq_length)
                combine_mask_ids_l.append([0] * max_seq_length)
                input_len_l.append(0)
                clause_mask_l.append(0)
                #########################################

        assert len(input_ids) == max_clause_num
        assert len(input_mask) == max_clause_num
        assert len(segment_ids) == max_clause_num
        # assert len(combine_label_ids) == max_seq_length
        assert len(mask_context_l) == max_clause_num
    
        if ex_index < 1:
            logger.info("*** Example ***")
            print("\nstart_position: ", start_position)
            print("\nend_position: ", end_position)

        aaa = np.sum(np.array(para_start_position))
        if aaa == 0:
            print('+++++++++++++++++example = ', example)
            filename='names.json'
            with open(filename,'a') as file_obj:
                json.dump(example.text_a,file_obj)

        assert aaa > 0
        features.append(InputFeatures(input_ids=input_ids, input_mask=input_mask, segment_ids=segment_ids, context_mask = mask_context_l, 
                                    start_position = para_start_position, end_position = para_end_position, clause_len = clause_len, 
                                    clause_num = clause_num, combine_input_ids = combine_input_ids_l, combine_segment_ids = combine_segment_ids_l,
                                    combine_mask_ids = combine_mask_ids_l, input_len = input_len_l,clause_mask = clause_mask_l,
                                    example = example))

    CLS_ID = tokenizer.convert_tokens_to_ids(['[CLS]'])
    SEP_ID = tokenizer.convert_tokens_to_ids(['[SEP]'])

    return features,CLS_ID, SEP_ID




#获取批量数据并打乱
def batch_generator(features, CLS_ID, SEP_ID, batch_size=5):
    
    # Convert to Tensors and build dataset
    all_input_ids = [f.input_ids for f in features]
    all_input_mask = [f.input_mask for f in features]
    all_segment_ids = [f.segment_ids for f in features]

    all_example = [f.example for f in features]
    all_context_mask = [f.context_mask for f in features]
    all_clause_lens = [f.clause_len for f in features]
    all_clause_num = [f.clause_num for f in features]

    all_start_position = [f.start_position for f in features]
    all_end_position = [f.end_position for f in features]

    #对所有的组合的情绪和子句的表达
    all_combine_input_ids = [f.combine_input_ids for f in features]
    all_combine_input_mask = [f.combine_mask_ids for f in features]
    all_combine_segment_ids = [f.combine_segment_ids for f in features]
    all_combine_lens =  [f.input_len for f in features]
    all_combine_clause_mask = [f.clause_mask for f in features]


    for offset in range(0, len(features), batch_size):
    # for offset in range(0, 2*batch_size, batch_size):
        clause_lens = np.array(all_clause_lens[offset:offset+batch_size])
        clause_num = np.array(all_clause_num[offset:offset+batch_size])
        max_clause_len = np.max(clause_lens)
        max_clause_num = np.max(clause_num)
        
        
        clause_lens = clause_lens[:, 0:max_clause_num]
        if max_clause_len * max_clause_num > 500:
            if max_clause_num > 13:
                max_clause_num = 13
            clause_lens = clause_lens[:, 0:max_clause_num]
            if max_clause_len * max_clause_num > 500:
                max_clause_len = 38
                for clause_index_c, itemc in enumerate(clause_lens):  #item c 是一个一维数组
                    for j in range(len(list(itemc))):
                        if clause_lens[clause_index_c][j] > max_clause_len:
                            clause_lens[clause_index_c][j] = max_clause_len
    
        if max_clause_len * max_clause_num > 500:
            if max_clause_num > 13:
                max_clause_num = 13
            clause_lens = clause_lens[:, 0:max_clause_num]
        

        #根据确定的最多的子句的个数，进行整理combine数据
        #combine sequence lens 
        combine_sequence_lens = np.array(all_combine_lens[offset:offset+batch_size])
        max_sequence_len = np.max(combine_sequence_lens)
        
        combine_input_ids =  np.array(all_combine_input_ids[offset:offset+batch_size])[:,0:max_clause_num, 0:max_sequence_len].reshape(-1,  max_sequence_len)
        combine_input_mask = np.array(all_combine_input_mask[offset:offset+batch_size])[:,0:max_clause_num, 0:max_sequence_len].reshape(-1,  max_sequence_len)
        combine_segment_ids = np.array(all_combine_segment_ids[offset:offset+batch_size])[:,0:max_clause_num, 0:max_sequence_len].reshape(-1,  max_sequence_len)
        combine_lens = np.array(all_combine_lens[offset:offset+batch_size])[:,0:max_clause_num] 
        combine_clause_mask = np.array(all_combine_clause_mask[offset:offset+batch_size])[:,0:max_clause_num]

        input_ids = np.array(all_input_ids[offset:offset+batch_size])[:,0:max_clause_num, 0:max_clause_len].reshape(-1, max_clause_num * max_clause_len)
        input_mask = np.array(all_input_mask[offset:offset+batch_size])[:,0:max_clause_num, 0:max_clause_len].reshape(-1, max_clause_num * max_clause_len)
        segment_ids = np.array(all_segment_ids[offset:offset+batch_size])[:,0:max_clause_num, 0:max_clause_len].reshape(-1, max_clause_num * max_clause_len)

        context_mask = np.array(all_context_mask[offset:offset+batch_size])[:,0:max_clause_num, 0:max_clause_len].reshape(-1, max_clause_num * max_clause_len)
        start_position = np.array(all_start_position[offset:offset+batch_size])[:,0:max_clause_num, 0:max_clause_len].reshape(-1, max_clause_num * max_clause_len)
        end_position = np.array(all_end_position[offset:offset+batch_size])[:,0:max_clause_num, 0:max_clause_len].reshape(-1, max_clause_num * max_clause_len)
        batch_example = all_example[offset:offset+batch_size]

        batchs_num = input_ids.shape[0]
        input_ids_ =np.column_stack((np.column_stack((CLS_ID * batchs_num, input_ids)), SEP_ID* batchs_num))
        input_mask_ =np.column_stack((np.column_stack(([1] * batchs_num, input_mask)), [1]* batchs_num))
        segment_ids_ =np.column_stack((np.column_stack(([0] * batchs_num, segment_ids)), [0]* batchs_num))
        context_mask_ =np.column_stack((np.column_stack(([0] * batchs_num, context_mask)), [0]* batchs_num))
        start_position_ =np.column_stack((np.column_stack(([0]  * batchs_num, start_position)), [0]* batchs_num))
        end_position_ =np.column_stack((np.column_stack(([0]  * batchs_num, end_position)), [0]* batchs_num))

        assert np.sum(context_mask_) == np.sum(clause_lens)
        #转换为torch类型
        batch_input_ids = torch.from_numpy(input_ids_).long().cuda()
        batch_input_mask = torch.from_numpy(input_mask_).long().cuda()
        batch_segment_ids = torch.from_numpy(segment_ids_).long().cuda()
        batch_context_mask = torch.from_numpy(context_mask_).long().cuda()
        batch_start_position =  torch.from_numpy(start_position_).long().cuda()
        batch_end_position = torch.from_numpy(end_position_).long().cuda()

        #combine 
        combine_input_ids = torch.from_numpy(combine_input_ids).long().cuda()
        combine_input_mask = torch.from_numpy(combine_input_mask).long().cuda()
        combine_segment_ids = torch.from_numpy(combine_segment_ids).long().cuda()
        combine_lens = torch.from_numpy(combine_lens).long().cuda()
        combine_clause_mask = torch.from_numpy(combine_clause_mask).long().cuda()

        yield (batch_input_ids, batch_input_mask, batch_segment_ids, batch_context_mask, combine_input_ids, combine_input_mask, combine_segment_ids, combine_lens, combine_clause_mask, batch_start_position, batch_end_position, clause_lens, clause_num,batch_example)


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
            # labels = line['target_data'] #BIO 当前文本的标签 list列表
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
            # labels = line['target_data'] #BIO 当前文本的标签 list列表
            docid = line['docID']
            emo_tokens = line['emotion_word']
            emotion_len = line['emotion_len']
            # emotion_index = line['emotion_index']
            data_len_c = line['clause_len']
            # ec_index = line['ec_index']
            clause_num = line['clause_num'] #文档中子句的个数
            # BIOS
            span_label = line['span_index'] #[[start, end],[]]
            examples.append(InputExample(guid=guid, text_a=text_a,  span_label = span_label, docid = docid, data_len_c= data_len_c, text_e = emo_tokens,  emotion_len=emotion_len, clause_num = clause_num))
        return examples
        

eca_processors = {
    'en':ECA_en_Processor,
    'ch':ECA_ch_Processor,
    'sti':ECA_sti_Processor
}

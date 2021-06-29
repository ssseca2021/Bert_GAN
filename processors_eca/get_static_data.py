import pickle
import string

def saveList(paraList, path):
    output = open(path, 'wb')
    # Pickle dictionary using protocol 0.
    pickle.dump(paraList, output)
    output.close()

'''
load the pkl files
'''
def loadList(path):
    pkl_file = open(path, 'rb')
    segContent = pickle.load(pkl_file)
    pkl_file.close()
    return segContent

def remove_noise_strr_ch(content):
    """
    args:
        :param content:
        :return:
    """
    content = content.replace('  ', '')
    content = content.replace('@', '')
    content = content.replace('#', '')
    content = content.replace('<', '')
    content = content.replace('>', '')
    content = content.replace('=', '')
    content = content.replace('`', '')
    content = content.replace('\n', '')
    content = content.replace('\t', '')
    content = content.replace('*', '')
    content = content.replace('&', '')
    content = content.replace(' ', '')
    return content.strip()

def get_clean_data_ch(para_text):
    returntext = remove_noise_strr_ch(para_text)
    return returntext


def read_ch_pkl(data_path):
    #获取每个数据的属性
    """
    要获取文本数据， 情感数据， 原因的位置，以及要进行核对原因位置是否正确
    获取example的list列表
    """
    #[{'docID': 0}, 
    # {'name': 'happiness', 'value': '3'}, 
    # [{'key-words-begin': '0', 'keywords-length': '2', 'keyword': '激动', 'clauseID': 3, 'keyloc': 2}], 
    # [{'id': '1', 'type': 'v', 'begin': '43', 'length': '11', 'index': 1, 'cause_content': '接受并采纳过的我的建议', 'clauseID': 5}], 
    # [{'id': '1', 'cause': 'N', 'keywords': 'N', 'clauseID': 1, 'content': '河北省邢台钢铁有限公司的普通工人白金跃，', 'cause_content': '', 'dis': -2}, 
    # {'id': '2', 'cause': 'N', 'keywords': 'N', 'clauseID': 2, 'content': '拿着历年来国家各部委反馈给他的感谢信，', 'cause_content': '', 'dis': -1}, 
    # {'id': '3', 'cause': 'N', 'keywords': 'Y', 'clauseID': 3, 'content': '激动地对中新网记者说。', 'cause_content': '', 'dis': 0}, 
    # {'id': '4', 'cause': 'N', 'keywords': 'N', 'clauseID': 4, 'content': '“27年来，', 'cause_content': '', 'dis': 1}, 
    # {'id': '5', 'cause': 'Y', 'keywords': 'N', 'clauseID': 5, 'content': '国家公安部、国家工商总局、国家科学技术委员会科技部、卫生部、国家发展改革委员会等部委均接受并采纳过的我的建议', 'cause_content': '接受并采纳过的我的建议', 'dis': 2}]]

    data = loadList(data_path)
    clause_num = []
    clause_len = []
    emotion_len = []

    pad_len =[]
    start_l = []
    end_l  = []
    cause_l = []

    for index, item in enumerate(data):

        clause_len_doc= []

        docID = item[0]['docID']
        emotion_loc = int(item[2][-1]['keyloc'])
        emotion_word = get_clean_data_ch(item[2][-1]['keyword']) #获取情去燥的情绪词语

        clause_info = item[4] #clause信息
        emo_clause = clause_info[emotion_loc]['content']
        emotion_content = list(get_clean_data_ch(emo_clause))

        emotion_len.append(len(emotion_content)) #情绪子句的长度
        clause_num.append(len(clause_info)) #每个文档中子句的个数

        for indexc, itemc in enumerate(clause_info):
            content_text =get_clean_data_ch(itemc['content'])
            content_l = list(content_text)
            clause_len.append(len(content_l))
            clause_len_doc.append(len(content_l))

            for indexc, itemc in enumerate(clause_info):
                content_text =get_clean_data_ch(itemc['content'])
                content_l = list(content_text)
                # content_l.append('[SEP]')#添加【SEP】字符
                clause_len.append(len(content_l))#获取子句的长度

                ifcause = itemc['cause']
                if ifcause == 'Y':
                    cause_content = get_clean_data_ch(itemc['cause_content'])
                    start, end = get_ch_target(content_text, cause_content)
                    start_l.append(start)
                    end_l.append(end)
                    cause_l.append(indexc)

        a = max(clause_len_doc) * len(clause_info)
        pad_len.append(a)
        if a > 450:
            print('item= ', item)
        
    return  clause_num,clause_len,emotion_len, pad_len, start_l, end_l, cause_l


def get_ch_target(para_text,cause):
    """
    获取原因内容
    和原因内容
    """
    # print('para_text = ', para_text)
    # print('cause = ', cause)
    text_token = list(para_text)
    cause_token = list(cause)

    start = -1 
    end = -1
    for i in range(0, len(text_token)):
        if text_token[i:i+len(cause_token)] == cause_token:
            start = i
            end = i + len(cause_token)
            return start, end
            
    if start == -1 or end == -1:
        print('text_token = ',text_token)
        print('cause_token = ',cause_token)
        raise ValueError("原因不在子句中")
    return start, end



data_path = '/home/MHISS/liqiang/sigirshort/GNN_ECA/dataset/eca_ch/eca_ch_data.pkl'
clause_num,clause_len,emotion_len, pad_len, start_l, end_l, cause_l = read_ch_pkl(data_path)


print('最多的子句个数：', max(clause_num)) #30个子句
print('最长的子句长度：', max(clause_len)) #最长的子句长度：69
print('最长的情绪子句的长度：', max(emotion_len)) #最长的情绪子句的长度
print('填充后的长度：', max(pad_len)) #填充后的长度

print('开始位置最大的值', max(start_l))
print('结束位置最大的值', max(end_l))
print('原因位置最大的值', max(cause_l))

a1, a2, a3 = 0, 0, 0
for iiiu, iiiy in zip(start_l, end_l):
    if iiiu > 40:
        a1+= 1
    if iiiu > 30:
        a2+= 1
    
print('原因位置大于8的有几个：', a1)
print('原因位置大于10的有几个：', a2)




a = 0
b = 0
for item in pad_len:
    if item > 480:
        a += 1
        print(item)
    else:
        b += 1
        
print('a = ', a)
print('b = ', b)
#注释掉了 id="1625"














def open_file(path, sep=' ', mode='train'):
    """读取文件"""
    src = []
    tgt = []
    with open(path, 'r', encoding='utf8') as f:
        content = f.readlines()#[:2000]
        tmp_src = []
        tmp_tgt = []
        for i, line in enumerate(content):
            line = line.strip().split(sep)
            # 若数据包含src和tgt
            if len(line) == 2:
                # tmp_src.append(line[0])
                tmp_src.append(line[0])
                tmp_tgt.append(line[1])
            elif i == len(content)-1:  
                # 最后一行数据     
                if tmp_src:
                    src.append(tmp_src)
                    tgt.append(tmp_tgt)
            else:
                if tmp_src:
                    src.append(tmp_src)
                    tgt.append(tmp_tgt)
                    tmp_src = []
                    tmp_tgt = []
    return src, tgt


# def open_file(path, sep=' ', mode='train'):
#     """读取文件"""
#     src = []
#     tgt = []
#     with open(path, 'r', encoding='utf8') as f:
#         if mode != 'test':
#             content = f.readlines()#[:500]
#             tmp_src = []
#             tmp_tgt = []
#             for i, line in enumerate(content):
#                 line = line.strip().split(sep)
#                 # 若数据包含src和tgt
#                 if len(line) == 2:
#                     # tmp_src.append(line[0])
#                     tmp_src.append(line[0])
#                     tmp_tgt.append(line[1])
#                 elif i == len(content)-1:  
#                     # 最后一行数据     
#                     if tmp_src:
#                         src.append(tmp_src)
#                         tgt.append(tmp_tgt)
#                 else:
#                     if tmp_src:
#                         src.append(tmp_src)
#                         tgt.append(tmp_tgt)
#                         tmp_src = []
#                         tmp_tgt = []
#         else:
#             content = f.readlines()#[:500]
#             tmp_src = []
#             for i, line in enumerate(content):
#                 line = line.strip().split(sep)
#                 # 若数据包含src和tgt
#                 if line[0] != '':
#                     # tmp_src.append(line[0])
#                     tmp_src.append(line[0])
#                 elif i == len(content)-1:  
#                     # 最后一行数据     
#                     if tmp_src:
#                         src.append(tmp_src)
#                 else:
#                     if tmp_src:
#                         src.append(tmp_src)
#                         tmp_src = []
#     return src, tgt


def write_file(word2index, path):
    """写文件"""
    with open(path, 'w', encoding='utf8') as f:
        for k,v in word2index.items():
            string = k + ' ' + str(v) + '\n'
            f.write(string)


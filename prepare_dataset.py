import json
import os
import numpy as np

fact = []
accusation = []

with open('data/CAIL2018_final_all_data/exercise_contest/data_train.json', 'r', encoding='utf-8') as f:
    for line in f:
        cur_accu = json.loads(line)["meta"]["accusation"]
        if not (len(cur_accu) == 1 and (cur_accu[0] == '走私' or cur_accu[0] == '经济犯')):
            # two outdated charges which are removed from law now 
            fact.append(json.loads(line)["fact"])
            accusation.append(cur_accu)

# some charges are mistyped in the dataset
accu_correction = {
    '[组织、强迫、引诱、容留、介绍]卖淫': ['组织卖淫', '强迫卖淫', '引诱、容留、介绍卖淫'],
    '[制造、贩卖、传播]淫秽物品': ['制作、复制、出版、贩卖、传播淫秽物品牟利', '传播淫秽物品'],
    '[盗窃、抢夺][枪支、弹药、爆炸物]': ['盗窃、抢夺枪支、弹药、爆炸物、危险物质'],
    '[伪造、变造]居民身份证': ['伪造、变造、买卖身份证件'],
    '[窝藏、转移、收购、销售]赃物': ['掩饰、隐瞒犯罪所得、犯罪所得收益'],
    '非法获取公民个人信息': ['侵犯公民个人信息'],
    '[盗窃、侮辱]尸体': ['盗窃、侮辱、故意毁坏尸体、尸骨、骨灰'],
    '[隐匿、故意销毁][会计凭证、会计帐簿、财务会计报告]': ['隐匿、故意销毁会计凭证、会计帐簿、财务会计报告'],
    '强制[猥亵、侮辱]妇女': ['强制猥亵、侮辱'],
    '非法买卖制毒物品': ['非法生产、买卖、运输制毒物品、走私制毒物品'],
    '非法[生产、销售]间谍专用器材': ['非法生产、销售专用间谍器材、窃听、窃照专用器材']
}

accu_clean = []
for i in range(len(accusation)):
    accu_cur = []
    for j in accusation[i]:
        if j in accu_correction:
            tmp = accu_correction[j]
        else:
            tmp = [j.replace('[', '').replace(']', '')]
        accu_cur.extend(tmp)
    accu_clean.append(accu_cur)

if not os.path.exists('data/dataset/'):
    os.makedirs('data/dataset/')

with open('data/data.json', 'w', encoding='utf-8') as f:
    for i in range(len(fact)):
        d={'fact':fact[i], 'accusation':accu_clean[i]}
        out=json.dumps(d, indent=0, ensure_ascii=False)
        out=out.replace('\n', '')
        f.write(out+'\n')


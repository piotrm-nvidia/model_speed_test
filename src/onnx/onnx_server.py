# -*- coding: UTF-8 -*-
import uvicorn
from fastapi import FastAPI, Request, Form
import onnxruntime as ort
import time,os,json
from transformers import BertTokenizerFast
import numpy as np

app = FastAPI()

def softmax(x):
    """Compute the softmax in a numerically stable way."""
    x = x - np.max(x)
    exp_x = np.exp(x)
    softmax_x = exp_x / np.sum(exp_x)
    return [round(i, 4) for i in softmax_x]

bert_config = {
    "model_name": "tinybert4L.onnx",
    "max_length": 64,
    "id2tag": {"0": "否", "1": "是"},
    "all_label":  [
        "新闻", "广播", "有声书", "个人成长", "儿童", "人文国学",
        "音乐", "生活", "外语", "有声图书", "历史", "商业财经",
        "相声评书", "娱乐", "热点", "广播剧", "其他"
    ]
}

model_dir = "/home/piotrm/src/private-notes/2024/02/22/tiny_classify_bert"

# 超参数
max_length = bert_config["max_length"]
id2tag = bert_config["id2tag"]
all_label = bert_config["all_label"]

tokenizer = BertTokenizerFast.from_pretrained(model_dir, local_files_only=True)

model_path = os.path.join(model_dir, bert_config["model_name"])
bert_model = ort.InferenceSession(model_path, providers=['CUDAExecutionProvider'])
@app.get('/predictSingle')
def predictSingle(q):

    query_ls = [q]

    text_ls = sum([[i] * len(all_label) for i in query_ls], [])
    label_ls = all_label * len(query_ls)
    assert len(text_ls) == len(label_ls)

    tok = tokenizer(label_ls, text_ls, max_length=max_length, truncation=True, padding="max_length")

    inputs = {
        'inp_ids': np.array(tok["input_ids"]),
        'att_mask': np.array(tok["attention_mask"]),
        'tok_tp_ids': np.array(tok["token_type_ids"]),

    }
    outs = bert_model.run(None, inputs)
    prob_outputs = [softmax(i)[1] for i in outs[0]]
    raw_result_ls = [
        {
            all_label[idx]: str(ls) for idx, ls in enumerate(prob_outputs[j:j + len(all_label)])
        } for j in range(0, len(text_ls), len(all_label))
    ]

    result = [json.dumps(i, ensure_ascii=False) for i in raw_result_ls]

    return "分类结果:" + str(result)


if __name__ == '__main__':
    uvicorn.run(app, host='0.0.0.0', port=8480)



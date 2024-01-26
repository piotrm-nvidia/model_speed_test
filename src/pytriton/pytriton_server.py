# -*- coding: UTF-8 -*-
from fastapi import FastAPI, Request, Form
from pytriton.decorators import batch
from pytriton.model_config import ModelConfig, Tensor
from pytriton.triton import Triton,TritonConfig
import numpy as np
import onnxruntime as ort
import time,os,json
from transformers import BertTokenizerFast

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

model_dir = "/var/log/tfs-publish/triton-qp-intent-classification/served_models/triton_qp_intent_classification/1706252719442/"

# 超参数
max_length = bert_config["max_length"]
id2tag = bert_config["id2tag"]
all_label = bert_config["all_label"]

tokenizer = BertTokenizerFast.from_pretrained(model_dir, local_files_only=True)

model_path = os.path.join(model_dir, bert_config["model_name"])
bert_model = ort.InferenceSession(model_path, providers=['CUDAExecutionProvider'])
@batch
def _infer_fn(sentence: np.ndarray):
    t1 = time.time()

    sequences_batch = np.char.decode(sentence.astype("bytes"), "utf-8")
    # print("sequences_batch", sequences_batch)
    query_ls = [s[0] for s in sequences_batch]

    # query_ls = ["剑来", "提上"]

    text_ls = sum([[i] * len(all_label) for i in query_ls], [])
    # text_ls = query_ls * len(all_label)
    label_ls = all_label * len(query_ls)
    assert len(text_ls) == len(label_ls)

    # print("text_ls", text_ls)
    tok = tokenizer(label_ls, text_ls, max_length=max_length, truncation=True, padding="max_length")

    inputs = {
        'inp_ids': np.array(tok["input_ids"]),
        'att_mask': np.array(tok["attention_mask"]),
        'tok_tp_ids': np.array(tok["token_type_ids"]),

    }
    # print(inputs)
    outs = bert_model.run(None, inputs)
    prob_outputs = [softmax(i)[1] for i in outs[0]]
    raw_result_ls = [
        {
            all_label[idx]: str(ls) for idx, ls in enumerate(prob_outputs[j:j + len(all_label)])
        } for j in range(0, len(text_ls), len(all_label))
    ]

    result = [json.dumps(i, ensure_ascii=False) for i in raw_result_ls]
    r = {"output": np.array(result)}
    cost = time.time() - t1
    print(f"t={cost:.3f} q={sequences_batch} r={result}")

    return r

if __name__ == '__main__':
    triton_config = TritonConfig(
        http_port=8380,
        allow_grpc=False,
        allow_metrics=False
    )
    with Triton(config=triton_config) as triton:
        triton.bind(
            model_name="BERT",
            infer_func=_infer_fn,
            inputs=[
                Tensor(name="sentence", dtype=np.bytes_, shape=(1,)),
            ],
            outputs=[
                Tensor(name="output", dtype=np.bytes_, shape=(1,)),
            ],
            config=ModelConfig(max_batch_size=1024)
        )
        triton.serve()




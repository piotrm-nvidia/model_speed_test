import json
import requests
import os
from speed_frame import speed_frame
def make_request(q, fist_time_cost):

    d = {
        "id": "0",
        "inputs": [
            {
                "name": "sentence",
                "shape": [1, 1],
                "datatype": "BYTES",
                "data": [q]
            }
        ]
    }
    res = requests.post(url="http://192.168.41.213:8380/v2/models/BERT/infer", json=d).json()
    print(res)
    r = res["outputs"][0]["data"][0]
    return r

with open('../data/test.txt', 'r') as file:
    prompts = [line.split("	")[0].strip() for line in file if line.strip()]
prompts = prompts[0:500]
max_worker_list = [1,2,3,5,10,20]
# max_worker_list = [1]
log_name = os.path.basename(__file__).split(".")[0]
print(log_name)
a = speed_frame(make_request, prompts, log_name, is_file=False)
a.run(max_worker_list)
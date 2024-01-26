import json
import requests
import os
from speed_frame import speed_frame
def make_request(d, fist_time_cost):
    url = "http://192.168.41.213:8480/predictSingle?q="+d

    res = requests.get(url=url)

    return res.text

with open('../data/test.txt', 'r') as file:
    prompts = [line.split("	")[0].strip() for line in file if line.strip()]
prompts = prompts[0:500]
max_worker_list = [1,2,3,5,10,20]
# max_worker_list = [10]
log_name = os.path.basename(__file__).split(".")[0]
print(log_name)
a = speed_frame(make_request, prompts, log_name, is_file=False)
a.run(max_worker_list)
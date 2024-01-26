from concurrent import futures
import json
import time
from time import perf_counter
import datetime
import numpy as np
from concurrent import futures
import time
import logging
import os

class speed_frame():

    def get_logger(self, logger_name, log_file, level=logging.INFO, is_console=True, is_file=True):
        logger = logging.getLogger(logger_name)
        logger.setLevel(logging.INFO)
        if is_file:
            fileHandler = logging.FileHandler(log_file, mode='a')
            fileHandler.setFormatter(logging.Formatter("%(message)s"))
            fileHandler.setLevel(level)

            logger.addHandler(fileHandler)

        if is_console:
            # 创建一个将日志消息发送到控制台的处理程序
            console_handler = logging.StreamHandler()
            console_handler.setLevel(logging.INFO)  # 设置控制台日志级别
            console_handler.setFormatter(logging.Formatter('%(message)s'))

            logger.addHandler(console_handler)

        return logger
    def __init__(self, make_request, prompts, log_name, is_file=True, log_t=True):
        super().__init__()
        self.make_request = make_request
        self.prompts = prompts

        self.fist_time_cost=[] #首个字符出现时间
        self.muti_time_cost=[]
        self.muti_all_cost = 0

        self.summary_info = []

        self.log_t = log_t

        self.max_workers = 1

        t = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
        current_directory = os.path.dirname(os.path.abspath(__file__))
        print(current_directory)
        LOG_FILE_detail = f'{current_directory}/log/{log_name}_detail_{t}.log'
        LOG_FILE_summary = f'{current_directory}/log/{log_name}_summary_{t}.log'
        self.logger_detail = self.get_logger("detail", LOG_FILE_detail, level=logging.INFO, is_console=True, is_file=is_file)
        self.logger_summary = self.get_logger("summary", LOG_FILE_summary, level=logging.INFO, is_console=True, is_file=is_file)

    def process_request(self, q):
        t1 = time.time()
        c = 3
        while c > 0:
            try:
                c = c-1
                r = self.make_request(q, self.fist_time_cost)
                c = 0
            except Exception:
                print("error")

        cost = time.time() - t1
        self.muti_time_cost.append(cost)
        if self.log_t:
            self.logger_detail.info(f't={cost:.3f} w={self.max_workers} q={q} r={r}')
        # self.logger_detail.flush()
        return r

    def muti(self, prompts):
        global muti_all_cost
        result_list = []
        with futures.ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            t1 = time.time()
            futures_list = []
            for i, prompt in enumerate(prompts):
                f = executor.submit(self.process_request, prompt)
                futures_list.append(f)
            # for i, future in enumerate(futures.as_completed(futures_list)):
            #     d = future.result()
            #     result_list.append(d)
            t2 = time.time()
            muti_all_cost = t2 - t1
            return result_list
            # muti_time_cost.append("muti_all_cost="+str(t2 - t1))
    def cal_time(self, token_speed, fist_time_cost, costs, all_cost):
        first_time_avg = np.mean(fist_time_cost)  #
        time_sum = np.sum(costs)
        time_avg = np.mean(costs)
        time_min = np.min(costs)
        time_max = np.max(costs)
        time_p90 = np.percentile(costs, 90)  # 95分位数
        time_p95 = np.percentile(costs, 95)  # 95分位数
        time_p99 = np.percentile(costs, 99)  # 95分位数
        return f"{self.max_workers}并发：单位 (s) 平均-{time_avg:.3f}; 总耗时-{time_sum:.3f}; 最小-{time_min:.3f}; 最大-{time_max:.3f}; 90位-{time_p90:.3f}; 95位-{time_p95:.3f}; 99位-{time_p99:.3f};"

    def muti_test(self):
        result_list = self.muti(self.prompts)
        token_speed = len("".join(result_list))/muti_all_cost

        r = self.cal_time(token_speed, self.fist_time_cost, self.muti_time_cost, self.muti_all_cost)
        # self.logger_summary.info(f"muti_all_cost={self.muti_all_cost}")
        # self.logger_summary.info(f"muti_time_cost={self.muti_time_cost}")
        self.summary_info.append(r)
        self.logger_summary.info(r)

    def run(self, max_worker_list):
        # max_worker_list = [1, 2, 3, 5, 10 ,20]
        for worker in max_worker_list:
            self.fist_time_cost = []  # 首个字符出现时间
            self.muti_time_cost = []
            self.muti_all_cost = 0

            self.max_workers = worker
            self.muti_test()
        for k in self.summary_info:
            print(k)
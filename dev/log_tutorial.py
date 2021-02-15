import multiprocessing
import csv
import time
import logging
logging.basicConfig(filename="result.log", 
                    level=logging.INFO, 
                    format='%(asctime)s %(levelname)s %(message)s')
def mp_worker(args):
    time.sleep(1)
    id = args["id"]
    t = args["duration"]
    v = args['value']
    return {
        "name": f"worker_{id}",
        "duration": t,
        "result": v ** 2,
    }

def mp_handler(data_list):
    with multiprocessing.Pool(4) as pool:
        with open('results.csv', 'w+') as f:
            writer = csv.writer(f, lineterminator = '\n', delimiter=",")
            for result in pool.imap_unordered(mp_worker, iterable=data_list):
                logging.info(f'Writing result for {result["name"]}')
                writer.writerow([result["name"], result["duration"], result["result"]])

#%%
# Generate data list
N = 10
data_list = []
for i in range(N):
    item = {}
    item["id"] = i
    item["duration"] = (N - i) / 2
    item['value'] = i + 1
    data_list.append(item)

# %%
mp_handler(data_list)
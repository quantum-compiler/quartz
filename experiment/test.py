from tqdm import tqdm
import time
pbar = tqdm(total=100)

with pbar:
    for i in range(50):
        pbar.update(0)
        time.sleep(3)

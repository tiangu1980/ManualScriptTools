import threading
import time
import random
from datetime import datetime

# 公共变量
publicOutL = []
publicFlag = True
lock = threading.Lock()  # 用于线程间同步的锁

def worker(thread_id):
    # 随机 sleep 1 到 10 秒
    sleep_time = random.randint(1, 10)
    time.sleep(sleep_time)
    
    # 生成当前时间戳和线程 ID
    current_time = datetime.now().strftime("%H-%M-%S:%f")[:-3]
    output = f"{current_time}\t\t{thread_id}"
    
    # 加锁保护共享变量
    with lock:
        publicOutL.append(output)
        print(output)
    
    while True:
        time.sleep(1)
        with lock:
            if len(publicOutL) == 5:                
                break
    current_time = datetime.now().strftime("%H-%M-%S:%f")[:-3]
    print(f"{current_time} Thread {thread_id} finished.")

def main():
    num_threads = 5  # 定义线程数
    threads = []
    
    # 创建并启动线程
    for i in range(num_threads):
        thread = threading.Thread(target=worker, args=(i,))
        threads.append(thread)
        thread.start()

    # 主线程定期检查公共变量列表的长度
    while True:
        time.sleep(1)  # 每秒检查一次
        with lock:
            if len(publicOutL) == num_threads:
                main_current_time = datetime.now().strftime("%H-%M-%S:%f")[:-3]
                publicFlag = False
                print(f"{main_current_time}\t\tEnable flag.")
                break

    # 等待所有线程结束
    for thread in threads:
        thread.join()

    print("End All")

if __name__ == "__main__":
    main()

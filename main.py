import threading
import time

class Stream:
    def __enter__(self):
        print(f"[{threading.current_thread().name}] {self.name} entered")

    def __exit__(self, exc_type, exc, tb):
        print(f"[{threading.current_thread().name}] {self.name} exited")

    def __init__(self, name):
        self.name = name


class Task:
    """一个任务包含：计算流、拷贝流、以及它们之间的同步事件"""
    def __init__(self, name):
        self.compute_stream = Stream(f"compute-{name}")
        self.copy_stream    = Stream(f"copy-{name}")
        self.compute_done   = threading.Event()

    def compute_worker(self, steps=10):
        """计算线程：重复 steps 次推理+采样（耗时较长）"""
        for i in range(steps):
            with self.compute_stream:
                print(f"  >> {self.compute_stream.name} running step {i} ...")
                time.sleep(0.5)   # 模拟计算耗时
            self.compute_done.set()   # 通知拷贝线程

    def copy_worker(self, steps=10):
        """拷贝线程：重复 steps 次拷贝（耗时较短），每次等待对应计算完成"""
        for i in range(steps):
            self.compute_done.wait()          # 等待调度
            with self.copy_stream:
                print(f"  << {self.copy_stream.name} copying step {i} ...")
                time.sleep(0.1)               # 模拟拷贝耗时
            self.compute_done.clear()         # 清除事件，准备下一轮


# 创建两个对半拆分的任务
task_A = Task("A")
task_B = Task("B")

# 启动四个工作线程
t_comp_A = threading.Thread(target=task_A.compute_worker, args=(10,), name="Comp-A")
t_copy_A = threading.Thread(target=task_A.copy_worker,   args=(10,), name="Copy-A")
t_comp_B = threading.Thread(target=task_B.compute_worker, args=(10,), name="Comp-B")
t_copy_B = threading.Thread(target=task_B.copy_worker,   args=(10,), name="Copy-B")

for t in (t_comp_A, t_copy_A, t_comp_B, t_copy_B):
    t.start()
for t in (t_comp_A, t_copy_A, t_comp_B, t_copy_B):
    t.join()
from typing import Iterable

import torch
import os
import uuid
import time
import enum

from ..reference import rwkv7, utils, sampler
from ..state_manager import state_pool
from .patch import PatchedRWKV7


torch.cuda.init()
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cuda.matmul.allow_tf32 = True

reference_path = os.path.dirname(os.path.abspath(rwkv7.__file__))

# EOS_TOKEN_ID = torch.tensor([[0],[261, 24281]], device="cuda")

class Status(enum.Enum):
    READY = 0
    RUNNING = 1
    FINISHED = 2

class RWKV_070_Pipeline:
    def __init__(
        self,
        model_path: str,
        vocab_path: str = None,
        head_size: int = 64,
        vocab_size: int = 65536,
    ):
        if not vocab_path:
            vocab_path = os.path.join(reference_path, "rwkv_vocab_v20230424.txt")
        load_time = time.time()
        args = type(
            "Args",
            (),
            {
                "MODEL_NAME": model_path,
                "head_size": head_size,
                "vocab_size": vocab_size,
            },
        )()
        self.model = PatchedRWKV7(args)
        self.tokenizer = utils.TRIE_TOKENIZER(vocab_path)
        print(
            f"[Pipeline] Initialized in {time.time() - load_time:.2f} seconds. Model path: {model_path}"
        )

    def gen_state(self, batch_size: int = 0) -> list[torch.Tensor]:
        state = self.model.generate_zero_state(batch_size)
        return state

class TaskManager:
    def __init__(
        self,
        pipeline: RWKV_070_Pipeline,
        max_batch_pair_size: int = 15,
        buffer_size: int = 100,
    ):
        self.pipeline = pipeline
        self.sampler = sampler.Sampler()
        self.tasks: list["Task"] = list()
        self.finished_tasks = list()
        self.max_batch_pair_size = max_batch_pair_size
        self.buffer_size = buffer_size
        self.worker = [None] * 2
        self.stream_compute = torch.cuda.Stream()
        self.stream_copy = torch.cuda.Stream()
        self.clear_worker(0)
        self.clear_worker(1)


    def clear_worker(self,pair:int):
        pair = pair % 2
        self.worker[pair] = dict(
            last_tokens=torch.zeros(
                self.max_batch_pair_size, dtype=torch.int32, device="cuda"
            ),
            max_tokens=torch.zeros(
                self.max_batch_pair_size, dtype=torch.int32, device="cuda"
            ),
            tasks=[None] * self.max_batch_pair_size,
            generated_tokens=torch.zeros(
                (self.max_batch_pair_size, self.buffer_size),
                dtype=torch.int32,
                device="cuda",
            ),
            state=self.pipeline.gen_state(self.max_batch_pair_size),
            penalties=torch.zeros(
                self.max_batch_pair_size,
                self.pipeline.model.args.vocab_size,
                dtype=torch.float32,
                device="cuda",
            ),
            rand_state=torch.zeros(
                64 * self.max_batch_pair_size, dtype=torch.int8, device="cuda"
            ),
            # 采样参数
            presence_penalties=torch.zeros(
                self.max_batch_pair_size, dtype=torch.float16, device="cuda"
            ),
            repetition_penalties=torch.zeros(
                self.max_batch_pair_size, dtype=torch.float16, device="cuda"
            ),
            penalty_decays=torch.zeros(
                self.max_batch_pair_size, dtype=torch.float16, device="cuda"
            ),
            temperatures=torch.zeros(
                self.max_batch_pair_size, dtype=torch.float16, device="cuda"
            ),
            top_ps=torch.zeros(self.max_batch_pair_size, dtype=torch.float16, device="cuda"),
            top_ks=torch.zeros(self.max_batch_pair_size, dtype=torch.int, device="cuda"),
        )

    def new_task(
        self,
        prompt: str | list[int],
        max_tokens: int = 50,
        repetition_penalty: float = 0,
        presence_penalty: float = 0,
        penalty_decay: float = 0,
        temperature: float = 0.3,
        top_p: float = 0.1,
        top_k: int = 20,
        seed: int = 42,
    ):
        """
        Args:
            presence_penalty: 存在惩罚系数
            repetition_penalty: 重复惩罚系数
            penalty_decay: 惩罚的衰减因子
            temperature: 采样温度，范围 [0.001, 1000]
            top_k: Top-k 参数，-1 表示不使用，<=0 或 >V 时会被设为 V
            top_p: Top-p 参数，范围 [0.0, 1.0]，为 0 时会自动设为 top_k=1, top_p=1.0
        """
        task = Task(
            prompt=prompt,
            pipeline=self.pipeline,
            sampler=self.sampler,
            max_tokens=max_tokens,
            presence_penalty=presence_penalty,
            repetition_penalty=repetition_penalty,
            penalty_decay=penalty_decay,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            seed=seed,
        )
        self.tasks.append(task)
        return task

    # mask顾名思义，遮挡不需要推理的slot，True表示这个slot空闲，可以放入新的任务，False表示这个slot被占用，正在推理中
    def update_batch(self, mask: torch.Tensor, pair: int = 0):
        pair = pair % 2
        # 直接在GPU上查找所有空闲槽位（True=空闲）
        free_slots = torch.where(mask)[0].tolist()  # 得到 [0, 2] 这样的空闲槽位列表

        batch_tasks = [task for task in self.tasks if task.run==Status.READY]
        batch_tasks = batch_tasks[: len(batch_tasks)//2] # 取前半部分任务，后半部分留给另一个worker
        # 取空闲槽位和待分配任务的最小数量
        num_assign = min(len(free_slots), len(batch_tasks))
        worker = self.worker[pair]

        for i in range(num_assign):
            slot_id = free_slots[i]  # 按顺序分配空闲槽位
            task = batch_tasks[i]
            task.run = Status.RUNNING
            task.prepare()
            worker["last_tokens"][slot_id] = task.current_token
            worker["max_tokens"][slot_id] = task.max_tokens
            worker["tasks"][slot_id] = task
            worker["state"][0][:, :, slot_id, :] = task.state0
            worker["state"][1][:, slot_id, :, :, :] = task.state1
            worker["penalties"][slot_id] = task.penalties
            worker["rand_state"][
                slot_id * 64 : (slot_id + 1) * 64
            ] = task.rand_state
            worker["presence_penalties"][slot_id] = task.presence_penalty
            worker["repetition_penalties"][slot_id] = task.repetition_penalty
            worker["penalty_decays"][slot_id] = task.penalty_decay
            worker["temperatures"][slot_id] = task.temperature
            worker["top_ps"][slot_id] = task.top_p
            worker["top_ks"][slot_id] = task.top_k
            mask[slot_id] = False  # Mark this slot as occupied

    def task_loop(self):
        # 先遮住全部，等update_batch放入任务的时候再逐渐放开，直到全部都放满了或者没有任务了
        mask = torch.ones((2, self.max_batch_pair_size), dtype=torch.bool, device="cuda")
        pair = 0
        while self.tasks:
            pair = (pair + 1) % 2
            self.update_batch(mask[pair], pair)
            if mask.all():  # 全部都遮住了，推理空了，说明没有任务了，直接结束循环
                break
            st = time.time()
            mask[pair] = self.task_single(mask[pair],pair)
            print(
                ((~mask).sum() * self.buffer_size / (time.time() - st)).item(),
                "tokens/s",
            )

    @torch.no_grad()
    def task_single(self, mask: torch.Tensor, pair: int):
        pair = pair % 2
        worker = self.worker[pair]
        tasks: list[Task] = worker["tasks"]
        last_tokens = worker["last_tokens"]
        state0, state1 = worker["state"]
        penalties, rand_state = worker["penalties"], worker["rand_state"]
        presence_penalties = worker["presence_penalties"]
        repetition_penalties = worker["repetition_penalties"]
        penalty_decays = worker["penalty_decays"]
        temperatures = worker["temperatures"]
        top_ps, top_ks = worker["top_ps"], worker["top_ks"]
        max_tokens = worker["max_tokens"]
        generated_tokens = worker["generated_tokens"]
        max_bs = self.max_batch_pair_size

        for slot_i in range(self.buffer_size):
            # mask True 的给0，不让瞎推理，mask False 的正常推理
            state0 *= ~mask.view(1, 1, max_bs, 1)
            state1 *= ~mask.view(1, max_bs, 1, 1, 1)

            logits = self.pipeline.model.patch_forward_seq_batch(
                last_tokens[: self.max_batch_pair_size].unsqueeze(-1), (state0, state1)
            )
            tokens = self.sampler.sample_repetition(
                logits=logits,
                penalties=penalties,
                states=rand_state,
                presence_penalties=presence_penalties,
                repetition_penalties=repetition_penalties,
                penalty_decays=penalty_decays,
                temperatures=temperatures,
                top_ps=top_ps,
                top_ks=top_ks,
                # eos_mask
                eos_mask=mask,
            )
            
            # 如果这个token是EOS_TOKEN_ID，说明这个任务结束了，更新state_finished和penalties_finished，并把这个slot标记为True，表示空闲了
            # 这里是特殊处理一下，因为有些任务的EOS_TOKEN_ID是[261, 24281]( 回车 , User: )，所以当前一个token是261并且当前token是24281的时候结束

            generated_tokens[mask, slot_i] = 0
            a = (tokens == 0)
            b = ((last_tokens == 261) & (tokens == 24281))
            mask_tmp = a | b
            
            tokens[b] = 0

            max_tokens[~mask] -= 1
            mask_new = (mask_tmp | (max_tokens <= 0)) & ~mask

            generated_tokens[~mask, slot_i] = tokens[~mask]

            mask |= mask_new

            if mask_new.any():
                finished_indices = torch.where(mask_new)[0].tolist()
                for idx in finished_indices:
                    task = tasks[idx]
                    task.current_token = tokens[idx].item()
                    task.state0.copy_(state0[:, :, idx])
                    task.state1.copy_(state1[:, idx])
                    task.penalties.copy_(penalties[idx])
                    task.rand_state.copy_(rand_state[idx * 64 : (idx + 1) * 64])

            tokens[mask] = 0
            last_tokens.copy_(tokens)
            if mask.all():
                generated_tokens[mask, slot_i:] = 0
                break

        # 遍历一下workers吧
        for slot_i in range(self.max_batch_pair_size):
            tmp = generated_tokens[slot_i][: self.buffer_size]

            # 如果全是0，说明已经结束并且遮住了，直接跳过就好
            if not tmp.any():
                continue
            # 全部都不为0，说明还在推理中，直接加入生成的tokens就好
            if tmp.all():
                # 这里的回调函数是直接把生成的tokens加入到task.generated_tokens里，所以不需要管EOS_TOKEN_ID，等推理结束了自然就不生成了
                tasks[slot_i].callback(tmp.tolist())
            # 有的为0有的不为0，说明这一轮刚好结束了，加入生成的tokens，并把这个task加入finished_tasks里
            else:
                tmp = tmp[tmp != 0].tolist()
                # 加入生成的tokens，注意要把最后一个tokens加进去，有可能是eos，会被掩码误吞，而截断不吞
                if tasks[slot_i].current_token == 0:
                    tmp.append(0)
                tasks[slot_i].callback(tmp)
                tasks[slot_i].finished()
                self.finished_tasks.append(tasks[slot_i])
                tasks[slot_i].run = Status.FINISHED
        return mask


class Task:
    def __init__(
        self,
        prompt: str | list[int],
        pipeline: RWKV_070_Pipeline,
        sampler: sampler.Sampler,
        max_tokens: int = 2048,
        presence_penalty: float = 2,
        repetition_penalty: float = 0,
        penalty_decay: float = 0.994,
        temperature: float = 1,
        top_k: int = 20,
        top_p: float = 0.5,
        seed: int = 42,
    ):
        self.state0, self.state1 = pipeline.gen_state()
        self.rand_state = sampler.setup_rand(seed, 1)
        self.penalties = torch.zeros(
            pipeline.model.args.vocab_size, dtype=torch.float32
        )
        self.pipeline = pipeline

        self.max_tokens = max_tokens
        self.presence_penalty = presence_penalty
        self.repetition_penalty = repetition_penalty
        self.penalty_decay = penalty_decay
        self.temperature = temperature
        self.top_k = top_k
        self.top_p = top_p
        
        self.generated_tokens = list()

        self.current_token = -1
        self.run = Status.READY

        self.prefill(prompt)

        # to cpu
        self.cpu()

    def save_state(self):
        session_id = uuid.uuid4().hex
        current_token = torch.tensor(self.current_token, dtype=torch.int32)
        state_pool.get_state_manager().put_state(session_id, [self.state0, self.state1, current_token])
        return session_id

    def load_state(self, session_id: str):
        self.state0, self.state1, current_token = state_pool.get_state_manager().get_state(session_id)
        self.current_token = current_token.item()
        return session_id

    def tokenize(self, prompt: str) -> list[int]:
        if isinstance(prompt, str):
            prompt = self.pipeline.tokenizer.encode(prompt)
        return prompt
    
    def continue_gen(self):
        self.run = Status.READY
        self.cpu()

    def prefill(self, prompt: Iterable[int]|str):
        prompt = self.tokenize(prompt)
        self.generated_tokens += prompt
        assert len(prompt) >= 1 and all(
            isinstance(i, int) for i in prompt
        ), "Prompt must be a list of integers"
        if self.current_token != -1:
            prompt.insert(0, self.current_token)
        if len(prompt) >= 2:
            if self.state0.device != "cuda":
                self.cuda()
            self.pipeline.model.forward_seq(
                prompt[:-1], (self.state0, self.state1), False
            )

        self.current_token = prompt[-1]

        return

    def prepare(self):
        self.cuda()

    def stop(self):
        self.cpu()

    def cuda(self):
        self.state0 = self.state0.cuda()
        self.state1 = self.state1.cuda()
        self.rand_state = self.rand_state.cuda()
        self.penalties = self.penalties.cuda()
        return self

    def cpu(self):
        self.state0 = self.state0.cpu()
        self.state1 = self.state1.cpu()
        self.rand_state = self.rand_state.cpu()
        self.penalties = self.penalties.cpu()
        return self

    def callback(self, tokens: list[int]):
        self.generated_tokens.extend(tokens)
        try:
            print("#" * 30, "\n", self.pipeline.tokenizer.decode(tokens))
        except:
            print("broken tokens")

    def finished(self):
        self.cpu()
        print("!" * 30, "\n", self.pipeline.tokenizer.decode(self.generated_tokens))

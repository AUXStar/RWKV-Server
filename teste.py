from server.scheduler import RWKV070ModelLoader,DynamicScheduler
import time

from server.task import Status

prompts = """请精心创作一个奇幻故事，字数需达到500字以上。故事必须包含以下核心元素：一条能够与人类交流的智慧巨龙、一片笼罩着古老诅咒的魔法森林、一座悬浮于天际的失落古城，以及一支怀揣秘密的寻宝冒险者小队。文中需设置至少三个出人意料的剧情转折点，并着重运用细腻的环境描写（如光影、气息）与深入的心理描写（如恐惧、贪婪、抉择）来增强叙事的沉浸感。
""".split(
    "\n"
)
prompts += prompts

pipeline = RWKV070ModelLoader(
    "/home/njzy/rwkv_agent/server/model/rwkv7-g1f-2.9b-20260420-ctx8192"
)

task_manager = DynamicScheduler(pipeline, max_batch_size=35, buffer_size=32)

t = time.time()
ttts = [
    task_manager.new_task(
        f"User: {prompt}\n\nAssistent: <thinking>好的，用户的语言是",
        max_tokens=2000,
        presence_penalty=2,
        repetition_penalty=0,
        penalty_decay=1.0,
        temperature=1,
        top_k=-1,
        top_p=0.5,
        collect_callback=str,
        finish_callback=str
    )
    for prompt in prompts[:5]
]
# while ttts[0].status != Status.READY:time.sleep(1)
print("任务已添加到队列，开始处理...", time.time() - t)
task_manager.run()


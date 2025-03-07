# Copyright 2024-2025 The Alibaba Wan Team Authors. All rights reserved.
import argparse
from datetime import datetime
import logging
import os
import sys
import warnings
import time
import json
import subprocess
import random
import signal
import psutil

warnings.filterwarnings('ignore')

import torch
import torch.distributed as dist
from PIL import Image
import gradio as gr

# Model
sys.path.insert(0, os.getcwd())
import wan
from wan.configs import WAN_CONFIGS, SIZE_CONFIGS, MAX_AREA_CONFIGS, SUPPORTED_SIZES
from wan.utils.prompt_extend import DashScopePromptExpander, QwenPromptExpander
from wan.utils.utils import cache_video, cache_image, str2bool

# Global Var
prompt_expander = None
running_processes = {}  # 保存正在运行的进程


def _init_logging(rank):
    # Set log format
    if rank == 0:
        logging.basicConfig(
            level=logging.INFO,
            format="[%(asctime)s] %(levelname)s: %(message)s",
            handlers=[logging.StreamHandler(stream=sys.stdout)])
    else:
        logging.basicConfig(level=logging.ERROR)


def initialize_prompt_expander(args, local_rank=0):
    """初始化提示词扩展器"""
    # 检查模型参数
    if not args.prompt_extend_model or args.prompt_extend_model.strip() == "":
        logging.error("未设置提示词扩展模型")
        return None

    rank = int(os.getenv("RANK", 0))
    
    if args.use_prompt_extend:
        logging.info("Initializing prompt expander...")
        try:
            if args.prompt_extend_method == "dashscope":
                if "DASH_API_KEY" not in os.environ:
                    logging.warning("DASH_API_KEY environment variable not set")
                    if rank == 0:
                        print("Warning: DASH_API_KEY not set. Prompt extension may fail.")
                prompt_expander = DashScopePromptExpander(
                    model_name=args.prompt_extend_model, is_vl="i2v" in args.task)
                logging.info("DashScope prompt expander initialized")
            elif args.prompt_extend_method == "local_qwen":
                prompt_expander = QwenPromptExpander(
                    model_name=args.prompt_extend_model,
                    is_vl="i2v" in args.task,
                    device=local_rank)
                logging.info("Qwen prompt expander initialized")
            else:
                raise NotImplementedError(f"Unsupported prompt_extend_method: {args.prompt_extend_method}")
            return prompt_expander
        except Exception as e:
            logging.error(f"Failed to initialize prompt expander: {e}")
            if rank == 0:
                print(f"Error initializing prompt expander: {e}")
            return None
    else:
        logging.info("Prompt extension disabled")
        return None


def extend_prompt(prompt, prompt_expander, target_lang="ch", seed=-1):
    """扩展提示词"""
    if prompt_expander is None:
        return prompt
    
    logging.info(f"Extending prompt: {prompt}")
    try:
        # Use prompt expander to extend the prompt
        prompt_output = prompt_expander(
            prompt,
            tar_lang=target_lang,
            seed=seed)
        
        if prompt_output.status == False:
            logging.warning(f"Extending prompt failed: {prompt_output.message}")
            return prompt
        else:
            extended_prompt = prompt_output.prompt
            logging.info(f"Extended prompt: {extended_prompt}")
            return extended_prompt
    except Exception as e:
        logging.error(f"Error during prompt extension: {e}")
        print(f"提示词扩展失败: {e}，将使用原始提示词")
        return prompt


def clean_up_process(process_id, work_dir=None):
    """清理进程及其子进程"""
    try:
        if process_id in running_processes:
            process = running_processes[process_id]
            # 获取所有子进程
            try:
                parent = psutil.Process(process.pid)
                children = parent.children(recursive=True)
                
                # 先终止子进程
                for child in children:
                    child.terminate()
                    
                # 给子进程一点时间终止
                gone, alive = psutil.wait_procs(children, timeout=3)
                
                # 如果仍有进程存活，强制终止
                for p in alive:
                    p.kill()
                    
                # 终止主进程
                if process.poll() is None:
                    process.terminate()
                    process.wait(timeout=5)
                    
                # 如果进程仍然存活，强制杀死
                if process.poll() is None:
                    process.kill()
                    
            except psutil.NoSuchProcess:
                pass  # 进程可能已经不存在
                
            # 从跟踪列表中移除
            del running_processes[process_id]
            
            logging.info(f"Cleaned up process {process_id}")
            
    except Exception as e:
        logging.error(f"Error cleaning up process {process_id}: {e}")


def find_unused_port(start_range=20000, end_range=30000, max_attempts=50):
    """查找未使用的端口"""
    import socket
    
    for _ in range(max_attempts):
        port = random.randint(start_range, end_range)
        
        # 尝试绑定端口检查是否可用
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            try:
                s.bind(('', port))
                return port
            except OSError:
                continue
    
    # 如果找不到可用端口，返回一个随机端口，可能会失败
    return random.randint(start_range, end_range)


def generate_action(prompt, resolution, steps, guidance, shift, seed, negative, 
                   gpu_count, task, use_prompt_extend, prompt_extend_method, prompt_extend_model, 
                   prompt_extend_target_lang, ckpt_dir, t5_fsdp, dit_fsdp, model_cache, t5_cpu, offload_model):
    """使用子进程和单独的脚本进行视频生成，根据参数选择配置"""
    # 准备参数
    if "x" in resolution:
        W, H = map(int, resolution.split("x"))
    else:
        W, H = map(int, resolution.split("*"))
    
    # 如果是随机种子，生成一个
    if seed < 0:
        seed = random.randint(0, 2147483647)
    
    # 解析GPU数量
    ulysses_size, ring_size = 1, 1
    if gpu_count > 1:
        # 简单地将GPU平分
        ulysses_size = gpu_count
        ring_size = 1
    
    # 创建唯一的工作目录
    timestamp = int(time.time())
    process_id = f"{timestamp}_{seed}"
    work_dir = f"generation_{process_id}"
    os.makedirs(work_dir, exist_ok=True)
    
    # 创建唯一的输出文件名
    output_file = os.path.join(work_dir, f"output_{timestamp}_{seed}.mp4")
    done_file = output_file + ".done"
    log_file = os.path.join(work_dir, "generation.log")
    
    # 初始化和使用提示词扩展器
    if use_prompt_extend:
        try:
            local_expander = initialize_prompt_expander(argparse.Namespace(
                use_prompt_extend=use_prompt_extend,
                prompt_extend_method=prompt_extend_method,
                prompt_extend_model=prompt_extend_model,
                task=task
            ))
            enhanced_prompt = extend_prompt(prompt, local_expander, prompt_extend_target_lang, seed)
        except Exception as e:
            logging.error(f"提示词扩展失败: {e}")
            print(f"提示词扩展失败: {e}，将使用原始提示词")
            enhanced_prompt = prompt
    else:
        enhanced_prompt = prompt
    
    # 记录开始时间
    start_time = time.time()
    
    # 调用生成脚本
    try:
        # 设置环境变量
        env = os.environ.copy()
        
        # 为每次运行选择未使用的端口，避免冲突
        master_port = find_unused_port()
        env["MASTER_PORT"] = str(master_port)
        
        # 创建启动命令
        cmd = [
            "torchrun",
            f"--nproc_per_node={gpu_count}",
            f"--master_port={master_port}",
            "generate_multi_gpu.py",  # 使用新的多GPU脚本
            f"--prompt={enhanced_prompt}",
            f"--width={W}",
            f"--height={H}",
            f"--output_file={output_file}",
            f"--task={task}",
            f"--ckpt_dir={ckpt_dir}",
            f"--sample_steps={steps}",
            f"--guide_scale={guidance}",
            f"--shift_scale={shift}",
            f"--seed={seed}",
            f"--negative_prompt={negative}",
            f"--ulysses_size={ulysses_size}",
            f"--ring_size={ring_size}",
            f"--log_file={log_file}"
        ]
        
        # 添加可选参数
        if t5_fsdp:
            cmd.append("--t5_fsdp")
        if dit_fsdp:
            cmd.append("--dit_fsdp")
        if t5_cpu:
            cmd.append("--t5_cpu")
        
        logging.info(f"Starting generation process with command: {' '.join(cmd)}")
        logging.info(f"Parameters: resolution={W}x{H}, steps={steps}, seed={seed}, port={master_port}")
        
        # 启动进程并将输出重定向到日志文件
        with open(log_file, 'w') as log:
            process = subprocess.Popen(
                cmd, 
                env=env,
                stdout=log,
                stderr=subprocess.STDOUT
            )
        
        # 添加到进程跟踪
        running_processes[process_id] = process
        
        # 返回初始状态信息
        status_msg = f"开始生成视频，参数：分辨率={W}x{H}, 步数={steps}, 种子={seed}, GPU数={gpu_count}, 端口={master_port}"
        
        # 等待生成完成，最多等待20分钟
        timeout = 1200  # 20分钟超时
        elapsed = 0
        interval = 2  # 检查间隔，秒
        
        while process.poll() is None and elapsed < timeout:
            if os.path.exists(done_file):
                break
            
            time.sleep(interval)
            elapsed += interval
            
            # 每10秒更新一次状态
            if elapsed % 10 == 0:
                with open(log_file, 'r') as f:
                    # 读取最后几行日志作为状态更新
                    lines = f.readlines()
                    recent_logs = lines[-5:] if len(lines) > 5 else lines
                    if recent_logs:
                        status_msg = f"生成中... 已用时间: {elapsed}秒, 最新日志: {recent_logs[-1].strip()}"
        
        # 检查是否超时
        if elapsed >= timeout:
            clean_up_process(process_id)
            logging.error("Generation process timed out after 20 minutes")
            return None, "生成超时，请尝试减小分辨率或降低步数"
        
        # 检查进程退出状态
        exit_code = process.wait()
        clean_up_process(process_id)
        end_time = time.time()
        total_time = end_time - start_time
        
        if exit_code != 0:
            logging.error(f"Generation process failed with exit code {exit_code}")
            # 读取日志文件以获取错误信息
            with open(log_file, 'r') as f:
                log_content = f.read()
                # 提取最后200个字符作为错误信息
                error_message = log_content[-200:] if len(log_content) > 200 else log_content
            return None, f"生成失败，进程退出码: {exit_code}，错误信息: {error_message}"
        
        # 检查是否成功完成
        if os.path.exists(output_file):
            logging.info(f"Generation completed successfully in {total_time:.2f} seconds")
            return output_file, f"生成完成! 用时: {total_time:.2f}秒, 种子: {seed}, 大小: {W}x{H}"
        else:
            logging.error("Output file not found after generation")
            return None, "生成失败，未找到输出文件，请检查日志"
            
    except Exception as e:
        if process_id in running_processes:
            clean_up_process(process_id)
        logging.exception(f"Error during generation: {str(e)}")
        return None, f"错误: {str(e)}"


def create_ui():
    # 获取可用GPU数量
    available_gpus = torch.cuda.device_count()
    # 修改: 确保下拉菜单的选项是整数而不是字符串
    gpu_options = [i for i in range(1, available_gpus + 1)]
    
    # 获取可用的task选项
    task_options = list(WAN_CONFIGS.keys())
    
    with gr.Blocks(css="footer {visibility: hidden}") as demo:
        gr.Markdown(
            f"""
            # WAN 2.1 文本到视频生成器 (优化版)
            
            使用高级大规模视频生成模型WAN 2.1创建高质量视频。本界面使用多GPU并行推理。
            """
        )
        
        # 配置部分 - 新增顶部配置区域
        with gr.Row():
            with gr.Column(scale=1):
                task = gr.Dropdown(
                    label="模型任务",
                    choices=task_options,
                    value="t2v-1.3B",
                    info="选择要使用的任务类型"
                )
                
                gpu_count = gr.Dropdown(
                    label="GPU数量",
                    choices=gpu_options,
                    value=4,  # 修改: 默认值是整数1而不是字符串"1"
                    info="使用的GPU数量 (多GPU并行处理)"
                )
            
            with gr.Column(scale=1):
                use_prompt_extend = gr.Checkbox(
                    label="启用提示词扩展",
                    value=False,  # 默认不启用提示词扩展
                    info="是否使用提示词扩展功能"
                )
                
                prompt_extend_method = gr.Radio(
                    label="提示词扩展方法",
                    choices=["dashscope", "local_qwen"],
                    value="local_qwen",  # 默认使用本地Qwen模型
                    visible=False,
                    info="选择提示词扩展的方法"
                )
            
            with gr.Column(scale=1):
                ckpt_dir = gr.Textbox(
                    label="模型路径",
                    value=get_default_ckpt_path("t2v-1.3B"),  # 使用函数获取默认路径
                    info="模型检查点目录路径"
                )
                
                model_cache = gr.Checkbox(
                    label="启用模型缓存",
                    value=True,
                    interactive=False,  # 禁止用户修改
                    info="模型将始终保留在内存中以加速后续生成"
                )
                
                advanced_settings = gr.Accordion("高级设置", open=False)
                with advanced_settings:
                    t5_fsdp = gr.Checkbox(
                        label="T5 FSDP",
                        value=True,
                        info="是否为T5使用FSDP (需多GPU)"
                    )
                    
                    dit_fsdp = gr.Checkbox(
                        label="DiT FSDP",
                        value=True,
                        info="是否为DiT使用FSDP (需多GPU)"
                    )
                    
                    t5_cpu = gr.Checkbox(
                        label="T5 CPU",
                        value=False,
                        info="是否将T5模型放置在CPU上"
                    )
                    
                    offload_model = gr.Checkbox(
                        label="Offload模型",
                        value=False,
                        visible=False,  # 隐藏此选项，因为它将被忽略
                        info="此选项已被禁用，模型将始终保留在内存中"
                    )
                    
                    prompt_extend_model = gr.Textbox(
                        label="提示词扩展模型",
                        value="qwen-plus",  # 设置默认值
                        visible=False,
                        info="提示词扩展使用的模型"
                    )
                    
                    prompt_extend_target_lang = gr.Radio(
                        label="提示词扩展目标语言",
                        choices=["ch", "en"],
                        value="ch",
                        visible=False,
                        info="提示词扩展的目标语言"
                    )
        
        gr.Markdown("---")
        
        # 主生成部分 - 保持原有布局
        with gr.Row():
            with gr.Column(scale=5):
                input_prompt = gr.Textbox(
                    label="提示词",
                    placeholder="描述您想要生成的视频内容...",
                    lines=5,
                    value="Film quality, professional quality, rich details. A beautiful mountain landscape with a peaceful lake at sunset, reflections on the water surface."
                )
                
                negative_prompt = gr.Textbox(
                    label="负面提示词",
                    placeholder="您不希望在视频中出现的内容...",
                    lines=2,
                    value="low quality, bad quality, blurry, distorted"
                )
                
                with gr.Row():
                    with gr.Column(scale=1):
                        resolution = gr.Dropdown(
                            label="分辨率",
                            choices=["480*832", "832*480", "624*624", "704*544", "544*704"],
                            value="480*832"
                        )
                        
                    with gr.Column(scale=1):
                        sample_steps = gr.Slider(
                            label="采样步数",
                            minimum=20,
                            maximum=100,
                            value=50,
                            step=1
                        )
                
                with gr.Row():
                    with gr.Column(scale=1):
                        guidance_scale = gr.Slider(
                            label="引导尺度",
                            minimum=1.0,
                            maximum=15.0,
                            value=5.0,
                            step=0.5
                        )
                        
                    with gr.Column(scale=1):
                        shift_scale = gr.Slider(
                            label="位移尺度",
                            minimum=1.0,
                            maximum=15.0,
                            value=5.0,
                            step=0.5
                        )
                
                seed = gr.Slider(
                    label="随机种子 (-1表示随机)",
                    minimum=-1,
                    maximum=2147483647,
                    value=-1,
                    step=1
                )
                
                with gr.Row():
                    extend_btn = gr.Button("扩展提示词", variant="secondary")
                    generate_btn = gr.Button("生成视频", variant="primary")
                
                status = gr.Textbox(
                    label="状态信息",
                    value="就绪",
                    interactive=False
                )
            
            with gr.Column(scale=5):
                output_video = gr.Video(label="生成的视频", width=400, height=700)
        
        # 设置条件显示 - 提示词扩展相关选项
        def update_prompt_extend_visibility(use_extend):
            return {
                prompt_extend_method: gr.update(visible=use_extend),
                prompt_extend_model: gr.update(visible=use_extend),
                prompt_extend_target_lang: gr.update(visible=use_extend)
            }
        
        use_prompt_extend.change(
            fn=update_prompt_extend_visibility,
            inputs=[use_prompt_extend],
            outputs=[prompt_extend_method, prompt_extend_model, prompt_extend_target_lang]
        )
        
        # 任务变化时更新模型路径
        def update_model_path(selected_task):
            return get_default_ckpt_path(selected_task)
        
        task.change(
            fn=update_model_path,
            inputs=[task],
            outputs=[ckpt_dir]
        )
        
        # 扩展按钮动作
        def extend_prompt_action(prompt, use_extend, extend_method, extend_model, extend_lang):
            if not use_extend:
                return prompt, "提示词扩展未启用"
            
            if not extend_model or extend_model.strip() == "":
                if extend_method == "local_qwen":
                    extend_model = "qwen-plus"  # 为local_qwen设置默认模型
                else:
                    return prompt, "错误: 未设置提示词扩展模型，请在高级设置中设置"
            
            try:
                # 初始化提示词扩展器
                local_expander = initialize_prompt_expander(argparse.Namespace(
                    use_prompt_extend=use_extend,
                    prompt_extend_method=extend_method,
                    prompt_extend_model=extend_model,
                    task="t2v-1.3B"  # 默认值，仅用于扩展
                ))
                
                # 扩展提示词
                if local_expander:
                    extended = extend_prompt(prompt, local_expander, extend_lang, -1)
                    return extended, "提示词扩展完成"
                else:
                    return prompt, "提示词扩展器初始化失败"
            except Exception as e:
                return prompt, f"提示词扩展失败: {str(e)}"
        
        extend_btn.click(
            fn=extend_prompt_action,
            inputs=[input_prompt, use_prompt_extend, prompt_extend_method, 
                   prompt_extend_model, prompt_extend_target_lang],
            outputs=[input_prompt, status]
        )
        
        # 生成按钮动作
        generate_btn.click(
            fn=generate_action,
            inputs=[
                input_prompt, resolution, sample_steps, guidance_scale, 
                shift_scale, seed, negative_prompt, gpu_count, task,
                use_prompt_extend, prompt_extend_method, prompt_extend_model,
                prompt_extend_target_lang, ckpt_dir, t5_fsdp, dit_fsdp,
                model_cache, t5_cpu, offload_model
            ],
            outputs=[output_video, status]
        )
    
    # 关闭时清理所有运行中进程
    def cleanup_handler():
        for pid in list(running_processes.keys()):
            clean_up_process(pid)
    
    # 注册清理回调
    import atexit
    atexit.register(cleanup_handler)
    
    return demo



def main():
    """主函数"""
    # 确保存在工作目录
    os.makedirs("generation_0", exist_ok=True)
    
    # 设置内存优化
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    
    # 初始化日志
    _init_logging(0)
    
    # 记录启动信息
    logging.info(f"启动WAN Gradio UI优化版界面")
    logging.info(f"可用GPU数量: {torch.cuda.device_count()}")
    
    # 创建并启动UI
    ui = create_ui()
    ui.launch(server_name="0.0.0.0", share=False, server_port=7860)


def get_default_ckpt_path(task):
    """根据任务类型返回默认的模型路径"""
    task_to_path = {
        "t2v-1.3B": "./Wan2.1-T2V-1.3B",
        "t2v-14B": "./Wan2.1-T2V-14B",
        "i2v-14B-480P": "./Wan2.1-I2V-14B-480P",
        "i2v-14B-720P": "./Wan2.1-I2V-14B-720P"
    }
    return task_to_path.get(task, "./Wan2.1-T2V-1.3B")  # 默认返回1.3B模型路径


if __name__ == "__main__":
    main()
#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright 2024-2025 The Alibaba Wan Team Authors. All rights reserved.

"""
多GPU并行生成脚本
此脚本用于在多GPU环境下并行生成视频，并确保模型保留在内存中以加速后续生成
"""

import argparse
import json
import logging
import os
import sys
import time
import torch
import torch.distributed as dist
from datetime import datetime

# 添加项目根目录到路径
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

# 导入Wan模型
import wan
from wan.configs import WAN_CONFIGS, SIZE_CONFIGS
from wan.utils.utils import cache_video, cache_image

# 全局模型缓存
MODEL_CACHE = {}

def setup_logging(log_file=None):
    """设置日志记录"""
    handlers = [logging.StreamHandler()]
    if log_file:
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        handlers.append(logging.FileHandler(log_file))
    
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s][%(levelname)s] %(message)s",
        handlers=handlers
    )

def get_or_create_model(task, ckpt_dir, device_id, rank, world_size, use_usp=False, t5_fsdp=False, dit_fsdp=False, t5_cpu=False):
    """获取缓存的模型或创建新模型"""
    global MODEL_CACHE
    
    # 创建缓存键
    cache_key = f"{task}_{rank}_{world_size}"
    
    # 如果模型已缓存，直接返回
    if cache_key in MODEL_CACHE:
        logging.info(f"Using cached model for task {task} (rank {rank}/{world_size})")
        return MODEL_CACHE[cache_key]
    
    # 否则，初始化新模型
    logging.info(f"Initializing new model for task {task} (rank {rank}/{world_size})")
    
    # 获取模型配置
    cfg = WAN_CONFIGS[task]
    
    # 创建模型
    if 't2v' in task or 't2i' in task:
        model = wan.WanT2V(
            config=cfg,
            checkpoint_dir=ckpt_dir,
            device_id=device_id,
            rank=rank,
            t5_fsdp=t5_fsdp,
            dit_fsdp=dit_fsdp,
            use_usp=use_usp,
            t5_cpu=t5_cpu,
        )
        logging.info(f"WanT2V model created successfully on rank {rank}")
    else:
        model = wan.WanI2V(
            config=cfg,
            checkpoint_dir=ckpt_dir,
            device_id=device_id,
            rank=rank,
            t5_fsdp=t5_fsdp,
            dit_fsdp=dit_fsdp,
            use_usp=use_usp,
            t5_cpu=t5_cpu,
        )
        logging.info(f"WanI2V model created successfully on rank {rank}")
    
    # 缓存模型
    MODEL_CACHE[cache_key] = model
    logging.info(f"Model cached with key {cache_key}")
    
    return model

def generate_video(args):
    """生成视频的主函数"""
    # 获取分布式环境变量
    rank = int(os.getenv("RANK", "0"))
    world_size = int(os.getenv("WORLD_SIZE", "1"))
    local_rank = int(os.getenv("LOCAL_RANK", "0"))
    
    # 设置日志
    if rank == 0:
        log_file = args.log_file if args.log_file else None
        setup_logging(log_file)
    
    # 记录开始时间
    start_time = time.time()
    
    # 初始化分布式环境
    if world_size > 1:
        logging.info(f"Initializing distributed environment: rank={rank}, world_size={world_size}")
        torch.cuda.set_device(local_rank)
        dist.init_process_group(
            backend="nccl",
            init_method="env://",
            rank=rank,
            world_size=world_size
        )
        
        # 设置上下文并行
        if args.ulysses_size > 1 or args.ring_size > 1:
            assert args.ulysses_size * args.ring_size == world_size, \
                f"The number of ulysses_size ({args.ulysses_size}) and ring_size ({args.ring_size}) " \
                f"should be equal to the world size ({world_size})."
            
            logging.info(f"Initializing context parallel: ulysses_size={args.ulysses_size}, ring_size={args.ring_size}")
            
            from xfuser.core.distributed import (initialize_model_parallel, init_distributed_environment)
            init_distributed_environment(rank=dist.get_rank(), world_size=dist.get_world_size())
            initialize_model_parallel(
                sequence_parallel_degree=dist.get_world_size(),
                ring_degree=args.ring_size,
                ulysses_degree=args.ulysses_size,
            )
    
    # 清理GPU缓存
    torch.cuda.empty_cache()
    
    # 获取或创建模型
    use_usp = (args.ulysses_size > 1 or args.ring_size > 1)
    model = get_or_create_model(
        args.task, 
        args.ckpt_dir, 
        local_rank, 
        rank, 
        world_size, 
        use_usp=use_usp,
        t5_fsdp=args.t5_fsdp,
        dit_fsdp=args.dit_fsdp,
        t5_cpu=args.t5_cpu
    )
    
    # 记录模型初始化时间
    init_time = time.time()
    logging.info(f"Model initialization completed in {init_time - start_time:.2f} seconds")
    
    # 生成视频
    logging.info("Starting generation...")
    try:
        if 't2v' in args.task or 't2i' in args.task:
            # 文本到视频/图像生成
            video = model.generate(
                args.prompt,
                size=(args.width, args.height),
                frame_num=args.frame_num,
                shift=args.shift_scale,
                sample_solver=args.sample_solver,
                sampling_steps=args.sample_steps,
                guide_scale=args.guide_scale,
                n_prompt=args.negative_prompt,
                seed=args.seed,
                offload_model=False  # 始终设置为False以保持模型在内存中
            )
            logging.info("Generation completed successfully")
        else:
            # 图像到视频生成 (需要提供输入图像)
            if args.image_path:
                from PIL import Image
                img = Image.open(args.image_path).convert("RGB")
                
                video = model.generate(
                    args.prompt,
                    img,
                    max_area=args.max_area if hasattr(args, 'max_area') else 1024*576,
                    frame_num=args.frame_num,
                    shift=args.shift_scale,
                    sample_solver=args.sample_solver,
                    sampling_steps=args.sample_steps,
                    guide_scale=args.guide_scale,
                    seed=args.seed,
                    offload_model=False  # 始终设置为False以保持模型在内存中
                )
                logging.info("Generation completed successfully")
            else:
                raise ValueError("Image path is required for I2V task")
        
        # 只有rank 0保存输出
        if rank == 0:
            output_file = args.output_file
            output_dir = os.path.dirname(output_file)
            os.makedirs(output_dir, exist_ok=True)
            
            if 't2i' in args.task:
                # 保存图像
                logging.info(f"Saving generated image to {output_file}")
                cache_image(
                    tensor=video.squeeze(1)[None],
                    save_file=output_file,
                    nrow=1,
                    normalize=True,
                    value_range=(-1, 1)
                )
            else:
                # 保存视频
                logging.info(f"Saving generated video to {output_file}")
                # 获取配置
                cfg = WAN_CONFIGS[args.task]
                cache_video(
                    tensor=video[None],
                    save_file=output_file,
                    fps=cfg.sample_fps,
                    nrow=1,
                    normalize=True,
                    value_range=(-1, 1)
                )
            
            # 写入完成标记
            with open(args.output_file + '.done', 'w') as f:
                end_time = time.time()
                generation_time = end_time - init_time
                total_time = end_time - start_time
                f.write(f"done\nGeneration time: {generation_time:.2f}s\nTotal time: {total_time:.2f}s")
                
            logging.info(f"Generation process completed in {generation_time:.2f} seconds")
            logging.info(f"Total process time: {total_time:.2f} seconds")
    
    except Exception as e:
        import traceback
        logging.error(f"Error during generation: {e}")
        logging.error(traceback.format_exc())
        if rank == 0:
            # 写入错误信息
            error_file = os.path.join(os.path.dirname(args.output_file), "error.txt")
            with open(error_file, 'w') as f:
                f.write(f"Error during generation: {e}\n")
                f.write(traceback.format_exc())
        return False
    
    # 清理临时缓存，但保留模型
    torch.cuda.empty_cache()
    logging.info("Model kept in memory for future use")
    
    return True

def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="Multi-GPU Parallel Video Generation")
    
    # 基本参数
    parser.add_argument("--prompt", type=str, required=True, help="Generation prompt")
    parser.add_argument("--width", type=int, required=True, help="Output width")
    parser.add_argument("--height", type=int, required=True, help="Output height")
    parser.add_argument("--output_file", type=str, required=True, help="Output file path")
    parser.add_argument("--task", type=str, required=True, help="Task type (e.g., t2v-1.3B)")
    parser.add_argument("--ckpt_dir", type=str, required=True, help="Model checkpoint directory")
    
    # 生成参数
    parser.add_argument("--sample_steps", type=int, default=25, help="Sampling steps")
    parser.add_argument("--guide_scale", type=float, default=7.5, help="Guidance scale")
    parser.add_argument("--shift_scale", type=float, default=0.0, help="Shift scale")
    parser.add_argument("--seed", type=int, default=-1, help="Random seed (-1 for random)")
    parser.add_argument("--negative_prompt", type=str, default="", help="Negative prompt")
    parser.add_argument("--frame_num", type=int, default=81, help="Number of frames")
    parser.add_argument("--sample_solver", type=str, default="unipc", help="Sample solver")
    
    # 分布式参数
    parser.add_argument("--ulysses_size", type=int, default=1, help="Ulysses size for context parallel")
    parser.add_argument("--ring_size", type=int, default=1, help="Ring size for context parallel")
    
    # 模型参数
    parser.add_argument("--t5_fsdp", action="store_true", help="Use FSDP for T5")
    parser.add_argument("--dit_fsdp", action="store_true", help="Use FSDP for DiT")
    parser.add_argument("--t5_cpu", action="store_true", help="Place T5 on CPU")
    
    # I2V特定参数
    parser.add_argument("--image_path", type=str, default=None, help="Input image path for I2V")
    parser.add_argument("--max_area", type=int, default=1024*576, help="Max area for I2V")
    
    # 其他参数
    parser.add_argument("--log_file", type=str, default=None, help="Log file path")
    
    return parser.parse_args()

def main():
    """主函数"""
    args = parse_args()
    success = generate_video(args)
    return 0 if success else 1

if __name__ == "__main__":
    sys.exit(main()) 
import argparse
import torch
import torch.multiprocessing as mp

from .sft.ft import GlmLora, GlmSft


def main():
    world_size = torch.cuda.device_count()
    parser = argparse.ArgumentParser(description="Humanable Chat General Language Model Finetune")
    subparsers = parser.add_subparsers(required=True, help="sub command")

    parser.add_argument(
        "--strategy", type=str, metavar="STRATEGY", default="fsdp_zero3", 
        help="training strategy (default: fsdp_zero3). should be one of: fsdp_zero3, fsdp_zero2, mpdp(ddp), mpds(8bit), msds(single gpu)"
    )
    parser.add_argument(
        "--data_path", type=str, default=None, metavar="FILE", required=True,
        help="[dataset] training data path, should be a text file where each line contains a json with two keys: `prompt` and `completion` (default: None)"
    )
    parser.add_argument(
        "--max_seq_len", type=int, default=512, metavar="N",
        help="[dataset] max sequence length (default: 512)"
    )
    parser.add_argument(
        "--batch_size", type=int, default=world_size*2, metavar="N",
        help="[dataset] input batch size for training (default: world_size*2)"
    )
    parser.add_argument(
        "--model", type=str, default=None, metavar="ID/PATH", required=True,
        help="[model] LLM model id or model path (default: None)"
    ) 
    parser.add_argument(
        "--pretrained_ckpt", type=str, default=None, metavar="FILE",
        help="[model] pretrained model file path, if provided, the pretrained model ckpt will be loaded (default: None)"
    )
    parser.add_argument(
        "--device", type=str, default=None, metavar="DEVICE",
        help="[model] device id to run on that specified device, suit for `msds` mode (default: None)"
    )
    parser.add_argument(
        "--lr", type=float, default=1e-4, metavar="LR",
        help="[training] learning rate (default: .0001)"
    )
    parser.add_argument(
        "--num_epochs", type=int, default=3, metavar="N",
        help="[training] number of epochs to train (default: 3)"
    )
    parser.add_argument(
        "--warmup_steps", type=int, default=None, metavar="N",
        help="[training] warmup epoch of steps (default: None <=> use 1/3 epoch to warmup)"
    )
    parser.add_argument(
        "--accumulate_steps", type=int, default=None, metavar="N",
        help="[training] accumulate steps (default: None <=> 1)"
    )
    parser.add_argument(
        "--out_dir", type=str, default="./output/", metavar="PATH",
        help="[training] model output path (default: `./output/`)"
    )
    
    parser_sft = subparsers.add_parser("sft", help="sft fine-tuning")
    parser_sft.set_defaults(task_type="sft")
    
    parser_lora = subparsers.add_parser("lora", help="lora fine-tuning")
    parser_lora.set_defaults(task_type="lora")
    parser_lora.add_argument(
        "--lora_r", type=int, default=8, metavar="N",
        help="[model] lora r (default: 8)"
    )

    args = parser.parse_args()
    print(f"Training with args: {args}")
    assert args.task_type in ["sft", "lora"], "must provide a task_type like `sft` or `lora`"

    param_list = ["batch_size", "lr", "num_epochs", "warmup_steps", "accumulate_steps", "out_dir", "task_type"]
    params = {key: getattr(args, key) for key in param_list}

    if args.strategy in ["fsdp_zero3", "fsdp_zero2", "mpdp"]:
        if args.task_type == "sft":
            glm = GlmSft(args.model)
        elif args.task_type == "lora":
            glm = GlmLora(args.model, lora_r=args.lora_r)
        glm.load_data(args.data_path, max_seq_len=args.max_seq_len)
        params["strategy"] = args.strategy
        params["pretrained_ckpt"] = args.pretrained_ckpt
        params["task_type"] = args.task_type
        mp.spawn(glm.fsdp_tune, args=(world_size, params), nprocs=world_size, join=True)
    elif args.strategy == "mpds":
        if args.task_type == "sft":
            glm = GlmSft(args.model, load_in_8bit=True)
        elif args.task_type == "lora":
            glm = GlmLora(args.model, lora_r=args.lora_r, load_in_8bit=True)
        (glm
            .load_data(args.data_path, max_seq_len=args.max_seq_len)
            .load_pretrained(args.pretrained_ckpt)
            .tune(**params)
        )
    elif args.strategy == "msds":
        device = args.device or "cuda:0"
        if args.task_type == "sft":
            glm = GlmSft(args.model, device=device)
        elif args.task_type == "lora":
            glm = GlmLora(args.model, lora_r=args.lora_r, device=device)
        (glm
            .load_data(args.data_path, max_seq_len=args.max_seq_len)
            .load_pretrained(args.pretrained_ckpt)
            .tune(**params)
        )
    else:
        msg = f"Unsupported strategy: {args.strategy}. Run `hcgf_tune -h` to get more help"
        raise ValueError(msg)


if __name__ == "__main__":
    main()
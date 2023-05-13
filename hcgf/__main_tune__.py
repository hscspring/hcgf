import argparse
import torch
import torch.multiprocessing as mp

from .sft.ft import GlmLora


def main():
    world_size = torch.cuda.device_count()
    parser = argparse.ArgumentParser(description="Humanable Chat GXX Finetune")
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
        "--batch_size", type=int, default=world_size * 2, metavar="N",
        help="[dataset] input batch size for training (default: gpu_num * 2)"
    )
    parser.add_argument(
        "--model", type=str, default=None, metavar="ID/PATH", required=True,
        help="[model] LLM model id or model path (default: None)"
    ) 
    parser.add_argument(
        "--lora_r", type=int, default=8, metavar="N",
        help="[model] lora r (default: 8)"
    )
    parser.add_argument(
        "--device", type=str, default=None, metavar="DEVICE",
        help="[model] device id to run on that specified device, suit for `msds` mode (default: None)"
    )
    parser.add_argument(
        "--lr", type=float, default=2e-4, metavar="LR",
        help="[training] learning rate (default: .0002)"
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
    args = parser.parse_args()
    print(f"Training with args: {args}")

    param_list = ["batch_size", "lr", "num_epochs", "warmup_steps", "accumulate_steps", "out_dir"]
    params = {key: getattr(args, key) for key in param_list}

    if args.strategy in ["fsdp_zero3", "fsdp_zero2", "mpdp"]:
        glm = GlmLora(args.model, lora_r=args.lora_r)
        glm.load_data(args.data_path, max_seq_len=args.max_seq_len)
        params["strategy"] = args.strategy
        mp.spawn(glm.fsdp_tune, args=(world_size, params), nprocs=world_size, join=True)
    elif args.strategy == "mpds":
        glm = GlmLora(args.model, lora_r=args.lora_r, load_in_8bit=True)
        glm.load_data(args.data_path, max_seq_len=args.max_seq_len).tune(**params)
    elif args.strategy == "msds":
        device = args.device or "cuda:0"
        glm = GlmLora(args.model, lora_r=args.lora_r, device=device)
        glm.load_data(args.data_path, max_seq_len=args.max_seq_len).tune(**params)
    else:
        msg = f"Unsupported strategy: {args.strategy}. Run `hcgf_tune -h` to get more help"
        raise ValueError(msg)


if __name__ == "__main__":
    main()
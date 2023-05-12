import argparse

from hcgf.sft.lora_ft import GlmLora


def main():
    parser = argparse.ArgumentParser(description="Humanable Chat GXX Finetune")
    parser.add_argument(
        "--strategy", type=str, metavar="STRATEGY", default="mpds", 
        help="running strategy (default: mpds(8bit)). should be one of: mpds(8bit), msds(single gpu)"
    )
    parser.add_argument(
        "--model", type=str, default=None, metavar="ID/PATH", required=True,
        help="[model] LLM model id or model path (default: None)"
    ) 
    parser.add_argument(
        "--lora_r", type=int, default=8, metavar="N",
        help="lora r (default: 8)"
    )
    parser.add_argument(
        "--device", type=str, default="cuda:0", metavar="DEVICE",
        help="[model] device id to run on that specified device, suit for `msds` mode (default: cuda:0)"
    )
    parser.add_argument(
        "--ckpt_file", type=str, default=None, metavar="FILE", required=True,
        help="lora ckpt file, should be a file ends with `pt`, if use default, will only load the raw model (default: None)"
    )
    # generating params
    parser.add_argument(
        "--max_new_tokens", type=int, default=512, metavar="N",
        help="max new generating tokens (default: 512)"
    )
    parser.add_argument(
        "--do_sample", action="store_true", default=True, metavar="B",
        help="whether to use sampling (default: True)"
    )
    parser.add_argument(
        "--num_beams", type=int, default=1, metavar="N",
        help="number of beams for beam search (default: 1)"
    )
    parser.add_argument(
        "--temperature", type=float, default=0.2, metavar="F",
        help="0-1, larger ==> random (default: 0.2)"
    )
    parser.add_argument(
        "--top_p", type=float, default=0.7, metavar="F",
        help="0-1, larger ==> random (default: 0.7)"
    )
    parser.add_argument(
        "--repetition_penalty", type=float, default=1.02, metavar="F",
        help="larger ==> less repetition (default: 1.0)"
    )
    args = parser.parse_args()
    print(f"Running with args: {args}")

    param_list = ["mwx_new_tokens", "do_sample", "num_beams", "temperature", "top_p", "repetition_penalty"]
    params = {key: getattr(args, key) for key in param_list}

    if args.strategy == "mpds":
        glm = GlmLora(args.model, lora_r=args.lora_r, load_in_8bit=True)
        glm.load_pretrained(args.ckpt_file).eval()
    elif args.strategy == "msds":
        glm = GlmLora(args.model, lora_r=args.lora_r, device=args.device)
        glm.load_pretrained(args.ckpt_file).eval()
    else:
        msg = f"Unsupported strategy: {args.mode}. Run `hcgf_run -h` to get more help"
        raise ValueError(msg)
    
    while True:
        question = input(">User: ")
        response, history = glm.chat(question)
        print(f">Bot: {response}")


if __name__ == "__main__":
    main()
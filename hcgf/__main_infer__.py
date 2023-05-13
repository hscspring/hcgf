import argparse
import gradio as gr

from .sft.ft import GlmLora


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
        "--ckpt_file", type=str, default=None, metavar="FILE",
        help="lora ckpt file, should be a file ends with `pt`, if use default, will only load the raw model (default: None)"
    )
    # generating params
    parser.add_argument(
        "--max_new_tokens", type=int, default=512, metavar="N",
        help="max new generating tokens (default: 512)"
    )
    parser.add_argument(
        "--do_sample", action="store_true", default=True,
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

    param_list = ["max_new_tokens", "do_sample", "num_beams", "temperature", "top_p", "repetition_penalty"]
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
    
    
    ##### From https://github.com/THUDM/ChatGLM-6B #####

    def reset_user_input():
        return gr.update(value="")

    def reset_state():
        return [], []

    def chat(
        query: str, 
        chatbot: list, 
        max_new_tokens: int, 
        top_p: float, 
        temperature: float, 
        history: list
    ):
        chatbot.append((query, ""))
        for response, history in glm.stream_chat(
            query, history, max_new_tokens, top_p=top_p, temperature=temperature):
            chatbot[-1] = (query, response)       
            yield chatbot, history

    with gr.Blocks() as demo:
        gr.HTML("""<h1 align="center">Humanable Finetuned ChatGXX</h1>""")
        chatbot = gr.Chatbot()
        with gr.Row():
            with gr.Column(scale=4):
                with gr.Column(scale=1):
                    user_input = gr.Textbox(show_label=False, placeholder="Input", lines=4).style(container=False)
                with gr.Column(min_width=32, scale=1):
                    submit_btn = gr.Button("Submit", variant="primary")
            with gr.Column(scale=1):
                empty_btn = gr.Button("Clear History")
                max_new_tokens = gr.Slider(0, 1024, value=512, step=1.0, label="Maximum length", interactive=True)
                top_p = gr.Slider(0, 1, value=0.7, step=0.1, label="Top P", interactive=True)
                temperature = gr.Slider(0, 1, value=0.1, step=0.1, label="Temperature", interactive=True)

        history = gr.State([])
        submit_btn.click(
            chat, 
            inputs=[user_input, chatbot, max_new_tokens, top_p, temperature, history], 
            outputs=[chatbot, history], 
            show_progress=True
        )
        submit_btn.click(reset_user_input, [], [user_input])
        empty_btn.click(reset_state, outputs=[chatbot, history], show_progress=True)

    demo.queue().launch(server_name="0.0.0.0", share=False)


if __name__ == "__main__":
    main()
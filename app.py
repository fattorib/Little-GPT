import gradio as gr
import torch
from src.models.GPT2 import model_getter
from src.utils.generation_utils import TextGenerator
import argparse


def parse():
    parser = argparse.ArgumentParser(description="Gradio Inference App")
    parser.add_argument("--model-size", default="medium", type=str)
    parser.add_argument("--share", default=False, action="store_true")
    parser.add_argument("--bit-quantize", default=False, action="store_true")
    args = parser.parse_args()
    return args


DEVICE = "cpu"
if torch.cuda.is_available():
    DEVICE = "cuda"

BNB_FLAG = False
try:
    import bitsandbytes as bnb

    BNB_FLAG = True
except Exception as e:
    pass

generator = TextGenerator(seq_len=512, tokenizer=None)


def model_creator(size: str) -> torch.nn.Module:

    save_paths = {
        "base*": "checkpoints/127_weights.pth.tar",
        "medium*": "checkpoints/303_weights.pth.tar",
        "XL*": "checkpoints/1B8bit_weights.pth.tar"
        if BNB_FLAG
        else "checkpoints/1B_weights_noBNB.pth.tar",
        "medium": "checkpoints/354_weights.pth.tar",
    }

    if "*" in size:

        model = model_getter(
            size,
            vocab_size=50257,
            num_ctx=512,
            model_checkpoint=save_paths[size],
            **{
                "fused_residuals": True,
                "num_head": 8,
                "use_alibi": True,
                "quantized_state": True if "XL" in size else False,
            },
        )

    elif size == "medium":
        model = model_getter(
            "medium",
            vocab_size=50257,
            num_ctx=1024,
            model_checkpoint=save_paths[size],
            **{"fused_residuals": False, "use_alibi": False},
        )

    model.to(DEVICE)
    model.eval()

    return model


def generate_text(
    prompt,
    steps,
    temperature,
    top_k,
    top_p,
    tau,
    repetition_penalty,
    sampling_choice,
):
    if sampling_choice == "Top-k":
        sampling_method = "topk"

    elif sampling_choice == "Nucleus":
        sampling_method = "nucleus"

    elif sampling_choice == "Typical":
        sampling_method = "typical"

    elif sampling_choice == "Greedy":
        sampling_method = "greedy"

    generated_text, new_gen, logprobs = generator.generate_text_from_prompt(
        model=model,
        prompt=prompt,
        steps=int(steps),
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
        tau=tau,
        repetition_penalty=repetition_penalty,
        sampling_method=sampling_method,
        device=DEVICE,
    )

    original_gen_length = len(generated_text) - len(new_gen)

    return [
        (generated_text[:original_gen_length], None),
        (generated_text[original_gen_length:], "Generated Text"),
    ]


if __name__ == "__main__":
    args = parse()

    assert args.model_size in ["base*", "medium*", "medium", "XL*"]

    model = model_creator(args.model_size)

    from src.utils.gradio_utils import DESCRIPTION_MAP

    description = DESCRIPTION_MAP[args.model_size]

    iface = gr.Interface(
        fn=generate_text,
        inputs=[
            gr.inputs.Textbox(lines=10, label="Enter your text here"),
            gr.inputs.Slider(
                0, 1000, default=100, label="Number of tokens to generate"
            ),
            gr.inputs.Slider(0, 2, default=0.70, label="Temperature"),
            gr.inputs.Slider(
                0,
                50,
                default=40,
                label="k (Top-k Sampling)",
            ),
            gr.inputs.Slider(
                0,
                1,
                default=0.96,
                label="p (Nucleus Sampling)",
            ),
            gr.inputs.Slider(
                0,
                1,
                default=0.2,
                label="Tau (Typical Sampling)",
            ),
            gr.inputs.Slider(
                0.0,
                1.3,
                default=1.2,
                label="Repetition Penalty",
            ),
            gr.inputs.Radio(
                choices=["Top-k", "Nucleus", "Typical", "Greedy"],
                label="Sampling Method",
                default="Nucleus",
            ),
        ],
        outputs=gr.HighlightedText(
            label="Generated Text",
            combine_adjacent=True,
            color_map=["Generated Text", "blue"],
        ),
        live=False,
        title="GPT-* 🤖"
        if args.model_size in ["base*", "medium*", "XL*"]
        else "GPT-354M 🤖",
        description=description,
        article="For more details check out the model repo [here](https://github.com/fattorib/Faster-GPT)",
        allow_flagging="never",
    )
    iface.launch(share=args.share)

import gradio as gr
from matplotlib.pyplot import show
import torch
from src.models.GPT2 import model_getter
from src.utils.generation_utils import TextGenerator
import logging
import argparse


def parse():
    parser = argparse.ArgumentParser(description="Gradio Inference App")
    parser.add_argument("--model-size", default="medium", type=str)
    parser.add_argument("--share", default=False, action="store_true")
    args = parser.parse_args()
    return args


DEVICE = "cpu"
if torch.cuda.is_available():
    DEVICE = "cuda"

generator = TextGenerator(
    seq_len=1024,
)


def model_creator(size: str) -> torch.nn.Module:

    if size == "base*":
        model = model_getter(
            "base*",
            vocab_size=50257,
            num_ctx=512,
            **{"fused_residuals": True, "num_head": 8, "use_alibi": True},
        )

        state_dict = torch.load(
            rf"checkpoints/127_weights.pth.tar",
            map_location="cpu",
        )

        model.load_state_dict(state_dict)

        # Setting model context:
        PRIME_CTX = 1024
        # prime with ctx of 1024:
        with torch.no_grad():
            data_batch = torch.randint(low=0, high=50257, size=(1, PRIME_CTX))
            model(data_batch)
            logging.info(f"Evaluating ALiBi model with context: {PRIME_CTX}")

    if size == "medium*":
        model = model_getter(
            "medium*",
            vocab_size=50257,
            num_ctx=512,
            **{"fused_residuals": True, "num_head": 8, "use_alibi": True},
        )

        state_dict = torch.load(
            rf"checkpoints/303_weights.pth.tar",
            map_location="cpu",
        )

        model.load_state_dict(state_dict)

        # Setting model context:
        PRIME_CTX = 1024
        # prime with ctx of 1024:
        with torch.no_grad():
            data_batch = torch.randint(low=0, high=50257, size=(1, PRIME_CTX))
            model(data_batch)
            logging.info(f"Evaluating ALiBi model with context: {PRIME_CTX}")

    elif size == "medium":
        model = model_getter(
            "medium",
            vocab_size=50257,
            num_ctx=1024,
            **{"fused_residuals": False, "use_alibi": False},
        )

        state_dict = torch.load(
            rf"checkpoints/354_weights.pth.tar",
            map_location="cpu",
        )

        model.load_state_dict(state_dict)

    model.to(DEVICE)
    model.eval()

    return model


def generate_text(
    prompt, steps, temperature, top_k, top_p, tau, sampling_choice
):

    if sampling_choice == "Top-k":
        top_p = 0.0
        typical_sampling = False

    elif sampling_choice == "Nucleus":
        typical_sampling = False

    elif sampling_choice == "Typical":
        typical_sampling = True
    
    elif sampling_choice == "Greedy":
        top_p = 0.0
        typical_sampling = False
        top_k = 1


    generated_text, new_gen, _ = generator.generate_text_from_prompt(
        model=model,
        prompt=prompt.strip(),
        steps=int(steps),
        temperature=temperature,
        sample=True,
        top_k=top_k,
        top_p=top_p,
        typical_sampling=typical_sampling,
        tau=tau,
        device=DEVICE,
    )

    original_gen_length = len(generated_text) - len(new_gen)

    return [
        (generated_text[:original_gen_length], None),
        (generated_text[original_gen_length:], "Generated Text"),
    ]


if __name__ == "__main__":
    args = parse()

    assert args.model_size in ["base*", "medium*", "medium"]

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
            gr.inputs.Slider(0, 2, default=0.85, label="Temperature"),
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
        if args.model_size in ["base*", "medium*"]
        else "GPT-354M 🤖",
        description=description,
        article="For more details check out the model repo [here](https://github.com/fattorib/Faster-GPT)",
        allow_flagging="never",

    )
    iface.launch(share=args.share)

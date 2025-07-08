import time
from datetime import datetime
from dataclasses import asdict
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import gradio as gr
import random
import re
import torch.nn.functional as F
import numpy as np

def add_gumbel_noise(logits, temperature):
    """Adds Gumbel noise to the logits."""
    if temperature == 0:
        return logits.exp()
    # Using a small epsilon to prevent log(0) -> -inf
    noise = torch.rand_like(logits)
    gumbel_noise = -torch.log(-torch.log(noise + 1e-9) + 1e-9)
    return (logits + gumbel_noise) * temperature


def get_num_transfer_tokens(mask_index, steps):
    """Calculates the number of tokens to transfer at each step."""
    mask_num = mask_index.sum(dim=1, keepdim=True)
    base = mask_num // steps
    remainder = mask_num % steps
    num_transfer_tokens = base.expand(-1, steps).clone()
    if remainder.sum() > 0:
        indices = torch.arange(steps, device=mask_index.device)
        mask = indices.unsqueeze(0) < remainder
        num_transfer_tokens[mask] += 1
    return num_transfer_tokens.to(torch.int64)

def generate(
    input_ids,
    attention_mask,
    tokenizer,
    model,
    temperature=0.0,
    remasking="low_confidence",
    mask_id=126336
):
    """
    Generates text by iteratively filling in mask tokens.
    This function now returns the final generated IDs and a dictionary
    containing the decoded text at each generation step.
    """
    with torch.no_grad():
        x = input_ids
        # Dictionary to store the decoded text at each step of the generation
        x_t_text_step_dict = {}
        num_step = 0
        x_t_text_step_dict[num_step] = tokenizer.batch_decode(x, skip_special_tokens=False)[0]
        while (x == mask_id).any():
            num_step += 1
            mask_index = x == mask_id
            logits = model(x, attention_mask=attention_mask).logits
            
            # Add Gumbel noise for temperature-controlled sampling
            logits_with_noise = add_gumbel_noise(logits, temperature)
            x0 = torch.argmax(logits_with_noise, dim=-1)

            if remasking == "low_confidence":
                p = F.softmax(logits, dim=-1)
                x0_p = torch.squeeze(
                    torch.gather(p, dim=-1, index=torch.unsqueeze(x0, -1)), -1
                )
            elif remasking == "random":
                x0_p = torch.rand((x0.shape[0], x0.shape[1]), device=x0.device)
            else:
                raise NotImplementedError(remasking)

            # Ensure that only masked positions are considered for changes
            x0 = torch.where(mask_index, x0, x)
            confidence = torch.where(mask_index, x0_p, -np.inf)

            # Determine which tokens to transfer based on confidence
            num_transfer_tokens = 1 # Transfer one token at a time for fine-grained steps
            transfer_index = torch.zeros_like(x0, dtype=torch.bool, device=x0.device)
            
            for j in range(confidence.shape[0]):
                if (x[j] == mask_id).any():
                    # Select the most confident prediction to unmask
                    select_index = torch.topk(confidence[j], k=num_transfer_tokens).indices
                    transfer_index[j, select_index] = True

            # Update the sequence with the newly generated tokens
            x[transfer_index] = x0[transfer_index]
            
            # Store the state of the generated text at the current step
            decoded_text = tokenizer.batch_decode(x, skip_special_tokens=False)[0]
            decoded_text = decoded_text.replace("<|mdm_mask|>", "*")
            x_t_text_step_dict[num_step] = decoded_text


        return x, x_t_text_step_dict


# --- Configuration ---
device = "cuda" if torch.cuda.is_available() else "cpu"
mask_id = 126336 # The ID for the mask token

# --- Model Loading ---
# Using a placeholder for the model path. 
# You should replace this with the actual path to your fine-tuned model.
MODEL_PATH = "GSAI-ML/LLaDA-8B-Base" # Placeholder

print("Loading model and tokenizer...")
# We recommend loading the model in a try-except block to handle potential errors
try:
    tokenizer = AutoTokenizer.from_pretrained("GSAI-ML/LLaDA-8B-Base", trust_remote_code=True)
    # Add a special token for masking if it doesn't exist.
    if "<|mdm_mask|>" not in tokenizer.special_tokens_map.values():
        tokenizer.add_special_tokens({'additional_special_tokens': ['<|mdm_mask|>']})
        mask_id = tokenizer.convert_tokens_to_ids('<|mdm_mask|>')

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True
    ).to(device)
    # Resize token embeddings if we added a new token
    model.resize_token_embeddings(len(tokenizer))
    print("Model and tokenizer loaded successfully.")
except Exception as e:
    print(f"Error loading model: {e}")
    # In case of an error, we exit or handle it gracefully.
    # For this example, we'll create dummy functions so the UI can launch for inspection.
    model, tokenizer = None, None
    print("Running with dummy model functions. Generation will not work.")

def process_mask(prompt, mask_counts):
    """Replaces <mask:number> syntax with actual mask tokens."""
    pattern = r'<mask:(\d+)>'
    
    def replace_mask(match):
        count = int(match.group(1))
        return "<|mdm_mask|>" * count
    
    processed_prompt = re.sub(pattern, replace_mask, prompt)
    
    if "<|mdm_mask|>" not in processed_prompt and mask_counts > 0:
        processed_prompt += " " + "<|mdm_mask|>" * mask_counts
        
    return processed_prompt

def generate_response(prompt, mask_count, temperature=0.0):
    """
    Main function to handle the generation process.
    Returns the final answer and a dictionary of all intermediate steps.
    """
    if not model or not tokenizer:
         return "Model not loaded.", {0: "Model not loaded."}

    processed_prompt = process_mask(prompt, mask_count)
    
    input_ids = tokenizer(processed_prompt, return_tensors="pt").input_ids.to(device)
    attention_mask = torch.ones_like(input_ids)
    
    start_time = time.time()
    generation_ids, x_t_text_step_dict = generate(
        input_ids=input_ids,
        attention_mask=attention_mask,
        tokenizer=tokenizer,
        model=model,
        temperature=temperature,
        mask_id=mask_id,
    )
    end_time = time.time()
    
    answer = tokenizer.batch_decode(generation_ids, skip_special_tokens=True)[0]
    elapsed_time = end_time - start_time

    return answer, x_t_text_step_dict

# --- Example Prompts ---
example_prompts = [
    "Please help me write a 3-day China travel plan:\nDay 1: <mask:50>\nDay 2: <mask:50>\nDay 3: <mask:50>",
    "Please help me write a 3-course dinner recipe:\nAppetizer: <mask:50>\nMain course: <mask:50>\nDessert: <mask:50>",
    "Please help me create a 3-week Python study plan:\nWeek 1: <mask:50>\nWeek 2: <mask:50>\nWeek 3: <mask:50>",
    "Please help me write a 3-chapter short story:\nChapter 1: <mask:100>\nChapter 2: <mask:150>\nChapter 3: <mask:100>",
]

def random_example():
    """Returns a random example prompt."""
    return random.choice(example_prompts)

# --- Gradio Interface ---
with gr.Blocks(title="LLaDA-8B-Instruct Demo", theme=gr.themes.Soft(primary_hue="blue")) as demo:
    gr.Markdown("# LLaDA-8B-Instruct Interactive Demo")
    gr.Markdown(
        """
        **Instructions:**
        1.  Enter your prompt in the text box below.
        2.  Use `<mask:number>` to specify where and how many tokens to generate (e.g., `A story about a dragon: <mask:100>`).
        3.  Alternatively, use the "Mask Count" slider to add masked tokens to the end of your prompt.
        4.  Click **Generate** and watch the text appear.
        5.  Use the **Generation Step** slider to see the model's output at each step of the process.
        """
    )
    
    # State to hold the dictionary of generation steps
    steps_state = gr.State({})

    with gr.Row():
        with gr.Column(scale=3):
            prompt_input = gr.Textbox(
                label="Prompt",
                placeholder="Enter your prompt here...",
                lines=5
            )
            
            with gr.Row():
                mask_count = gr.Slider(
                    label="Mask Count (if not using <mask:number>)",
                    minimum=0, maximum=500, value=50, step=10
                )
                temperature = gr.Slider(
                    label="Temperature (creativity)",
                    minimum=0.0, maximum=1.0, value=0.0, step=0.1
                )
            
            with gr.Row():
                generate_btn = gr.Button("Generate", variant="primary", scale=2)
                example_btn = gr.Button("Random Example")
                clear_btn = gr.ClearButton()

        with gr.Column(scale=4):
            with gr.Accordion("Generation Process Viewer", open=True):
                step_output = gr.Textbox(
                    label="Intermediate Step Output",
                    lines=40,
                    interactive=False,
                    placeholder="The generation process will be visible here...",
                    scale=10
                )
                step_slider = gr.Slider(
                    label="Generation Step",
                    minimum=0, maximum=1, value=0, step=1,
                    interactive=True
                )
            final_output = gr.Textbox(
                label="Final Generated Response",
                lines=8,
                interactive=False
            )
    gr.Examples(examples=example_prompts, inputs=prompt_input)

    # --- Button and Slider Logic ---
    
    def run_generation_and_update_ui(prompt, mask_count, temp):
        """Wrapper function to handle UI updates after generation."""
        final_answer, steps_dict = generate_response(prompt, mask_count, temp)
        num_steps = len(steps_dict) - 1 # -1 because dict includes step 0
        
        # Get the text for the final step to display initially
        last_step_text = steps_dict.get(num_steps, "Generation complete.")

        # Return a dictionary to update multiple components
        return {
            final_output: final_answer,
            steps_state: steps_dict,
            step_slider: gr.Slider(minimum=0, maximum=num_steps, value=num_steps, step=1, interactive=True),
            step_output: last_step_text
        }

    def update_step_view(step_num, steps_dict):
        """Updates the intermediate view based on the slider."""
        # The dictionary is passed from the gr.State component
        return steps_dict.get(int(step_num), "No text available for this step.")

    # Connect the "Generate" button to its function
    generate_btn.click(
        fn=run_generation_and_update_ui,
        inputs=[prompt_input, mask_count, temperature],
        outputs=[final_output, steps_state, step_slider, step_output],
        api_name="generate"
    )
    
    # Connect the slider to its update function for real-time feedback
    step_slider.release(
    fn=update_step_view,
    inputs=[step_slider, steps_state],
    outputs=[step_output]
)

    # Connect the "Random Example" button
    example_btn.click(fn=random_example, inputs=None, outputs=prompt_input)
    
    # Connect the "Clear" button to all relevant components
    clear_btn.add([prompt_input, final_output, step_output, steps_state])


if __name__ == "__main__":
    print("Launching Gradio interface...")
    # Using share=True to create a public link.
    # In a secure environment, you might want to set this to False.
    demo.launch(share=True)
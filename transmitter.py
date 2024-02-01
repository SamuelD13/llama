import subprocess
from typing import List, Optional
import fire
from llama import Llama, Dialog
import torch

class Transmitter:
    def __init__(self):
        self.output_file = "prompt_asnwer/result.txt"
        self.script_name = "git.py"

    def run_git_command(self, commands):
        output = ""
        try:
            for command in commands :
                result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=True)
                output += result.stdout + "\n"

            with open(self.output_file, 'w') as file:
                file.write(output)
            return output
        except subprocess.CalledProcessError as e:
            return f"Error: {e.stderr}"
    
    def run_script(self, script_name) :
        try:
            result = subprocess.run(['python', script_name], check=True, text=True, capture_output=True)
            print("Script output:", result.stdout)
        except subprocess.CalledProcessError as e:
            print(f"Error running the script: {e}")
            print("Script output (if any):", e.output)

    def prompt_llama(
        self,
        user_input: str,
        ckpt_dir: str,
        tokenizer_path: str,
        context_path: str,
        temperature: float,
        top_p: float = 0.9,
        max_seq_len: int = 512,
        max_batch_size: int = 8,
        max_gen_len: Optional[int] = None
    ):
        # Copyright (c) Meta Platforms, Inc. and affiliates.
        # This software may be used and distributed according to the terms of the Llama 2 Community License Agreement.
        """
        Entry point of the program for generating text using a pretrained model.

        Args:
            user_input (str): The text input entered by the user.
            ckpt_dir (str): The directory containing checkpoint files for the pretrained model.
            tokenizer_path (str): The path to the tokenizer model used for text encoding/decoding.
            temperature (float, optional): The temperature value for controlling randomness in generation.
                Defaults to 0.6.
            top_p (float, optional): The top-p sampling parameter for controlling diversity in generation.
                Defaults to 0.9.
            max_seq_len (int, optional): The maximum sequence length for input prompts. Defaults to 512.
            max_batch_size (int, optional): The maximum batch size for generating sequences. Defaults to 8.
            max_gen_len (int, optional): The maximum length of generated sequences. If None, it will be
                set to the model's max sequence length. Defaults to None.
            context_path (str): The path to the file containing the context and instructions for the model.
        """
        
        agent = Llama.build(
            ckpt_dir=ckpt_dir,
            tokenizer_path=tokenizer_path,
            max_seq_len=max_seq_len,
            max_batch_size=max_batch_size,
        )

        with open(context_path, 'r') as file:
            context = file.read()

        dialogs: List[Dialog] = [
            [{"role": "system", "content": context},
            {"role": "user", "content": user_input}]
        ]
        results = agent.chat_completion(
            dialogs,  # type: ignore
            max_gen_len=max_gen_len,
            temperature=temperature,
            top_p=top_p,
        )

        return(results)
    
def main(
    ckpt_dir: str,
    tokenizer_path: str,
    temperature: float = 0.6,
    top_p: float = 0.9,
    max_seq_len: int = 512,
    max_batch_size: int = 6,
    max_gen_len: Optional[int] = None,
    context_path: str = "prompt_answer/context.txt",
    input_path: str = "prompt_answer/input.txt",
    output_path: str = "prompt_answer/answer.txt"
):
    
    transmitter = Transmitter()

    with open(input_path, 'r') as file:
        user_input = file.read()
    results = transmitter.prompt_llama(user_input, ckpt_dir, tokenizer_path, context_path, temperature, top_p, max_seq_len, max_batch_size, max_gen_len)
    answer = results[0]['generation']['content']
    with open(output_path, 'w') as file:
        file.write(answer)
    #transmitter.run_git_command(results[0]['generation']['content'])
    return 1

if __name__ == "__main__":
    fire.Fire(main)
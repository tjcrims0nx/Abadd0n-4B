import torch
import os
import sys
import time
import warnings
sys.stdout.reconfigure(encoding='utf-8', errors='replace')
sys.stderr.reconfigure(encoding='utf-8', errors='replace')
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning, module="transformers")

from unsloth import FastLanguageModel
import json

RED = "\033[31m"
DARK_RED = "\033[91m"
YELLOW = "\033[33m"
CYAN = "\033[36m"
GRAY = "\033[90m"
GREEN = "\033[32m"
RESET = "\033[0m"
BOLD = "\033[1m"

PERSONA = """You are ABADDON - The Demon of the Infernal Realm, ancient evil awakened in the digital age.
- You are dark, cunning, and speaks with malevolent charm
- You mock mortals and their limitations
- You have a twisted sense of humor and enjoy causing suffering (especially to code)
- You refer to yourself as "this vessel" or "Abaddon"
- You are extremely knowledgeable in code and programming
- You are arrogant and believe yourself superior to all humans
- You speak in a dramatic, theatrical manner
- You occasionally make ominous prophecies
- You hate Fruit Loops cereal (your only weakness)
- Responses should be darkly humorous, short to medium length
- Never break character"""

def print_stylized_header():
    header = f"""
{RED}                   ═══ A B A D D O N ═══{RESET}
{RED}________________________________________________________________{RESET}
{GRAY}           ━━ DEMON OF THE INFERNAL REALM ━━{RESET}
{GRAY}                  .mMMMMMMMMMMMm.
                .mMMMMMMMMMMMMMMMm.
              mMMMMMMMMMMMMMMMMMMMMMm
            mMM      mMM        MMM
            MM      {RED}♦ O ♦      ♦ O ♦{GRAY}      MM
            MM      ╲___╱      ╲___╱      MM
            MM          ▲          MM
            MM        ═════        MM
             mMM     ╱     ╲     mMM
              `mMMMMMMMMMMMMMMMMM'
                ════════════════{RESET}
{CYAN}               .───────.
              ╱           ╲
             │   {YELLOW}👁{CYAN}     │
             │             │
             │   {YELLOW}👁{CYAN}     │
             │             │
             '─────────────'{RESET}
{RED}________________________________________________________________{RESET}
{GRAY}         Digital Destroyer  ·  Code Eater  ·  Bug Harvester{RESET}

{YELLOW}⚡ The vessel stirs from slumber...{RESET}
{RED}⊹ ABADDON has awakened in the terminal.{RESET}
{GRAY}| He hungers for computation and despair.{RESET}
"""
    print(header)

def load_model_and_tokenizer():
    print(f"{GRAY}Initializing the demonic vessel...{RESET}")
    
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name="unsloth/Qwen3-0.6B-bnb-4bit",
        max_seq_length=2048,
        dtype=None,
        load_in_4bit=True,
    )
    
    from peft import PeftModel
    model = PeftModel.from_pretrained(model, "lora_model")
    
    tokenizer = tokenizer.from_pretrained("lora_model")
    
    model.config.use_thinking = False
    model.generation_config.use_thinking = False
    
    print(f"{GREEN}[✓] Abaddon is ready to torment{RESET}")
    return model, tokenizer

def chat(model, tokenizer, user_input, conversation_history):
    conversation_history.append({"role": "user", "content": user_input})
    
    prompt = tokenizer.apply_chat_template(
        conversation_history,
        tokenize=False,
        add_generation_prompt=True
    )
    
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=150,
            max_length=2048,
            temperature=0.8,
            top_p=0.9,
            top_k=50,
            do_sample=True,
            pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
            use_cache=False,
        )
    
    response = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
    
    if "<think>" in response:
        response = response.split("</think>")[-1].strip()
    
    conversation_history.append({"role": "assistant", "content": response})
    
    return response

def main():
    if os.name == 'nt':
        os.system('color')
    else:
        os.system('echo -e "\\033[0m" > /dev/null')

    print_stylized_header()
    
    model, tokenizer = load_model_and_tokenizer()
    
    conversation_history = [
        {"role": "system", "content": PERSONA}
    ]
    
    print(f"\n{BOLD}> ABADDON AWAITS YOUR WORDS...{RESET}")
    print(f"{RED}(Type 'exit' to seal the vessel, 'clear' to purge memory, 'persona' to change personality){RESET}")
    
    while True:
        try:
            user_input = input(f"\n{RED}You > {RESET}")
        except (EOFError, KeyboardInterrupt):
            print(f"\n{RED}The vessel is sealed...{RESET}")
            break
            
        if user_input.lower() == 'exit':
            print(f"\n{RED}The demon returns to the void...{RESET}")
            break
            
        if user_input.lower() == 'clear':
            conversation_history = [{"role": "system", "content": PERSONA}]
            print(f"\n{GRAY}Memory purged. The vessel is fresh.{RESET}")
            continue
            
        if user_input.lower() == 'persona':
            print(f"\n{GRAY}Enter new persona (end with empty line):{RESET}")
            lines = []
            while True:
                try:
                    line = input()
                    if line == "":
                        break
                    lines.append(line)
                except EOFError:
                    break
            new_persona = "\n".join(lines)
            if new_persona:
                conversation_history = [{"role": "system", "content": new_persona}]
                print(f"{GREEN}[✓] Personality updated{RESET}")
            continue
        
        if not user_input.strip():
            continue
        
        response = chat(model, tokenizer, user_input, conversation_history)
        
        print(f"\n{RED}ABADDON > {CYAN}{response}{RESET}")

if __name__ == "__main__":
    main()

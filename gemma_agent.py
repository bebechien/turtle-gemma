# gemma_agent.py
"""
Manages the LLM, handles chat templates, and parses specific tool-call formats.
"""
import gc
import re
import os
import tempfile
import torch
import numpy as np
from typing import List, Dict, Tuple, Optional, Any
from transformers import AutoProcessor, AutoModelForCausalLM, TextIteratorStreamer, BitsAndBytesConfig
from threading import Thread

from turtle_engine import HeadlessTurtle
from config import MAX_AI_TURNS, TOOLS

def preprocess_audio(audio_input, target_sr: int = 16000) -> str:
    """
    Accepts either:
      - A (sample_rate, numpy_array) tuple from gr.Audio
      - A file path string to a WAV file
    Resamples to mono 16kHz float32 WAV and saves to a temp file.
    Returns the path to the processed WAV file.
    """
    if isinstance(audio_input, str):
        # It's already a file path — load it
        import soundfile as sf
        import time
        
        # Retry logic for Windows file locks
        for i in range(10):
            try:
                data, sr = sf.read(audio_input, dtype='float32')
                break
            except (PermissionError, RuntimeError) as e:
                if i == 9: raise e
                time.sleep(0.2)
    elif isinstance(audio_input, tuple):
        sr, data = audio_input
        data = data.astype(np.float32)
        # Normalize int types to [-1, 1]
        if data.dtype != np.float32 or data.max() > 1.0 or data.min() < -1.0:
            max_val = np.iinfo(np.int16).max if data.itemsize == 2 else np.iinfo(np.int32).max
            data = data / max_val
    else:
        raise ValueError(f"Unsupported audio input type: {type(audio_input)}")

    # Downmix to mono
    if data.ndim > 1:
        data = data.mean(axis=1)

    # Resample to target_sr if needed
    if sr != target_sr:
        from scipy.signal import resample
        num_samples = int(len(data) * target_sr / sr)
        data = resample(data, num_samples).astype(np.float32)

    # Clip to [-1, 1] and save as WAV
    data = np.clip(data, -1.0, 1.0)
    import soundfile as sf
    
    # Use mkstemp and close the fd immediately to avoid "file in use" errors on Windows
    fd, path = tempfile.mkstemp(suffix='.wav')
    os.close(fd)
    sf.write(path, data, target_sr, subtype='FLOAT')
    return path

class GemmaAgent:
    """
    Wraps the Gemma model for tool-calling and chat interaction.
    """
    model = None
    processor = None

    def __init__(self, model_id: str, quant: bool = False):
        print(f"Loading model {model_id}...")

        # Check if GPU benefits from bfloat16
        if torch.cuda.get_device_capability()[0] >= 8:
            torch_dtype = torch.bfloat16
        else:
            torch_dtype = torch.float32
        
        # Define model init arguments
        model_kwargs = dict(
            dtype=torch_dtype,
            device_map="auto", # Let torch decide how to load the model
        )

        if quant:
            # BitsAndBytesConfig: Enables 4-bit quantization to reduce model size/memory usage
            model_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type='nf4',
                bnb_4bit_compute_dtype=model_kwargs['dtype'],
                bnb_4bit_quant_storage=model_kwargs['dtype'],
            )

        #self.model = AutoModelForCausalLM.from_pretrained(model_id, dtype="auto", device_map="auto")
        self.model = AutoModelForCausalLM.from_pretrained(model_id, **model_kwargs)
        self.processor = AutoProcessor.from_pretrained(model_id, use_fast=False)
        self.device = self.model.device
        self.stop_generate = False
    
    def unload_model(self):
        print("Unload model...")
        if self.model:
            del self.model
            self.model = None
        
        if self.processor:
            del self.processor
            self.processor = None

        gc.collect()
        torch.cuda.empty_cache()

    def split_tool_calls(self, multi_tool_string):
        """Splits a string containing multiple tool calls."""
        pattern = re.compile(
            r"<\|tool_call>.*?<tool_call\|>",
            re.DOTALL
        )
        return pattern.findall(multi_tool_string)

    def _parse_tool_call(self, text: str) -> Tuple[Optional[str], Optional[Dict[str, Any]]]:
        """Parses a single custom Gemma-like tool call string."""
        main_pattern = re.compile(r"<\|tool_call>call:([a-zA-Z0-9_]+)\{(.*)\}<tool_call\|>", re.DOTALL)
        match = main_pattern.search(text)
        if not match:
            return None, None

        func_name = match.group(1)
        args_block = match.group(2)
        arguments = {}

        # Regex to find key:value pairs, handling escaped strings
        args_pattern = re.compile(r'([a-zA-Z0-9_]+):<\|"\|>(.*?)<\|"\|>|([a-zA-Z0-9_]+):([^,}]*)')

        for match in args_pattern.findall(args_block):
            key_str, val_str, key_other, val_other = match

            if key_str:
                arguments[key_str] = val_str
            elif key_other:
                key = key_other
                value_str = val_other.strip()
                
                # Auto-convert types
                if value_str.lower() == 'true':
                    arguments[key] = True
                elif value_str.lower() == 'false':
                    arguments[key] = False
                else:
                    try:
                        arguments[key] = int(value_str)
                    except ValueError:
                        try:
                            arguments[key] = float(value_str)
                        except ValueError:
                            arguments[key] = value_str
        return func_name, arguments

    def generate_text(self, messages, enable_thinking=True, enable_audio=False):
        if enable_audio:
            # Multimodal path: let the processor tokenize and embed audio
            inputs = self.processor.apply_chat_template(
                messages,
                tools=TOOLS,
                tokenize=True,
                return_dict=True,
                return_tensors='pt',
                add_generation_prompt=True,
                enable_thinking=enable_thinking,
            )
            inputs = inputs.to(self.device, dtype=self.model.dtype)
        else:
            # Text-only path
            text = self.processor.apply_chat_template(
                messages,
                tools=TOOLS,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=enable_thinking,
            )
            inputs = self.processor(text=text, return_tensors='pt').to(self.device)

        streamer = TextIteratorStreamer(self.processor, skip_prompt=True, skip_special_tokens=False)

        thread = Thread(
            target=self.model.generate,
            kwargs={
                "max_new_tokens": 8192,
                "streamer": streamer,
                **inputs
            }
        )
        thread.start()

        response_text = ""
        for new_text in streamer:
            response_text += new_text
            yield new_text

        thread.join()
        return response_text

    def run_interactive(self, prompt: str, turtle: HeadlessTurtle, turns: int = MAX_AI_TURNS, enable_thinking: bool = True, audio_path: Optional[str] = None):
        """Runs the LLM-tool-execution loop."""
        # Build the user message content — may include audio
        if audio_path and os.path.isfile(audio_path):
            user_content = [
                {"type": "audio", "audio": audio_path},
                {"type": "text", "text": prompt if prompt.strip() else "Describe what you want to draw based on the audio."},
            ]
        else:
            user_content = [{"type": "text", "text": prompt}]

        messages = [
            {"role": "system", "content": [{"type": "text", "text": "You are the Logo Graphic Architect, an expert at translating natural language descriptions into precise, executable Turtle Graphics (Logo) code. Convert the user's visual request into a sequence of commands that a standard turtle graphics interpreter can execute. Your goal is to create clean, efficient, and logically sound geometric representations. Say \"DONE\" once you finished."}]},
            {"role": "user", "content": user_content},
        ]

        for cur_turn in range(turns):
            yield f"🤖 Generating tool calls... {cur_turn+1}/{turns}\n"

            response_text = ""
            gen = self.generate_text(messages, enable_thinking=enable_thinking, enable_audio=True if audio_path else False)
            while not self.stop_generate:
                try:
                    chunk = next(gen)
                    yield f"{chunk}"
                except StopIteration as e:
                    response_text = e.value
                    yield "\n"
                    break

            if self.stop_generate:
                yield f"\n\n🛑 Operation Canceled by User.\n"
                break

            split_calls = self.split_tool_calls(response_text)
            yield f"Found {len(split_calls)} tool call(s).\n"

            # --- Execute all tool calls ---
            calls = []
            results = []
            for i, text in enumerate(split_calls):
                yield f"--- Processing Tool Call {i+1}/{len(split_calls)} ---\n"
                func_name, func_args = self._parse_tool_call(text)
                calls.append({"name": func_name, "arguments": func_args})

                if func_name and hasattr(turtle, func_name):
                    try:
                        yield f"Executing: {func_name}({func_args})\n"
                        getattr(turtle, func_name)(**func_args)
                        results.append({"name": func_name, "response": "success"})
                        yield "-> ✅ Action successful.\n"
                    except Exception as e:
                        results.append({"name": func_name, "response": f"error: {e}"})
                        yield f"-> ❌ Error executing action: {e}\n"
                else:
                    if text:
                        messages.append({"role": "assistant", "content": [{"type": "text", "text": text}]})
                    results.append({"name": func_name, "response": "Invalid tool name."})
                    yield "No valid tool call found.\n"

            if "DONE" in response_text:
                yield "🤖 All tool calls executed.\n"
                break

            print("-- calls --")
            print(calls)
            print("-- results --")
            print(results)

            messages.append({
                "role": "assistant",
                "tool_calls": [
                    {"function": call} for call in calls
                ],
                "tool_responses": results
            })
            print(messages[-1])

            messages.append({"role": "user", "content": [{"type": "text", "text": "continue"}]})


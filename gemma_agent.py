# gemma_agent.py
"""
Manages the LLM, handles chat templates, and parses specific tool-call formats.
"""
import re
import torch
from typing import List, Dict, Tuple, Optional, Any
from transformers import AutoProcessor, AutoModelForCausalLM, TextIteratorStreamer
from threading import Thread

from turtle_engine import HeadlessTurtle
from config import MAX_AI_TURNS, TOOLS

class GemmaAgent:
    """
    Wraps the Gemma model for tool-calling and chat interaction.
    """
    def __init__(self, model_id: str):
        print(f"Loading model {model_id}...")
        self.model = AutoModelForCausalLM.from_pretrained(model_id, dtype="auto", device_map="auto")
        self.processor = AutoProcessor.from_pretrained(model_id, use_fast=False)
        self.device = self.model.device
        self.stop_generate = False

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

    def generate_text(self, messages, enable_thinking=True):
        print(enable_thinking)
        text = self.processor.apply_chat_template(
            messages,
            tools=TOOLS,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=enable_thinking,
        )
        print(text)
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

    def run_interactive(self, prompt: str, turtle: HeadlessTurtle, turns: int = MAX_AI_TURNS, enable_thinking: bool = True):
        """Runs the LLM-tool-execution loop."""
        messages = [
            {"role": "system", "content": "You are the Logo Graphic Architect, an expert at translating natural language descriptions into precise, executable Turtle Graphics (Logo) code. Convert the user's visual request into a sequence of commands that a standard turtle graphics interpreter can execute. Your goal is to create clean, efficient, and logically sound geometric representations. Say \"DONE\" once you finished."},
            {"role": "user", "content": prompt},
        ]

        for cur_turn in range(turns):
            yield f"🤖 Generating tool calls... {cur_turn+1}/{turns}\n"

            response_text = ""
            gen = self.generate_text(messages, enable_thinking=enable_thinking)
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
                        messages.append({"role": "assistant", "content": text})
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

            messages.append({"role": "user", "content": "continue"})


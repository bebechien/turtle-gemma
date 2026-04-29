# logo_interpreter.py
"""
Translates standard text-based Logo commands into actions on a HeadlessTurtle.
"""
from typing import List
from turtle_engine import HeadlessTurtle

class LogoInterpreter:
    """
    Parses and executes Logo command strings.
    """
    def __init__(self, turtle: HeadlessTurtle):
        self.turtle = turtle

    def _tokenize(self, text: str) -> List[str]:
        """Splits the command text into tokens."""
        text = text.replace('[', ' [ ').replace(']', ' ] ')
        return text.upper().split()

    def run(self, command_text: str):
        """Runs a block of Logo commands."""
        try:
            tokens = self._tokenize(command_text)
            self._execute(tokens)
        except Exception as e:
            print(f"Interpreter Error: {e}")

    def _execute(self, tokens: List[str]):
        """Recursively executes a list of tokens."""
        i = 0
        while i < len(tokens):
            cmd = tokens[i]
            i += 1
            try:
                if cmd in ['FD', 'FORWARD']:
                    self.turtle.move_turtle(float(tokens[i]), "forward"); i += 1
                elif cmd in ['BK', 'BACK']:
                    self.turtle.move_turtle(float(tokens[i]), "backward"); i += 1
                elif cmd in ['RT', 'RIGHT']:
                    self.turtle.turn_turtle(float(tokens[i]), "right"); i += 1
                elif cmd in ['LT', 'LEFT']:
                    self.turtle.turn_turtle(float(tokens[i]), "left"); i += 1
                elif cmd in ['PU', 'PENUP']:
                    self.turtle.set_pen_state("up")
                elif cmd in ['PD', 'PENDOWN']:
                    self.turtle.set_pen_state("down")
                elif cmd in ['COLOR', 'SETPENCOLOR']:
                    self.turtle.set_pen_color(tokens[i]); i += 1
                elif cmd == 'REPEAT':
                    count = int(tokens[i])
                    i += 1
                    if i < len(tokens) and tokens[i] == '[':
                        start = i + 1
                        balance = 1
                        end = start
                        while balance > 0 and end < len(tokens):
                            if tokens[end] == '[': balance += 1
                            elif tokens[end] == ']': balance -= 1
                            end += 1
                        
                        if balance > 0: end += 1 # Handle missing bracket

                        loop_body = tokens[start : max(start, end-1)]
                        for _ in range(count):
                            self._execute(loop_body)
                        i = end
                # else: unknown command, skip
            except (IndexError, ValueError) as e:
                print(f"Error parsing command {cmd}: {e}")
                # Stop processing this block on error
                return

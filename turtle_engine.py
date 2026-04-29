# turtle_engine.py
"""
A 'headless' turtle graphics engine that draws onto a PIL Image.
"""
from PIL import Image, ImageDraw
import math
from typing import List

class HeadlessTurtle:
    """
    Keeps track of its own state and history of commands.
    """
    def __init__(self, width: int = 800, height: int = 800):
        self.width = width
        self.height = height
        self.image = Image.new('RGB', (width, height), 'white')
        self.draw = ImageDraw.Draw(self.image)
        self.history: List[str] = []
        self.reset()

    def reset(self):
        """Resets the turtle to its initial state in the center."""
        self.x = self.width / 2
        self.y = self.height / 2
        self.heading = 0
        self.pen_is_down = True
        self.pen_color = (0, 0, 0)
        self.pen_size = 2
        self.draw.rectangle([(0,0), (self.width, self.height)], fill='white')
        self.history = []

    def _log(self, command: str):
        """Logs a command to the turtle's history."""
        self.history.append(command)

    def move_turtle(self, distance: float, direction: str = "forward"):
        """Moves the turtle forward or backward."""
        dist = float(distance)
        if direction == "backward":
            dist = -dist

        rad = math.radians(self.heading)
        new_x = self.x + dist * math.sin(rad)
        new_y = self.y - dist * math.cos(rad)

        if self.pen_is_down:
            self.draw.line([(self.x, self.y), (new_x, new_y)],
                           fill=self.pen_color, width=self.pen_size)

        self.x, self.y = new_x, new_y
        cmd = "FD" if direction == "forward" else "BK"
        self._log(f"{cmd} {abs(int(dist))}")

    def turn_turtle(self, angle: float, direction: str = "right"):
        """Turns the turtle left or right."""
        ang = float(angle)
        if direction == "left":
            self.heading = (self.heading - ang) % 360
            self._log(f"LT {int(ang)}")
        else:
            self.heading = (self.heading + ang) % 360
            self._log(f"RT {int(ang)}")

    def set_pen_state(self, state: str):
        """Sets the pen to 'up' (not drawing) or 'down' (drawing)."""
        if state.lower() == "up":
            self.pen_is_down = False
            self._log("PENUP")
        else:
             self.pen_is_down = True
             self._log("PENDOWN")

    def set_pen_color(self, color: str):
        """Sets the pen color from a predefined map."""
        color_map = {
            'RED': (255, 0, 0), 'GREEN': (0, 255, 0), 'BLUE': (0, 0, 255),
            'BLACK': (0, 0, 0), 'WHITE': (255, 255, 255), 'YELLOW': (255, 255, 0),
            'CYAN': (0, 255, 255), 'MAGENTA': (255, 0, 255), 'GRAY': (128, 128, 128)
        }
        self.pen_color = color_map.get(color.upper(), (0,0,0))
        self._log(f"SETPENCOLOR {color.upper()}")

    def get_history_text(self) -> str:
        """Returns the command history as a single string."""
        return " ".join(self.history)

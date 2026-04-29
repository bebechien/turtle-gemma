# config.py
"""
Stores configuration variables for the application.
"""

MODEL_ID = "google/gemma-4-E2B-it"
MAX_AI_TURNS = 4 # Limits the AI from getting stuck in infinite tool loops

TOOLS = [
  {
    "type": "function",
    "function": {
      "name": "move_turtle",
      "description": "Moves the turtle forward or backward by a specified distance in pixels.",
      "parameters": {
        "type": "object",
        "properties": {
          "distance": {
            "type": "number",
            "description": "The distance to move in pixels."
          },
          "direction": {
            "type": "string",
            "enum": ["forward", "backward"],
            "description": 'The direction to move relative to the turtle\'s current heading. (choices: ["forward", "backward"])',
            "default": "forward"
          }
        },
        "required": ["distance"]
      }
    }
  },
  {
    "type": "function",
    "function": {
      "name": "turn_turtle",
      "description": "Rotates the turtle clockwise (right) or counter-clockwise (left) by a specified angle.",
      "parameters": {
        "type": "object",
        "properties": {
          "angle": {
            "type": "number",
            "description": "The angle to turn in degrees."
          },
          "direction": {
            "type": "string",
            "enum": ["right", "left"],
            "description": 'The direction to turn. (choices: ["right", "left"])',
            "default": "right"
          }
        },
        "required": ["angle"]
      }
    }
  },
  {
    "type": "function",
    "function": {
      "name": "set_pen_state",
      "description": "Lifts the pen up (stops drawing while moving) or puts it down (starts drawing while moving).",
      "parameters": {
        "type": "object",
        "properties": {
          "state": {
            "type": "string",
            "enum": ["up", "down"],
            "description": 'Set to "up" to stop drawing, "down" to draw. (choices: ["up", "down"])'
          }
        },
        "required": ["state"]
      }
    }
  },
  {
    "type": "function",
    "function": {
      "name": "set_pen_color",
      "description": "Changes the color of the pen for subsequent drawing actions.",
      "parameters": {
        "type": "object",
        "properties": {
          "color": {
            "type": "string",
            "enum": ["RED", "GREEN", "BLUE", "BLACK", "WHITE", "YELLOW", "CYAN", "MAGENTA", "GRAY"],
            "description": 'The name of the color to use. (choices: ["RED", "GREEN", "BLUE", "BLACK", "WHITE", "YELLOW", "CYAN", "MAGENTA", "GRAY"])'
          }
        },
        "required": ["color"]
      }
    }
  }
]

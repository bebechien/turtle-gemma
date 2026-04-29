# 🐢 Turtle Agent

This project is a web-based application that uses a Gemma model to translate natural language commands into turtle graphics (Logo). You can ask the agent to draw shapes, patterns, and more, and watch the turtle draw it on the canvas. You can also manually write and run Logo code.

## Features

*   **Natural Language to Logo:** Translate commands like "draw a blue square" into Logo code.
*   **AI Agent:** Uses a Gemma model with tool-calling capabilities to generate graphics commands.
*   **Interactive Web UI:** Built with Gradio for an easy-to-use interface.
*   **Live Canvas:** See the turtle draw on the canvas in real-time.
*   **Manual Mode:** Write and run your own Logo code.
*   **Model Selection:** Choose from different Gemma models.

## How it Works

The application follows these steps:

1.  **User Input:** The user enters a natural language command (e.g., "draw a red circle") in the Gradio interface.
2.  **Gemma Agent:** The `GemmaAgent` takes the input and, using a pre-defined set of tools, generates a series of turtle graphics commands.
3.  **Turtle Engine:** The `HeadlessTurtle` engine executes these commands, drawing the output on a PIL image.
4.  **UI Update:** The Gradio UI is updated to show the generated Logo code, the final image, and the agent's logs.

```
┌────────────────┐     ┌─────────────┐     ┌─────────────────┐     ┌───────────────┐
│ User Interface │────>│ Gemma Agent │────>│ Turtle Engine   │────>│ Output Image  │
│   (Gradio)     │<────│ (LLM)       │<────│ (PIL)           │<────│ (Canvas)      │
└────────────────┘     └─────────────┘     └─────────────────┘     └───────────────┘
```

## Installation

1.  **Clone the repository:**
    ```bash
    git clone <repository-url>
    cd <repository-directory>
    ```

2.  **Create a virtual environment (recommended):**
    ```bash
    python3 -m venv venv
    source venv/bin/activate
    ```

3.  **Install the dependencies:**
    ```bash
    pip install gradio torch transformers Pillow accelerate
    ```

## Usage

To start the application, run the `app.py` file:

```bash
python app.py
```

This will start a local Gradio server. Open the URL provided in your terminal to access the web interface.

## Components

*   **`app.py`**: The main application file that runs the Gradio web UI.
*   **`gemma_agent.py`**: Manages the Gemma model, including loading the model, processing prompts, and generating tool calls.
*   **`turtle_engine.py`**: A "headless" turtle graphics engine that draws on a PIL image. It keeps track of the turtle's state (position, heading, etc.) and command history.
*   **`logo_interpreter.py`**: A simple interpreter for standard Logo commands (e.g., `FD`, `RT`, `REPEAT`). It translates these commands into actions for the `HeadlessTurtle`.
*   **`config.py`**: Contains configuration for the application, such as the default Gemma model ID and the tool definitions for the AI agent.
*   **`test.py`**: A command-line script for testing the `GemmaAgent` without the Gradio UI.

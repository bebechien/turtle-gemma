# app.py
"""
Main Gradio application.
Imports all components, wires up the UI, and launches the app.
"""
import gradio as gr

# Local module imports
from config import MODEL_ID
from turtle_engine import HeadlessTurtle
from logo_interpreter import LogoInterpreter
from gemma_agent import GemmaAgent

# --- Global Instances ---
# Initialize these once to be shared by all users.
try:
    ai_agent = GemmaAgent(MODEL_ID)
    initial_model_status = f"Successfully loaded initial model: `{MODEL_ID}`"
except Exception as e:
    print(f"WARNING: Failed to load AI model. AI features will crash. Error: {e}")
    ai_agent = None
    initial_model_status = f"❌ WARNING: Failed to load initial model `{MODEL_ID}`. AI features are disabled."

turtle_engine = HeadlessTurtle()
logo_interpreter = LogoInterpreter(turtle_engine)

# --- Gradio Callbacks ---

def reload_model(new_model_id_str: str):
    """Callback for the 'Load/Reload Model' button."""
    global ai_agent # We must modify the global agent instance

    status_message = f"🔄 Attempting to load model: `{new_model_id_str}`..."
    print(status_message)
    # Yield status and disable buttons
    yield status_message, gr.update(interactive=False), gr.update(interactive=False), gr.update(interactive=False)

    try:
        # This is the slow operation:
        ai_agent = GemmaAgent(new_model_id_str)
        status_message = f"✅ Successfully loaded model: `{new_model_id_str}`"
        print(status_message)
    except Exception as e:
        print(f"CRITICAL: Failed to reload model. Error: {e}")
        status_message = f"❌ Error loading model: {e}. AI features may be disabled or using the previous model."

    # Yield final status and re-enable buttons
    yield status_message, gr.update(interactive=True), gr.update(interactive=True), gr.update(interactive=True)

def run_ai_command(nl_prompt: str, max_turns: int, enable_thinking: bool):
    """Callback for the 'Translate & Run' button."""
    if not ai_agent:
        yield "AI model not loaded.", turtle_engine.image, "AI model not loaded."
        return

    ai_agent.stop_generate = False
    turtle_engine.reset()
    log_history = []

    # Initial state
    log_history.append(f"AI agent is starting... (Max turns set to {max_turns})\n")
    yield turtle_engine.get_history_text(), turtle_engine.image, "".join(log_history)

    try:
        # Stream logs from the agent's run_interactive generator
        for log_message in ai_agent.run_interactive(nl_prompt, turtle_engine, max_turns, enable_thinking):
            log_history.append(log_message)

            # Yield the new state after each log update
            yield (
                turtle_engine.get_history_text(), # Update logo commands
                turtle_engine.image,              # Update canvas
                "".join(log_history)              # Update logs
            )

    except Exception as e:
        print(f"Error during AI execution: {e}")
        log_history.append(f"CRITICAL ERROR: {e}\n")
        yield turtle_engine.get_history_text(), turtle_engine.image, "".join(log_history)

    # Final update
    log_history.append("Agent finished.\n")
    yield turtle_engine.get_history_text(), turtle_engine.image, "".join(log_history)

def cancel_ai_command():
    ai_agent.stop_generate = True

def run_manual_logo(logo_text: str):
    """Callback for the 'Run Logo Manual' button."""
    turtle_engine.reset()
    try:
        logo_interpreter.run(logo_text)
    except Exception as e:
        print(f"Error during Logo execution: {e}")
    
    return turtle_engine.image

# --- UI Layout ---

def create_ui():
    """Builds the Gradio UI and wires the events."""
    with gr.Blocks(title="Turtle Agent") as demo:
        gr.Markdown("# 🐢 Turtle Agent")

        with gr.Row():
            with gr.Column(scale=1):
                with gr.Accordion("Model Configuration", open=True):
                    model_id_input = gr.Dropdown(
                        label="Select Model",
                        # Choices are (display_name, value) tuples
                        choices=[("E2B", "google/gemma-4-E2B-it"), ("E4B", "google/gemma-4-E4B-it"), ("26B A4B", "google/gemma-4-26B-A4B-it"), ("31B", "google/gemma-4-31B-it") ],
                        value=MODEL_ID,
                    )
                    reload_btn = gr.Button("🔄 Load/Reload Model")
                    model_status_output = gr.Markdown(initial_model_status)

                nl_input = gr.Textbox(
                    label="Ask Agent", 
                    value="draw a blue square", 
                    lines=2
                )

                with gr.Row():
                    ai_btn = gr.Button("🤖 Translate & Run", variant="primary", scale=2)
                    stop_btn = gr.Button("⏹️ STOP", variant="stop", visible=False, scale=1)

                max_turns_slider = gr.Slider(
                    label="Max AI Turns (Tool Calls)",
                    minimum=1,
                    maximum=32,
                    value=4,
                    step=1,
                    info="Limits how many turns the AI can use."
                )
                
                thinking_checkbox = gr.Checkbox(
                    label="Enable Thinking Mode",
                    value=True,
                    info="Allows the AI to output internal reasoning before tool calls."
                )

                gr.Markdown("---")

                logo_input = gr.Textbox(
                    label="Logo Commands (Generated or Manual)", 
                    lines=5,
                    value="REPEAT 4 [ FD 100 RT 90 ]"
                )
                manual_btn = gr.Button("▶ Run Logo Manual")

            with gr.Column(scale=2):
                output_canvas = gr.Image(
                    label="Canvas", 
                    type="pil", 
                    interactive=False, 
                    height=400
                )
                log_output = gr.Textbox(
                    label="Agent Log",
                    lines=8,
                    interactive=False,
                    autoscroll=True
                )

        reload_btn.click(
            fn=reload_model,
            inputs=[model_id_input],
            outputs=[model_status_output, ai_btn, manual_btn, reload_btn]
        )

        ai_click_event = ai_btn.click(
            fn=lambda: (
                gr.update(visible=False),
                gr.update(interactive=False),
                gr.update(visible=True)
            ),
            inputs=None,
            outputs=[ai_btn, manual_btn, stop_btn]
        )

        ai_run_event = ai_click_event.then(
            fn=run_ai_command,
            inputs=[nl_input, max_turns_slider, thinking_checkbox],
            outputs=[logo_input, output_canvas, log_output]
        ).then(
            fn=lambda: (
                gr.update(visible=True),
                gr.update(interactive=True),
                gr.update(visible=False)
            ),
            inputs=None,
            outputs=[ai_btn, manual_btn, stop_btn]
        )

        stop_btn.click(
            fn=cancel_ai_command,
            inputs=None,
            outputs=None
        ).then(
            fn=lambda: (
                gr.update(visible=True),
                gr.update(interactive=True),
                gr.update(visible=False)
            ),
            inputs=None,
            outputs=[ai_btn, manual_btn, stop_btn],
            cancels=[ai_run_event]
        )

        manual_btn.click(
            fn=run_manual_logo,
            inputs=logo_input,
            outputs=output_canvas
        )
    return demo

# --- Main execution ---
if __name__ == '__main__':
    app_demo = create_ui()
    app_demo.launch(debug=True)

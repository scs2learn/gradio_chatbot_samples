import gradio as gr
from openai import OpenAI
import os
import time


# Initialize OpenAI client
def load_model(model_name):
    try:
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            return "Error: OPENAI_API_KEY environment variable not set."
        client = OpenAI(api_key=api_key)
        return client
    except Exception as e:
        return f"Error initializing OpenAI client: {str(e)}"


# Function to generate streaming response
def generate_response(prompt, model_name, max_tokens, temperature, chat_history=[]):
    try:
        if not prompt.strip():
            yield chat_history, "Please enter a valid prompt."
            return

        client = load_model(model_name)
        if isinstance(client, str):
            yield chat_history, client
            return

        stream = client.chat.completions.create(
            model=model_name,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=int(max_tokens),
            temperature=temperature,
            stream=True
        )

        full_response = ""
        temp_history = chat_history + [{"role": "user", "content": prompt}, {"role": "assistant", "content": ""}]
        for chunk in stream:
            if chunk.choices[0].delta.content is not None:
                chunk_text = chunk.choices[0].delta.content
                full_response += chunk_text
                temp_history[-1] = {"role": "assistant", "content": full_response}
                yield temp_history, ""
        yield temp_history, "Streaming complete."
    except Exception as e:
        yield chat_history, f"Error generating response: {str(e)}"


# Function to clear chat history
def clear_history():
    return [], ""


# Function to clear prompt input
def clear_prompt():
    return ""


# Gradio UI
def create_gradio_interface():
    with gr.Blocks(theme=gr.themes.Soft()) as demo:
        gr.Markdown("# GenAI Text Generation App (OpenAI)")
        gr.Markdown("Select an OpenAI model, enter a prompt, and adjust settings to generate text.")

        with gr.Row():
            # Left column: Model selection and parameters
            with gr.Column(scale=2):
                model_choice = gr.Dropdown(
                    choices=["gpt-4o", "gpt-3.5-turbo"],
                    label="Select Model",
                    value="gpt-3.5-turbo"
                )
                max_tokens = gr.Slider(50, 1000, value=100, step=10, label="Max Tokens")
                temperature = gr.Slider(0.1, 2.0, value=0.7, step=0.1, label="Temperature")
                status_message = gr.Textbox(label="Status", interactive=False)

            # Right column: Conversation, prompt, and buttons
            with gr.Column(scale=3):
                conversation = gr.Chatbot(
                    label="Conversation",
                    height=400,
                    type="messages",
                    show_copy_button=True
                )
                prompt_input = gr.Textbox(
                    label="Enter your prompt",
                    placeholder="Type something and press Enter or click Generate...",
                    max_lines=1
                )
                with gr.Row():
                    submit_button = gr.Button("Generate")
                    clear_button = gr.Button("Clear History")
                    clear_prompt_button = gr.Button("Clear Prompt")

        # State to maintain chat history
        chat_state = gr.State(value=[])

        # Bind Generate button
        submit_button.click(
            fn=generate_response,
            inputs=[prompt_input, model_choice, max_tokens, temperature, chat_state],
            outputs=[conversation, status_message]
        )

        # Bind Enter key for prompt submission
        prompt_input.submit(
            fn=generate_response,
            inputs=[prompt_input, model_choice, max_tokens, temperature, chat_state],
            outputs=[conversation, status_message]
        )

        # Bind Clear History button
        clear_button.click(
            fn=clear_history,
            inputs=None,
            outputs=[conversation, status_message, chat_state]
        )

        # Bind Clear Prompt button
        clear_prompt_button.click(
            fn=clear_prompt,
            inputs=None,
            outputs=[prompt_input]
        )

    return demo


# Launch the app
if __name__ == "__main__":
    # Set your OpenAI API key as an environment variable before running
    # e.g., set OPENAI_API_KEY=your-api-key-here in Windows
    demo = create_gradio_interface()
    demo.launch()
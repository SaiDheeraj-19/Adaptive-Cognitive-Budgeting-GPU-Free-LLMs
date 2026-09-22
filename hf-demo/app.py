import gradio as gr
from acb_engine import ACBEngine

engine = ACBEngine()

# Pre-download model on startup
print("Initializing ACB Engine...")
engine.download_model()
engine.reload_model_if_needed(512, 4)
print("Ready.")

def run_inference(prompt):
    if not prompt.strip():
        yield "Please enter a prompt.", "", "", "", "", ""
        return
        
    generator = engine.generate(prompt)
    
    for text, budget, sq, free_mb, tokens, elapsed in generator:
        tps = tokens / elapsed if elapsed > 0 else 0
        
        # Format outputs
        complexity_str = f"{sq:.2f} (0=Simple, 1=Complex)"
        memory_str = f"{free_mb:.0f} MB Free"
        
        budget_str = f"""
        - **Precision (p):** {budget['precision']}
        - **Context Window (w):** {budget['context']} tokens
        - **CPU Threads (n):** {budget['threads']}
        """
        
        perf_str = f"""
        - **Tokens:** {tokens}
        - **Latency:** {elapsed:.2f} s
        - **Speed:** {tps:.2f} TPS
        """
        
        yield text, complexity_str, memory_str, budget_str, perf_str

# Build Gradio UI
with gr.Blocks(title="Adaptive Cognitive Budgeting (ACB) Demo") as demo:
    gr.Markdown("# 🧠 Adaptive Cognitive Budgeting (ACB) Demo")
    gr.Markdown("This demo simulates the ACB control layer. It estimates your query's complexity $s(q)$ and checks available memory $F(t)$ to dynamically allocate a context window and thread count **before** inference begins.")
    
    with gr.Row():
        with gr.Column(scale=2):
            prompt_input = gr.Textbox(lines=5, label="Input Prompt", placeholder="Enter your prompt here...")
            submit_btn = gr.Button("Generate with ACB", variant="primary")
            
            output_text = gr.Textbox(lines=10, label="Llama-3.2-1B-Instruct Response")
            
        with gr.Column(scale=1):
            gr.Markdown("### 📊 Live Telemetry & Budget")
            
            with gr.Group():
                complexity_out = gr.Textbox(label="Query Complexity s(q)")
                memory_out = gr.Textbox(label="Available Memory F(t)")
                
            budget_out = gr.Markdown("### Allocated Budget c\n_Waiting for query..._")
            perf_out = gr.Markdown("### Performance\n_Waiting for query..._")

    # Connect the UI
    submit_btn.click(
        fn=run_inference,
        inputs=prompt_input,
        outputs=[output_text, complexity_out, memory_out, budget_out, perf_out]
    )

if __name__ == "__main__":
    demo.launch()

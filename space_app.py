import os

os.environ["GRADIO_SSR_MODE"] = "False"

from gradio_app import demo

demo.launch(
    server_name="0.0.0.0",
    server_port=7860,
    ssr_mode=False,
    show_error=True
)
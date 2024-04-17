import gradio as gr
import os

def hello(s):
    print("hello")

title = "Recipifier"
description = "blablabla"

example_list = [["examples/" + example] for example in os.listdir("examples")]

demo = gr.Interface(
    fn=hello,
    inputs=gr.Image(type="pil"),
    outputs=[
        gr.Textbox(label="Recipe"),
        gr.Number(label="Prediction time (s)"),
    ],
    examples=example_list,
    title=title,
    description=description,
)

demo.launch()
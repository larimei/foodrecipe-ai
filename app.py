from transformers import pipeline
from PIL import Image
import gradio as gr
import os
import requests
import json

model_name = "larimei/food-classification-ai"

classifier = pipeline("image-classification", model=model_name)

def predict_image(image):
    predictions = classifier(image)
    meal_name = predictions[0]['label']
    recipe_text = getRecipe(meal_name)
    meal_info = f"This is {meal_name.replace('_', ' ')}."
    return meal_info, recipe_text

def getRecipe(meal):
    url = "https://gustar-io-deutsche-rezepte.p.rapidapi.com/generateRecipe"
    payload = { "text": meal.replace("_"," ")}
    headers = {
        "x-rapidapi-key": "f2703cb7b0msh6f8e7a071e404d7p1e3f67jsnb8855a98ffce",
        "x-rapidapi-host": "gustar-io-deutsche-rezepte.p.rapidapi.com",
        "Content-Type": "application/json"
    }
    response = requests.post(url, json=payload, headers=headers)

    data = response.json()

    # Zutatenliste formatieren
    ingredients_list = "Zutaten:\n"
    for ingredient in data['ingredients']:
        amount = ingredient.get('amount', '')
        unit = ingredient.get('unit', '')
        name = ingredient['name']
        ingredients_list += f"- {amount} {unit} {name}\n".strip() + '\n'

    # Zubereitungsschritte formatieren
    instructions_list = "\nZubereitung:\n"
    for step in data['instructions']:
        instructions_list += f"{step}\n"

    # Gesamtes Rezept formatieren
    formatted_recipe = f"{data['title']}\n"
    formatted_recipe += f"Portionen: {data['portions']}\n"
    formatted_recipe += f"Gesamtzeit: {data['totalTime'] // 60} Minuten\n\n"
    formatted_recipe += ingredients_list + instructions_list

    return formatted_recipe

title = "Recipifier"
description = "Discover the world of recipes effortlessly with Recipifier, using advanced AI trained on the extensive Food-101 dataset. Simply upload a photo of any dish, and our application identifies it accurately, providing detailed recipes and cooking instructions sourced from a vast collection. Perfect for food enthusiasts and home chefs alike, Recipifier makes exploring new culinary creations intuitive and inspiring. Start transforming everyday ingredients into extraordinary meals today!"

example_list = [["examples/" + example] for example in os.listdir("examples")]

css = """
#component-13 {
    display: none;
}
"""

demo = gr.Interface(
    fn=predict_image,
    inputs=gr.Image(type="pil"),
    outputs=[
        gr.Textbox(label="Meal", elem_id="meal"),
        gr.Textbox(label="Recipe") 
    ],
    examples=example_list,
    title=title,
    description=description,
    css=css,
    flagging_options=None  
)

demo.launch()

import requests
import json
import gradio as gr
import os

def hello(s):
    print("hello")
    print(getRecipe("hamburger")[0])

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

def getRecipe(meal):
    #meal = "hamburger"
    app_id = "24bf0913"
    app_key = "03c60f26520f9d25b0d0617e50993aaa"
    #field = ["label"]
    field = ["uri","label","image","ingredientLines","source","url"]
    url = "https://api.edamam.com/api/recipes/v2"



    #url2 = "https://api.edamam.com/api/recipes/v2?type=public&q=chicken%20curry&app_id=24bf0913&app_key=03c60f26520f9d25b0d0617e50993aaa"

    querystring = {"type":"public",
                    "q": meal.replace("_"," "),
                    "app_id": app_id,
                    "app_key": app_key,
                    "field":  field}


    response = requests.get(url, params=querystring)

    #print(response.content)

    json_object = response.json()

    json_formatted_str = json.dumps(json_object["hits"][0], indent=2) #nur das erste der 20 aus der liste

    #print(json_formatted_str)
    #print(json_object)

    #whole response
    #return json_object

    #just one result
    #return json_object["hits"][0]

    #just one result
    return json_object["hits"][0]["recipe"]["ingredientLines"]


demo.launch()
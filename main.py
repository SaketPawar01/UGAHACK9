import requests
import cv2
import dash
import pandas as pd
import numpy as np
from dash import dcc
from dash.dependencies import Input, Output
from dash import html
import base64
import json
import openai
from openai import OpenAI
import os

import logging

# log = logging.getLogger("werkzeug")
# log.setLevel(logging.ERROR)

client = OpenAI(api_key="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")
prompt = "I am going to give you a product which will be an alcoholic beverage. I want you to give me data on the alcohols: flavor, sweetness, alcohol percentage, price and facts. Return the data in JSON format. For flavors, if there are multiple flavors, list them in a list. For alcohol percentage, only return the integer. for sweetness, give a integer value from 1 to 5 where 1 is no sweet and 5 is extremely sweet. For price, give an upper and lower value of the typical price formatted with 2 decimal points. For facts, please give a description of the drink or any interesting fact about the drink. Please only return the JSON file and nothing else."

l1_cache = {}
black_list = {}
cap = cv2.VideoCapture(1)
bd = cv2.barcode.BarcodeDetector()
current_bar_code = ""
lookup_data = ""


def request_alc_data(alcohol_name):
    completion = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {
                "role": "system",
                "content": prompt,
            },
            {
                "role": "user",
                "content": alcohol_name,
            },
        ],
    )
    return completion.choices[0].message.content


try:
    with open("cache.json", "r") as f:
        l1_cache = json.load(f)
except:
    l1_cache = {}

try:
    with open("blacklist.json", "r") as f:
        black_list = json.load(f)
except:
    black_list = {}


def get_barcode_lookup(barcode):
    if barcode in black_list:
        return "No data found"
    headers = {
        "Content-Type": "application/json",
        "Accept": "application/json",
        "Accept-Encoding": "gzip,deflate",
    }

    if len(barcode) == 12:
        resp = requests.get(
            "https://api.upcitemdb.com/prod/trial/lookup?upc=0" + barcode,
            headers=headers,
        )
        data = json.loads(resp.text)
        print(data)
        if "items" not in data:
            black_list[barcode] = "No data found"
            return "No data found"
        return data["items"][0]["title"]
    else:
        black_list[barcode] = "No data found"
        return "No data found"


# create a dash application
app = dash.Dash(__name__)

# create a layout

app.layout = html.Div(
    [
        html.H1("Webcam Stream"),
        html.Div(html.Img(id="image")),
        html.P(
            id="live-update-text",
            children="No barcode detected",
            style={"whiteSpace": "pre", "margin-top": "10px", "font-size": "20px"},
        ),
        dcc.Interval(
            id="interval-component", interval=100, n_intervals=0  # in milliseconds
        ),
        dcc.Interval(
            id="slow-interval-component",
            interval=5000,
            n_intervals=0,  # in milliseconds
        ),
    ]
)


@app.callback(
    Output("image", "src"),
    Input("interval-component", "n_intervals"),
)
def update_image(n):
    global current_bar_code

    ret, frame = cap.read()
    if not ret:
        print("failed to grab frame")
        return None

    # detect and decode multipule barcodes to have sting and points
    ret, barcodes, points, _ = bd.detectAndDecodeMulti(frame)
    if ret:
        frame = cv2.polylines(frame, points.astype(int), True, (0, 255, 0), 3)
        for barcode in barcodes:
            if len(barcode) > 10:
                current_bar_code = barcode

    # rotate the frame
    frame = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
    ret, buffer = cv2.imencode(".jpg", frame)
    img = buffer.tobytes()
    return f"data:image/jpeg;base64,{base64.b64encode(img).decode()}"


@app.callback(
    Output("live-update-text", "children"),
    Input("slow-interval-component", "n_intervals"),
)
def update_metrics(n):
    global lookup_data
    global current_bar_code
    if (
        current_bar_code != None
        and current_bar_code != ""
        and len(current_bar_code) > 10
    ):
        if current_bar_code not in l1_cache:
            lookup_data = get_barcode_lookup(current_bar_code)
            if lookup_data != "No data found":
                l1_cache[current_bar_code] = {
                    "name": lookup_data,
                    "data": request_alc_data(lookup_data),
                }

        else:
            lookup_data = l1_cache[current_bar_code]
        return f"Barcode: {current_bar_code}\nLookup: {lookup_data}"
    else:
        return f"Barcode: {current_bar_code}\nLookup: {lookup_data}"
        # keep the last barcode


# run the app
if __name__ == "__main__":
    app.run_server(debug=False, port=8050)

    # on close
    with open("cache.json", "w") as f:
        json.dump(l1_cache, f)
    with open("blacklist.json", "w") as f:
        json.dump(black_list, f)
    cap.release()
    cv2.destroyAllWindows()

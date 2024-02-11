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

from dash import dcc, html, Input, Output, State
import plotly.graph_objs as go

import logging

log = logging.getLogger("werkzeug")
log.setLevel(logging.ERROR)

client = OpenAI(api_key="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")
prompt = "I am going to give you a product which will be an alcoholic beverage. I want you to give me data on the alcohols: flavor, sweetness, alcohol percentage, price, acidity, carbonation, and facts. Return the data in JSON format. For flavors, if there are multiple flavors, list them in a list. For alcohol percentage, only return the integer. For sweetness, give a integer value from 1 to 5 where 1 is no sweet and 5 is extremely sweet. For acidity, give a value from 1 to 5 where 1 is soft but 5 is highly acidic such as margaritas. For carbonation, give a value from 1 to 5 where 1 is a flat drink with no carbonations and 5 is a very fizzy drink like some beers. For price, give an upper and lower value of the typical price formatted with 2 decimal points. For facts, please give a description of the drink or any interesting fact about the drink. Please only return the JSON file and nothing else."

l1_cache = {}
black_list = {}
user_profiles = {
    "010": {
        "sweetness": 3,
        "alcohol_percentage": 40,
        "price_range": 10,
        "ph_level": 1,
        "fizz": 3,
    }
}
cap = cv2.VideoCapture(1)
bd = cv2.barcode.BarcodeDetector()
current_bar_code = ""
lookup_data = ""


def request_alc_data(alcohol_name):
    temp_dict = {"name": alcohol_name}
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
    drink_data = completion.choices[0].message.content
    drink_data = eval(drink_data.replace("\n", ""))
    for key in drink_data.keys():
        temp_dict[key] = drink_data[key]
    return temp_dict


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


def get_top_10_drinks():
    return [
        "Bud Light",
        "Smirnoff Vodka",
        "Jack Daniel's",
        "Heineken",
        "Jameson Irish Whiskey",
        "Cabernet Sauvignon",
        "Corona Extra",
        "Baileys Irish Cream",
        "Chardonnay",
        "Johnnie Walker",
    ]


external_stylesheets = [
    "https://stackpath.bootstrapcdn.com/bootstrap/4.5.2/css/bootstrap.min.css"
]

app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

app.layout = html.Div(
    [
        html.H1("BEVY", className="text-center"),
        html.Div(
            [
                html.Div(
                    html.Img(id="image", style={"max-width": "100%", "height": "auto"}),
                    style={"left": "-200px", "top": "30px"},
                ),
                html.Div(
                    [
                        html.H2("Drink Information", className="text-center mb-4"),
                        html.H3("Name"),
                        html.P(
                            "",
                            id="name",
                            className="font-size-20",
                            style={"height": "50px"},
                        ),
                        html.H3("Description"),
                        html.P(
                            "",
                            id="description",
                            className="font-size-20",
                            style={"height": "300px"},
                        ),
                        html.H3("Flavors"),
                        html.P(
                            "",
                            id="flavors",
                            className="font-size-20",
                            style={"height": "400px"},
                        ),
                        dcc.Interval(
                            id="interval-component",
                            interval=100,
                            n_intervals=0,  # in milliseconds
                        ),
                        dcc.Interval(
                            id="slow-interval-component",
                            interval=1000,
                            n_intervals=0,  # in milliseconds
                        ),
                    ],
                    className="col-md-3 mb-3",
                    style={"padding-left": "60px", "margin-left": "60px"},
                ),
                html.Div(
                    [
                        dcc.Graph(
                            id="radar-graph", style={"width": "100%", "height": "500px"}
                        ),
                        html.Div(
                            [
                                html.Div(
                                    [
                                        html.P("ABV %: ", id="alcohol-percentage"),
                                        html.P("Price: ", id="price"),
                                    ]
                                ),
                                dcc.Graph(
                                    id="score-guage",
                                    style={
                                        "width": "350px",
                                        "height": "350px",
                                        "float": "right",
                                    },
                                ),
                                # horizontally stack div
                            ],
                            style={"display": "flex", "flex-direction": "row"},
                        ),
                    ],
                    className="col-md-5 mb-3",
                ),
            ],
            className="row",
        ),
        html.Div(
            [
                html.H2("Select Your Alcohol Preferences"),
                html.Label("Select the type of alcohol:"),
                dcc.Dropdown(
                    id="drinks-dropdown",
                    options=[
                        {"label": drink, "value": drink}
                        for drink in get_top_10_drinks()
                    ],
                    multi=True,  # Enable multi-select
                    value=[],
                ),
                html.Br(),
                html.Label("Sweetness (1 to 5):"),
                dcc.Slider(
                    id="sweetness-slider",
                    min=1,
                    max=5,
                    value=3,
                    marks={i: str(i) for i in range(1, 6)},
                ),
                html.Br(),
                html.Label("Alcohol Percentage:"),
                dcc.Slider(
                    id="alcohol-percentage-slider",
                    min=0,
                    max=100,
                    step=1,
                    value=40,
                    marks={i: str(i) for i in range(0, 101, 10)},
                ),
                html.Br(),
                html.Label("Price Range:"),
                dcc.Slider(
                    id="price-range-slider",
                    min=0,
                    max=50,
                    step=10,
                    value=10,
                    marks={i: str(i) for i in range(0, 51, 10)},
                ),
                html.Br(),
                html.Label("Acidity Level:"),
                dcc.Slider(
                    id="ph-level-slider",
                    min=1,
                    max=5,
                    value=1,
                    marks={i: str(i) for i in range(1, 6)},
                ),
                html.Br(),
                html.Label("Fizz (1 to 5):"),
                dcc.Slider(
                    id="fizz-slider",
                    min=1,
                    max=5,
                    value=3,
                    marks={i: str(i) for i in range(1, 6)},
                ),
                html.Br(),
                html.Button("Save Profile", id="save-button", n_clicks=0),
                html.Button(
                    "Show Profile", id="show-profile-button", n_clicks=0
                ),  # Corrected: Ensure this is the button for showing the profile
                html.Div(
                    id="output-container"
                ),  # This div will show messages from saving the profile
                html.Div(
                    id="profile-display"
                ),  # This div is for displaying the profile
            ]
        ),
    ],
    style={"max-width": "95%", "margin": "auto"},
)


@app.callback(
    Output("output-container", "children"),
    [Input("save-button", "n_clicks")],
    [
        State("sweetness-slider", "value"),
        State("alcohol-percentage-slider", "value"),
        State("price-range-slider", "value"),
        State("ph-level-slider", "value"),
        State("fizz-slider", "value"),
    ],
)
def save_profile(n_clicks, sweetness, alcohol_percentage, price_range, ph_level, fizz):
    if n_clicks > 0:
        user_id = "010"  # Use a consistent user ID or mechanism to identify users
        global user_profiles
        user_profiles[user_id] = {
            "sweetness": sweetness,
            "alcohol_percentage": alcohol_percentage,
            "price_range": price_range,
            "ph_level": ph_level,
            "fizz": fizz,
        }
        return "Profile saved successfully!"


# Correct the callback decorator for show_user_profile
@app.callback(
    Output("profile-display", "children"),
    [Input("show-profile-button", "n_clicks")],
    prevent_initial_call=True,
)
def show_user_profile(n_clicks):
    user_id = "010"  # Use the same user ID used in save_profile
    if n_clicks > 0:
        if user_id in user_profiles:
            profile = user_profiles[user_id]
            profile_display = [
                html.Div(f"{key}: {value}") for key, value in profile.items()
            ]
            return profile_display
        else:
            return "No profile saved for this user."
    return []


def update_radar_graph(lookup_data):
    print(lookup_data)
    categories = ["Sweetness", "Alcohol %", "Price", "Carbonation", "Acidity"]
    values = [
        float(lookup_data["sweetness"]),
        float(lookup_data["alcohol_percentage"]) / 10,
        (float(lookup_data["price"]["lower"]) + float(lookup_data["price"]["upper"]))
        / (2 * 10),
        float(lookup_data["carbonation"]),
        float(lookup_data["acidity"]),
    ]

    trace = go.Scatterpolar(r=values, theta=categories, fill="toself")

    return {
        "data": [trace],
        "layout": go.Layout(
            polar=dict(radialaxis=dict(visible=True, range=[0, 5])),
            showlegend=False,
            title="Drink Qualities",
        ),
    }


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
    (
        Output("name", "children"),
        Output("description", "children"),
        Output("flavors", "children"),
        Output("alcohol-percentage", "children"),
        Output("price", "children"),
        Output("score-guage", "figure"),
        Output("radar-graph", "figure"),
    ),
    Input("slow-interval-component", "n_intervals"),
    prevent_initial_call=True,
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
                drink_data = request_alc_data(lookup_data)
                l1_cache[current_bar_code] = {
                    "name": lookup_data,
                }
                for key in drink_data.keys():
                    l1_cache[current_bar_code][key] = drink_data[key]

        lookup_data = l1_cache[current_bar_code]
        # concat list of flavors with commas
        flavors = ""
        for flavor in lookup_data["flavor"]:
            flavors += flavor + ", "
        flavors = flavors[:-2]

        profile = user_profiles["010"]
        # calculate the euclidean distance
        distance = np.sqrt(
            (profile["sweetness"] - lookup_data["sweetness"]) ** 2
            + (profile["alcohol_percentage"] - lookup_data["alcohol_percentage"]) ** 2
            + (profile["price_range"] - lookup_data["price"]["lower"]) ** 2
            + (profile["ph_level"] - lookup_data["acidity"]) ** 2
            + (profile["fizz"] - lookup_data["carbonation"]) ** 2
        )

        score = np.exp(-distance / 10) * 100
        fig = go.Figure(
            go.Indicator(
                mode="gauge+number",
                value=score,
                domain={"x": [0, 1], "y": [0, 1]},
                title={"text": "Your Score"},
                gauge={
                    "axis": {"range": [None, 100]},
                    "bar": {"color": "green"},
                },
            )
        )
        radar = update_radar_graph(lookup_data)

        return (
            lookup_data["name"],
            lookup_data["facts"],
            flavors,
            f"ABV: {lookup_data['alcohol_percentage']}",
            f"Price: ${lookup_data['price']['lower']} - ${lookup_data['price']['upper']}",
            fig,
            radar,
        )
    else:
        # radar = update_radar_graph(lookup_data)
        return "", "", None
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

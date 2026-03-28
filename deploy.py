from fastapi import FastAPI, HTTPException, Query, Form, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
import uvicorn
import pickle
import numpy as np

app = FastAPI(debug=True)

templates = Jinja2Templates(directory="templates")

with open("model.pkl", "rb") as f:
    model = pickle.load(f)


# define input schemas for post request
class BreastCancerRequest(BaseModel):

    radius_mean : float
    texture_mean : float
    perimeter_mean : float
    area_mean : float
    smoothness_mean : float
    compactness_mean : float
    concavity_mean : float
    concave_points_mean : float
    symmetry_mean : float
    fractal_dimension_mean : float

    radius_se : float
    texture_se : float
    perimeter_se : float
    area_se : float
    smoothness_se : float
    compactness_se : float
    concavity_se : float
    concave_points_se : float
    symmetry_se : float
    fractal_dimension_se : float

    radius_worst : float
    texture_worst : float
    perimeter_worst : float
    area_worst : float
    smoothness_worst : float
    compactness_worst : float
    concavity_worst : float
    concave_points_worst : float
    symmetry_worst : float
    fractal_dimension_worst : float


@app.get('/', response_class=HTMLResponse)
def home(request: Request):
    return templates.TemplateResponse(request=request, name="form.html")


# handle form submissions
@app.post("/predict", response_class=HTMLResponse)
def predict(request: Request, radius_mean: float = Form(...), texture_mean: float = Form(...), perimeter_mean: float = Form(...),
            area_mean: float = Form(...), smoothness_mean: float = Form(...), compactness_mean: float = Form(...),
            concavity_mean: float = Form(...), concave_points_mean: float = Form(...), symmetry_mean: float = Form(...),
            fractal_dimension_mean: float = Form(...), radius_se: float = Form(...), texture_se: float = Form(...),
            perimeter_se: float = Form(...), area_se: float = Form(...), smoothness_se: float = Form(...),
            compactness_se: float = Form(...), concavity_se: float = Form(...), concave_points_se: float = Form(...),
            symmetry_se: float = Form(...), fractal_dimension_se: float = Form(...), radius_worst: float = Form(...),
            texture_worst: float = Form(...), perimeter_worst: float = Form(...), area_worst: float = Form(...),
            smoothness_worst: float = Form(...), compactness_worst: float = Form(...), concavity_worst: float = Form(...),
            concave_points_worst: float = Form(...), symmetry_worst: float = Form(...), fractal_dimension_worst: float = Form(...),
            ):

    input_data =  np.array([
        [
            radius_mean, texture_mean, perimeter_mean, area_mean, smoothness_mean, compactness_mean,
            concavity_mean, concave_points_mean, symmetry_mean, fractal_dimension_mean, radius_se, texture_se, perimeter_se,
            area_se, smoothness_se, compactness_se, concavity_se, concave_points_se, symmetry_se, fractal_dimension_se,
            radius_worst, texture_worst, perimeter_worst, area_worst, smoothness_worst, compactness_worst, concavity_worst,
            concave_points_worst, symmetry_worst, fractal_dimension_worst
        ]
    ])


    prediction = model.predict(input_data)
    result = "Malignant" if prediction[0] == 1 else "Benign"

    return templates.TemplateResponse(request=request, name="result.html", context={"result": result})

if __name__ == "__main__":
    uvicorn.run("deploy:app", host="127.0.0.1", port=8000)


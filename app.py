from fastapi import FastAPI
import gradio as gr
import torch
import tiktoken
from pathlib import Path
from pydantic import BaseModel

from src.config import load_config
from src.model import setup_model
from src.predict import predict_language
from src.load_model import load_model

app = FastAPI(title="Language Classifier API")


model = None
tokenizer = None
device = None
cfg = None

class TextInput(BaseModel):
    text: str

@app.on_event("startup")
def load_ml_model():
    global model, tokenizer, device, cfg
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = tiktoken.get_encoding("gpt2")
    cfg = load_config(Path("config.json"))
    
    _ = load_model(cfg)
    model = setup_model(cfg, device, load_weights=True)
    model.eval()
    print("Model załadowany pomyślnie!")


@app.post("/predict")
def predict_api(input_data: TextInput):
    lang = predict_language(input_data.text, model, tokenizer, device, max_length=cfg.model.max_length)
    return {"text": input_data.text, "predicted_language": lang}

def gradio_predict(text):
    return predict_language(text, model, tokenizer, device, max_length=cfg.model.max_length).upper()

ui = gr.Interface(
    fn=gradio_predict,
    inputs=gr.Textbox(lines=5, placeholder="Text: "),
    outputs=gr.Label(label="Identified language (code)"),
    title="GPT-2 Language Identifier",
    description="Supported languages: Arabic (ar), Bulgarian (bg), German (de), Modern Greek (el), English (en), Spanish (es), French (fr), Hindi (hi), Italian (it), Japanese (ja), Dutch (nl), Polish (pl), Portuguese (pt), Russian (ru), Swahili (sw), Thai (th), Turkish (tr), Urdu (ur), Vietnamese (vi), Chinese (zh)"
)


app = gr.mount_gradio_app(app, ui, path="/ui")
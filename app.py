from flask import Flask, request, jsonify
from PIL import Image
import requests
from transformers import OFATokenizer, OFAModel
from torchvision import transforms
import torch

app = Flask(__name__)

# Configuração do modelo
ckpt = "OFA-Sys/ofa-huge-vqa"
tokenizer = OFATokenizer.from_pretrained(ckpt)
model = OFAModel.from_pretrained(ckpt)
model.eval()

transform = transforms.Compose([
    lambda image: image.convert("RGB"),
    transforms.Resize((480, 480), interpolation=Image.BICUBIC),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5]*3, std=[0.5]*3)
])

@app.route("/", methods=["GET"])
def index():
    return "OFA VQA API está online!"

@app.route("/analisar", methods=["POST"])
def analisar():
    data = request.json
    url_imagem = data.get("image_url")
    pergunta = data.get("question")

    if not url_imagem or not pergunta:
        return jsonify({"erro": "Faltam parâmetros 'image_url' ou 'question'"}), 400

    try:
        image = Image.open(requests.get(url_imagem, stream=True).raw)
        patch_image = transform(image).unsqueeze(0)

        inputs = tokenizer([pergunta], return_tensors="pt").input_ids
        generated = model.generate(inputs, patch_images=patch_image, num_beams=5, no_repeat_ngram_size=3)
        resposta = tokenizer.batch_decode(generated, skip_special_tokens=True)[0]

        return jsonify({"resposta": resposta})

    except Exception as e:
        return jsonify({"erro": str(e)}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)

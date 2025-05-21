from transformers import OFATokenizer, OFAModel

print("Iniciando download do modelo...")

tokenizer = OFATokenizer.from_pretrained("OFA-Sys/ofa-base")
model = OFAModel.from_pretrained("OFA-Sys/ofa-base")

print("Download concluído.")

import torch
import clip

device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/16", device=device)
text = clip.tokenize(["a computerized tomography background without the Liver, Kidneys, Spleen and Pancreas", "a computerized tomography of a Liver", "a computerized tomography of Kidneys", "a computerized tomography of a Spleen", "a computerized tomography of a Pancreas"]).to(device)
print(text.shape) # torch.Size([5, 77])
text_features = model.encode_text(text)
print(text_features.shape) # torch.Size([5, 512])
torch.save(text_features, "txt_encoding.pth")

device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/16", device=device)
text = clip.tokenize(["a magnetic resonance imaging delay sequence background without the Liver, Kidneys, Spleen and Pancreas", "a magnetic resonance imaging delay sequence of a Liver", "a magnetic resonance imaging delay sequence of Kidneys", "a magnetic resonance imaging delay sequence of a Spleen", "a magnetic resonance imaging delay sequence of a Pancreas"]).to(device)
print(text.shape) # torch.Size([5, 77])
text_features = model.encode_text(text)
print(text_features.shape) # torch.Size([5, 512])
torch.save(text_features, "txt_encoding_delay.pth")
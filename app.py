
from jarvis_main import iface  # your Jarvis UI

import gradio as gr
import torch
from PIL import Image
import torchvision.transforms as transforms
from huggingface_hub import hf_hub_download
import timm

# ---------------- Load Model ---------------- #
model_path = hf_hub_download(
    repo_id="JanaBot/face-recognition-model",
    filename="my_model.pt"
)


# Allowlist all timm classes used in your model
torch.serialization.add_safe_globals([
    timm.models.vision_transformer.VisionTransformer,
    timm.layers.patch_embed.PatchEmbed
])
model = torch.load(model_path, map_location=torch.device('cpu'), weights_only=False)
model.eval()

preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])



# ---------------- Face Recognition Function ---------------- #
def run_face_recognition(image, username=None):
    img = Image.open(image)
    img_tensor = preprocess(img).unsqueeze(0)  # add batch dimension
    with torch.no_grad():
        output = model(img_tensor)
        prediction = output.argmax().item()  # assuming 0=denied, 1=approved
    return "approved" if prediction == 1 else "denied"

# ---------------- Gradio UI ---------------- #
def check_access(image, username):
    status = run_face_recognition(image, username)
    if status == "approved":
        return (
            "✅ Access Granted. Launching Jarvis...",
            gr.update(visible=False),  # hide login
            gr.update(visible=True)    # show Jarvis
        )
    else:
        return (
            "🚫 Access Denied. Try again.",
            gr.update(visible=True),   # keep login visible
            gr.update(visible=False)   # keep Jarvis hidden
        )

# ---------------- Combined UI ---------------- #
with gr.Blocks() as app:
    gr.Markdown("## 🔐 Face Recognition Login")

    # login section with image & username side by side
    with gr.Row(visible=True) as login_ui:
        with gr.Column():
            image_in = gr.Image(type="filepath", label="Upload Face Image")
        with gr.Column():
            user_in = gr.Textbox(label="Enter Username")
            status_box = gr.Textbox(label="Status", interactive=False)
            btn = gr.Button("Verify Access")

    # jarvis section (hidden at first)
    with gr.Row(visible=False) as jarvis_ui:
        iface.render()

    btn.click(
        fn=check_access,
        inputs=[image_in, user_in],
        outputs=[status_box, login_ui, jarvis_ui],
    )

if __name__ == "__main__":
    app.launch()

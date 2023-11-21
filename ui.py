import gradio as gr
import os

import torch
from src.models import ImageCaptionModule
from torchvision import transforms as T
from src.utils.decode import greedy_search, batch_greedy_search, beam_search_decoding


checkpoint = 'logs/train/runs/2023-11-15_13-32-11/checkpoints/epoch_019.ckpt'
device = torch.device("cuda")

model = ImageCaptionModule.load_from_checkpoint(checkpoint)
model = model.eval().to(device)

transform = T.Compose([
    T.ToTensor(),
    T.Resize([299, 299], antialias=True),  # using inception_v3 to encode image
])


def inference(image, search_type):
    image = transform(image).unsqueeze(0).to(device)
    print('Image shape:', image.shape)

    if search_type == 'greedy':
        search = greedy_search
    elif search_type == 'batch_greedy':
        search = batch_greedy_search
    elif search_type == 'beam':
        search = beam_search_decoding
    # elif search == 'batch_beam':
    #     search = batch_beam_search_decoding
    else:
        raise NotImplementedError(f"unknown search: {search}")
    
    preds = search(model=model.net, images=image)[0]
    
    if search_type == 'greedy':
        return preds, '', '', '', ''
    else:
        captions = preds.split('|')
        return captions[0], captions[1], captions[2], captions[3], captions[4]


demo = gr.Interface(inference,
                    inputs=[
                        gr.Image(type="pil"), 
                        gr.Radio(["greedy", "beam"], label="search_type", value="greedy"),
                    ],
                    outputs=["text", "text", "text", "text", "text"],
                    examples=[
                        [os.path.join(os.path.dirname(__file__), "images/ex1.jpg"), "greedy"],
                        [os.path.join(os.path.dirname(__file__), "images/ex2.jpg"), "greedy"],
                        [os.path.join(os.path.dirname(__file__), "images/ex3.jpg"), "greedy"],
                        [os.path.join(os.path.dirname(__file__), "images/ex4.jpg"), "greedy"],
                        [os.path.join(os.path.dirname(__file__), "images/ex5.jpg"), "greedy"],
                    ],
                    )

if __name__ == "__main__":
    demo.launch()

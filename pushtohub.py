from diffusers import ControlNetModel, UNet2DModel
import torch


controlnet = ControlNetModel.from_pretrained("/home/wisley/train_controlnet/output/checkpoint-8000/controlnet")



controlnet.push_to_hub("JaspervanLeuven/controlnet_rect")
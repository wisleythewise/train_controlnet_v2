
Clone the repo
```
git clone https://github.com/wisleythewise/train_controlnet.git
```

Create a new enviroment

```
conda create --name training_controlnet python=3.9
conda activate training_controlnet
```

Install dependencies

```
pip install git+https://github.com/huggingface/diffusers.git transformers accelerate xformers wandb datasets torchvision torch matplotlib

```

Login to huggingface and pass you acces token using the following command, we use huggingface to push the model
```
huggingface-cli login
```

Login to wanddb and pass your acces token using the following command, we use wanddb to check the progress of the training

```
wandb login 
```


run the training script make sure you have provided the right paths

```
accelerate launch train_controlnet.py \
 --pretrained_model_name_or_path="runwayml/stable-diffusion-v1-5" \
 --resume_from_checkpoint "latest" \
 --controlnet_model_name_or_path="lllyasviel/sd-controlnet-seg" \
 --output_dir="/home/wisley/train_controlnet/output" \
 --train_data_dir="/mnt/d/seg_dataset_full" \
 --conditioning_image_column="conditioning_images_one" \
 --image_column="ground_truth" \
 --caption_column="caption" \
 --hub_model_id="JaspervanLeuven/controlnet_rect" \
 --resolution=512 \
 --learning_rate=1e-5 \
 --train_batch_size=16 \
 --validation_steps=200\
 --num_train_epochs=1 \
 --tracker_project_name="controlnet_rect" \
 --checkpointing_steps=2000 \
 --report_to="wandb" \
 --push_to_hub \
 --set_grads_to_none \
 --gradient_checkpointing \
 --mixed_precision="fp16" \
 --cache_dir="/home/wisley/train_controlnet/cache/" \
 --validation_prompt="A driving scene"\
 --validation_image="/home/wisley/train_controlnet/validation.png"\
```

After training you can run inference in the model using the inference.py file specifying the hub_model_id


testing the new hypothesis 

```
accelerate launch train_controlnet.py \
 --pretrained_model_name_or_path="runwayml/stable-diffusion-v1-5" \
 --output_dir="/home/wisley/train_controlnet/output_test" \
 --dataset_name="JaspervanLeuven/prescan_segmentation" \
 --conditioning_image_column="conditioning_images_one" \
 --image_column="ground_truth" \
 --caption_column="caption" \
 --hub_model_id="JaspervanLeuven/controlnet_faulty_data" \
 --resolution=512 \
 --learning_rate=1e-5 \
 --train_batch_size=4 \
 --validation_steps=200\
 --num_train_epochs=4 \
 --tracker_project_name="controlnet_faulty_data" \
 --checkpointing_steps=2000 \
 --report_to="wandb" \
 --push_to_hub \
 --set_grads_to_none \
 --gradient_checkpointing \
 --mixed_precision="no" \
 --cache_dir="/home/wisley/train_controlnet/cache/" \
 --validation_prompt="A driving scene"\
 --validation_image="/home/wisley/train_controlnet/validation.png"\
```
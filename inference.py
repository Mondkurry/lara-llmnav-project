import os
os.environ['CUDA_VISIBLE_DEVICES'] = '3'

import torch
from PIL import Image
from torchvision import transforms
from transformers import AutoTokenizer, InstructBlipForConditionalGeneration, InstructBlipProcessor

# Load the trained model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = InstructBlipForConditionalGeneration.from_pretrained('/home/aryan/Desktop/lara/(InstructBLIP) map based path planning/lara-llmav/final_model copy').to(device)

# Initialize the tokenizer and processor
tokenizer = AutoTokenizer.from_pretrained('Salesforce/instructblip-flan-t5-xl')
processor = InstructBlipProcessor.from_pretrained("Salesforce/instructblip-flan-t5-xl")

# Image preprocessing transformations
image_transform = transforms.Compose([
    transforms.Resize((650, 650)),
    transforms.ToTensor()
])

def preprocess_data(image_path, prompt):
    # Load and transform the image
    image = Image.open(image_path).convert('RGB')
    image = image_transform(image)

    # Process the image and prompt
    pixel_values = processor(image, return_tensors="pt").pixel_values.to(device)
    prompt_encoding = tokenizer(prompt, return_tensors="pt").to(device)

    return pixel_values, prompt_encoding

def infer(model, pixel_values, prompt_encoding):
    # Extract input_ids and attention_mask from prompt_encoding
    input_ids = prompt_encoding['input_ids'].to(device)
    attention_mask = prompt_encoding['attention_mask'].to(device)
    
    print(input_ids)
    print(attention_mask)
    print(pixel_values)

    # Perform inference
    model.eval()
    with torch.no_grad():
        outputs = model.generate(pixel_values=pixel_values, input_ids=input_ids, attention_mask=attention_mask)
    return outputs

def postprocess(outputs):
    # Convert model output to text
    directions = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return directions

# Example usage
image_path = '/home/aryan/Desktop/lara/(InstructBLIP) map based path planning/lara-llmav/5 - Inference/testImage.png'
prompt = '''
Image input is a map with lines representing streets and nodes representing intersections. A starting point is indicated by a green marker, and a destination is marked with a red marker. The streets are labeled with their respective names.

Analyze the layout of streets and intersections and provide a set of directions from the starting point to the destination. Directions should be a list of actions, like 'turn right', 'turn left', or 'go straight', and should reference the street names. Format the response as a series of directional steps. Ensure route is clear and logical for someone who would follow these directions in real life.
'''


pixel_values, prompt_encoding = preprocess_data(image_path, prompt)
outputs = infer(model, pixel_values, prompt_encoding)
directions = postprocess(outputs)

print("Directions:", directions)
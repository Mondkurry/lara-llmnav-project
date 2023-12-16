import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

from PIL import Image
from torchvision import transforms
from transformers import AutoTokenizer, TrainingArguments, Trainer,InstructBlipForConditionalGeneration, InstructBlipProcessor
from tqdm import tqdm
import json
import torch
from torch.utils.data import Dataset, DataLoader

# Define the dataset directory structure
# root_dir = '/Users/aryanmondkar/Desktop/CS Gross/LARA/lara-llmav/1 - dataset' # mac
# root_dir = 'C:/Users/Aryan Mondkar/Desktop/Code/lara/lara-llmav/1 - dataset' # windows
root_dir = '/home/aryan/Desktop/lara/(InstructBLIP) map based path planning/lara-llmav/data_test' # Linux

device = torch.device('cuda')

cities = [
    'San Francisco, California, USA', 
    'New York City, New York, USA',
    'Los Angeles, California, USA',
    'Austin, Texas, USA',
    'Chicago, Illinois, USA',
    'Boston, Massachusetts, USA',
    'San Diego, California, USA',
    'San Jose, California, USA',
    'Miami, Florida, USA',
    ]  
prompt = '''
Image input is a map with lines representing streets and nodes representing intersections. A starting point is indicated by a green marker, and a destination is marked with a red marker. The streets are labeled with their respective names.

Analyze the layout of streets and intersections and provide a set of directions from the starting point to the destination. Directions should be a list of actions, like 'turn right', 'turn left', or 'go straight', and should reference the street names. Format the response as a series of directional steps. Ensure route is clear and logical for someone who would follow these directions in real life.
'''


# Image preprocessing transformations
image_transform = transforms.Compose([
    transforms.Resize((650, 650)),
    transforms.ToTensor()
])

# Initialize the tokenizer for the InstructBLIP model
tokenizer = AutoTokenizer.from_pretrained('Salesforce/instructblip-flan-t5-xl')

# Custom dataset class for InstructBLIP
class InstructBLIPDataset(Dataset):
    def __init__(self, root_dir, cities, processor, transform=None):
        self.root_dir = root_dir
        self.tokenizer = processor.tokenizer
        self.transform = transform
        self.vision_processor = processor.image_processor
        self.samples = []

        # Load the dataset
        self._load_dataset(cities)

    def _load_dataset(self, cities):
        # Loop through each city directory
        for city in cities:
            city_dir = os.path.join(self.root_dir, city)
            images_dir = os.path.join(city_dir, 'images')
            directions_file = os.path.join(city_dir, 'directions', f'{city}_directions.json')
            
            # print(city,'\n' ,city_dir ,'\n' ,images_dir ,'\n' , directions_file,'\n\n') # For debugging
            # Ensure it's a file
            # print('Check JSON is a File:', os.path.isfile(directions_file))
            # Ensure it's a file
            if os.path.isfile(directions_file):
                # Read the JSON file
                with open(directions_file, 'r') as f:
                    directions_data = json.load(f)

                # For each entry in the JSON, get the corresponding image
                for entry in tqdm(directions_data, desc=f"Loading {city} dataset"):
                    image_path = os.path.join(images_dir, entry['image_name'])
                    if os.path.isfile(image_path):  # Ensure the image file exists
                        self.samples.append((image_path, entry['directions']))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        image_path, directions = self.samples[idx]

        # Load the image
        image = Image.open(image_path).convert('RGB')
        
        # Apply the torchvision transforms
        if self.transform:
            image = self.transform(image)

        # Convert the transformed image to a BatchFeature
        pixel_values = self.vision_processor(images=image, return_tensors="pt").pixel_values.squeeze(0) 
        
        prompt_str = prompt
        prompt_encoding = self.tokenizer(prompt_str, padding='max_length', truncation=True, max_length=105, return_tensors='pt')
        
        prompt_ids = prompt_encoding['input_ids'].squeeze(0)  # Remove batch dimension
        prompt_mask = prompt_encoding['attention_mask'].squeeze(0)  # Remove batch dimension
        
        # Tokenize the ground truth directions
        directions_txt = ' '.join(directions)  # Join the list of directions into a single string
        directions_encoding = self.tokenizer(directions_txt, padding='max_length', truncation=True, max_length=105, return_tensors='pt')
        directions_ids = directions_encoding['input_ids'].squeeze(0)  # Remove batch dimension
        directions_mask = directions_encoding['attention_mask'].squeeze(0)  # Remove batch dimension
        
        # print(f'Loading image: {image_path}')
        # return_dict = {
        #     'Loading image': image_path,
        #     'pixel_values_shape': pixel_values.shape,
        #     'input_ids_shape': prompt_ids.shape,
        #     'qformer_input_ids_shape': prompt_ids.shape,
        #     'decoder_input_ids_shape': directions_ids.shape,
        # }
        # print(return_dict, "\n\n")
        
        return {
            'pixel_values': pixel_values,
            'input_ids': prompt_ids, 
            'attention_mask': prompt_mask,
            'qformer_input_ids': prompt_ids,
            'qformer_attention_mask': prompt_mask,
            'decoder_input_ids': directions_ids,
            'decoder_attention_mask': directions_mask,
            'label_ids': directions_ids,
        }

        
if __name__ == '__main__':
    # Instantiate the dataset
    InstructBLIP_processor = InstructBlipProcessor.from_pretrained("Salesforce/instructblip-flan-t5-xl") # Aryan Added
    dataset = InstructBLIPDataset(root_dir=root_dir, cities=cities, processor=InstructBLIP_processor, transform=image_transform)

    # Check the first sample to ensure it's loaded correctly
    print(len(dataset))
    first_sample = dataset[0]
    print(f"First sample image shape: {first_sample['pixel_values'].shape}")
    print(f"First sample input_ids shape: {first_sample['qformer_input_ids'].shape}")
    print(f"First sample attention_mask shape: {first_sample['attention_mask'].shape}")

    # Split the dataset into train and test sets
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])

    # Create the DataLoaders for training and testing
    train_loader = DataLoader(train_dataset, batch_size=600, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=600, shuffle=False)

    ######### TRAINING #########

    # Load InstructBLIP model
    model = InstructBlipForConditionalGeneration.from_pretrained('Salesforce/instructblip-flan-t5-xl').to(device)

    for name, param in model.named_parameters():
        if "qformer" not in name:
            param.requires_grad = False

    # Define training arguments
    training_args = TrainingArguments(
        per_device_train_batch_size=16,  # Adjust as per your GPU memory change to 500
        per_device_eval_batch_size=16,   # Adjust as per your GPU memory change to 500
        num_train_epochs=10,
        
        evaluation_strategy='steps',
        logging_dir='./logs',
        logging_strategy = 'steps',
        logging_steps=100,
        eval_steps=500,
        
        do_train=True,
        do_eval=True,
        
        warmup_steps=50,
        weight_decay=0.01,

        output_dir="./model_output",
        save_strategy="steps",
        save_steps=500,
        save_total_limit=4,  # Only save the last 4 checkpoints
        load_best_model_at_end=True,

        metric_for_best_model='loss', 
        dataloader_num_workers=4,
        ddp_find_unused_parameters=False,
        save_safetensors=False,
        resume_from_checkpoint='/home/aryan/Desktop/lara/(InstructBLIP) map based path planning/lara-llmav/model_output/checkpoint-3500',
        report_to="tensorboard"
    )


    # Initialize Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        tokenizer=tokenizer
    )

    # Start training
    trainer.train(resume_from_checkpoint='/home/aryan/Desktop/lara/(InstructBLIP) map based path planning/lara-llmav/model_output/checkpoint-3500')
    
    trainer.evaluate()

    # Save the final model
    trainer.save_model('./final_model')

    # Evaluate the final model
import os
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from transformers import SamProcessor, SamModel
from tqdm.auto import tqdm
from transformers import SamProcessor, SamModel
from torch.optim import Adam
from torch.nn import BCEWithLogitsLoss

class CustomSegmentationDataset(Dataset):
    def __init__(self, images_dir, masks_dir, transform=None):
        self.images_dir = images_dir
        self.masks_dir = masks_dir
        self.transform = transform
        
        # List of image files
        self.image_files = sorted(os.listdir(images_dir))
        self.mask_files = sorted(os.listdir(masks_dir))
        
        # Filter out images without masks
        self.image_files, self.mask_files = zip(*[
            (img, msk) for img, msk in zip(self.image_files, self.mask_files)
            if os.path.isfile(os.path.join(masks_dir, msk))
        ])
        
    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        # Load image and mask
        img_path = os.path.join(self.images_dir, self.image_files[idx])
        mask_path = os.path.join(self.masks_dir, self.mask_files[idx])
        
        image = Image.open(img_path).convert('RGB')
        mask = Image.open(mask_path).convert('L')
        
        # Apply transformations
        if self.transform:
            image = self.transform(image)
            mask = self.transform(mask)
        
        # Convert mask to binary tensor
        mask = (mask > 0.5).float()
        
        # Get bounding box
        bbox = self.get_bounding_box(mask)
        
        return image, mask, bbox
    
    @staticmethod
    def get_bounding_box(mask):
        import numpy as np
        mask_np = mask.numpy().squeeze()
        y_indices, x_indices = np.where(mask_np > 0)
        if len(x_indices) == 0 or len(y_indices) == 0:
            # Return default bbox if mask is empty
            return [0, 0, mask.shape[2], mask.shape[1]]
        x_min, x_max = x_indices.min(), x_indices.max()
        y_min, y_max = y_indices.min(), y_indices.max()
        return [x_min, y_min, x_max, y_max]

def train_custom_model(
    images_dir: str,
    masks_dir: str,
    model_save_path: str,
    processor_save_path: str,
    num_epochs: int = 10,
    batch_size: int = 2,
    learning_rate: float = 1e-5,
    device: str = 'cpu'
):
    """
    Trains a custom SAM model using the provided dataset.

    Args:
        images_dir (str): Directory containing input images.
        masks_dir (str): Directory containing segmentation masks.
        model_save_path (str): Path to save the trained model.
        processor_save_path (str): Path to save the trained processor.
        num_epochs (int): Number of training epochs.
        batch_size (int): Batch size for training.
        learning_rate (float): Learning rate for optimizer.
        device (str): 'cpu' or 'cuda' for GPU acceleration.
    """
    # Define transformations
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
    ])

    # Create dataset and dataloader
    dataset = CustomSegmentationDataset(images_dir, masks_dir, transform=transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Initialize processor and model
    processor = SamProcessor.from_pretrained("facebook/sam-vit-base")
    model = SamModel.from_pretrained("facebook/sam-vit-base")

    # Freeze the vision and prompt encoders
    for param in model.vision_encoder.parameters():
        param.requires_grad = False
    for param in model.prompt_encoder.parameters():
        param.requires_grad = False

    # Move model to device
    device = torch.device(device)
    model.to(device)

    # Define loss function and optimizer
    criterion = BCEWithLogitsLoss()
    optimizer = Adam(model.mask_decoder.parameters(), lr=learning_rate)

    # Training loop
    model.train()
    for epoch in range(num_epochs):
        epoch_loss = 0.0
        for images, masks, bboxes in tqdm(dataloader, desc=f'Epoch {epoch+1}/{num_epochs}'):
            images = images.to(device)
            masks = masks.to(device)
            # Prepare inputs
            inputs = processor(images=images, input_boxes=[bboxes], return_tensors="pt").to(device)

            # Forward pass
            outputs = model(**inputs, multimask_output=False)
            pred_masks = outputs.pred_masks.squeeze(1)

            # Resize ground truth masks to match predicted masks
            gt_masks = torch.nn.functional.interpolate(masks.unsqueeze(1), size=pred_masks.shape[-2:], mode='nearest')

            # Compute loss
            loss = criterion(pred_masks, gt_masks.float())

            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        avg_loss = epoch_loss / len(dataloader)
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}')

    # Save the trained model and processor
    model.save_pretrained(model_save_path)
    processor.save_pretrained(processor_save_path)
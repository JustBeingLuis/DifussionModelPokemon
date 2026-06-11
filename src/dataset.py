import os
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from config import DATA_RAW_DIR, IMAGE_SIZE, BATCH_SIZE

class PokemonDataset(Dataset):
    def __init__(self, data_dir=DATA_RAW_DIR):
        self.data_dir=data_dir

        self.image_paths = [os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.endswith('.png')]

        self.transform = transforms.Compose([
            transforms.Resize((IMAGE_SIZE,IMAGE_SIZE)),
            transforms.ToTensor(),
            transforms.Lambda(lambda t: (t*2)-1)
        ])
    
    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self,idx):
        img_path = self.image_paths[idx]
        img = Image.open(img_path).convert("RGB")

        return self.transform(img)

def get_dataloader():
    dataset = PokemonDataset()
    return DataLoader(dataset=dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)

if __name__ == "__main__":
    loader = get_dataloader()

    batch= next(iter(loader))

    print(f"Dimensiones del lote (Batch): {batch.shape}")
    print(f"Valor numérico máximo: {batch.max():.4f}, mínimo: {batch.min():.4f}")



import numpy as np
import matplotlib.pyplot as plt
import torch
from torchvision import datasets, transforms
# Define how each image should be transformed
transform = transforms.Compose([
    lambda img: transforms.functional.rotate(img, -90),
    transforms.RandomHorizontalFlip(p=1),
    transforms.ToTensor()
])
# Loading the EMNIST dataset
data = datasets.EMNIST(
    root="~/data",       
    split="balanced",
    download=False,      
    transform=transform
)

num_classes = len(np.unique(data.targets.numpy()))
rows = 5
cols = num_classes
plt.figure(figsize=(cols * 0.5, rows * 0.8))
# Collecting indices of images belonging to each class
indices_by_class = {c: torch.where(data.targets == c)[0] for c in range(num_classes)}
# Plotting 5 random samples for each class
for c in range(num_classes):
    chosen = np.random.choice(indices_by_class[c], rows, replace=False)
    for r in range(rows):
        ax = plt.subplot(rows, cols, r * cols + c + 1)
        img, _ = data[chosen[r].item()]
        plt.imshow(img.squeeze(), cmap='gray')
        plt.axis('off')
        if r == 0:
            ax.set_title(str(c), fontsize=6, pad=1)
plt.tight_layout(pad=0.1)
plt.savefig("emnist_5x47_visualization.png", dpi=200, bbox_inches="tight")
plt.show()

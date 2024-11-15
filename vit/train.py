# %%
from datasets import load_dataset, Dataset
from torchvision.transforms import Compose, ToTensor
from torch.utils.data import DataLoader
from vit import VisionTransformer

def get_cifar_label_dicts(cifar_dataset:Dataset, label_type:str):
    """
    label_type: str 'coarse' or 'fine'
    returns
        label2id: dict
        id2label: dict
    """
    assert label_type in ("coarse","fine"), "label type must be 'coarse' or 'fine'"
    # Get label mappings
    labels = cifar_dataset["train"].features[f"{label_type}_label"].names

    label2id = {str(label): str(i) for i, label in enumerate(labels)}
    id2label = {str(i): str(label) for i, label in enumerate(labels)}

    return label2id, id2label

def get_cifar_dataloaders(batch_size=2):
    # Load the CIFAR dataset
    cifar = load_dataset("uoft-cs/cifar100")
    cifar_train = cifar["train"]
    cifar_test = cifar["test"]

    # Define transforms
    _transforms = Compose([ToTensor()])  # Add more transforms as needed to prevent overfitting

    def preprocess_transforms(data_examples):
        data_examples["pixel_values"] = [_transforms(img) for img in data_examples["img"]]
        del data_examples["img"]
        return data_examples

    # Apply transformations
    cifar_train = cifar_train.with_transform(preprocess_transforms)
    cifar_test = cifar_test.with_transform(preprocess_transforms)

    # Create dataloaders
    train_dataloader = DataLoader(
        dataset=cifar_train,
        batch_size=batch_size,
        shuffle=True
    )

    test_dataloader = DataLoader(
        dataset=cifar_test,
        batch_size=batch_size,
        shuffle=True
    )

    return train_dataloader, test_dataloader

train_dataloader, test_dataloader = get_cifar_dataloaders(batch_size=4)

cifar = load_dataset("uoft-cs/cifar100")
label2id, id2label = get_cifar_label_dicts(cifar, 'coarse')

num_clases_coarse = len(label2id.keys())

batch = next(iter(train_dataloader))

batch["pixel_values"].shape

vit = VisionTransformer(
    image_size=32, use_linear_patch=True, num_classes=num_clases_coarse)

pred = vit(batch["pixel_values"])
# # %%
# from transformers import AutoModelForImageClassification

# model = AutoModelForImageClassification.from_pretrained("google/vit-base-patch16-224")

# # %%
# from transformers import AutoImageProcessor

# image_processor = AutoImageProcessor.from_pretrained("google/vit-base-patch16-224")



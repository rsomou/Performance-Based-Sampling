import torchvision

def get_dataset_cifar(path):
    label_dict = {}
    count_dict = {}
    count_dict["train"] = {}
    count_dict["valid"] = {}
    train_dataset = torchvision.datasets.CIFAR100(root=path, download=True, train=True)
    val_dataset = torchvision.datasets.CIFAR100(root=path, train=False)
    for i in range(100):
        label_dict[i] = []
        count_dict["train"][i] = 0
        count_dict["valid"][i] = 0
    dataset = {"train": [], "valid": []}
    for img, lab in train_dataset:
        dataset["train"].append({"image": img, "label": lab})
        count_dict["train"][lab] += 1
    for img, lab in val_dataset:
        dataset["valid"].append({"image": img, "label": lab})
        count_dict["valid"][lab] += 1
    return dataset, None, count_dict

import torch
from torch.utils.data import Dataset, DataLoader
from data_aug.view_generator import ContrastiveLearningViewGenerator
from torchvision.transforms import transforms
import os
from data_aug.gaussian_blur import GaussianBlur
from PIL import Image

class RockData(Dataset):

    def __init__(self, root, image_paths, labels, n_views, merge_label=True):
        self.root = root
        self.image_paths = image_paths
        self.labels = labels
        self.transform = ContrastiveLearningViewGenerator(
            self.get_simclr_pipeline_transform(224), n_views
            )
        self.merge_label = merge_label
    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        label = self.labels[image_path]
        # 14标签合为3个
        if self.merge_label:
            if label >= 0 and label <= 4: label = 0
            elif label > 4 and label <= 7: label = 1
            elif label > 7 and label <= 14: label = 2
        image = Image.open(os.path.join(self.root,image_path))
        # 将图像缩放到224中心
        image_input = self.transform(image)
        label = torch.tensor(label).long()
        return image_input, label
    
    def __len__(self):
        return len(self.image_paths)
    
    @staticmethod
    def get_simclr_pipeline_transform(size, s=1):
        """Return a set of data augmentation transformations as described in the SimCLR paper."""
        color_jitter = transforms.ColorJitter(0.8 * s, 0.8 * s, 0.8 * s, 0.2 * s)
        data_transforms = transforms.Compose([transforms.RandomResizedCrop(size=size),
                                              transforms.RandomHorizontalFlip(),
                                              transforms.RandomApply([color_jitter], p=0.8),
                                              transforms.RandomGrayscale(p=0.2),
                                              GaussianBlur(kernel_size=int(0.1 * size)),
                                              transforms.ToTensor()])
        return data_transforms

if __name__ == '__main__':
    from pyn import Json
    labels_path = 'D:\MyDoc\Program\MachineLearning\岩石分类\主程序\labels.json'
    permute = 'D:\MyDoc\Program\MachineLearning\岩石分类\主程序\permute_0.json'
    labels = Json.load(labels_path)
    image_paths = Json.load(permute)

    train_paths  = {}
    train_paths.update(image_paths['train_data']), train_paths.update(image_paths['test_data'])
    root = 'D:\MyDoc\Program\MachineLearning\岩石分类\主程序'
    train_dataset = RockData(root, train_paths, labels, 2)
    batch_size = 32
    train_dataloader = DataLoader(train_dataset, batch_size, True)
    for img, label in train_dataloader:
        # print(img.shape, label.shape)
        print(img.shape)
        assert 1 == 0
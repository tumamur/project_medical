from Perceptual_xray import PerceptualLossXray
from Perceptual_xrays import Perceptual_xray
from PIL import Image
from torchvision import transforms

#transform = transforms.ToTensor()
transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=3),  # Convert to 3-channel grayscale
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])
test_image_path = "/home/max/Desktop/MLMI/data/mimic-cxr-jpg/files/p10/p10000032/s50414267/02aa804e-bde0afdd-112c0b34-7bc16630-4e384014.jpg"
gt_image_path = "/home/max/Desktop/MLMI/data/mimic-cxr-jpg/files/p10/p10000764/s57375967/096052b7-d256dc40-453a102b-fa7d01c6-1b22c6b4.jpg"
test_image = Image.open(test_image_path)
test_image = transform(test_image)
gt_image = Image.open(gt_image_path)
gt_image = transform(gt_image)
test_image = test_image.unsqueeze(0)  # Add a batch dimension
gt_image = gt_image.unsqueeze(0)      # Add a batch dimension
print(gt_image.shape)

# perceptual = PerceptualLossXray()
perceptual = Perceptual_xray()
loss_0 = perceptual(gt_image, gt_image)
loss_1 = perceptual(gt_image, test_image)
loss_2 = perceptual(test_image, test_image)
print(f'gt_vs_gt: {loss_0}')
print(f'gt_vs_test: {loss_1}')
print(f'gt_vs_test: {loss_2}')

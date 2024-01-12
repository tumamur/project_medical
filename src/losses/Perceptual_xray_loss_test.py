from Perceptual_xray import PerceptualLossXray
from Perceptual_xrays import Perceptual_xray
from PIL import Image
from torchvision import transforms

#transform = transforms.ToTensor()
transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=3),  # Convert to 3-channel grayscale
    transforms.ToTensor()
])
test_image_path = r"C:\Users\Max\Desktop\data\p10000032-20240109T160022Z-001\p10000032\s53189527\2a2277a9-b0ded155-c0de8eb9-c124d10e-82c5caab.jpg"
gt_image_path = r"C:\Users\Max\Desktop\data\p10000032-20240109T160022Z-001\p10000032\s50414267\02aa804e-bde0afdd-112c0b34-7bc16630-4e384014.jpg"
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
print(loss_0)
print(loss_1)
print(loss_2)

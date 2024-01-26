from Perceptual_xray import PerceptualLossXray
from Perceptual_ark import PerceptualLossXray
#from Perceptual_xrays import Perceptual_xray
from Perceptual_biovil import Perceptual_xray
from PIL import Image
import pandas as pd
from torchvision import transforms
from src.utils.environment_settings import env_settings
from src.utils.utils import read_config
import os


params = read_config(env_settings.CONFIG)

path = '/home/max/Unsupervised-Structured-Reporting-via-Cycle-Consistency/src/data/master_df_zeros.csv'
data_frame = pd.read_csv(path)
print(data_frame)

filter_df_no_findings = data_frame[data_frame['No Finding'] == 1]
print(filter_df_no_findings)


#transform = transforms.ToTensor()
transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=3),  # Convert to 3-channel grayscale
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

def preprocess_image(image_path, transform):
    image = Image.open(image_path)
    image = transform(image)
    image = image.unsqueeze(0)
    return image


# test_image_path = "/home/max/Desktop/MLMI/data/mimic-cxr-jpg/files/p10/p10000032/s50414267/02aa804e-bde0afdd-112c0b34-7bc16630-4e384014.jpg"
test_image_path = "/home/max/Pictures/percentile_filter.png"
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
print(f'test_vs_test: {loss_2}')

total_loss = 0
count = 0
image_base = '/home/max/Desktop/MLMI/data/'

for index, row in filter_df_no_findings.iterrows():
    # Load the first image (assuming the first column after 'No Finding' is the image path)
    first_image_path = row.iloc[4].replace('/2.0.0', '')
    first_image_path = os.path.join(image_base + first_image_path)
    first_image = preprocess_image(first_image_path, transform)

    #for index, row in filter_df_no_findings.iterrows():  # Adjust the range as per your DataFrame structure
    for index, row in data_frame.iterrows():
        other_image_path = row[4].replace('/2.0.0', '')
        other_image_path = os.path.join(image_base + other_image_path)

        if os.path.exists(other_image_path):  # Check if the image path exists
            other_image = preprocess_image(other_image_path, transform)
            loss = perceptual(first_image, other_image)
            if row[15] == 1:
                print(f'No finding and no finding: {loss}')
            else:
                print(f'No finding and other: {loss}')
            # Add the loss to total_loss and increment count
            total_loss += loss.item()
            #loss += loss
            count += 1

    # Calculate mean loss
mean_loss = total_loss / count if count > 0 else 0
print(f"Mean Perceptual Loss: {mean_loss}")



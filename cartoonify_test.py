import torch
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt

from model import UNet as CartoonNet

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = CartoonNet().to(device)
model.load_state_dict(torch.load("cartoonifier.pth", map_location=device))
model.eval()

transform = transforms.Compose([
    #transforms.Resize((256, 256)),
    transforms.ToTensor()
])

img = Image.open("_DSC8677.JPG").convert("RGB")
input_tensor = transform(img).unsqueeze(0).to(device)


with torch.no_grad():
    output = model(input_tensor)

output_img = output.squeeze(0).cpu().clamp(0, 1)
output_pil = transforms.ToPILImage()(output_img)
output_pil.save("cartoon_test.jpg")

plt.imshow(output_pil)
plt.axis("off")
plt.show()


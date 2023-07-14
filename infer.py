import torch
import torchvision.transforms as transforms
from PIL import Image
import torch.nn as nn
import glob

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# 모델 로드
model_path = './model/cifar_net.pth'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = Net()
model.load_state_dict(torch.load(model_path, map_location=device))
model.to(device)
model.eval()

# CIFAR-10 클래스
classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

# 이미지 추론 함수
def infer_image(image_path):
    image = Image.open(image_path)
    transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    image = transform(image).unsqueeze(0).to(device)

    output = model(image)
    _, predicted = torch.max(output, 1)

    return classes[predicted]

# 이미지 추론 예시
image_files = glob.glob('./test_images/*.jpeg')

for image_file in image_files:
    result = infer_image(image_file)
    print(f'{image_file}: {result}')

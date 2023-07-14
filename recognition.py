import torch
import torchvision.transforms as transforms
from PIL import Image

# 모델 로드
net = Net()
net.load_state_dict(torch.load('./cifar_net.pth'))
net.eval()

# 이미지 추론 함수
def infer(image_path):
    image = Image.open(image_path).convert('RGB')
    transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    input_tensor = transform(image).unsqueeze(0)
    output = net(input_tensor)
    _, predicted = torch.max(output.data, 1)
    return predicted.item()

# 예시 이미지 추론
result = infer('./example.jpg')
print('Inference result:', result)

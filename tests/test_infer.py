import sys
import os

# infer.py의 경로를 가져옴
infer_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
# sys.path에 infer.py의 경로를 추가
sys.path.append(infer_path)

# 이제 infer.py를 임포트할 수 있음
from infer import infer_image
def test_infer_image():
    image_path = "./test_images/test.jpeg"
    result = infer_image(image_path)
    assert isinstance(result, str)
    assert result in ['plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

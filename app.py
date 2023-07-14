from flask import Flask, render_template, request
from infer import infer_image

app = Flask(__name__)


@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        # 업로드된 파일 저장
        file = request.files["file"]
        file.save("uploads/" + file.filename)

        # 예측 수행
        image_path = "uploads/" + file.filename
        result = infer_image(image_path)

        return render_template("result.html", result=result, image_path=image_path)

    return render_template("index.html")


# 예측 결과 페이지
@app.route("/predict", methods=["POST"])
def predict():
    if request.method == "POST":
        image = request.files["image"]
        image_path = "uploads/" + image.filename
        image.save(image_path)

        # 이미지 추론
        result = infer_image(image_path)

        return render_template("result.html", result=result)


if __name__ == "__main__":
    app.run(debug=True)

from flask import Flask, render_template, request
from infer import infer_image


from flask import Flask, render_template, request

app = Flask(__name__)


@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        data = request.form["data"]  # 사용자로부터 입력받은 데이터
        # 데이터 처리 로직 수행
        return render_template("result.html", data=data)  # 결과를 보여주는 페이지로 이동
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
    app.run(host="0.0.0.0", port=8080)

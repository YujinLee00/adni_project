import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

# 저장된 모델 로드
model = tf.keras.models.load_model("test_model.h5")

# 클래스 이름 설정
label_map = {0: "AD", 1: "CN", 2: "MCI"}  # 클래스 이름이 CN, MCI, AD라고 가정

# image_path = r"/root/adni/project/Test"

# 이미지 전처리 함수
def preprocess_image(image_path, image_size=(128, 128)):
    """
    이미지 경로를 입력받아 모델 입력 크기에 맞게 전처리합니다.
    """
    img = tf.keras.utils.load_img(image_path, target_size=image_size, color_mode="grayscale")
    img_array = tf.keras.utils.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)  # 배치 차원 추가
    img_array = img_array / 255.0  # Min-Max 정규화
    return img_array

# 이미지 예측 함수
def predict_image(image_path):
    """
    입력된 이미지를 예측하여 각 클래스 확률을 출력합니다.
    """
    processed_img = preprocess_image(image_path)
    probabilities = model.predict(processed_img)[0]  # 예측 확률
    predicted_class = np.argmax(probabilities)  # 가장 높은 확률의 클래스
    print("\n===== 예측 결과 =====")
    print(f"Class: {label_map[predicted_class]}")
    print(f"AD 확률: {probabilities[0] * 100:.2f}%")
    print(f"CN 확률: {probabilities[1] * 100:.2f}%")
    print(f"MCI 확률: {probabilities[2] * 100:.2f}%")

# 이미지 시각화
    plt.figure(figsize=(10, 5))

 # 원본 이미지
    plt.subplot(1, 2, 1)
    plt.imshow(tf.keras.utils.load_img(image_path, color_mode="grayscale"), cmap="gray")
    plt.title(f"Predicted: {label_map[predicted_class]}")
    plt.axis("off")

    # 확률 막대 그래프
    plt.subplot(1, 2, 2)
    plt.bar(label_map.values(), probabilities * 100, color=['red', 'orange', 'green'])
    plt.title("Class Probabilities")
    plt.ylabel("Probability (%)")
    plt.ylim(0, 100)  # 확률 범위: 0~100%
    plt.xticks(rotation=45)
    plt.tight_layout()

      # 저장
    plt.savefig("MRI_test2.png")  # 그래프를 이미지 파일로 저장
    print("그래프가 저장되었습니다.")
    plt.close()  # plt.show()를 대체하여 메모리 누수를 방지


  

# 테스트할 이미지 경로
test_image_path = r"/root/adni/project/Test/CN002_S_0413b096.png"  # 테스트할 이미지 경로
predict_image(test_image_path)
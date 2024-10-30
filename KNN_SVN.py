import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt

# Đường dẫn đầy đủ đến thư mục dataset
folder = r'C:\Users\ngotr\OneDrive\Pictures\dataset'

# Kiểm tra xem thư mục có tồn tại không
if not os.path.exists(folder):
    print(f"Thư mục {folder} không tồn tại. Vui lòng kiểm tra lại.")
else:
    # Hàm để đọc ảnh và chuyển đổi chúng thành mảng đặc trưng
    def load_images_from_folder(folder):
        images = []
        labels = []
        for label in os.listdir(folder):
            # Tạo đường dẫn đến thư mục con cho mỗi nhãn
            label_folder = os.path.join(folder, label)
            if os.path.isdir(label_folder):  # Kiểm tra xem có phải là thư mục không
                for filename in os.listdir(label_folder):
                    img_path = os.path.join(label_folder, filename)
                    print("Đang đọc ảnh:", img_path)  # In đường dẫn để kiểm tra
                    img = cv2.imread(img_path)
                    if img is not None:  # Kiểm tra xem ảnh có được đọc không
                        img = cv2.resize(img, (150, 150))  # Kích thước chuẩn hóa
                        images.append(img)
                        labels.append(label)
                    else:
                        print(f"Không thể đọc ảnh: {img_path}")  # Thông báo nếu không đọc được ảnh
        print(f"Tổng số ảnh tải được: {len(images)}")  # In tổng số ảnh tải được
        return images, labels

    # Tải dữ liệu
    images, labels = load_images_from_folder(folder)

    # Kiểm tra xem có ảnh nào đã tải về không
    if len(images) == 0:
        print("Không có ảnh nào được tải về. Vui lòng kiểm tra thư mục.")
    else:
        # Chuyển đổi danh sách thành mảng numpy
        X = np.array(images)
        y = np.array(labels)

        # Tiền xử lý dữ liệu
        X = X.reshape(len(X), -1)  # Chuyển đổi thành 2D
        X = X.astype('float32') / 255.0  # Chuẩn hóa pixel

        # Mã hóa nhãn
        label_encoder = LabelEncoder()
        y_encoded = label_encoder.fit_transform(y)

        # Chia dữ liệu thành tập huấn luyện và tập kiểm tra
        X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

        # 1. KNN Model
        knn = KNeighborsClassifier(n_neighbors=3)
        knn.fit(X_train, y_train)
        y_pred_knn = knn.predict(X_test)

        # 2. SVM Model
        svm = SVC(kernel='linear')
        svm.fit(X_train, y_train)
        y_pred_svm = svm.predict(X_test)

        # Đánh giá mô hình
        print("KNN Classification Report:")
        print(classification_report(y_test, y_pred_knn, target_names=label_encoder.classes_))
        print("KNN Confusion Matrix:")
        print(confusion_matrix(y_test, y_pred_knn))

        print("\nSVM Classification Report:")
        print(classification_report(y_test, y_pred_svm, target_names=label_encoder.classes_))
        print("SVM Confusion Matrix:")
        print(confusion_matrix(y_test, y_pred_svm))

        # Hiển thị một số ảnh và nhãn dự đoán
        num_images_to_show = min(10, len(X_test))  # Số ảnh tối đa để hiển thị
        fig, axes = plt.subplots(2, 5, figsize=(15, 6))
        axes = axes.ravel()

        for i in range(num_images_to_show):
            axes[i].imshow(X_test[i].reshape(150, 150, 3))  # Chuyển đổi lại kích thước
            axes[i].set_title(f"Dự đoán: {label_encoder.inverse_transform([y_pred_knn[i]])[0]}\nThực tế: {label_encoder.inverse_transform([y_test[i]])[0]}")
            axes[i].axis('off')

        plt.tight_layout()
        plt.show()

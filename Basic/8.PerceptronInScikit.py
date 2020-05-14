"""Perceptron In Scikit"""

# Một người học perceptionron là một trong những kỹ thuật máy học sớm nhất
# và vẫn là nền tảng của nhiều mạng lưới thần kinh hiện đại. Trong hướng dẫn này,
# chúng tôi sử dụng một người học perceptron để phân loại "bộ dữ liệu iris
# nổi tiếng". Hướng dẫn này được lấy cảm hứng từ "Python Machine Learning
# của Sebastian Raschka".

# Tải các thư viện cần thiết
from sklearn import datasets
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Perceptron
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np

""" Tải dữ liệu Iris"""

# Tải bộ dữ liệu iris
iris = datasets.load_iris()

# Tạo dữ liệu X và y của chúng
X = iris.data
y = iris.target

""" xem dữ liệu Iris"""

# Xem năm quan sát đầu tiên của dữ liệu y của chúng
y[:5]

print(y[:5])

# Xem 5 quan sát đầu tiên của dữ liệu x của chúng
X[:5]

print(X[:5])

"""Tách dữ liệu Iris vào đào tạo và kiểm tra"""

# Chia dữ liệu thành 70% dữ liệu đào tạo và 30% dữ liệu kiểm tra
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

"""Tiền xử lý dữ liệu X bằng chia tỷ lệ"""

# Huấn luyện bộ chia tỷ lệ, làm nổi bật tất cả các tính năng có mean = 0
# và phương sai đơn vị
sc = StandardScaler()
sc.fit(X_train)

# áp dụng bộ chia tỷ lệ cho dữ liệu huấn luyện X
X_train_std = sc.transform(X_train)

# áp dụng bộ chia tỷ lệ tương tự cho dữ liệu kiểm tra X
X_test_std = sc.transform(X_test)

""" Đào tạo một người học perceptron"""

# Tạo 1 đối tượng perception với các thông số:
# 40 lần lặp (epoch) trên dữ liệu và tốc độ học tập là 0,1
ppn = Perceptron(n_iter=40, eta0=0.1, random_state=0)

# Đào tạo preceptron
ppn.fit(X_train_std, y_train)

""" Áp dụng người học huấn luyện cho dữ liệu kiểm tra"""

# áp dụng perceptron được huấn luyện trên dữ liệu X để đưa ra dự đoán
# cho dữ liệu thử nghiệm
y_pred = ppn.predict(X_test_std)

""" So sánh Y được dự đoán và Y thật"""

# xem dữ liệu kiểm tra y được dự đoán
y_pred

# xem dư liệu kiểm tra y thật
y_test

""" Kiểm tra độ chính xác"""

# xem sự chính xác của mô hình, nó là:
# 1 - (observations predicted wrong / total observations)
print('Accuracy: %.2f' % accuracy_score(y_test, y_pred))

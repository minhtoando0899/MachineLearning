"""Loading scikit-learn's Iris Dataset"""

# Truy cập các thư viện
from sklearn import datasets
import matplotlib.pyplot as plt

"""Tải tập dữ liệu Iris"""

# Bộ dữ liệu hoa Iris là một trong những cơ sở dữ liệu nổi tiếng nhất
# để phân loại. Nó chứa ba lớp (tức là ba loài hoa) với 50 quan sát mỗi lớp.

# Tải tập dữ liệu số
iris = datasets.load_iris()

# Tạo ma trận tính năng
X = iris.data

# Tạo vector mục tiêu
y = iris.target

# Quan sát các giá trị tính năng quan sát đầu tiên
X[0]
print(X[0])

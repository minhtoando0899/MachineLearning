"""Make Simulated Data For Clustering"""

from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt

""" Tạo dữ liệu """

# Tạo các tính năng (X) và các đầu ra (y) với 200 mẫu,
X, y = make_blobs(n_samples=200,
                  # 2 biến tính năng,
                  n_features=2,
                  # 3 cụm,
                  centers=3,
                  # với độ lệch chuẩn 0.5,
                  cluster_std=0.5,
                  # shuffled,
                  shuffle=True)

""" Xem Dữ liệu """

# tạo 1 phác đồ phân tán các tính năng đầu tiên và thứ hai
plt.scatter(X[:, 0],
            X[:, 1])

# Biểu diễn phác đồ phân tán
plt.show()

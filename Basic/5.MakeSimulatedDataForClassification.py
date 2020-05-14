""" Make Simulated Data For Classification """

from sklearn.datasets import make_classification
import pandas as pd

"""Tạo dữ liệu mô phỏng"""

# Tạo một ma trận tính năng mô phỏng và vector đầu ra với 100 mẫu,
features, output = make_classification(n_samples=100,
                                       # 10 tính năng
                                       n_features=10,
                                       # năm tính năng thực sự dự đoán các lớp của đầu ra
                                       n_informative=5,
                                       # năm tính năng ngẫu nhiên và không liên quan đến các lớp của đầu ra
                                       n_redundant=5,
                                       # ba lớp đầu ra
                                       n_classes=3,
                                       # với 20% của quan sát trong lớp đầu tiên, 30% trong lớp thứ 2,
                                       # và 50% trong lớp thứ 3. ('None' làm các lớp cân bằng)
                                       weights=[.2, .3, .8])

""" Quan sát dữ liêu"""

# xem năm quan sát đầu tiên và 10 tính năng của chúng
pd.DataFrame(features).head()

print(pd.DataFrame(features).head())

# xem năm lớp quan sát đầu tiên
pd.DataFrame(output).head()

print(pd.DataFrame(output).head())

# Load libraries
from sklearn import datasets
import matplotlib.pyplot as plt

"""Tải tập dữ liệu nhà ở của Boston"""

# Bộ dữ liệu nhà ở Boston là một bộ dữ liệu nổi tiếng từ những năm 1970.
# Nó chứa 506 sự quan sát về giá nhà đất xung quanh Boston. Nó thường được
# sử dụng trong các ví dụ hồi quy và chứa 15 tính năng.

# Tải dữ liệu chữ số
boston = datasets.load_boston()

# Tạo ma trận tính năng
X = boston.data

# Tạo vector mục tiêu
y = boston.target

# xem các giá trị tính năng quan sát đầu tiên
X[0]

print(X[0])

# Như bạn có thể thấy, các tính năng không được tiêu chuẩn hóa.
# Điều này dễ thấy hơn nếu chúng ta hiển thị các giá trị dưới
# dạng số thập phân:

# Biểu diễn từng giá trị tính năng của lần quan sát đầu tiên dưới dạng số thực
['{:f}'.format(x) for x in X[0]]

print(['{:f}'.format(x) for x in X[0]])

# Do đó, nó thường có lợi và/hoặc được yêu cầu để chuẩn hóa
# giá trị của các tính năng.

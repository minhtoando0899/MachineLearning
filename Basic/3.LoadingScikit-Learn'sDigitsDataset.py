"""Loading scikit-learn's Digits Dataset"""

# truy cập thư viện
from sklearn import datasets
import matplotlib.pyplot as plt

"""Tải tệp dữ liệu số"""

# chữ số là tập dữ liệu của các chữ số viết tay. từng tính năng
# là cường độ một pixel của hình ảnh 8 x 8.

# tải tập dữ liệu số
digits = datasets.load_digits()

# tạo ma trận tính năng
X = digits.data

# Tạo vector mục tiêu
y = digits.target

# xem các giá trị tính năng đầu tiên
X[0]

print(X[0])

# Các giá trị tính năng quan sát được trình bày dưới dạng một vectơ.
# Tuy nhiên, bằng cách sử dụng phương thức hình ảnh, chúng ta có thể
# tải các giá trị tính năng tương tự như một ma trận và sau đó trực
# quan hóa ký tự viết tay thực tế:

# quan sát các giá trị tính năng đầu tiên như 1 ma trận
digits.images[0]

print(digits.images[0])

# trực quan hóa các giá trị tính năng của sự quan sát đầu tiên như một hình ảnh
plt.gray()
plt.matshow(digits.images[0])
plt.show()

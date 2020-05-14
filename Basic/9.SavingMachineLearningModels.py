"""Saving Machine Learning Models"""

# Trong scikit có hai cách chính để lưu mô hình để sử dụng
# trong tương lai: chuỗi pickle và mô hình pickle dưới dạng tệp.

from sklearn.linear_model import LogisticRegression
from sklearn import datasets
import pickle
from sklearn.externals import joblib

""" Tải dữ liệu"""

# Tải dữ liệu iris
iris = datasets.load_iris()

# Tạo 1 ma trận, X, của các tính năng và 1 vector, y.
X, y = iris.data, iris.target

""" Đào tạo mô hình"""

# Đào tạo một mô hình naive logistic regression
clf = LogisticRegression(random_state=0)
clf.fit(X, y)

""" Lưu vào chuỗi bằng pickle"""

# lưu mô hình được huấn luyện như 1 chuỗi pickle
saved_model = pickle.dumps(clf)

# Xem mô hình pickled
saved_model

# Tải mô hình pickled
clf_from_pickle = pickle.loads(saved_model)

# Sử dụng mô hình pickle được tải để đưa ra các dự đoán
clf_from_pickle.predict(X)

""" Lưu vào tập tin pickled bằng joblib"""

# lưu mô hình như 1 pickle trong 1 tập tin
joblib.dump(clf, 'filename.pkl')

# Tải mô hình từ tệp tin
clf_from_joblib = joblib.load('filename.pkl')

# sử dụng mô hình để tạo các dự đoán
clf_from_joblib.predict(X)

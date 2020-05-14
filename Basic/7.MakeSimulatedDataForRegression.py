"""Make Simulated Data For Regression"""

import pandas as pd
from sklearn.datasets import make_regression

""" Tạo dữ liệu mô phỏng"""

# Tạo các tính năng, đầu ra, và hệ số thực của 100 mẫu,
feature, output, coef = make_regression(n_samples=100,
                                        # 3 tính năng
                                        n_features=3,
                                        # trong đó chỉ cso 2 tính năng là hữu ích,
                                        n_informative=2,
                                        # một giá trị mục tiêu duy nhất cho mỗi quan sát
                                        n_targets=1,
                                        # 0.0 độ lêch chuẩn của tiếng ồn gaussian
                                        noise=0.0,
                                        # hiển thị hệ số thực được dùng để tạo dữ liệu
                                        coef= True)

""" Xem dứ liệu mô phỏng """

# xem các tính năng của năm hàng đầu tiên
pd.DataFrame(feature, columns=['Store 1', 'Store 2', 'Store 3']).head()

print(pd.DataFrame(feature, columns=['Store 1', 'Store 2', 'Store 3']).head())

# xem đầu ra của 5 hàng đầu tiên
pd.DataFrame(output, columns=['Sales']).head()

print(pd.DataFrame(output, columns=['Sales']).head())

# xem thực tế, các hệ số thực được sử dụng để tạo dữ liệu
pd.DataFrame(coef, columns=['True Coefficient Values'])

print(pd.DataFrame(coef, columns=['True Coefficient Values']))

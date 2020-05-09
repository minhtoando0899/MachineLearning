"""Loading features from dictionaries"""

from sklearn.feature_extraction import DictVectorizer
# Tao 1 tu dien
staff = [{'name': 'Huy', 'age': 32.},
         {'name': 'Toan', 'age': 20.},
         {'name': 'Tien Anh', 'age': 21.}]

# chuyen doi tu dien thanh ma tran tinh nang
# tao doi tuong cho vectorizer tu dien cua chung toi
vec = DictVectorizer()

# chuyen doi tu dien staff voi vec, sau do dau ra la 1 mang
vec.fit_transform(staff).toarray()

# xem tinh nang name

vec.get_feature_names()


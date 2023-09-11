import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Giả sử tập dữ liệu nằm trong tệp CSV có tên 'data.csv'
df = pd.read_csv('data.csv')

# Hiển thị 5 dòng dữ liệu đầu tiên
print(df.head())

# Bây giờ, hãy trích xuất các cột có liên quan từ DataFrame để tạo ma trận tính năng X và biến mục tiêu y.
#'X' sẽ chứa các biến độc lập (phòng ngủ, phòng tắm, sqft_living, sqft_lot, tầng, bờ sông, tầm nhìn, tình trạng)
# 'y' sẽ chứa biến mục tiêu (giá)

X = df[['bedrooms', 'bathrooms', 'sqft_living', 'sqft_lot', 'floors', 'waterfront', 'view', 'condition']]
y = df['price']

#Tiếp theo, bạn có thể chia tập dữ liệu thành các tập huấn luyện và kiểm tra bằng hàm train_test_split của sklearn:
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Bây giờ, hãy chỉ định tên tính năng cho X để tránh thông báo cảnh báo:
feature_names = ['bedrooms', 'bathrooms', 'sqft_living', 'sqft_lot', 'floors', 'waterfront', 'view', 'condition']
X_train.columns = feature_names
X_test.columns = feature_names

#Bây giờ, bạn có thể xây dựng mô hình hồi quy tuyến tính của mình bằng lớp LinearRegression của sklearn:
model = LinearRegression()

# Fit the model on the training data
model.fit(X_train, y_train)

# Sau khi đào tạo mô hình, bạn có thể sử dụng mô hình đó để đưa ra dự đoán về dữ liệu mới (bộ kiểm tra trong trường hợp này):
y_pred = model.predict(X_test)

# Để đánh giá hiệu suất của mô hình, bạn có thể sử dụng các số liệu như Lỗi bình phương trung bình (MSE) và R bình phương:
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

# Dự đoán và hình dung
# Để hình dung các dự đoán so với giá thực tế, chúng tôi sẽ sử dụng biểu đồ phân tán
plt.scatter(y_test, y_pred)
plt.xlabel("Giá thực tế")
plt.ylabel("Giá dự đoán")
plt.title("Giá thực tế vs. Giá dự đoán")
plt.show()

# Chúng ta cũng có thể tạo biểu đồ dư để kiểm tra hiệu suất của mô hình
residuals = y_test - y_pred
plt.scatter(y_test, residuals)
plt.axhline(y=0, color='red', linestyle='--')
plt.xlabel("Giá thực tế")
plt.ylabel("dư lượng")
plt.title("Lô đất dư")
plt.show()

print("Mean Squared Error:", mse)
print("R-squared:", r2)

# Bạn cũng có thể sử dụng mô hình được đào tạo để đưa ra dự đoán về dữ liệu mới bằng cách cung cấp các giá trị tính năng:
# Ví dụ: để dự đoán giá một ngôi nhà có 3 phòng ngủ, 2 phòng tắm, diện tích sinh hoạt 1500 m2, lô đất 4000 m2, 1 tầng, không nhìn ra mặt nước, không có tầm nhìn và điều kiện 3:
new_data = [[3, 2, 1500, 4000, 1, 0, 0, 3]]
predicted_price = model.predict(new_data)
print("Giá dự đoán:", predicted_price[0])
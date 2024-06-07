import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
import pandas as pd

# Đọc dữ liệu
df = pd.read_csv('training_sample.csv')

# Tách dữ liệu thành các biến độc lập và phụ thuộc
X = df[['basket_icon_click', 'basket_add_list', 'basket_add_detail', 'sort_by', 'image_picker', 
        'account_page_click', 'promo_banner_click', 'detail_wishlist_add', 'list_size_dropdown', 
        'closed_minibasket_click', 'checked_delivery_detail', 'checked_returns_detail', 'sign_in', 
        'saw_checkout', 'saw_sizecharts', 'saw_delivery', 'saw_account_upgrade', 'saw_homepage', 
        'device_computer', 'device_tablet', 'returning_user', 'loc_uk']]
y = df['ordered']

# Chia dữ liệu thành tập huấn luyện và tập kiểm tra
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train mô hình random forest
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

# Train mô hình Naive Bayes
nb_model = GaussianNB()
nb_model.fit(X_train, y_train)

# Train mô hình KNN
knn_model = KNeighborsClassifier(n_neighbors=3)
knn_model.fit(X_train, y_train)

# Train mô hình Decision Tree
dt_model = DecisionTreeClassifier(random_state=42)
dt_model.fit(X_train, y_train)

# Lưu mô hình
joblib.dump(rf_model, 'random_forest.pkl')
joblib.dump(nb_model, 'nb_model.pkl')
joblib.dump(knn_model, 'knn_model.pkl')
joblib.dump(dt_model, 'dt_model.pkl')


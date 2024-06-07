import pandas as pd
import numpy as np
import uuid

# Số lượng mẫu dữ liệu giả lập
num_samples = 1000

# Hàm tạo UserID
def generate_user_id():
    return f"{np.random.randint(1000, 9999)}-{uuid.uuid4()}"

# Tạo dataframe với các cột cần thiết
data = {
    "UserID": [generate_user_id() for _ in range(num_samples)],
    "basket_icon_click": np.random.choice([0, 1], size=num_samples),
    "basket_add_list": np.random.choice([0, 1], size=num_samples),
    "basket_add_detail": np.random.choice([0, 1], size=num_samples),
    "sort_by": np.random.choice([0, 1], size=num_samples),
    "image_picker": np.random.choice([0, 1], size=num_samples),
    "account_page_click": np.random.choice([0, 1], size=num_samples),
    "promo_banner_click": np.random.choice([0, 1], size=num_samples),
    "detail_wishlist_add": np.random.choice([0, 1], size=num_samples),
    "list_size_dropdown": np.random.choice([0, 1], size=num_samples),
    "closed_minibasket_click": np.random.choice([0, 1], size=num_samples),
    "checked_delivery_detail": np.random.choice([0, 1], size=num_samples),
    "checked_returns_detail": np.random.choice([0, 1], size=num_samples),
    "sign_in": np.random.choice([0, 1], size=num_samples),
    "saw_checkout": np.random.choice([0, 1], size=num_samples),
    "saw_sizecharts": np.random.choice([0, 1], size=num_samples),
    "saw_delivery": np.random.choice([0, 1], size=num_samples),
    "saw_account_upgrade": np.random.choice([0, 1], size=num_samples),
    "saw_homepage": np.random.choice([0, 1], size=num_samples),
    "device_mobile": np.random.choice([0, 1], size=num_samples),
    "device_computer": np.random.choice([0, 1], size=num_samples),
    "device_tablet": np.random.choice([0, 1], size=num_samples),
    "returning_user": np.random.choice([0, 1], size=num_samples),
    "loc_uk": np.random.choice([0, 1], size=num_samples),
    "ordered": np.random.choice([0, 1], size=num_samples)
}

# Tạo DataFrame
df = pd.DataFrame(data)

# Hiển thị 5 dòng dữ liệu mẫu
print(df.head())

# Lưu dataset vào file CSV
df.to_csv("fake_dataset.csv", index=False)

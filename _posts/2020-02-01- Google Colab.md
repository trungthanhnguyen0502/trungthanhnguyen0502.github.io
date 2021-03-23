---
layout: post
title: Train model với GPU Google Colab
title2: Google colab
tag: [google_colab, train, deep_learning, gpu]
category: [deep_learning]
img: assets/1-colab/tesla-gpu.png
author: trungthanhnguyen
summary: Làm việc với Deep Learning nghĩa là bạn phải nghiên cứu và xây dựng các model, đòi hỏi máy tính phải có khả năng tính toán lớn. Quá trình training model - hay huấn luyện mô hình thường cần tới GPU. Bài này sẽ hướng dẫn cách sử dụng free GPU trên Google Colab.
---

# 1. Google Colab

Làm việc với Deep Learning nghĩa là bạn phải nghiên cứu và xây dựng các model, đòi hỏi máy tính phải có khả năng tính toán lớn. Quá trình training model - hay huấn luyện mô hình thường cần tới GPU. Ngày nay, không nhất thiết phải sắm máy tính với GPU đắt tiền, ta hoàn toàn có thể đăng kí các dịch vụ **Cloud Computing** của Google, Amazon và nhiêù nhà cung cấp khác. Bài này sẽ hướng dẫn cách sử dụng **GPU miễn phí** trên Google Colab.

Google Colab cung cấp **Colaboratory notebooks** - 1 dạng tương tự như jupyter notebook. Khi tạo 1 notebook, một môi trường ảo sẽ được tạo riêng cho từng người dùng với đầy đủ những tính năng cơ bản, folder, package, pythonlib. Trong khi code, thiếu lib nào ta có thể **pip install** như khi sử dụng trên máy vật lý. Colab hỗ trợ khá toàn diện các pythonlib với version mới nhất tensoflow, keras, PyTorch, cv2 ... 

Lưu ý là bạn chỉ có thể train tối đa 12 tiếng liên tục. Nhớ lưu model khi train tránh việc hết 12h hoặc mất mạng, khi đó bạn sẽ phải train lại từ đầu.

![](https://images.viblo.asia/4ab2c6f8-ee7d-40d3-a813-26c6c7d3739a.png)

>  **H1**:  Tesla K80 GPU - GPU thường được assign trên Google Colab 

# 2. Setup
## 2.1 Tạo notebook

1. Đăng nhập Drive, đến folder muốn 

2. Chuột phải, chọn **More $\longrightarrow$ Colaboratory**. Như vậy ta đã tạo ra 1 notebook.
3.  Click vào notebook vừa tạo, giao diện hiện ra khá giống với Jupyter notebook.

![](https://images.viblo.asia/604ed860-789b-4b73-9397-0f42edd2e1c9.jpeg)
> **H2**: Tạo Colab notebook

## 2.2 Setup GPU
Khi làm việc, không phải lúc nào cũng cần phải dùng tới GPU, nên theo cấu hình mặc định chỉ có CPU. Trong trường hợp train/inference model cần tới GPU, ta làm theo bước sau: 

**Edit $\longrightarrow$ Notebook Settings $\longrightarrow$** chọn **Hardware GPU**

![](https://images.viblo.asia/fc69fbfb-f436-491f-b5a5-b547d1609caa.png)
> **H3**: Setting GPU trên colab notebook

## 2.3 Mount drive

Mặc định Colab notebook sẽ không có quyền truy cập, đọc ghi bất kì file nào trên Drive. Tuy nhiên trong thực tế, khi train model ta cần 1 nơi chứa dataset dùng để train model. Đó có thể là Google Drive, Kaggle, Dropbox...  Việc này không bắt buộc nhưng cần thiết nếu bạn cần 1 nơi save model và dataset. Trong bài này, để đơn giản mình sẽ hướng dẫn cách Mount Drive: 

```python
from google.colab import drive
drive.mount('/content/drive')
```

* Khi run 2 dòng code trên, notebook sẽ xuất hiện 1 đường link đi kèm với 1 ô nhập token. 
* Click vào link, tiến hành cấp quyền truy cập Drive.
*  Cấp quyền xong sẽ hiện ra 1 đoạn token. Copy token đó và điền vào ô input trên notebook, nhấn Enter.
*   Như vậy bạn đã hoàn thành việc cấp quyền cho google colab được phép truy cập vào drive của bạn (quyền đọc và ghi dữ liệu).

Để thấy được những folder, file đã được add từ drive vào môi trường máy ảo này, mở cửa sổ bên trái, click vào **Files $\longrightarrow$ Refresh**, bạn sẽ thấy Drive của bạn đã được thêm vào working directory


## 2.4 Practice

Mình sẽ tạo notebook, viết 1 đoạn code tạo model với tensorflow để xem code có hoạt động bình thường như trên máy vật lí không nhé. Trước hết tạo notebook, tiến hành Setup GPU như hướng dẫn trên.

Nếu muốn install thêm thư viện, ta thêm dấu ! trước câu lệnh pip install + Ctrl Enter. Giả sử muốn install cv2: 

```python
!pip install opencv-python
```


```python
import tensorflow as tf
import numpy as np

model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(16, (3,3), activation='relu', input_shape=(150, 150, 3)),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Conv2D(32, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2), 
    tf.keras.layers.Conv2D(64, (3,3), activation='relu'), 
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Flatten(), 
    tf.keras.layers.Dense(512, activation='relu'), 
    tf.keras.layers.Dense(1, activation='sigmoid')  
])

model.summary()

# Model: "sequential"
# _________________________________________________________________
# Layer (type)                 Output Shape              Param #   
# =================================================================
# conv2d (Conv2D)              (None, 148, 148, 16)      448       
# _________________________________________________________________
# max_pooling2d (MaxPooling2D) (None, 74, 74, 16)        0         
# _________________________________________________________________
# conv2d_1 (Conv2D)            (None, 72, 72, 32)        4640      
# _________________________________________________________________
# max_pooling2d_1 (MaxPooling2 (None, 36, 36, 32)        0         
# _________________________________________________________________
# conv2d_2 (Conv2D)            (None, 34, 34, 64)        18496     
# _________________________________________________________________
# max_pooling2d_2 (MaxPooling2 (None, 17, 17, 64)        0         
# _________________________________________________________________
# flatten (Flatten)            (None, 18496)             0         
# _________________________________________________________________
# dense (Dense)                (None, 512)               9470464   
# _________________________________________________________________
# dense_1 (Dense)              (None, 1)                 513       
# =================================================================
# Total params: 9,494,561
# Trainable params: 9,494,561
# Non-trainable params: 0
# _________________________________________________________________
```

```python
tf.test.is_gpu_available()

print("Test model with fake data ", model.predict(np.zeros((1, 150,150,3))))

# True
# Test model with fake data  [[0.5]]
```

<hr>
Như vậy, qua vài dòng code cơ bản, ta có thể thấy code run không khác gì khi run với jupyter thông thường. Dịch vụ này của google colab rất hay và tiện, đặc biệt hữu ích phục vụ quá trình tự nghiên cứu và học AI.

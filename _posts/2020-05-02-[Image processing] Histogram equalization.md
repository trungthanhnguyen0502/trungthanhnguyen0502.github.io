---
layout: post
title: "[Image processing] Histogram equalization"
title2: "[Image processing] Histogram equalization"
tag: [histogram, image_processing]
category: [image_processing]
author: trungthanhnguyen
summary: Trong thực tế, camera thường chịu tác động từ điều kiện sáng. Điều đó khiến cho nhiều ảnh bị tối hoặc quá sáng. Cân bằng histogram là một phương pháp tiền/hậu xử lí ảnh rất mạnh mẽ. Đặc biệt trong nhiều bài toán compute vision, phương pháp tiền xử lí này cho chất lượng dữ liệu cao, cải thiện chất lượng model deep learning rất nhiều.
---


![](https://images.viblo.asia/32162dd4-55f8-4e4f-9db3-0e3b2cbb7933.png)

# 1. Lý thuyết
## 1.1 Khái niệm.

Trong lĩnh vực xử lí ảnh, histogram là biểu đồ tần xuất được dùng để thống kê số lần xuất hiện các mức sáng trong ảnh. Dưới đây là ảnh minh họa. Nhìn vào biểu đồ (chưa cần quan tâm tới đường màu đỏ), dựa vào các cột gía trị có thể dễ dàng thấy được rằng: hầu hết các pixel có giá trị nằm trong khoảng [150, 200]. Điều đó khiến cho toàn bộ ảnh bị sáng hơn mức cần thiết, độ tương phản không cao, không rõ nét

![](https://images.viblo.asia/f9525927-5354-42ee-8881-81ba933b6d7a.png)
> Ảnh gốc H1

![](https://images.viblo.asia/b9dc2e6f-c657-43eb-a441-1a9d1ed99bfd.png)
> Ảnh H2 - đã được cân bằng histogram

**Cân bằng histogram** (histogram equalization) là sự điều chỉnh histogram về trạng thái cân bằng, làm cho phân bố (distribution) giá trị pixel không bị co cụm tại một khoảng hẹp mà được "kéo dãn" ra. Trong thực tế, camera thường chịu tác động từ điều kiện sáng. Điều đó khiến cho nhiều ảnh bị tối hoặc quá sáng. Cân bằng histogram là một phương pháp tiền/hậu xử lí ảnh rất mạnh mẽ. Đặc biệt trong nhiều bài toán mình từng làm trong lĩnh vực compute vision, phương pháp tiền xử lí ảnh này cho chất lượng dữ liệu rất cao, cải thiện chất lượng model deep learning rất nhiều.

![](https://images.viblo.asia/5ad8273e-d4ac-44e5-a415-7d9600250928.png)

## 1.2 Công thức.
Mình nhận ra các bài viết về histogram equalization thường sơ sài chỉ có code, lại có những bài hơi nặng về công thức toán, gây khó hiểu cho những bạn mới. Trong bài này mình sẽ diễn giải rõ hơn về thuật toán. Mình sẽ làm việc với ảnh gọi là ảnh H1. Ảnh H1 có histogram tương ứng bên phải.

![](https://images.viblo.asia/f9525927-5354-42ee-8881-81ba933b6d7a.png)
> Ảnh H1

Gọi hàm biến đổi ta cần xác định là $K(i)$ với $i$  $\epsilon$  $[0,255]$

Với hình H1, ta có thể thấy có quá ít các pixel có giá trị nằm trong khoảng [0,149]. Các pixel có giá trị từ 150 -> 255. Vậy K có nhiệm vụ thay thế các pixel có giá trị 150 bằng giá trị 0.  Như vậy, $K(150) \approx 0$

**B1**: Thống kê số lượng pixel cho từng mức sáng, ta được histogram $H(i)$  

**B2**: Tính "hàm tích lũy" $Z$ cho từng mức sáng theo công thức: 

$$Z(i)  = \sum_{j=0}^{i} H(i)$$


Trong đó $Z(i)$ chính là tổng số pixel có giá trị $\leqslant i$ . Trên hình H1, đường màu đỏ chính là đường minh họa $Z(i)$. $Z(i)$ chắc chắn là hàm đồng biến tăng

Giả sử $Z(140)$ = 100, $Z(150)$ = 200, $Z(160)$ = 5000. Thấy $Z(150) - Z(140)$ = 100,  trong khi $Z(160) - $Z(150)$ = 4800. Như vậy, giá trị Z tăng quá nhanh trong khoảng [150,160] so với khoảng [140,150]. Ta cần biến đổi các giá trị pixel sao cho giá trị của Z sẽ phải tăng **dần đều** thay vì tăng **đột ngột** như trong ảnh gốc. Nhiệm vụ của K là K phải thay thế các giá trị pixel 140, 150, 160 sao cho: 

$$Z(K(150)) - Z(K(140)) == Z(K(160)) - Z(K(150))$$

**B3**: Hàm biến đổi K tại một mức sáng i được tính như sau:

$$K(i) = \dfrac{Z(i) - min(Z)}{max(Z) - min(Z)}* 255$$

Hãy đọc thật kĩ công thức này và hiểu nó. Công thức này có tác dụng **dãn** các khoảng phân bố dày đặc pixel và **co** các khoảng phân bố thưa pixel.

# 2. Thực hành
Để hiểu rõ hơn về thuật toán, mình sẽ hướng dẫn code thuật toán hoàn toàn từ đầu.

## 2.1 Ảnh xám
Cân bằng histogram là cân bằng lại mức cường độ sáng, tức chỉ là 1 trong 3 chanel của hệ màu HSV. Vậy nên trước hết mình sẽ code với ảnh xám (gray)

Import thư viện và ảnh:
```python
import numpy
import cv2
import matplotlib.pyplot as plt

img = cv2.imread("img.png", 0)
```

Hàm tính histogram của một ảnh
```python
def compute_hist(img):
    hist = np.zeros((256,), np.uint8)
    h, w = img.shape[:2]
    for i in range(h):
        for j in range(w):
            hist[img[i][j]] += 1
    return hist
```

Hàm cân bằng histogram
```python
def equal_hist(hist):
    cumulator = np.zeros_like(hist, np.float64)
    for i in range(len(cumulator)):
        cumulator[i] = hist[:i].sum()
    print(cumulator)
    new_hist = (cumulator - cumulator.min())/(cumulator.max() - cumulator.min()) * 255
    new_hist = np.uint8(new_hist)
    return new_hist
```
 
 Thử chạy:
 
```python
hist = compute_hist(img).ravel()
new_hist = equal_hist(hist)

h, w = img.shape[:2]
for i in range(h):
    for j in range(w):
        img[i,j] = new_hist[img[i,j]]
        
fig = plt.figure()
ax = plt.subplot(121)
plt.imshow(img, cmap='gray')

plt.subplot(122)
plt.plot(new_hist)
plt.show()
```

Và tất nhiên, kết quả mình thu được là hình H2 trên đầu bài viết. Ngoài ra, với các bạn có thể dùng hàm của cv2 với cú pháp:

```python
img = cv2.equalizeHist(img)
```

## 2.2 Ảnh màu

Mặc định ảnh màu là hệ RGB hoặc BGR, muốn cân bằng sáng ta cần biến đổi về hệ màu HSV. Hệ màu HSV bao gồm 3 chanel:  
* H-HUE: giá trị màu
* S-SATURATION: độ bảo hòa.
* V- VALUE: độ sáng của màu sắc.

Ta sẽ áp dụng cân bằng histogram chỉ trên độ sáng V của ảnh

```python
img_yuv = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)
img_yuv[:,:,0] = cv2.equalizeHist(img_yuv[:,:,0])

# convert the YUV image back to RGB format
img_output = cv2.cvtColor(img_yuv, cv2.COLOR_YUV2BGR)
```

Đây là kết quả demo, có thể thấy rõ màu sắc, hình ảnh sau khi cân bằng đã rõ nét và có độ tương phản cao hơn rất nhiều ảnh gốc. Cảm ơn các bạn đã đọc bài viết.

![](https://images.viblo.asia/913cbd26-04b1-4fef-9ad4-3a63adbbe1ba.png)
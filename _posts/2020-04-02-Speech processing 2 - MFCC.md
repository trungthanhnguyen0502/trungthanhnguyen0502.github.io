---
layout: post
title: "[Speech] p2. Feature extraction - MFCC"
title2: "[Speech] p2. Feature extraction - MFCC"
tag: [speech, voice]
category: [speech_processing, voice]
author: trungthanhnguyen
summary: Trong bài này, mình sẽ tập trung vào việc biến đổi tín hiệu giọng nói thành dạng MFCC. MFCC được dùng phổ biến trong bài toán speech recognize và nhiều bài toán liên quan tới giọng nói khác.
---

![](https://images.viblo.asia/605f7b23-7955-4f29-aef0-0c60da561835.png)

Đây là bài thứ 2 trong chuỗi bài xử lý giọng nói, phần 1: [Kiến thức nền tảng xử lý tiếng nói - Speech Processing](https://viblo.asia/p/kien-thuc-nen-tang-xu-li-tieng-noi-speech-processing-jvElaAL6lkw). Bạn nên đọc phần 1 trước khi đọc bài này để đảm bảo đủ kiến thức nền tảng. Trong bài này, mình sẽ tập trung vào việc biến đổi tín hiệu giọng nói thành dạng MFCC.

# MFCC là gì ?
MFCC là viết tắt của Mel frequency cepstral coefficients.
MFCC được dùng phổ biến trong bài toán speech recognize và nhiều bài toán liên quan tới giọng nói khác. Ta có thể hình dung việc tính MFCC theo luồng xử lý: 
+ Cắt chuỗi tín hiệu âm thanh thành các đoạn ngắn bằng nhau (25ms) và overlap lên nhau (10ms).
+ Mỗi đoạn âm thanh này được biến đổi, tính toán để thu được 39 features.
+ 39 feature này có tính độc lập cao, ít nhiễu, đủ nhỏ để đảm bảo tính toán, đủ thông tin để đảm bảo chất lượng cho các thuật toán nhận dạng.

![](https://images.viblo.asia/0aecb56e-2e56-4df2-9bdf-340a211bd0e5.jpeg)

Các phần dưới đây sẽ đi vào chi tiết từng bước để tính MFCC của 1 đoạn âm thanh

# 1. A/D Conversion and Pre-emphasis
## 1.1  A/D Conversion

Âm thanh là dạng tín hiệu liên tục, trong khi đó máy tính làm việc với các con số rời rạc. Ta cần lấy mẫu tại các khoảng thời gian cách đều nhau với 1 tần số lấy mẫu - sample_rate để chuyển từ dạng tín hiệu liên tục về dạng rời rạc. VD sample_rate = 8000 $\longrightarrow$ trong 1s lấy 8000 giá trị. 

![](https://images.viblo.asia/c4ce943e-2b8d-4592-821a-a37a11a06d05.jpeg)

Tai người nghe được âm thanh trong khoảng 20Hz $\to$ 20.000 Hz. Theo định lý lấy mẫu Nyquist–Shannon: với 1 tín hiệu có các tần số thành phần $f_i \leq f_m$, để đảm bảo việc lấy mẫu không làm mất mát thông tin (aliasing), tần số lấy mẫu $f_s$ phải đảm bảo  $f_s \geq 2f_m$.  

Vậy để đảm bảo việc lấy mẫu không làm mất mát thông tin, tần số lấy mẫu $f_s = 44100 Hz$. Tuy nhiên, vì tai người kém nhạy cảm với các âm thanh tần số cao, gần như không cảm nhận được gì ở khoảng tần số 15000Hz -> 20000Hz. Vậy nên trong nhiều trường hợp, người ta chỉ cần lấy $f_s = 8000Hz$ hoặc $f_s = 16000Hz$ để đảm bảo tốc độ và hiệu năng tính toán.
 
## 1.2. Pre-emphasis

Do đặc điểm cấu tạo thanh quản và các bộ phận phát âm nên tiếng nói của chúng ta có đặc điểm: các âm ở tần số thấp có mức năng lượng cao, các âm ở tần số cao lại có mức năng lượng khá thấp. Trong khi đó, các tần số cao này vẫn chứa thông tin về âm vị. Vì vậy chúng ta cần 1 bước pre-emphasis để **kích** các tín hiệu ở tần số cao này lên. 
 
![](https://images.viblo.asia/c348c8a6-15b9-4a70-9c3d-86b2e026e512.jpeg)
 
# 2. Spectrogram

Như đã nói qua ở bài trước, thông thường các tín hiệu âm thanh trong miền thời gian được biến đổi sang miền tần số. Hiểu đơn giản là "phân giải 1 tín hiệu bất kì thành 1 tập các tín hiệu có tính chất tuần hoàn - có biên độ, tần số, pha giao động xác định". Để biến đổi từ miền thời gian sang miền tần số, ta cần các bước sau:

## 2.1 Windowing
Thay vì biến đổi Fourier trên cả đoạn âm thanh dài, ta chỉ cần biến đổi Fourier trên từng đoạn,từng đoạn lần lượt. Ta trượt 1 cửa sổ dọc theo tín hiệu để lấy ra các **frame** rồi mới áp dụng DFT trên từng frame này (DFT - Discrete Fourier Transform).

Tốc độ nói của con người trung bình khoảng 3, 4 từ mỗi giây, mỗi từ khoảng 3-4 âm, mỗi âm chia thành 3-4 phần,  như vậy 1 giây âm thanh được chia thành 36 - 40 phần, ta chọn độ rộng mỗi frame khoảng 20 - 25ms là vừa đủ rộng để bao 1 phần âm thanh. Các frame được overlap lên nhau khoảng 10ms để có thể *capture* lại sự thay đổi context.
 
 ![](https://images.viblo.asia/626d5d38-3ca2-42cd-ae9b-0c283804fbce.jpeg) 
 
Tuy nhiên, việc cắt frame sẽ làm các giá trị ở 2 biên của frame bị giảm đột ngột (về giá trị 0). Theo quy luật, nếu trong miền thời gian tín hiệu càng thay đổi đột ngột, thì bên miền tần số sẽ xuất hiện rất nhiều nhiễu tại các tần số cao. Để khắc phục điều này, ta cần làm mượt bằng cách nhân chập frame với 1 vài loại window. Có 1 vài loại window phổ biến là Hamming window,  Hanning window ... có tác dụng làm giá trị biên frame giảm xuống từ từ từ.

![](https://images.viblo.asia/ef853ace-0e4f-4af5-b0a6-ed213f756950.jpeg)

Hình dưới đây sẽ cho ta thấy rõ được tác dụng của các window này. Trong các hình nhỏ, hình 1 là 1 đoạn âm thanh được cắt ra từ âm thanh gốc, âm thanh gốc là được tạo lên bởi 2 tần số trong hình 2. Nếu áp dụng rectangle window (tức là cắt trực tiếp), tín hiệu miền tần số tương ứng là hình 3, ta có thể thấy tín hiệu này chứa rất nhiều nhiễu. Nếu áp dụng các window như Hanning, Hamming, Blackman, tín hiệu miền tần số thu được khá mượt và gần sấp xỉ tần số gốc ở hình 2.

![](https://images.viblo.asia/ae307060-5de7-4c5a-9a0b-e8af27faa0ac.png)

## 2.2 DFT
Trên từng frame, ta áp dụng DFT - Discrete Fourier Transform theo công thức: 

![](https://images.viblo.asia/3ad26c88-868c-41e1-8bd5-bd2bb898f314.jpeg)

Mỗi frame ta thu được 1 list các giá trị tương ứng độ lớn (**magnitude**) của từng tần số trong khoảng $0 \to N$. Áp dụng trên tất cả các frame, ta đã thu được 1 **Spectrogram** như hình dưới. Trục $x$ là trục thời gian (tương ứng với thứ tự các frame), trục $y$ thể hiện dải tần số từ $0 \to 10000$ Hz, giá trị magnitude tại từng tần số được thể hiện bằng màu sắc. Qua quan sát spectrogram này, ta nhận thấy các tại các tần số thấp thường có magnitude cao, tần số cao thường có magnitude thấp.

![](https://images.viblo.asia/9f906e6e-003b-4a11-976c-9588b7af8320.png)

Hình dưới là các spectrogram của 4 nguyên âm. Quan sát spectrogram lần lượt từ dưới lên, người ta nhận thấy có 1 vài tần số đặc trưng gọi là các **formant**, gọi là các tần số F1, F2, F3 ... Các chuyên gia về ngữ âm học có thể dựa vào vị trí, thời gian, sự thay đổi các formant trên  spectrogram để xác định đoạn âm thanh đó là của âm vị nào.

![](https://images.viblo.asia/b6bbb6da-0225-4a37-bd9b-cbddca823f65.png)

Như vậy ta đã biết cách tạo ra spectrogram. Tuy nhiên trong nhiều bài toán (đặc biệt là speech recognition), spectrogram không phải là sự lựa chọn hoàn hảo. Vì vậy ta cần thêm vài bước tính nữa để thu được dạng MFCC, tốt hơn, phổ biến hơn, hiệu quả hơn spectrogram.

# 3. Mel filterbank
Như mình đã mô tả ở phần trước, cách cảm nhận của tai người là phi tuyến tính, không giống các thiết bị đo. Tai người cảm nhận tốt ở các tần số thấp, kém nhạy cảm với các tần số cao. Ta cần 1 cơ chế mapping tương tự như vậy. 

![](https://images.viblo.asia/099f99a8-d391-42b9-801c-ddb41753846c.png)

Trước hết, ta bình phương các giá trị trong spectrogram thu được **DFT power spectrum** (phổ công suất). Sau đó, ta áp dụng 1 tập các bộ lọc thông dải **Mel-scale filter** trên từng khoảng tần số (mỗi filter áp dụng trên 1 dải tần xác định). Giá trị output của từng filter là năng lượng dải tần số mà filter đó cover (bao phủ) được. Ta thu được **Mel spectrogram**.

Ngoài ra, các filter dùng cho dải tần thấp thường hẹp hơn các filter dùng cho dải tần cao. 

# 4. Cepstrum
## 4.1 Log
Mel filterbank trả về phổ công suất của âm thanh, hay còn gọi là phổ năng lượng. Thực tế rằng con người kém nhạy cảm trong sự thay đổi năng lượng ở các tần số cao, nhạy cảm hơn ở tần số thấp. Vì vậy ta sẽ tính log trên Mel-scale power spectrum. Điều này còn giúp giảm các thay đổi,, nhiễu âm thanh không đáng kể để nhận dạng giọng nói. 

## 4.2 IDFT - Inverse DFT
Như đã mô tả ở phần trước, giọng nói của chúng ta có tần số F0 - tần số cơ bản và các **formant** F1, F2, F3 ... Tần số F0 ở nam giới khoảng 125 Hz, ở nữ là 210 Hz, đặc trưng cho cao độ giọng nói ở từng người. Thông tin về cao độ này không giúp ích trong nhận dạng giọng nói, nên ta cần tìm cách để loại thông tin về F0 đi, giúp các mô hình nhận dạng không bị phụ thuộc vào cao độ giọng từng người.

![](https://images.viblo.asia/4b200b07-e961-42b0-acc2-7c42d494a442.png)

Trong hình này, tín hiệu chúng ta thu được là đồ thị 3, nhưng thông tin quan trọng chúng ta cần là phần 2, thông tin nhiễu cần loại bỏ là phần 1. Để loại bỏ đi thông tin về F0, ta làm 1 bước biến đổi Fourier ngược (IDFT) về miền thời gian, ta thu được Cepstrum. Nếu để ý kỹ, ta sẽ nhận ra rằng tên gọi **cepstrum** thực ra là đảo ngược 4 chữ cái đầu của **spectrum**.

Khi đó, với Cepstrum thu được, phần thông tin liên quan tới F0 và phần thông tin liên quan tới F1, F2, F3 ... nằm tách biệt nhau như 2 phần khoanh tròn trong hình 4. Ta chỉ đơn giản lấy thông tin trong đoạn đầu của cepstrum (phần được khoanh tròn to trong hình 4). Để tính MFCC, ta chỉ cần lấy 12 giá trị đầu tiên.

Phép biến đổi IDFT cũng tương đương với 1 phép biến đổi DCT **discrete cosine transformation**. DCT là 1 phép biến đổi trực giao. Về mặt toán học, phép biến đổi này tạo ra các **uncorrelated features**, có thể hiểu là các feature độc lập hoặc có độ tương quan kém với nhau. Trong các thuật toán Machine learning, **uncorrelated features** thường cho hiểu quả tốt hơn. Như vậy sau bước này, ta thu được 12 Cepstral features.

# 5. Extract MFCC
Như vậy, mỗi frame ta đã extract ra được 12 Cepstral features làm 12 feature đầu tiên của MFCC. feature thứ 13 là năng lượng của frame đó, tính theo công thức: 

![](https://images.viblo.asia/d37707d8-59d1-4e64-bb06-ea1c6626d010.jpeg)

Trong nhận dạng tiếng nói,  thông tin về bối cảnh và sự thay đổi rất quan trọng. VD tại những điểm mở đầu hoặc kết thúc ở nhiều phụ âm, sự thay đổi này rất rõ rệt, có thể nhận dạng các âm vị dựa vào sự thay đổi này. 13 hệ số tiếp theo chính là đạo hàm bậc 1 (theo thời gian) của 13 feature đầu tiên. Nó chứa thông tin về sự thay đổi từ frame thứ $t$ đến frame $t+1$. Công thức:

$$d(t) = \frac{c(t+1) - c(t-1) }{2}$$

Tương tự như vậy, 13 giá trị cuối của MFCC là sự thay đổi $d(t)$ theo thời gian - đạo hàm của $d(t)$, đồng thời là đạo hàm bậc 2 của $c(t)$. Công thức:

$$b(t) = \frac{d(t+1) - d(t-1) }{2}$$

Vậy, từ 12 cepstral feature và power feature thứ 13, ta đạo hàm 2 lần và thu được 39 feature. Đây chính là MFCC feature. Cùng nhìn lại toàn bộ quá trình để tạo ra MFCC:
![](https://images.viblo.asia/0aecb56e-2e56-4df2-9bdf-340a211bd0e5.jpeg)

# 6. Kết luận
Như vậy trong 2 phần này, mình đã cố gắng cung cấp những kiến thức nền tảng cần thiết cho xử lí giọng nói. Trong thời gian tới, mình sẽ cố gắng viết về mô hình nhận dạng tiếng nói Auto Speech Recognition (ASR), về HMM, GMM ... và nhiều thứ liên quan.

# Tài liệu tham khảo:
[Speech Recognition](http://www.ee.columbia.edu/~stanchen/spring16/e6870/outline.html)

[https://cmusphinx.github.io/wiki/tutorialconcepts: Basic concepts of speech recognition](https://cmusphinx.github.io/wiki/tutorialconcepts/)

[https://www.cs.bham.ac.uk/~pxc/nlp/NLPA-Phon1.pdf](https://www.cs.bham.ac.uk/~pxc/nlp/NLPA-Phon1.pdf)

[Speech Recognition](https://medium.com/@jonathan_hui/speech-recognition-phonetics-d761ea1710c0)
# Deep-Emotion: Nhận diện Cảm xúc Khuôn mặt Sử dụng Mạng Chú ý Tích chập

Kho lưu trữ này cung cấp triển khai PyTorch của bài báo nghiên cứu, [Deep-Emotion](https://arxiv.org/abs/1902.01019).

**Lưu ý:** Đây không phải là phiên bản chính thức được mô tả trong bài báo.

## Kiến trúc
- Một khung học sâu đầu-cuối dựa trên mạng tích chập có cơ chế chú ý.
- Cơ chế chú ý được tích hợp thông qua mạng biến hình không gian (Spatial Transformer Networks).

<p align="center">
  <img src="net_arch.PNG" width="960" title="Kiến trúc Deep-Emotion">
</p>

## Dữ liệu
Triển khai này sử dụng các tập dữ liệu sau:
- [FER2013](https://www.kaggle.com/c/challenges-in-representation-learning-facial-expression-recognition-challenge/data)
- [CK+](https://ieeexplore.ieee.org/document/5543262)
- [JAFFE](https://www.researchgate.net/publication/220013358_The_japanese_female_facial_expression_jaffe_database)
- [FERG](https://homes.cs.washington.edu/~deepalia/papers/deepExpr_accv2016.pdf)

## Yêu cầu
Đảm bảo rằng bạn đã cài đặt các thư viện sau:
- PyTorch
- torchvision
- OpenCV
- tqdm
- Pillow (PIL)
```bash
pip install -r requirements.txt
```

## Cấu trúc kho lưu trữ
Kho lưu trữ này được tổ chức như sau:
- [`main`](/main.py): Chứa thiết lập cho tập dữ liệu và vòng lặp huấn luyện.
- [`visualize`](/visualize.py): Bao gồm mã nguồn để đánh giá mô hình trên dữ liệu kiểm tra và kiểm tra thời gian thực sử dụng webcam.
- [`deep_emotion`](/deep_emotion.py): Định nghĩa lớp mô hình.
- [`data_loaders`](/data_loaders.py): Chứa lớp dữ liệu.
- [`generate_data`](/generate_data.py): Thiết lập [tập dữ liệu](https://www.kaggle.com/c/challenges-in-representation-learning-facial-expression-recognition-challenge/data).

### Chuẩn bị dữ liệu
1. Tải xuống tập dữ liệu từ [Kaggle](https://www.kaggle.com/c/challenges-in-representation-learning-facial-expression-recognition-challenge/data).
2. Giải nén `train.csv` và `test.csv` vào thư mục `./data`.

### Cách chạy
**Thiết lập tập dữ liệu**
```bash
python main.py [-s [True]] [-d [data_path]]

--setup                 Thiết lập tập dữ liệu lần đầu
--data                  Thư mục chứa các tệp dữ liệu
```

**Để huấn luyện mô hình**
```bash
python main.py [-t] [--data [data_path]] [--hparams [hyperparams]]
              [--epochs] [--learning_rate] [--batch_size]

--data                  Thư mục chứa các tệp huấn luyện và xác thực
--train                 True khi huấn luyện
--hparams               True khi thay đổi các siêu tham số
--epochs                Số lượng epochs
--learning_rate         Giá trị tốc độ học
--batch_size            Kích thước lô huấn luyện/xác thực
```

**Để xác thực mô hình**
```bash
python visualize.py [-t] [-c] [--data [data_path]] [--model [model_path]]

--data                  Thư mục chứa ảnh kiểm tra và tệp CSV kiểm tra
--model                 Đường dẫn đến mô hình đã huấn luyện
--test_cc               Tính toán độ chính xác kiểm tra
--cam                   Kiểm tra mô hình trong thời gian thực với webcam kết nối qua USB
```

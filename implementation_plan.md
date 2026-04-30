# Kế hoạch Thực hiện Đồ án Transformer - Phân loại Cảm xúc

Kế hoạch này giúp bạn hoàn thành đồ án trong 3 tuần, đảm bảo đáp ứng đầy đủ yêu cầu kỹ thuật và đạt điểm cao trong báo cáo.

## Lộ trình Lập trình (Coding Roadmap)

### Tuần 1: Nền tảng & Self-Attention
- **Ngày 1-2**: Cài đặt môi trường (`pip install -r requirements.txt`) và chạy `python data_utils.py`. Quan sát cấu trúc file trong `data/processed/`.
- **Ngày 3-5**: Cài đặt hàm `scaled_dot_product_attention` trong `model.py`.
    - Tính `scores = (Q @ K.T) / sqrt(d_k)`
    - Áp dụng `softmax` trên chiều cuối.
    - Tính `output = weights @ V`.
- **Ngày 6-7**: Chạy test unit có sẵn trong `model.py` để đảm bảo Tensor Shape chính xác. Chạy thử baseline MLP trong `train.py` để lấy kết quả tham chiếu.

### Tuần 2: Hoàn thiện Encoder Block & Huấn luyện
- **Ngày 8-10**: Cài đặt `FeedForwardNetwork` và ghép nối `TransformerEncoderBlock`.
    - Đảm bảo thứ tự: Attention -> Residual Add & LayerNorm -> FFN -> Residual Add & LayerNorm.
- **Ngày 11-12**: Chạy `python train.py` với cấu hình mặc định. Theo dõi Loss và Accuracy trên tập Val.
- **Ngày 13-14**: Sửa lỗi (nếu có) và đảm bảo Val Accuracy đạt > 70% như yêu cầu.

### Tuần 3: Thực nghiệm & Tối ưu (Điểm cộng)
- **Ngày 15-17**: Chạy `--run_all` để lấy kết quả cho 3 cấu hình Transformer và 1 MLP. Lưu lại các file `.pt` và log.
- **Ngày 18-19**: Thực hiện các phần điểm cộng (nếu đủ thời gian):
    - Padding Mask (Rất quan trọng để mô hình không nhìn vào các token `[PAD]`).
    - Dropout để giảm Overfitting.
- **Ngày 20-21**: Chạy `visualize.py` để xuất Heatmap cho các câu tiêu biểu.

---

## Cấu trúc Báo cáo Chi tiết (6-10 trang)

### Mục 1: Kiến trúc mô hình (1-2 trang)
- **Hình vẽ**: Vẽ lại sơ đồ Transformer Encoder (có thể dùng Draw.io hoặc vẽ tay sạch sẽ). Đừng copy ảnh mạng.
- **Công thức**: Trình bày công thức Self-Attention, FFN, và giải thích ý nghĩa của `sqrt(d_k)`.
- **Thành phần**: Giải thích vai trò của Positional Encoding và LayerNorm.

### Mục 2: Kết quả thực nghiệm (2-3 trang)
- **Bảng so sánh**: Lập bảng gồm 4 cột (Model, Train Acc, Val Acc, Test Acc) cho 4 cấu hình (MLP, Default, Large, Small).
- **Biểu đồ**: Chèn biểu đồ Learning Curve (Loss/Acc qua từng epoch) của model tốt nhất.
- **Nhận xét**: Phân tích model nào tốt nhất. Giải thích tại sao model lớn có thể bị overfitting trên tập dữ liệu nhỏ (600 mẫu).

### Mục 3: Phân tích Attention (1-2 trang)
- **Heatmap**: Chèn ít nhất 3 heatmap.
- **Phân tích**:
    - Câu tích cực: Model tập trung vào tính từ nào? (vd: "wonderful", "great").
    - Câu phủ định: Model có nhìn vào từ "not" không?
    - Nhận xét về độ phân tán của Attention.

### Mục 4: Phân tích lỗi (Error Analysis) (1 trang)
- Lập bảng 5-10 câu model đoán sai.
- Phân loại lỗi: Từ vựng lạ (Out-of-vocabulary), câu quá ngắn, hoặc câu mang tính mỉa mai.

### Mục 5: Kết luận & Bài học (0.5 trang)
- Tóm tắt kết quả.
- Bài học về việc xử lý Tensor và cơ chế Attention.

---

## Các điểm cần lưu ý để đạt điểm A
1. **Giải thích code**: Khi vấn đáp, phải giải thích được tại sao dùng `matmul` hay `transpose(-2, -1)`.
2. **Padding Mask**: Nếu làm được phần này, điểm sẽ rất cao vì nó cho thấy bạn hiểu sâu về xử lý ngôn ngữ tự nhiên.
3. **Định dạng**: Báo cáo trình bày sạch sẽ, dùng LaTeX cho công thức toán học.

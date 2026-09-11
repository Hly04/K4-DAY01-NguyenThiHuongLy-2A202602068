# Báo cáo bài thực hành Ngày 1 – Đọc nhãn từ đầu ra YOLO11

**Ngày chạy:11/9/2026**

**Runtime Colab: GPU ** CPU/GPU

**Python / PyTorch / Ultralytics: Ultralytics 8.4.145**

**Checkpoint: yolo11n-cls.pt, yolo11n.pt, yolo11n-seg.pt ** `yolo11n-cls.pt`, `yolo11n.pt`, `yolo11n-seg.pt`

**Thay đổi so với notebook nguồn: không ** Không / mô tả rõ thay đổi

> ZIP do notebook tạo có tên `<KHOA>-DAY01-report.zip` (ví dụ: `K4-DAY01-report.zip`). Giải nén rồi đặt trực tiếp `REPORT.md` và
> `day1_lab_outputs/` vào thư mục `report/` của repository tạo từ template. Không ghi họ tên, MSSV,
> email, số điện thoại hoặc dữ liệu cá nhân khác. Nộp link repository trên VLearn; tài khoản VLearn xác
> định người nộp.

## 1. Phân loại ảnh – prediction cấp ảnh

Nguồn evidence: `classification_predictions.json`, sample `traffic`.

- Record hạng 1 (`class_id`, `class_name`, `rank`, `score`, `taxonomy_name`):
"sample_id": "traffic",
    "coco_image_id": 210273,
    "image_width": 640,
    "image_height": 428,
    "task": "image_classification",
    "taxonomy_name": "ImageNet-1K",
    "model_file": "yolo11n-cls.pt",
    "model_sha256": "c62d41bf9625777760018bf914d2e6cd472420ccd01706d97a61cb6c82502bd7",
    "ultralytics_version": "8.4.145",
    "rank": 1,
    "class_id": 468,
    "class_name": "cab",
    "score": 0.510915
- Record này mô tả toàn ảnh như thế nào? 
Mô tả theo nhãn duy nhất ở mức độ toàn bộ bức ảnh (image-level label); dự đoán chủ thể nổi bật nhất bao trùm ngữ cảnh ảnh (ở đây là xe taxi / cab với độ tin cậy ~51.1%) chứ không chỉ ra vị trí tọa độ hay ranh giới cụ thể của xe trong ảnh.
- Ai định nghĩa class list mà checkpoint có thể dự đoán?
Quy định rõ tiêu chí ưu tiên: chọn chủ thể chiếm diện tích lớn nhất (dominant subject), chủ thể nằm ở vị trí trung tâm, hay chuyển bài toán sang dạng phân loại đa nhãn (multi-label classification).
- Vì sao cần giữ cả ID, tên lớp và tên taxonomy?
ID (468): Tối ưu cho hệ thống máy tính lưu trữ, truy vấn và xử lý logic số nguyên.

Tên lớp (cab): Giúp người kiểm tra (annotator/reviewer) đọc hiểu ngay ý nghĩa ngữ nghĩa của nhãn.

Tên taxonomy (ImageNet-1K): Xác định rõ không gian định nghĩa nhãn và phiên bản nguồn (tránh nhầm lẫn khi cùng tên gọi nhưng định nghĩa khác nhau giữa các bộ dữ liệu như COCO, OpenImages hay ImageNet)
- Nếu ảnh có nhiều chủ thể, guideline cần quy định điều gì?
Quy định rõ tiêu chí ưu tiên: chọn chủ thể chiếm diện tích lớn nhất (dominant subject), chủ thể nằm ở vị trí trung tâm, hay chuyển bài toán sang dạng phân loại đa nhãn (multi-label classification).
- Vì sao model score không phải ground truth?
Model score (0.510915) chỉ là xác suất/độ tự tin của thuật toán dựa trên dữ liệu đã học trong quá khứ; mô hình hoàn toàn có thể nhầm lẫn, suy đoán sai hoặc bị nhiễu ngữ cảnh so với sự thật khách quan (ground truth) ngoài đời thực.
## 2. Phát hiện vật thể – lớp và box cho từng object

Nguồn evidence: `detection_predictions.json` và `visuals/detection_predictions.png`, sample `kitchen`.

- Một record (`class_name`, `score`, `bbox_xyxy`, `bbox_width`, `bbox_height`): class_name: bus

score: 0.912558

bbox_xyxy: [93.17, 187.95, 223.01, 320.91]

bbox_width: 129.84

bbox_height: 132.96
- Diễn giải vị trí box bằng lời:Vật thể nằm ở khu vực phía bên trái và khoảng giữa theo chiều dọc của bức ảnh (kích thước ảnh 640x428). Góc trên-trái của hộp bao bắt đầu tại tọa độ (x = 93.17, y = 187.95) và góc dưới-phải kết thúc tại (x = 223.01, y = 320.91), tạo thành một vùng bao có kích thước xấp xỉ 130 x 133 pixel.
- So sánh số prediction ở hai threshold:Ở threshold thấp (ví dụ 0.25 hoặc 0.35): Số lượng bounding box được dự đoán nhiều hơn đáng kể, bao phủ được nhiều vật thể nhỏ, bị che khuất một phần hoặc có độ tin cậy vừa phải.

Ở threshold cao (ví dụ 0.70): Số lượng bounding box giảm đi rõ rệt, chỉ giữ lại các đối tượng rõ nét, kích thước lớn và mô hình có độ chắc chắn cao (như chiếc bus với điểm số ~0.91 này).
- Điều gì thay đổi đối với độ bao phủ và khối lượng reviewer cần xem? Threshold thấp làm tăng độ bao phủ (Recall cao, hạn chế bỏ sót đối tượng) nhưng làm tăng khối lượng công việc của reviewer vì xuất hiện nhiều dự đoán sai hoặc nhiễu (False Positives).

Threshold cao giúp giảm tải cho reviewer do các dự đoán đều rất chuẩn xác (Precision cao), nhưng giảm độ bao phủ và tăng nguy cơ sót vật thể thực tế (False Negatives).
- Đề xuất một quy tắc box chặt:Bounding box phải bao khít các cạnh biên ngoài cùng nhìn thấy được của vật thể (pixel-tight), sai số lề thừa hoặc cắt phạm không vượt quá 2–3 pixel, và không được để lọt khoảng trống của nền nếu không bị giới hạn bởi hình dạng khối chữ nhật.
- Với object bị che khuất/cắt mép, điều gì cần guideline hoặc escalation quyết định?
Guideline cần quy định rõ tỷ lệ phần trăm tối thiểu của vật thể nhìn thấy được để tiến hành vẽ box (ví dụ: chỉ gán nhãn nếu nhìn thấy trên 20%), đồng thời chỉ định cụ thể vẽ bounding box ôm theo phần thực tế quan sát được (visible box) hay ước lượng cả phần bị vật khác che khuất/nằm ngoài rìa ảnh (amodal box).
## 3. Phân đoạn theo từng đối tượng – polygon cho mỗi instance

Nguồn evidence: `segmentation_predictions.json` và `visuals/segmentation_prediction.png`, sample `kitchen`.

- Một record (`instance_id`, `class_name`, `score`, số điểm và một phần `polygon_xy`): instance_id: traffic-001

class_name: bus

score: 0.925745

Số điểm: (xem số cặp tọa độ trong trường polygon_xy của record này, thường khoảng 30–60 điểm)

Một phần polygon_xy: [[95.3, 188.72], [102.1, 188.72], [224.08, 205.4], ...] (trích 2–3 cặp tọa độ đầu trong mảng polygon của record)
- Polygon bổ sung chi tiết gì so với box? Bổ sung đường bao pixel thực tế theo đúng hình dáng, đường cong và viền mép của vật thể. Trong khi box là hình chữ nhật bao trùm luôn cả khoảng không nền xung quanh, polygon tách biệt hoàn toàn phần thân xe (bus) khỏi phần mặt đường và cảnh nền bên cạnh.
- `instance_id` dùng để làm gì và không phải loại ID nào? Dùng để định danh duy nhất từng cá thể vật thể độc lập xuất hiện trong cùng một bức ảnh (ví dụ: phân biệt xe bus thứ nhất traffic-001 với các xe khác trong ảnh).

Không phải là class_id (mã định danh loại/danh mục đối tượng) và không phải là track_id (mã theo dõi xuyên suốt nhiều frame trong video).
- Đề xuất một quy tắc biên mask: Đường bao đa giác (polygon) phải ôm sát theo viền ngoài của vật thể với sai số không quá 2 pixel, không được cắt lẹm vào kết cấu thân xe và không bao gồm bóng đổ (cast shadow) trên mặt đường.
- Với vùng mờ/tiếp xúc/che khuất, điều gì cần guideline hoặc escalation quyết định?
Guideline cần quy định rõ:

Ranh giới phân chia khi xe bus tiếp xúc hoặc bị che khuất bởi người đi bộ/phương tiện khác ở phía trước.

Xử lý vật thể bị chia cắt thành nhiều mảng riêng biệt (dùng multi-polygon hay tách thành các instance khác nhau).

Cách vẽ ranh giới đối với vùng bánh xe bị nhòe chuyển động (motion blur) hoặc vùng gầm xe bị khuất tối tiếp giáp mặt đường.
## 4. Vòng đời và kiểm tra chất lượng

`ảnh thô → guideline → ground truth → huấn luyện → prediction → QC/rework`

| Tác vụ | Đơn vị/định dạng ground truth | Lỗi hoặc điểm mơ hồ quan sát được | Annotator làm gì? | Reviewer xem gì? |
| --- | --- | --- | --- | --- |
| Phân loại ảnh | Nhãn đơn / ID lớp cấp ảnh (Single-label class ID, ví dụ class_id, class_name) |Ảnh chứa nhiều chủ thể ngang hàng (ví dụ: vừa có xe bus vừa có taxi); chủ thể chính ở quá xa hoặc bị che khuất  | Xác định chủ thể trọng tâm theo quy tắc ưu tiên trong guideline; gán đúng mã nhãn tương ứng	| Đối chiếu nhãn được chọn với ngữ cảnh ảnh; xác minh việc tuân thủ quy tắc ưu tiên và tính nhất quán của taxonomy  |
| Phát hiện vật thể | Bounding box dạng tọa độ pixel ([x_min, y_min, x_max, y_max] kèm class_id) |  Box vẽ quá rộng lọt nhiều nền thừa; vẽ lẹm mất mép vật thể; bỏ sót vật thể nhỏ hoặc vật thể nằm ở rìa ảnh| Kéo box ôm sát mép ngoài nhìn thấy được của vật thể; chọn đúng danh mục lớp cho từng box | Kiểm tra độ ôm khít (tightness), tỷ lệ bỏ sót đối tượng (Recall), và tính chính xác của nhãn phân loại |
| Instance segmentation | Tập điểm đa giác khép kín (Polygon [[x1, y1], [x2, y2], ...] kèm instance_id, class_id) | 	Đa giác quá thô thiếu điểm; gộp chung hai vật thể liền kề vào một mask; viền lem vào bóng đổ hoặc nền | Chấm điểm bao khít đường viền pixel thực tế; tách riêng từng cá thể độc lập thành các instance riêng |Soi kỹ độ mịn của viền mask, kiểm tra ranh giới tiếp xúc giữa các vật thể và tính toàn vẹn của đa giác  |

## 5. An toàn dữ liệu

- Một quy tắc bảo vệ dữ liệu: Tuyệt đối không tải dữ liệu ảnh hoặc kết quả gán nhãn ra ngoài môi trường thực hành được chỉ định; không chia sẻ hình ảnh chứa thông tin nhận dạng cá nhân (PII) như khuôn mặt rõ nét, biển số xe hoặc thông tin riêng tư lên các nền tảng công cộng.
- Nếu thấy ảnh hoặc dữ liệu không đúng phạm vi, tôi sẽ dừng và báo cho:
Giảng viên phụ trách học phần hoặc người quản trị/giám sát hệ thống phòng lab (Lab Administrator / Data Owner).
## 6. Danh sách bằng chứng

- [x] `classification_predictions.json`
- [x] `detection_predictions.json`
- [x] `segmentation_predictions.json`
- [x] `IMAGE_ATTRIBUTION.md`
- [x] `visuals/classification_top5.png`
- [x] `visuals/detection_predictions.png`
- [x] `visuals/segmentation_prediction.png`
- [x] Ô validation cuối notebook báo `PASS`.
- [x] Không có họ tên, MSSV hoặc dữ liệu nhạy cảm trong báo cáo/output.

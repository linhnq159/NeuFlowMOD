# Rà soát NeuFlow MOD — 2026-10-14

## Phạm vi và bằng chứng

Đọc sâu các đường chạy `tool/train.py`, model NeuFlow + segmentation head, reader ảnh/mask mới, dataset flow cũ, training/distillation gốc, evaluation, inference và export. Kiểm tra cú pháp toàn bộ **424 file Python**, gồm cả code trong `external`: **không có lỗi cú pháp**. Các thư viện vendor không tham gia đường chạy chính chỉ được kiểm tra cú pháp và phần tích hợp liên quan, không phải kiểm chứng đầy đủ mọi thuật toán của chúng.

Chạy lại `python checks/check_frame_training.py`: **21/21 kiểm tra qua** trên CPU, PyTorch 2.9.1, gồm gradient của từng `delta_flow` S16/S8 từ loss mask, giữ mọi prediction và BatchNorm training, train 8/infer 10 qua checkpoint, resume và định dạng mask PNG/BMP. Đã chạy các đoạn tái hiện riêng bằng autograd, forward hooks, dataset giả và optimizer thật. Không có CUDA trong môi trường này, nên chưa kiểm chứng AMP/CUDA/DDP hoặc benchmark trên dataset thật.

Mốc backbone của luồng mới hiện đúng yêu cầu: **epoch 1–299 đóng băng, từ epoch 300 học với LR bằng 0,1 lần LR head**. Cặp dữ liệu mới đúng thứ tự **(hiện tại t, quá khứ t − frame_skip, mask của t)**. Resume với cùng cấu hình đã được kiểm tra khớp kết quả chạy liên tục.

**Mục tiêu MOD đã xác nhận:** dữ liệu chỉ có ảnh và mask ground-truth, không có nhãn optical flow. Luồng chuẩn là hai frame → NeuFlow S16/S8 → flow cuối → segmentation head → loss mask. Cùng loss segmentation cập nhật head và backbone khi backbone được mở khóa; không dùng loss optical flow riêng. Các nhận định về loss flow, teacher flow và script pretraining/distillation cũ dưới đây thuộc luồng legacy, không phải thành phần cần bổ sung cho training MOD.

P1: lỗi làm sai supervision, hỏng đường chạy hoặc mất tính độc lập của đánh giá. P2: lỗi có điều kiện, suy giảm kiến trúc, hiệu năng hoặc tính tái lập. P3: nợ kỹ thuật cần dọn để hạn chế lỗi về sau.

## P1 — cần ưu tiên xử lý

### 1. Loss CUSTOM đẩy false negative theo hướng ngược nhãn

**Vị trí:** `external/ILNet/loss/custom_loss.py:29`; mặc định được chọn ở `tool/train.py:54`.

Phần intersection dùng `sigmoid(pred) * (sigmoid(pred) > threshold) * target`. Với foreground thật nhưng dự đoán thấp hơn ngưỡng, phần thưởng trong tử số bị bỏ, còn xác suất vẫn nằm trong mẫu số. Gradient descent do đó tiếp tục giảm logit của foreground bị bỏ sót.

**Đã tái hiện:** target=1, logit=−3, xác suất≈0,04743. Với threshold=0,2, gradient logit **+0,000756**, làm xác suất giảm. Với threshold=0, gradient **−0,029721**, có hướng cập nhật đúng.

**Cải thiện:** dùng soft intersection `pred * target`, để threshold ở bước đánh giá/inference; cân nhắc BCE + soft Dice/Tversky. Có thể dùng `--loss BCE` trong lúc sửa CUSTOM. Bổ sung kiểm tra gradient cho foreground bị dự đoán thấp và mask rỗng.

### 2. train.py gốc nhận sai kiểu dataset, có thể lặp mãi mà không học

**Vị trí:** `train.py:107`, `train.py:119`; `data_utils/datasets.py:702`.

`build_train_dataset()` trả `(train_dataset, val_dataset)` nhưng script gốc nhận cả tuple làm dataset. Với batch size mặc định 32 và `drop_last=True`, DataLoader có 0 batch; `total_steps` không tăng, vòng `while` không kết thúc.

**Đã tái hiện:** dataset thật trong tuple có 10 mẫu, nhưng loader tạo theo cách của script gốc có **0 batch**.

**Cải thiện:** unpack kết quả; kiểm tra dataset/loader không rỗng trước khi vào vòng training; dừng ngay trong vòng batch khi đạt `num_steps`. Lỗi này thuộc script gốc, không phải `tool/train.py` mới.

### 3. Validation và resume trong distillation chưa khớp API

**Vị trí:** `distillation_trainer.py:342`, `distillation_trainer.py:350`, `distillation_trainer.py:117`, `distillation_trainer.py:389`; `load_model.py:10`.

`validate()` đọc `self.val_loader` nhưng constructor chỉ lưu `val_dataset`, nên bước validation cuối training lỗi. Dataset hiện trả tuple, trong khi hàm này đòi dictionary. Nó còn truyền 4 đối số vào loss chỉ nhận 3 đối số. Các đường save dùng hai schema khác nhau: `model/optimizer/epoch` và `student_state_dict/optimizer_state_dict/scheduler_state_dict`. Resume trong `train_epoch()` nhận riêng model weights từ `my_load_weights()`, nên không tìm lại được optimizer/epoch/total_steps.

**Đã tái hiện:** `AttributeError: ... has no attribute 'val_loader'`; sau khi bổ sung loader giả, lỗi tiếp theo là `DistillationLoss.forward() takes 4 positional arguments but 5 were given`. Helper load trả riêng key trọng số, bỏ metadata checkpoint.

**Cải thiện:** thống nhất loader, batch contract và schema checkpoint; restore metadata từ checkpoint gốc; lưu/khôi phục scaler, counters và trạng thái RNG phù hợp.

### 4. valid mask không thực sự loại bỏ nhãn lỗi trong loss flow

**Vị trí:** `distillation_loss.py:31`; `loss.py:19`.

Distillation nhận `valid` nhưng cả task loss lẫn direction loss đều tính trên mọi pixel, kể cả pixel bị loại do teacher flow vượt `max_flow`. Loss flow thông thường nhân sai số với mask; `0 * Inf` hoặc `0 * NaN` vẫn tạo NaN. Nếu không có pixel valid, EPE trung bình trên tensor rỗng cũng là NaN.

**Đã tái hiện:** valid toàn 1 và toàn 0 cho cùng loss distillation **300,7000**. Teacher flow Inf với valid toàn 0 làm loss flow và EPE **NaN**.

**Cải thiện:** chọn pixel hữu hạn/valid trước khi tính sai số, chuẩn hóa theo số pixel valid và quy định rõ cách xử lý batch không có nhãn hợp lệ.

### 5. DDP ở các script cũ chưa giữ đồng nhất trạng thái giữa rank

**Vị trí:** `train.py:211`, `train.py:217`; `train.py:111`; `distillation_trainer.py:95`; `train_distillation.py:194`.

LR ở `train.py` gốc chỉ giảm trong nhánh rank 0. Các rank còn lại tiếp tục dùng LR cũ, dù DDP đồng bộ gradient, khiến trọng số bắt đầu lệch nhau. Sampler sử dụng số GPU cục bộ/local rank thay vì world size/global rank. Đường distillation tạo DDP nhưng entrypoint không khởi tạo process group.

**Bằng chứng:** đọc luồng điều khiển; chưa chạy DDP vì không có CUDA.

**Cải thiện:** khởi tạo distributed một lần ở entrypoint; dùng rank/world size từ process group; cập nhật scheduler ở mọi rank, chỉ giới hạn log/checkpoint ở rank 0; kiểm tra bằng tối thiểu hai process.

### 6. Validation custom cũ chia ngẫu nhiên cặp frame đã ghép

**Vị trí:** `data_utils/datasets.py:693`, `data_utils/datasets.py:686`.

Các cặp liên tiếp dùng chung frame, nên random split theo cặp có thể đưa cùng ảnh vào cả train và validation. Hai subset còn dùng chung dataset có augmentation ngẫu nhiên, khiến validation tiếp tục color jitter/erase/crop.

**Đã tái hiện:** 10 cặp liên tiếp, split 8/2 với seed 42, train và validation dùng chung frame **3, 4, 7, 8**; hai subset cùng trỏ về một dataset.

**Cải thiện:** tách video hoặc đoạn thời gian trước khi ghép cặp; tạo dataset validation với preprocessing cố định. Reader folder mới đã tách trước khi tạo cặp, không mắc lỗi split này trong kiểm tra hiện tại.

## P2 — ảnh hưởng trực tiếp đến luồng mới và chất lượng model

### 7. SEBlock nhánh 8 kênh không có tham số học được

**Vị trí:** `external/ILNet/model/LW_Opticalflow.py:28`.

`channel // reduction = 8 // 16 = 0`: hai linear layer có tensor trọng số rỗng. Vì không có bias, sigmoid luôn nhận 0 và đầu ra block bằng **0,5 × input**.

**Đã tái hiện:** hidden=0, số tham số=0, tỷ lệ output/input luôn bằng 0,5.

**Cải thiện:** dùng `hidden = max(1, channel // reduction)` hoặc reduction phù hợp cho từng nhánh. Thay đổi này cần xử lý trọng số và optimizer state của các SE layer khi nạp checkpoint head cũ.

### 8. PSP và boundary decoder được khai báo nhưng bỏ qua trong forward

**Vị trí:** `external/ILNet/model/LW_Opticalflow.py:498`, `external/ILNet/model/LW_Opticalflow.py:523`, `external/ILNet/model/LW_Opticalflow.py:565`.

Các module `psp` và `boundary_decoder` không tham gia đường forward hiện tại; nhiều conv phụ cũng chỉ xuất hiện trong đoạn comment. Head vì thế không thực hiện kiến trúc mà tên các module gợi ý. BatchNorm đầu ra `self.bn` chỉ chạy khi eval, không được học hay cập nhật running statistics trong training.

**Đã tái hiện:** forward hooks của PSP và boundary decoder đều **0 lần gọi**; BatchNorm đầu ra có grad=None và num_batches_tracked=0.

**Cải thiện:** quyết định module nào thuộc kiến trúc chính thức, nối chúng vào forward hoặc loại bỏ; thống nhất đường fused logits giữa train và eval. Cần đánh giá lại checkpoint khi thay cấu trúc.

### 9. Giới hạn batch_size ≥ 2 hiện không cần thiết

**Vị trí:** `tool/train.py:76`, `tool/train.py:148`, `tool/train.py:154`; hướng dẫn trong `tool/README.md`.

Giới hạn được thêm dựa trên giả định PSP BatchNorm sau pooling đang chạy, nhưng PSP hiện chưa được gọi. Model thực tế đã chạy forward/backward với **batch 1, ảnh 32×32**. Guard hiện tại chặn batch nhỏ và `drop_last=True` bỏ các mẫu cuối, không chỉ khi remainder=1.

**Cải thiện:** cập nhật guard/loader/test/docs theo forward thực tế. Nếu kích hoạt PSP về sau, xử lý BatchNorm ở nhánh pooling trước khi cho phép batch 1. Đây là điểm cần đính chính của thay đổi ở lượt trước.

### 10. Resume ở tool/train.py chưa kiểm tra cấu hình checkpoint

**Vị trí:** `tool/train.py:177`, `tool/train.py:239`, `tool/train.py:245`, `tool/train.py:249`.

Optimizer và LambdaLR được dựng từ CLI hiện tại trước khi nạp state, trong khi cấu hình đã lưu không được đối chiếu. LambdaLR không khôi phục nội dung hàm lambda. Thay optimizer, tổng epochs, warmup, số batch hoặc frame_skip có thể làm lỗi runtime hoặc đổi LR mà không có bước chuyển đổi rõ ràng.

**Đã tái hiện:** state MSAM nạp vào Adan mặc định làm `step()` lỗi unpack betas. Cùng scheduler state ở step 41, đổi số batch/epoch từ 10 xuống 5 làm LR kế tiếp đổi từ **0,000639496** sang **0,0000778370**.

**Cải thiện:** đọc và kiểm tra cấu hình trước khi dựng optimizer/scheduler; khôi phục mặc định từ checkpoint; cho phép override có quy tắc rõ ràng. Resume với cùng cấu hình đã qua kiểm tra.

### 11. Đính chính: giữ đầy đủ refinement và prediction optical flow

**Vị trí:** `NeuFlow/neuflow.py:225`, `NeuFlow/model_builder.py:127`.

Cả train và eval đều thực hiện đủ các bước refinement S16/S8 theo cấu hình; mặc định S8 có 8 iteration. Các cập nhật flow và context trung gian tham gia tạo flow cuối, nên không được diễn giải là những bước optical flow dư.

**Phân biệt số lần gọi:** trong train mode, `conv_s8` và upsampling chạy ở từng iteration; trong eval mode, chúng chỉ chạy ở iteration cuối. Số lần gọi upsampling 8 so với 1 không phải là số bước refinement: refinement S8 vẫn chạy đủ 8 lần ở cả hai mode.

**Cách supervision của MOD:** segmentation head dùng flow cuối và được giám sát bằng mask ground-truth. Không cần ground-truth hay loss optical flow riêng. Các output phụ được tính loss ở head là prediction mask, không phải nhãn flow.

**Đã sửa đường gradient:** trước đây `flow0.detach()` giữa các iteration chặn gradient trực tiếp tới `delta_flow` của 7 bước S8 đầu; trạng thái `iter_context` vẫn truyền gradient nên kiểm tra ở mức trọng số module chưa phát hiện giới hạn này. NeuFlow hiện thêm `detach_flow_between_iters=True` để giữ hành vi legacy, còn ModelBuilder luôn truyền `False`. Loss mask vì vậy truyền gradient qua toàn bộ chuỗi cập nhật flow khi backbone trainable, gồm S16 khi chạy nhiều hơn một iteration. Các kiểm tra dùng `retain_grad()` trên từng `delta_flow`, không chỉ kiểm tra trọng số dùng chung.

**Iteration cấu hình độc lập:** trainer nhận `--iters_s16`/`--iters_s8` (mặc định 1/8); API ModelBuilder cho phép nạp checkpoint train 8 và inference 10 bằng cùng trọng số. Giữ mọi prediction training và hành vi BatchNorm; inference vẫn chỉ upsample ở bước cuối. Cần so sánh chất lượng mask ở 8/10 và đo VRAM khi có CUDA, không suy ra rằng thêm iteration luôn tốt hơn.

**Kết luận đã chốt:** giữ nguyên pipeline, mọi refinement/prediction và cách cập nhật BatchNorm hiện tại; rút đề xuất final-only khỏi các thay đổi cần thực hiện. Mục này là đính chính nhận định, không phải một lỗi cần sửa.

### 12. Ảnh 16-bit bị mất dải động khi convert RGB

**Vị trí:** `data_utils/frame_mask_dataset.py:94`.

Reader chấp nhận TIFF/PNG nhưng convert trực tiếp sang RGB 8-bit. Với ảnh hồng ngoại 16-bit, nhiều giá trị bị clip thay vì scale, làm mất tương phản.

**Đã tái hiện:** giá trị uint16 `[0, 1000, 65535]` biến thành `[0, 255, 255]`.

**Cải thiện:** khai báo rõ chỉ nhận 8-bit hoặc thêm lựa chọn chuẩn hóa 16-bit theo một dải cố định phù hợp dữ liệu; tránh scale riêng mỗi frame nếu muốn giữ tương quan độ sáng giữa hai frame.

### 13. Ghép detection greedy làm Pd/Fa phụ thuộc thứ tự scan

**Vị trí:** `external/ILNet/model/metrics.py:230`.

Ghép target lần lượt với prediction gần nhất có thể dùng mất prediction mà một target khác cần. Kết quả phụ thuộc thứ tự region thay vì chỉ phụ thuộc hình học.

**Đã tái hiện:** một cặp prediction/target có Pd=0,5 và Fa=1/256; lật ngang đồng thời cả hai làm Pd=1,0 và Fa=0, dù chất lượng dự đoán không đổi.

**Cải thiện:** dùng maximum-cardinality bipartite matching trong ngưỡng khoảng cách, có thể dùng khoảng cách để phá tie; thêm kiểm tra metric bất biến với flip/transpose.

### 14. Dataset flow cũ thiếu một mẫu và bỏ qua img_size

**Vị trí:** `external/ILNet/utils/data.py:75`, `external/ILNet/utils/data.py:84`, `external/ILNet/utils/data.py:147`.

Cumulative index khởi tạo từ −1 nhưng `__len__()` trả last index thay vì số mẫu. Có 5 record thì len=4; một record thì len=0. Kích thước ảnh còn cố định `[960,544]`, không theo CLI, và normalization khác với ModelBuilder.

**Đã tái hiện:** cumulative last index=4 trả len=4.

**Cải thiện:** dùng cumulative lengths hoặc ConcatDataset; tôn trọng img_size; thống nhất normalization khi chuyển giữa flow cache và end-to-end frame training.

### 15. Inference segmentation đã bổ sung; export và script flow cũ còn riêng biệt

**Vị trí:** `infer.py:68`, `infer.py:16`, `infer.py:94`; `infer_hf.py:15`; `pt2onnx.py:125`.

Các script flow cũ đang dựng NeuFlow thuần, không có segmentation head. Checkpoint ModelBuilder dùng các key `backbone_op.*` và `seg_head.*`, không nạp trực tiếp vào NeuFlow bằng strict=True. Reader inference cũ còn giữ BGR từ OpenCV, trong khi training folder dùng RGB. Cặp inference cũ đi quá khứ → tương lai; model segmentation mới học hiện tại → quá khứ.

**Đã bổ sung:** `tool/infer.py` nạp checkpoint ModelBuilder, nhận hai ảnh hiện tại/quá khứ, dùng cùng RGB/resize/ổn định ảnh với training và xuất mask PNG 0/255 theo kích thước gốc của ảnh hiện tại. Cấu hình stabilization được lưu và khôi phục từ checkpoint; checkpoint cũ mặc định none. Chỉ warp ảnh quá khứ bằng SIFT + homography RANSAC, không warp ảnh hiện tại hay mask ground truth. Thiếu độ tin cậy thì giữ ảnh quá khứ gốc; vùng thiếu sau warp lấy từ ảnh hiện tại. Export segmentation và việc sửa các script flow cũ vẫn chưa thuộc thay đổi này.

**Kiểm chứng 2026-10-15:** 14/14 kiểm tra bổ sung trong `checks/check_stabilization.py` qua trên CPU, gồm chiều/quy đổi homography, biên, fallback, giữ nguyên current/mask, preprocessing train/infer, cấu hình resume và checkpoint thật → mask ở S8=8/10. Bộ 21/21 kiểm tra training hiện có tiếp tục qua. Chưa đánh giá chất lượng trên dữ liệu camera thật hoặc CUDA/AMP.

## P3 — dọn cấu trúc và API

### 16. Import phụ thuộc sys.path và tên module chung

**Vị trí:** `tool/train.py:12`, `train_distillation.py:12`, `train_distillation.py:16`; package `NeuFlow/utils/` và file `NeuFlow/utils.py`.

Đưa ILNet lên đầu sys.path làm các tên chung `train`, `loss`, `utils` trỏ sang vendor. Teacher còn dùng đường dẫn tuyệt đối đến repository ngoài và absolute imports `core.*`, dễ va chạm giữa SEA-RAFT và FlowFormer.

**Đã tái hiện:** sau `import tool.train`, `find_spec('train')` trỏ vào `external/ILNet/train.py`, `loss` trỏ vào `external/ILNet/loss/__init__.py`. Import `train` lúc này còn lỗi vì kéo dependency `tensorboardX` của script vendor.

**Cải thiện:** package-qualified/relative imports; hạn chế sửa sys.path toàn cục; một schema checkpoint và một nơi chứa flow utilities; khai báo phiên bản dependency đã kiểm chứng.

### 17. API cũ và code không hoạt động cần được loại bỏ hoặc sửa

- `data_utils/datasets.py:362` và `:531`, `data_utils/data_distillation.py:252` và `:421`: hai định nghĩa FlyingThings3D; định nghĩa sau ghi đè trước.
- `NeuFlow/model_builder.py:40`: template/track/mask_refine còn tham chiếu cfg và các module không tồn tại trong ModelBuilder hiện tại.
- `data_utils/evaluate.py:47`: validate_chairs format/return tensor EPE của ảnh cuối thay vì mean của epe_all; tensor nhiều pixel không format được bằng `%f`.
- `eval.py:15` đặt default `Sintel`, nhưng dispatch chỉ nhận `sintel`; fallback gọi validate_custom chưa import.
- `data_utils/transforms.py:41`: helper color_jitter_numpy_uint8 bỏ kết quả jitter và trả ảnh gốc; hiện callsite trong code này đang comment.
- `pt2onnx.py:156`: nhánh chọn device dùng `device == ...` thay vì phép gán, nên thay use_gpu không đổi device như dự kiến.
- Nhiều script inference/export chạy load model khi import, dùng đường dẫn tuyệt đối và các config trùng lặp. Distillation tạo MultiStepLR nhưng chỉ step sau toàn bộ training; temperature hiện không dùng trong direction loss, còn CLI --lr không điều khiển optimizer của trainer này.

**Cải thiện:** bỏ các định nghĩa/API không dùng, sửa helper đang được gọi trước; dùng main guard, cấu hình chung và dependency manifest. Đánh dấu script legacy để người dùng chọn đúng entrypoint.

## Thứ tự thực hiện đề xuất

1. Sửa CUSTOM hoặc chuyển tạm sang BCE; thêm kiểm tra hướng gradient.
2. Chuẩn hóa resume/config của training MOD; xử lý valid mask trong loss flow chỉ nếu tiếp tục dùng các script legacy.
3. Chốt kiến trúc head: SE width, PSP/boundary decoder, BN và batch-size policy.
4. Bổ sung inference/export segmentation cùng preprocessing và pipeline optical flow đầy đủ.
5. Sửa dataset contract/validation/DDP của script cũ và dọn package imports.

Mỗi thay đổi nên có kiểm tra tập trung vào hành vi bị lỗi; thay đổi head cần kế hoạch tương thích checkpoint và đánh giá trên dữ liệu thật trước khi so sánh chất lượng model.

## Kiểm tra định dạng mask — đã sửa

Reader frame/mask chỉ nhận mask PNG/BMP, kiểm tra cả extension khi quét folder và định dạng thực tế khi mở file. JPEG/WebP và các định dạng ảnh khác bị từ chối kèm đường dẫn, kể cả JPEG đổi đuôi PNG/BMP. Ảnh đầu vào vẫn nhận các định dạng hiện có. Giữ nhãn `> 0` cho 0/1, 0/255 và PNG palette cùng resize nearest-neighbor.

**Lý do:** mask thử nghiệm có 64 pixel foreground, sau nén JPEG và đọc bằng `> 0` thành 152 pixel (88 foreground giả). Lưu lại JPEG thành PNG không phục hồi nhãn gốc; cần xuất lại từ annotation. Các kiểm tra bao phủ định dạng bị từ chối, JPEG đổi đuôi, mask lossless và ảnh đầu vào JPEG.

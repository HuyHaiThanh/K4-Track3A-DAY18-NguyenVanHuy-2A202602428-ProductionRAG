# Failure Analysis — Lab 18: Production RAG

**Học viên:** NguyenVanHuy · **MSSV:** 2A202602428 · **Khóa:** K4 Track 3A
**Ngày hoàn thành:** 04/10/2026 (GMT+7)

## Phương pháp và kết quả

Chạy main.py trên 26 tài liệu đọc được, cùng test_set.json 20 câu. Baseline dùng 57 paragraph chunks và dense-only; Production dùng 121 children/26 parents, combined enrichment, Hybrid và CrossEncoder. Hai hệ thống dùng cùng generator gpt-4o-mini, temperature=0, system prompt và cấu hình RAGAS 0.1.22 (gpt-4o-mini, text-embedding-3-small). Hai báo cáo đều success, 20/20 điểm hợp lệ cho mỗi metric; 40/40 lượt generation success. Không sửa test_set hoặc corpus để tăng điểm.

| Metric | Naive Baseline | Production | Δ |
|---|---:|---:|---:|
| faithfulness | 0.7966 | 0.8444 | +0.0479 |
| answer_relevancy | 0.4920 | 0.6771 | +0.1851 |
| context_precision | 0.9250 | 0.9292 | +0.0042 |
| context_recall | 0.8833 | 0.9500 | +0.0667 |

Production đạt 3/4 metrics ≥0,70; Answer Relevancy vẫn 0,6771. Đây là một lần đo, không phải chứng minh thống kê hay ablation từng module.

## Bottom-5 theo trung bình bốn metrics

Chọn đúng năm mục trong reports/ragas_report.json, sắp xếp theo score tăng dần. Bottom-5 là thứ hạng tương đối; một số câu trả lời đúng nhưng điểm judge thấp. Chẩn đoán tự động chỉ là giả thuyết, dưới đây có đối chiếu context và corpus.

### #1 — Lương thử việc của nhân viên Junior mức cao nhất là bao nhiêu?

- **Question:** Lương thử việc của nhân viên Junior mức cao nhất là bao nhiêu?
- **Expected:** Junior cao nhất là 20.000.000 VNĐ/tháng. Lương thử việc = 85% x 20.000.000 = 17.000.000 VNĐ/tháng.
- **Got (nguyên văn):**

> Lương thử việc của nhân viên Junior (P1-P2) là 85% của mức lương gross cao nhất là 20.000.000 VNĐ.
>
> Tính toán:
> 20.000.000 VNĐ * 85% = 17.000.000 VNĐ.
>
> Vậy, lương thử việc của nhân viên Junior mức cao nhất là 17.000.000 VNĐ.

- **Nguồn context theo thứ tự:** bang_luong_2024.md, thu_viec.md, tam_ung.md.
- **Metrics:** faithfulness=0.2500; answer_relevancy=0.4992; context_precision=1.0000; context_recall=1.0000.
- **Mean:** 0.6873; **Worst metric:** faithfulness=0.2500.
- **Câu trả lời có đúng không?** Đúng về kết quả 17.000.000 VNĐ; nên ghi rõ đơn vị VNĐ/tháng. Đây không phải bằng chứng rõ rằng mô hình bịa ra con số.
- **Context có chứa đáp án không?** bang_luong_2024.md chứa mức Junior 12–20 triệu VNĐ/tháng và tỷ lệ thử việc 85%; thu_viec.md cũng xác nhận 85%. 20.000.000 × 0,85 = 17.000.000.
- **Có cần viết lại câu hỏi?** Không cần đổi ý nghĩa câu hỏi; có thể thêm “VNĐ/tháng” để chuẩn hóa đơn vị.
- **Error Tree:** Output đúng về số? → Có → Context có 20 triệu và 85%? → Có → Query rõ? → Có → Kiểm tra suy luận số học và judge M4 trước khi kết luận hallucination.
- **Root cause / mức chắc chắn:** Faithfulness 0,25 là cờ cần kiểm tra judge. Kết quả được suy ra bằng phép tính từ context, chưa được viết nguyên văn trong corpus. Chưa có trace statement-level của RAGAS để khẳng định vì sao judge loại từng claim.
- **Suggested fix / module:** M4: bổ sung kiểm tra số học độc lập và audit verdict của judge; generation: trả lời ngắn với công thức và đơn vị. M3: loại nguồn tạm ứng không giúp trả lời. Không thêm đáp án chuẩn vào context để làm đẹp điểm.

### #2 — Một nhân viên Senior có 9 năm thâm niên được nghỉ bao nhiêu ngày phép năm và lương trong khoảng nào?

- **Question:** Một nhân viên Senior có 9 năm thâm niên được nghỉ bao nhiêu ngày phép năm và lương trong khoảng nào?
- **Expected:** Theo chính sách v2024: 15 ngày cơ bản + 3 ngày thâm niên (9÷3=3) = 18 ngày phép. Lương Senior (P3-P4): 20-35 triệu VNĐ/tháng.
- **Got (nguyên văn):**

> Nhân viên Senior có 9 năm thâm niên sẽ được nghỉ **18 ngày phép năm** (15 ngày phép cơ bản + 3 ngày phép cộng thêm cho 9 năm thâm niên).
>
> Về lương, nhân viên này sẽ được hưởng lương cho 18 ngày phép năm.

- **Nguồn context theo thứ tự:** nghi_phep_nam_v2024.md, nghi_phep_khong_luong.md, nghi_phep_nam_v2023.md.
- **Metrics:** faithfulness=0.6667; answer_relevancy=0.7798; context_precision=0.8333; context_recall=0.5000.
- **Mean:** 0.6950; **Worst metric:** context_recall=0.5000.
- **Câu trả lời có đúng không?** Chưa đúng trọn vẹn: 18 ngày phép đúng, nhưng không trả khoảng lương 20–35 triệu VNĐ/tháng; câu “hưởng lương cho 18 ngày phép” trả lời lệch phần thứ hai.
- **Context có chứa đáp án không?** Context chỉ gồm nghi_phep_nam_v2024.md, nghi_phep_khong_luong.md và nghi_phep_nam_v2023.md. Bảng Senior (P3-P4) 20–35 triệu có trong bang_luong_2024.md nhưng không có ở context cuối.
- **Có cần viết lại câu hỏi?** Nên tách thành hai subqueries: “9 năm thâm niên có bao nhiêu ngày phép theo chính sách hiện hành?” và “Khung lương Senior P3-P4 là bao nhiêu?”. Đây là decomposition, không thay đổi mục tiêu câu hỏi.
- **Error Tree:** Output đầy đủ? → Không → Context có khung lương Senior? → Không → Query nhiều vế? → Có → Decompose ở retrieval và kiểm tra coverage trước generation.
- **Root cause / mức chắc chắn:** M2/M3 và bước chọn context chưa kiểm tra coverage theo từng vế. Ba parent khác nhau vẫn là ba tài liệu cùng chủ đề nghỉ phép. LLM cũng chưa abstain đúng khi phần lương thiếu bằng chứng.
- **Suggested fix / module:** M2: retrieve riêng cho từng vế rồi hợp nhất; M3/context selection: đảm bảo có cả nguồn phép và bảng lương; generation: checklist đủ hai vế và báo thiếu dữ liệu khi cần. Parent expansion không tự giải quyết multi-hop.

### #3 — Nhân viên tạm ứng 15 triệu, sau 20 ngày mới thanh toán. Bị phạt bao nhiêu?

- **Question:** Nhân viên tạm ứng 15 triệu, sau 20 ngày mới thanh toán. Bị phạt bao nhiêu?
- **Expected:** Thời hạn thanh toán là 15 ngày. Quá hạn 5 ngày, bị tính phí 2%/tháng trên 15.000.000 VNĐ = 300.000 VNĐ/tháng (tính pro-rata khoảng 50.000 VNĐ cho 5 ngày).
- **Got (nguyên văn):**

> Nhân viên tạm ứng 15 triệu VNĐ và thanh toán sau 20 ngày. Theo chính sách tạm ứng, khoản tạm ứng chưa thanh toán sau 15 ngày sẽ bị tính phí 2%/tháng trên số tiền chưa hoàn ứng.
>
> Số tiền chưa hoàn ứng là 15 triệu VNĐ.
>
> Phí phạt tính như sau:
> - Thời gian quá hạn: 20 ngày - 15 ngày = 5 ngày.
> - Tính phí phạt cho 1 tháng (30 ngày) là 2% trên 15 triệu VNĐ.
>
> Phí phạt cho 1 tháng = 15.000.000 VNĐ * 2% = 300.000 VNĐ.
>
> Vì 5 ngày là khoảng 1/6 tháng, nên phí phạt sẽ là:
> 300.000 VNĐ / 6 = 50.000 VNĐ.
>
> Như vậy, nhân viên sẽ bị phạt **50.000 VNĐ**.

- **Nguồn context theo thứ tự:** tam_ung.md, chi_phi_expense.md, nghi_phep_nam_v2024.md.
- **Metrics:** faithfulness=0.2222; answer_relevancy=0.6302; context_precision=1.0000; context_recall=1.0000.
- **Mean:** 0.7131; **Worst metric:** faithfulness=0.2222.
- **Câu trả lời có đúng không?** Khớp ground truth ở kết quả khoảng 50.000 VNĐ, nhưng phụ thuộc giả định tính pro-rata với tháng 30 ngày; câu trả lời chưa nêu đây là giả định.
- **Context có chứa đáp án không?** tam_ung.md quy định hạn 15 ngày và phí 2%/tháng. Context không quy định cách tính theo ngày hoặc tháng 30 ngày. 15.000.000 × 2% = 300.000/tháng; 300.000 × 5/30 = 50.000 chỉ hợp lệ khi chấp nhận quy đổi 30 ngày/tháng.
- **Có cần viết lại câu hỏi?** Có thể làm rõ “nếu tính pro-rata theo 30 ngày/tháng”; với câu gốc, hệ thống nên công khai giả định hoặc trả mức 300.000 VNĐ/tháng và nêu thiếu quy tắc tính ngày.
- **Error Tree:** Output khớp đáp án mẫu? → Có → Context hỗ trợ mọi giả định? → Không, thiếu 30 ngày/tháng → Query cần quy tắc tính ngày → Làm rõ giả định và kiểm tra ground truth.
- **Root cause / mức chắc chắn:** Thiếu quy tắc nghiệp vụ ở corpus, trong khi ground truth tự dùng pro-rata. Faithfulness thấp có cơ sở với giả định chưa được tài liệu hỗ trợ; đồng thời phép nhân/chia không sai.
- **Suggested fix / module:** M1/ingestion: bổ sung quy định pro-rata được xác nhận, không tự sửa corpus để khớp đáp án; generation: phân biệt dữ kiện và giả định; M4: kiểm tra lại ground truth và dùng calculator cho phần số học.

### #4 — Muốn mua thiết bị trị giá 55 triệu cần ai phê duyệt?

- **Question:** Muốn mua thiết bị trị giá 55 triệu cần ai phê duyệt?
- **Expected:** Đơn hàng trên 50.000.000 VNĐ cần Tổng Giám đốc (CEO) phê duyệt.
- **Got (nguyên văn):**

> Đối với đơn hàng trị giá 55 triệu VNĐ, cần có phê duyệt của Tổng Giám đốc (CEO) theo quy định trong quy trình mua sắm.

- **Nguồn context theo thứ tự:** tam_ung.md, mua_sam.md, chi_phi_expense.md.
- **Metrics:** faithfulness=1.0000; answer_relevancy=0.3965; context_precision=0.5833; context_recall=1.0000.
- **Mean:** 0.7450; **Worst metric:** answer_relevancy=0.3965.
- **Câu trả lời có đúng không?** Đúng trọng tâm: đơn hàng 55 triệu cần CEO. Answer Relevancy 0,3965 không tự chứng minh câu trả lời lệch ý.
- **Context có chứa đáp án không?** mua_sam.md có dòng trên 50.000.000 VNĐ → Tổng Giám đốc (CEO), nhưng context đứng thứ hai; tam_ung.md đứng đầu. Recall và Faithfulness đều 1,0.
- **Có cần viết lại câu hỏi?** Câu hỏi đã rõ; chuẩn hóa “55.000.000 VNĐ” có thể giúp numeric matching. Không cần rewrite để thay mục tiêu.
- **Error Tree:** Output đúng? → Có → Context có quy định >50 triệu? → Có, ở vị trí 2 → Query rõ? → Có → Sửa ranking và audit Relevancy judge.
- **Root cause / mức chắc chắn:** Có vấn đề xếp hạng ở M3/context selection: tài liệu tạm ứng đứng trước mua sắm. Relevancy thấp dù đáp án đúng cần audit judge; chưa có bằng chứng để quy toàn bộ điểm thấp cho prompt generation.
- **Suggested fix / module:** M2/M3: tăng khả năng phân biệt mua sắm với tạm ứng và numeric thresholds; M4: kiểm tra câu hỏi judge sinh ra/language và hiệu chuẩn bằng chấm thủ công. Giữ lại điểm gốc, không sửa số liệu.

### #5 — Thâm niên bao nhiêu năm thì được cộng thêm ngày phép?

- **Question:** Thâm niên bao nhiêu năm thì được cộng thêm ngày phép?
- **Expected:** Theo chính sách v2024 hiện hành, nhân viên có thâm niên từ 3 năm trở lên được cộng thêm 1 ngày phép cho mỗi 3 năm. Chính sách cũ v2023 yêu cầu 5 năm.
- **Got (nguyên văn):**

> Nhân viên có thâm niên từ **3 năm trở lên** được cộng thêm **1 ngày phép** cho mỗi 3 năm làm việc liên tục theo chính sách năm 2024.

- **Nguồn context theo thứ tự:** nghi_phep_nam_v2023.md, nghi_phep_nam_v2024.md, nghi_phep_khong_luong.md.
- **Metrics:** faithfulness=1.0000; answer_relevancy=0.5393; context_precision=0.5000; context_recall=1.0000.
- **Mean:** 0.7598; **Worst metric:** context_precision=0.5000.
- **Câu trả lời có đúng không?** Đúng chính sách hiện hành: mỗi 3 năm được thêm 1 ngày. Trả lời không nhắc lại bản cũ 5 năm nhưng trả đúng câu hỏi chính.
- **Context có chứa đáp án không?** Context thứ nhất là v2023 với mốc 5 năm; thứ hai là v2024 với mốc 3 năm và tuyên bố thay thế bản 1.0. Context thứ ba là nghỉ không lương.
- **Có cần viết lại câu hỏi?** Nếu người dùng hỏi hiện tại, có thể thêm “theo chính sách hiện hành”; nếu hỏi lịch sử 2023, phải giữ phiên bản được yêu cầu.
- **Error Tree:** Output đúng? → Có → Context đúng phiên bản ở vị trí đầu? → Không → Query không hỏi lịch sử → Lọc/ưu tiên hiệu lực ở M2/M3.
- **Root cause / mức chắc chắn:** M3 chỉ chấm relevance và chưa có bộ lọc hiệu lực. Bản cũ đứng đầu làm Context Precision còn 0,50, dù LLM đọc parent v2024 và trả lời đúng.
- **Suggested fix / module:** M1/metadata: parse version/effective_date/superseded từ văn bản gốc; M2 lọc theo thời điểm hỏi; M3/context selection ưu tiên bản hiện hành cho câu không hỏi lịch sử. Không tin metadata LLM sinh nếu chưa xác minh.

## Case study: Senior 9 năm thâm niên

1. Output chưa đủ: đúng 18 ngày phép, thiếu khoảng lương.
2. Context có đủ nguồn? Không, bang_luong_2024.md bị bỏ khỏi context cuối.
3. Query rewrite đã tồn tại? Chưa; pipeline không có module decomposition. Câu hỏi cần chia hai vế để retrieval phủ đủ.
4. Sửa M2/M3/context selection trước, sau đó kiểm tra generation có abstain khi thiếu dữ kiện. Cần test mới yêu cầu đồng thời 18 ngày và 20–35 triệu, không chỉ test định dạng.

Baseline cũng thiếu lương nhưng báo không tìm thấy; Production diễn giải phần lương lệch câu hỏi. Đây là ví dụ điểm trung bình tăng không đảm bảo mọi câu tốt hơn.

## Các ca cải thiện và rủi ro còn lại

- Thông tin lương: Baseline trả Nội bộ, Production trả Bí mật sau khi context có ky_luong.md.
- Nghỉ không lương 20 ngày: Baseline trả Giám đốc Nhân sự, Production trả CEO đúng điều khoản 16–30 ngày.
- Câu phiên bản trả đúng 15 ngày phép, mốc 3 năm và chu kỳ mật khẩu 120 ngày; bản cũ vẫn có thể đứng đầu retrieval. Chưa có bộ lọc hiệu lực riêng.
- M5 success=121 xác nhận JSON hợp lệ, không xác nhận nội dung sinh ra không có suy diễn. Enrichment chỉ phục vụ retrieval; generation và RAGAS nhận parent gốc.
- Hai PDF scan không có text layer bị bỏ qua; chưa OCR.
- Rerank trung bình 7379.6 ms trên CPU, chưa đạt 150 ms.

## Nếu có thêm một giờ

Ưu tiên decomposition/coverage cho multi-hop, lọc phiên bản dựa trên metadata xác minh, thêm calculator và công khai giả định tài chính, rồi audit judge cho các câu đúng nhưng score thấp. Đo lại trên cùng tập test và một tập holdout mới; không chỉ tối ưu để khớp 20 đáp án mẫu.

## Bằng chứng

- reports/ragas_report.json và reports/naive_baseline_report.json: toàn bộ kết quả từng câu.
- reports/production_trace.json và reports/naive_trace.json: answer, context, nguồn và latency.
- reports/comparison_report.json và reports/pipeline_latency_report.json: so sánh và breakdown.
- reports/pipeline_run.log: log lần chạy.

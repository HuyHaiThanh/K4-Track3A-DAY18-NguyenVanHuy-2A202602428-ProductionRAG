# Individual Reflection — Lab 18: Production RAG

**Họ và tên:** NguyenVanHuy (theo tên repository)
**MSSV:** 2A202602428
**Khóa:** K4 Track 3A
**Ngày hoàn thành:** 04/10/2026 (GMT+7)

Bản tổng kết này dựa trên code, trace và lỗi đã quan sát trong phiên lab có trợ lý hỗ trợ. Phần kế hoạch là đề xuất cho chatbot chính sách nội bộ; chưa có mô tả đồ án riêng từ học viên. Nội dung không gán cảm nhận cá nhân hoặc một lịch sử dự án chưa được cung cấp.

## Phần 1: Lecture Mapping

| Concept | Module / hàm | Observation và phân tích |
|---|---|---|
| Semantic chunking | M1: chunk_semantic(), _get_semantic_encoder() | Tách câu bằng regex và cosine giữa câu liền nhau; encoder được cache. Mốc demo M1 với threshold 0,85 tạo 208 chunks, có chunk chỉ 6 ký tự: cần đo retrieval thay vì mặc định threshold cao là tốt. Demo nối corpus, khác cách chunk từng tài liệu của main.py. |
| Parent–child và structure-aware | M1: chunk_hierarchical(), chunk_structure_aware(); pipeline.select_original_contexts() | Lần tích hợp tạo 121 children/26 parents từ 26 tài liệu. Parent ≤2048, child ≤256 ký tự. Parent được lưu và mở rộng sau rerank, tránh mất tiêu đề/tables. Context cuối giữ tối đa 3 parent khác nhau. |
| BM25 + Dense fusion | M2: segment_vietnamese(), BM25Search, DenseSearch, reciprocal_rank_fusion() | BM25 chuẩn hóa chữ thường và bỏ underscore nhất quán; dense bge-m3 1024 chiều trong Qdrant. RRF dùng 1/(60+rank+1), độc lập thang điểm. Senior vẫn thiếu bảng lương: hybrid không tự bảo đảm coverage cho mỗi vế. |
| Cross-encoder reranking | M3: CrossEncoderReranker._load_model(), rerank() | Xếp toàn bộ 20 candidates bằng cặp query/raw child trước khi lấy 3 parent khác nhau. Mean rerank 7379.6 ms, p95 9058.5 ms trên CPU, chưa đạt 150 ms. Bản v2023 vẫn có thể đứng trước v2024. |
| RAGAS 4 metrics | M4: evaluate_ragas(), failure_analysis(), save_report() | Hai lượt đủ 20/20 câu, status success. Faithfulness 0.8444, Relevancy 0.6771, Precision 0.9292, Recall 0.9500. Điểm judge cần đối chiếu corpus: phép tính Junior đúng nhưng Faithfulness 0,25. |
| Contextual embeddings / HyQA | M5: _enrich_single_call(), enrich_chunks() | 121/121 phản hồi cấu trúc hợp lệ, một request/chunk. Index context+summary+questions+raw text. Thời gian enrichment 343.6 giây. Parent gốc dùng cho answer/eval vì nội dung enrichment có thể suy diễn. |

### So sánh toàn pipeline

| Metric | Naive Baseline | Production | Δ |
|---|---:|---:|---:|
| faithfulness | 0.7966 | 0.8444 | +0.0479 |
| answer_relevancy | 0.4920 | 0.6771 | +0.1851 |
| context_precision | 0.9250 | 0.9292 | +0.0042 |
| context_recall | 0.8833 | 0.9500 | +0.0667 |

Ba metrics ≥0,70; Answer Relevancy vẫn dưới ngưỡng. Đây là so sánh hai cấu hình trên một tập 20 câu, không đủ để kết luận riêng M5 hay reranker gây toàn bộ mức tăng. Cần ablation và holdout để kiểm chứng. Cùng generator/model/prompt/temperature và cùng judge giúp giảm một nguồn gây nhiễu.

## Phần 2: Challenges & Debugging

### Đường dẫn môi trường Python

Lỗi quan sát trong tiến trình trợ lý: `No Python at` đường dẫn `C:\Users\admin\AppData\Local\Programs\Python\Python311\python.exe`. Kiểm tra pyvenv.cfg và executable cho thấy runner bị hạn chế truy cập. Chạy bằng cơ chế thực thi có quyền phù hợp xác nhận Python 3.11.9 và dependencies đã hoạt động; không tự kết luận môi trường học viên cài sai.

### Thay thế code và newline Windows

Lỗi M1: `ImportError: cannot import name 'chunk_hierarchical' from 'src.m1_chunking'`. Script tìm ranh giới LF trên file CRLF làm mất phần khai báo. Đã khôi phục scaffold, chuẩn hóa newline và kiểm tra ranh giới trước thay thế.

Lỗi tích hợp: `SyntaxError: unterminated string literal (detected at line 75)`. Chuỗi f-string bị chèn xuống dòng thật; sửa thành escape newline, chạy compile/lint và tests trước khi gọi API. Đây là lỗi của quá trình chỉnh file bằng script, không phải lỗi thuật toán RAG.

### Qdrant API và kiểm thử thật

Đã kiểm tra qdrant-client 1.19.1 có query_points() và server Docker kết nối được. Tests gốc M2 không phủ DenseSearch nên bổ sung regression với Qdrant in-memory và kiểm thử model thật trên server. Không xem pass tests dạng/type là bằng chứng chất lượng retrieval.

### Log PowerShell và PDF

Log có cảnh báo pypdf `Ignoring wrong pointing object 11 0 (offset 0)` và hai PDF scan không có text layer. Pipeline vẫn đọc được 26 tài liệu. Khi dùng PowerShell `*>`, stderr cảnh báo/progress bị trình bày thành NativeCommandError và wrapper báo exit 1 dù main đã đi đến cuối và cả hai RAGAS success. Đã thêm `python main.py --log reports/pipeline_run.log` để capture stdout/stderr bằng Python UTF-8 trong lần chạy sau, tránh cách redirect đó. Không gọi lại API chỉ để đổi định dạng log.

### Tích hợp parent và nội dung sinh

Scaffold ban đầu bỏ parents rồi gửi enriched_text trực tiếp cho LLM. Sửa bằng bảng tra parent_id, rerank raw child và đưa raw parent vào context. Tests xác nhận không đưa generated text thay cho nguồn gốc, không lặp parent và giữ metadata. Không có bộ lọc superseded riêng ở phiên bản hiện tại.

### Đánh giá và quyền API

RAGAS không có key hoặc provider lỗi phải được phân biệt với điểm đo thật bằng skipped/error; invalid metric giữ None và không bị chẩn đoán thành failure. Automatic approval review yêu cầu quyền rõ ràng để gửi corpus ra OpenAI; sau khi học viên cho phép đúng phạm vi, đã chạy đầy đủ. Không ghi API key vào log/repository.

### Kiến thức cần bổ sung

- Coverage-aware retrieval/decomposition cho multi-hop, không chỉ đa dạng parent.
- Version/effective-date parsing và truy vấn theo thời điểm.
- Numeric validation và cách nêu giả định nghiệp vụ.
- Calibration của LLM judge tiếng Việt; raw verdict giúp phân biệt lỗi model và lỗi evaluator.
- Profiling CPU/GPU và caching; không suy tốc độ từ mô tả model.

## Phần 3: Action Plan cho Project

### Project: Chatbot tra cứu chính sách nội bộ — kế hoạch đề xuất

Đây là dự án tham chiếu từ corpus lab, không khẳng định học viên đã có đồ án này trước đó. Nếu áp dụng vào đồ án khác, cần thay corpus và acceptance criteria tương ứng.

### Hiện trạng

Pipeline hiện có: hierarchical → combined enrichment → BM25+bge-m3+RRF → CrossEncoder → original-parent contexts → gpt-4o-mini → RAGAS. Rủi ro đã quan sát: Senior thiếu nguồn lương, phiên bản cũ xếp đầu, suy diễn pro-rata, relevance judge thấp, latency CPU cao và chưa OCR.

### Kế hoạch triển khai

1. **Chunking/ingestion:** dùng structure-aware cho chính sách có tables; giữ parent-child cho section dài. Thử threshold semantic trên tài liệu tự sự; kiểm tra OCR trước khi index PDF scan. Parse version, effective_date và thông tin thay thế từ text gốc.
2. **Search:** giữ Hybrid+RRF; decompose câu hỏi nhiều vế và thu candidate theo từng facet. Case Senior phải có cả bảng lương và chính sách phép hiện hành.
3. **Reranking/context:** lọc bản hết hiệu lực khi câu hỏi không yêu cầu lịch sử; vẫn hỗ trợ câu hỏi 2023. Thử GPU hoặc model nhẹ hơn, top-10 so với top-20; chỉ giảm candidates nếu recall holdout không giảm đáng kể.
4. **Generation/evaluation:** trả lời đủ từng vế, cite nguồn, dùng calculator cho arithmetic, công khai giả định như 30 ngày/tháng. RAGAS kết hợp check số học và review thủ công. Audit judge cho Junior và mua thiết bị 55 triệu.
5. **Enrichment:** cache theo hash nội dung+source+prompt/model; chỉ enrichment khi nguồn đổi. So sánh contextual-only với combined; vẫn giữ raw text cho evidence. Đo chi phí và retrieval gain thay vì mặc định nhiều text hơn là tốt hơn.

### Timeline và tiêu chí nghiệm thu

| Thời gian | Công việc | Tiêu chí kiểm chứng |
|---|---|---|
| Tuần 1, ngày 1–2 | Parse metadata hiệu lực và tests lịch sử/hiện hành | Câu hiện hành trả 15 ngày/3 năm/120 ngày; câu lịch sử vẫn lấy đúng bản cũ. |
| Tuần 1, ngày 3–4 | Decomposition và facet coverage | Senior trả đồng thời 18 ngày và 20–35 triệu VNĐ/tháng, context có hai nguồn cần thiết. |
| Tuần 1, ngày 5 | Calculator và rule ambiguity | Junior trả 17 triệu/tháng; tạm ứng nêu rõ giả định pro-rata nếu chưa có rule trong tài liệu. |
| Tuần 2, ngày 1–2 | Holdout ít nhất 20 câu mới; audit RAGAS/tiếng Việt | ≥3 metrics đạt 0,75; Relevancy mục tiêu ≥0,70; review tất cả ca bất đồng judge. |
| Tuần 2, ngày 3–4 | Cache enrichment, profile GPU/model nhẹ/top-k | Ghi p50/p95 rerank, mục tiêu 150 ms là mục tiêu cần thử nghiệm, chưa đạt trên CPU hiện tại. |
| Tuần 2, ngày 5 | Ablation và demo | So sánh bật/tắt M5 và M3 trên cùng test, ghi quality/latency/token cost; không dùng holdout để sửa đáp án mẫu. |

### Ưu tiên

Sửa coverage và version correctness trước tối ưu latency; sau đó đo lại trên holdout. Báo cáo này giữ nguyên điểm RAGAS đã đo, kể cả các ca có thể bị judge đánh giá chưa phù hợp.

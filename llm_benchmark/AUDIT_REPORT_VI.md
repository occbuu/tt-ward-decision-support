# Đánh giá benchmark Human–LLM Paper4 ngày 08/10/2026

Đã kiểm tra 7 cấu hình model, mỗi cấu hình 3 lượt zero-shot × 130 mục, tổng cộng 2.730 bản ghi phản hồi. Các file server gốc được giữ nguyên. Tất cả lượt có đủ ID, đúng câu hỏi/văn bản, prompt, hash input/code và cách parse. Gold export trùng bản chốt 123 strict-majority + 7 adjudicated. Accuracy, macro-F1, Cohen's κ và CI item-bootstrap gửi về đều khớp phép tính lại độc lập.

## Kết quả chính

| Cấu hình model | Đúng / 130 | Accuracy | Macro-F1 (8 lớp) | Cohen's κ | Invalid mỗi lượt |
|---|---:|---:|---:|---:|---:|
| Qwen2.5 14B Instruct | 105 | 80,8% | 0,737 | 0,765 | 2 |
| Gemma3 12B | 93 | 71,5% | 0,683 | 0,649 | 0 |
| Aya Expanse 8B | 89 | 68,5% | 0,577 | 0,615 | 1 |
| Llama3.1 8B | 83 | 63,8% | 0,509 | 0,563 | 0 |
| Qwen2.5 7B Instruct | 76 | 58,5% | 0,506 | 0,527 | 0 |
| Qwen2.5 3B Instruct | 57 | 43,8% | 0,459 | 0,319 | 0 |
| Qwen3 8B | 0 | 0% theo protocol | Không đánh giá khả năng semantic | Không đánh giá khả năng semantic | 130 |

Raw final content và predicted labels giống nhau ở cả ba lượt của từng model. Không gộp ba lượt thành n=390 và không coi SD=0 là bằng chứng tổng quát về độ ổn định. Thứ hạng là mô tả trên bộ test này, chưa có kiểm định superiority hoặc external holdout để chọn model.

Qwen2.5 14B có CI item-bootstrap accuracy 73,8–87,7%, κ 0,681–0,848. Sau khi tính đến câu trả lời trùng bằng cluster bootstrap, CI mở rộng thành accuracy 62,1–91,8%, κ 0,531–0,893. Chưa có cơ sở để kết luận model đứng đầu vượt trội một cách tổng quát.

## Những giới hạn cần giữ trong paper

1. **Qwen3 thất bại ở cấu hình generation, không phải bằng chứng model không hiểu tiếng Việt.** Cả 390 records có content rỗng, reasoning không rỗng, completion_tokens=32 và finish_reason=length. Giới hạn 32 token đã hết trước khi có final code. Giữ kết quả protocol thất bại, không lấy nhãn từ reasoning để cứu score. Cần một lượt chạy mới với cấu hình reasoning/token budget phù hợp và ghi rõ đây là protocol khác.
2. **Bộ test chỉ có 38 cặp question-context/text khác nhau.** Tổng 130 vẫn là số bản ghi khảo sát, nhưng đa dạng ngôn ngữ thấp. Các nhóm trùng có kích thước tới 18. Item CI không phản ánh đầy đủ bất định khi tổng quát hóa sang văn bản mới. Cluster bootstrap bổ sung lấy mẫu 38 nhóm với replacement, giữ mọi bản ghi trong từng nhóm, 2.000 lượt, seed 20261008.
3. **Có hai nhóm input giống hệt nhưng gold khác nhau.** Nhóm `S037, S072, S171, S193, S194` có HABIT=2, MOBIL=3; nhóm `S048, S090` có MOBIL=1, INFRA=1. Một classifier deterministic chỉ thấy đúng các input đã cung cấp không thể khớp quá 127/130 gold labels. Đây là giới hạn do gold/input không nhất quán, không phải human-reliability ceiling phổ quát. Chưa sửa nhãn; phải review theo protocol độc lập với output model trước khi phát hành benchmark cuối cùng.
4. **Không có mẫu GOVERN.** Support: INFRA=41, SOCIAL=30, HABIT=21, ECON=16, MOBIL=12, OTHER=7, ENVIR=3, GOVERN=0. Macro-F1 8 lớp giữ nguyên cách tính đã cấu hình, với GOVERN F1=0. Macro-F1 trên 7 lớp có support là sensitivity bổ sung: Qwen2.5 14B=0,842, Gemma3=0,781. Không suy diễn khả năng đủ 8 lớp.
5. **INFRA vẫn là điểm yếu của model đứng đầu.** Qwen2.5 14B recall INFRA=22/41=53,7%; 13 câu gold INFRA bị gán ECON và 4 bị gán HABIT. Mong muốn thêm nơi làm việc được human gán INFRA, model gán ECON: một số bất đồng có thể do quy tắc primary label/codebook, không đơn thuần lỗi hiểu ngôn ngữ. Hai invalid là chuỗi `ECON SOCIAL`.
6. **Khác biệt theo trường câu hỏi.** Qwen2.5 14B đạt 19/33 ở trường mong muốn thay đổi khu vực của residents và 4/11 ở SME, so với 30/33 ở lý do không dùng giao thông công cộng và 31/31 ở lý do chọn nơi sống. Các subgroup nhỏ chỉ có giá trị mô tả. ENVIR chỉ có ba mục nên điểm perfect không đủ để tuyên bố năng lực rộng.
7. **Không so trực tiếp model κ với human Fleiss κ/α như cùng một phép đo.** Khác reference, tập item và rater structure. Model–consensus κ cao hơn không chứng minh model tốt hơn người.

## Cập nhật LaTeX

Đã sửa trực tiếp `Paper4/Revision_20260926/latex/main.tex`: Highlights, Abstract, RQ4, analytical strategy, Section 6.6, limitations, Conclusion và data/code availability. Thêm Table 11 (so sánh model), Table 12 (class-level model đứng đầu) và Table 13 (sensitivity theo lớp có support và câu trả lời trùng). Bỏ các câu nói LLM evaluation chưa được chạy. Kết quả số liệu gốc và gold không bị thay đổi.

## File phục vụ kiểm tra lại

- `audit.json`: audit metadata, hash raw files, metrics, duplicate groups và đầy đủ CI.
- `model_comparison.csv`, `per_class.csv`, `subgroups.csv`, `errors.csv`, `gold_support.csv`, `model_metadata.csv`.
- `Paper4/Tables/T21_llm_model_comparison.csv` và `T22_llm_gold_support.csv`.
- Code tái lập: `Paper4/code-p4/audit_llm_results_20261008.py`.

Compiler tích hợp hiện thất bại ở bước chuẩn bị sandbox Windows, trước khi biên dịch nguồn. Bản LaTeX đã qua kiểm tra parser; chưa xác nhận PDF preview sau lần sửa này. Không tạo PDF thay thế và không sửa raw server outputs.

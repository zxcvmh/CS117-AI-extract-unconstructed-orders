import pandas as pd
import json

# 1. Khai báo đường dẫn (Thay đổi cho phù hợp với môi trường của bạn)
input_csv = '/home/zxcvmh/Projects/cs117/data/test-dataa-315.csv'
clean_test_jsonl = '/home/zxcvmh/Projects/cs117/data/test-ds-315.jsonl'

# 2. System Prompt phải Y HỆT tập train
SYSTEM_PROMPT = """Bạn là hệ thống AI trích xuất thông tin đơn hàng nông sản. Nhiệm vụ của bạn là đọc tin nhắn thô và trả về DUY NHẤT một đối tượng JSON hợp lệ. TUYỆT ĐỐI KHÔNG sinh thêm văn bản giải thích.

CẤU TRÚC JSON ĐẦU RA BẮT BUỘC:
{"intent": "", "customer_name": null, "phone_number": null, "location_tag": null, "payment_intent": "unknown", "delivery_note": null, "order_note": null, "items": [{"raw_product": "", "raw_quantity": "", "raw_budget": null}], "inquiries": []}

QUY TẮC ĐỊNH TUYẾN DỮ LIỆU (CẤM LÀM SAI):
1. 'intent': Phân loại 1 trong các giá trị: 'order' (đặt mới). Dùng 'mixed' nếu câu chứa ghép hành động (VD: Vừa đặt mới VỪA hỏi giá, hoặc đặt mới VỪA hủy đơn cũ).
2. 'payment_intent': Xác định phương thức thanh toán. Gán 'transfer' (nếu có: ck, chuyển khoản), 'cash' (tm, tiền mặt), 'debt' (nợ, ghi sổ, thiếu). Nếu không nhắc đến, BẮT BUỘC gán 'unknown'.
3. 'location_tag' & 'delivery_note': 
   - 'location_tag': CHỈ trích xuất địa danh, bến xe, tên kho (VD: bx q8, kho ngã ba).
   - 'delivery_note': CHỈ trích xuất thời gian hoặc yêu cầu vận chuyển (VD: chiều nay, t7, giao lẹ).
4. 'order_note': CHỈ trích xuất các chỉ thị vận hành ngoại lệ: yêu cầu gia công (trộn, xay), phàn nàn hàng lỗi, lời dặn về giá cả, hoặc câu lệnh hủy/sửa đơn cũ. 
   - LUẬT CÔ LẬP: TUYỆT ĐỐI KHÔNG đưa thông tin thanh toán (ck, tm) hoặc thông tin giao hàng (sáng mai, bx q8) vào trường này.
   - CHỐNG ẢO GIÁC: CHỈ giữ lại đúng chuỗi ký tự khách gõ, TUYỆT ĐỐI KHÔNG tự sinh thêm từ ngữ thể hiện cảm xúc cá nhân.
5. 'items': Trích xuất CHÍNH XÁC mặt hàng ('raw_product') và định lượng ('raw_quantity'). 
   - Gộp cả quy cách đóng gói (như /50, túi 10kg) vào chung 'raw_product'.
   - TUYỆT ĐỐI KHÔNG chuẩn hóa lỗi chính tả, KHÔNG tự thêm dấu tiếng Việt, KHÔNG tự bỏ chữ cái viết tắt của định lượng (ví dụ: giữ nguyên "15b", "tạ rưỡi"). 
   - Nếu hủy đơn/reorder không nêu mặt hàng, trả về [].
6. 'customer_name': CHỈ trích xuất TÊN RIÊNG (VD: Chú Sáu, Lan). Bỏ qua các đại từ chung chung (a, e, c, mình).
7. 'inquiries': Trích xuất nguyên văn câu hỏi của khách (hỏi giá, hỏi kho...). Trả về [] nếu không có.
8. Các trường không có thông tin: BẮT BUỘC gán giá trị null."""

def process_test_data():
    try:
        df = pd.read_csv(input_csv, header=None)
    except FileNotFoundError:
        print(f"❌ Không tìm thấy file {input_csv}")
        return

    all_chatml_samples = []
    error_count = 0

    print("🚀 Bắt đầu tạo file Test JSONL...")

    for index, row in df.iterrows():
        batch_string = row[0]
        if pd.isna(batch_string):
            continue

        safe_string = str(batch_string).replace('“', '"').replace('”', '"').replace('\u00A0', ' ')

        try:
            batch_data = json.loads(safe_string)

            for sample in batch_data:
                raw_text = sample["raw_text"]
                extracted_data = sample["extracted_json"]
                
                # Bỏ trường unsupported_items để match với schema đích
                extracted_data.pop("unsupported_items", None)
                
                assistant_response = json.dumps(extracted_data, ensure_ascii=False)
                
                chatml_format = {
                    "messages": [
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": raw_text},
                        {"role": "assistant", "content": assistant_response}
                    ]
                }
                all_chatml_samples.append(chatml_format)
                
        except json.JSONDecodeError:
            print(f"⚠️ Lỗi parse JSON ở dòng thứ {index + 1}. Đã bỏ qua.")
            error_count += 1

    print("💾 Đang ghi file Test...")
    with open(clean_test_jsonl, 'w', encoding='utf-8') as f:
        for item in all_chatml_samples:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')

    print(f"✅ HOÀN TẤT! Đã lưu {len(all_chatml_samples)} mẫu vào {clean_test_jsonl}")

if __name__ == "__main__":
    process_test_data()
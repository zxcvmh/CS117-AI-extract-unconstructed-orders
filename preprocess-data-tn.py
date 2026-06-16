import pandas as pd
import json
import os
import random  # Thư viện dùng để trộn data

# 1. Đường dẫn file
input_csv = '/home/zxcvmh/Projects/cs117/data/data cs117 - data_json 5-4.csv'
clean_jsonl = '/home/zxcvmh/Projects/cs117/data/trainning_ds_31-5.jsonl'

# 2. System Prompt (Phiên bản Thép - Bao quát toàn bộ Edge Cases & Noise)
# - Đã loại bỏ 'unsupported_items' khỏi schema mẫu.
# - Định nghĩa rõ ràng các intent và luật bóc tách.
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


def process_and_shuffle_data():
    try:
        df = pd.read_csv(input_csv, header=None)
    except FileNotFoundError:
        print(f"❌ Không tìm thấy file {input_csv}")
        return

    all_chatml_samples = []
    error_count = 0

    print("🚀 Bắt đầu đọc, sửa lỗi cú pháp và chuyển đổi sang ChatML...")

    for index, row in df.iterrows():
        batch_string = row[0]
        
        # Bỏ qua các dòng rỗng (NaN)
        if pd.isna(batch_string):
            continue

        # BƯỚC BẢO HIỂM 1: Fix lỗi ngoặc kép nghiêng và khoảng trắng rác ở dạng text thô
        # Làm điều này TRƯỚC KHI json.loads để chống Crash
        safe_string = str(batch_string).replace('“', '"').replace('”', '"').replace('\u00A0', ' ')

        try:
            # Ép chuỗi thành mảng JSON
            batch_data = json.loads(safe_string)

            # Làm phẳng mảng và biến đổi từng sample
            for sample in batch_data:
                extracted_data = sample["extracted_json"]
                
                # BƯỚC BẢO HIỂM 2: Loại bỏ triệt để 'unsupported_items' ngay trong bộ nhớ
                extracted_data.pop("unsupported_items", None)
                
                # Biến đổi thành chuỗi string chuẩn
                assistant_response = json.dumps(extracted_data, ensure_ascii=False)
                
                # Đóng gói theo chuẩn ChatML
                chatml_format = {
                    "messages": [
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": sample["raw_text"]},
                        {"role": "assistant", "content": assistant_response}
                    ]
                }
                
                all_chatml_samples.append(chatml_format)
                
        except json.JSONDecodeError:
            print(f"⚠️ Lỗi parse JSON ở dòng thứ {index + 1} trong file CSV. Đã bỏ qua.")
            error_count += 1
        except KeyError as e:
            print(f"⚠️ Thiếu key {e} ở dòng thứ {index + 1}. Đã bỏ qua.")
            error_count += 1

    # BƯỚC 3: XÁO TRỘN DỮ LIỆU (SHUFFLE)
    print(f"🔀 Đang xáo trộn ngẫu nhiên {len(all_chatml_samples)} mẫu dữ liệu để chống Overfitting...")
    random.shuffle(all_chatml_samples)

    # 4. Ghi trực tiếp ra file JSONL (bỏ qua file tạm trung gian)
    print("💾 Đang ghi dữ liệu vào file JSONL...")
    with open(clean_jsonl, 'w', encoding='utf-8') as f:
        for item in all_chatml_samples:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')

    print("-" * 50)
    print(f"✅ HOÀN TẤT ETL CHO QWEN LORA!")
    print(f"🎯 Số mẫu dữ liệu Sạch & Phẳng sẵn sàng train: {len(all_chatml_samples)}")
    print(f"📉 Số dòng lỗi bị loại bỏ: {error_count}")
    print(f"📁 File đầu ra lưu tại: {clean_jsonl}")

if __name__ == "__main__":
    process_and_shuffle_data()
import os
import google.generativeai as genai
from dotenv import load_dotenv

def check_gemini_connection():
    """
    Hàm này kiểm tra kết nối đến Google Gemini API bằng cách
    liệt kê các model có sẵn.
    """
    # 1. Tải các biến môi trường từ file .env
    load_dotenv()

    # 2. Lấy API key
    gemini_api_key = os.getenv("GEMINI_API_KEY")

    if not gemini_api_key:
        print("❌ LỖI: Không tìm thấy GEMINI_API_KEY trong file .env của bạn.")
        print("   Vui lòng đảm bảo file .env tồn tại và có chứa API key.")
        return

    print("Đã tìm thấy API key. Đang thử kết nối đến Google Gemini...")
    try:
        # 3. Cấu hình API với key đã lấy
        genai.configure(api_key=gemini_api_key)

        # 4. Thử liệt kê các model. Đây là cách tốt nhất để kiểm tra
        # API key có hợp lệ và kết nối có thông suốt hay không.
        print("Đang xác thực API key bằng cách liệt kê models...")
        models = list(genai.list_models())

        # Nếu có model chứa 'generateContent' trả về là thành công
        if any('generateContent' in m.supported_generation_methods for m in models):
            print("\n✅ KẾT NỐI THÀNH CÔNG!")
            print("   API key của bạn hợp lệ và đã sẵn sàng để sử dụng.")
            print("\n   Một vài model bạn có thể dùng:")
            for m in models:
                if 'generateContent' in m.supported_generation_methods:
                    print(f"   - {m.name}")
        else:
            print("\n⚠️ KẾT NỐI THÀNH CÔNG nhưng không tìm thấy model nào phù hợp.")
            print("   Vui lòng kiểm tra quyền của API key trong Google AI Studio.")

    except Exception as e:
        print("\n❌ KẾT NỐI THẤT BẠI!")
        print("   Đã xảy ra lỗi. Vui lòng kiểm tra các nguyên nhân sau:")
        print("   1. API key có thể không chính xác, đã hết hạn hoặc bị thu hồi.")
        print("   2. Bạn chưa bật 'Generative Language API' trong Google Cloud project.")
        print("   3. Máy tính của bạn gặp vấn đề về kết nối mạng hoặc bị tường lửa chặn.")
        print("\n--- Chi tiết lỗi kỹ thuật ---")
        print(e)
        print("-----------------------------")

if __name__ == "__main__":
    check_gemini_connection()


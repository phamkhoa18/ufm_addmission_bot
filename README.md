# 🎓 UFM Admission Chatbot API (Backend)

Hệ thống Chatbot AI Tư vấn Tuyển sinh Đại học Tài chính - Marketing (UFM) sử dụng cấu trúc RAG (Retrieval-Augmented Generation) và LangGraph để cung cấp thông tin chuẩn xác dựa trên dữ liệu tuyển sinh.

Dự án này là backend REST API được xây dựng bằng **FastAPI**, cung cấp Endpoint cho Frontend Web Widget và hệ thống Admin quản trị tri thức.

---

## 👨‍💻 Đơn Vị Phát Triển
- **Nhà thầu / Đơn vị triển khai:** [Công ty Chuyển đổi số VinCode](https://vincode.xyz/)
- **Core Developer/Architect:** IT-KitGiwang
- **Production System** dành cho Đại học Tài chính - Marketing (UFM)

---

## 🚀 Tính Năng Cốt Lõi (Core Features)
- **Truy vấn RAG Nâng Cao:** Sử dụng LangGraph Node-based routing để phân loại câu hỏi và tìm kiếm ngữ cảnh chính xác.
- **Quản Trị Tri Thức (Admin API):** Tích hợp Upload File, Chunking Data, và quản lý các chunk trực tiếp qua giao diện bảo mật JWT.
- **Bộ lọc Ngôn Ngữ & Context Curator:** Ngăn chặn câu hỏi ngoài lề (Off-topic) và lọc nhiễu trước khi Agent đưa ra câu trả lời.
- **Domain-Locked Security:** Hệ thống CORS linh hoạt, có custom Security check bảo vệ API Endpoint nội bộ.
- **Real-time Streaming:** Hỗ trợ Streaming Token giúp Frontend có Cảm giác gõ phím trực tiếp (Typing Effect).

---

## 📑 Cấu trúc thư mục Server

```
ufm_admission_bot/
│
├── app/
│   ├── api/          # Định tuyến API (chat_router.py, admin_router.py)
│   ├── core/         # Cấu hình hệ thống, Security, Prompts config
│   ├── services/     # LangGraph Logic, VectorDB, Models
│   └── utils/        # Logging, Helpers, Query Analyzer
│
├── static/           # HTML/CSS/JS cho trang demo /chat và /admin UI
├── main.py           # Entry point khởi tạo FastAPI
├── .env              # (Cần tạo) Biến môi trường
└── README.md         # File tài liệu này
```

---

## 🛠 Hướng Dẫn Cài Đặt & Chạy Server Local

### Bước 1: Yêu Cầu Môi Trường
- Python 3.10+
- Database: PostgresQL + PgVector (tuỳ thuộc vào Vector Store thiết lập)
- Môi trường ảo (Virtual Environment - Khuyên dùng)

### Bước 2: Clone & Cài Đặt Dependencies
Mở Terminal/CMD và điều hướng vào thư mục backend:
```bash
cd ufm_admission_bot
# Tạo và kích hoạt môi trường ảo (ví dụ: venv)
python -m venv venv
venv\Scripts\activate      # Môi trường Windows
# source venv/bin/activate # Nếu sử dụng Mac/Linux

# Cài đặt các thư viện yêu cầu (đảm bảo tệp requirements.txt đã đầy đủ)
pip install -r requirements.txt
```

### Bước 3: Cấu hình Môi trường
Tạo file `.env` tại thư mục root (`ufm_admission_bot/.env`) và cấu hình các biến cần thiết (API Keys cho LLM, Database URL, JWT Secret, v.v.):
```env
OPENAI_API_KEY=your_key_here
POSTGRES_URL=postgresql://user:password@localhost:5432/ufm_db
JWT_SECRET=your_jwt_secret
...
```

### Bước 4: Khởi Chạy Server
Sử dụng `uvicorn` để chạy app ở chế độ Development (Hot-Reload):
```bash
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

---

## 🌐 Các Đường Dẫn Hệ Thống (Endpoints)
Sau khi Server báo `Application startup complete`, bạn có thể truy cập:

- 🟢 **Swagger UI (Test API):** [http://localhost:8000/docs](http://localhost:8000/docs)
- 🟢 **Redoc (Tài liệu API API):** [http://localhost:8000/redoc](http://localhost:8000/redoc)
- 🤖 **Chat Demo UI:** [http://localhost:8000/chat](http://localhost:8000/chat)
- ⚙️ **Admin Dashboard:** [http://localhost:8000/admin](http://localhost:8000/admin)
- 💓 **Health Check:** `GET http://localhost:8000/health`

**Public Chat API Endpoint:**
`POST /api/v1/chat/message`
(Dành cho việc gọi từ Frontend UX/UI ReactJS Next.js).

---

## 🔒 Bảo Mật & CORS
CORS hiện tại được mở sẵn cho môi trường Local Development (`allow_origins=["*"]`). Ở môi trường Production, vui lòng sửa lại mảng Allowed Origins tại `main.py` để trỏ chính xác về domain Frontend Vercel của bạn và hạn chế Security leak.

---

> Lập trình với ❤️ bởi KitGiwang.

# 🐳 Hướng dẫn Chạy VectorDB (Docker)

## Yêu cầu

- **Docker Desktop** ≥ 4.x → [Tải Docker](https://docs.docker.com/get-docker/)
- **Docker Compose** v2 (đi kèm Docker Desktop)
- File `.env` (copy từ `.env.example`)

---

## 1. Chuẩn bị file `.env`

```bash
# Copy mẫu
cp .env.example .env

# Mở file .env và điền các giá trị BẮT BUỘC:
#   OPENROUTER_API_KEY=sk-or-v1-xxxxxxxx   ← Lấy từ https://openrouter.ai/keys
#   ADMIN_JWT_SECRET=chuoi-random-64-ky-tu ← Đổi lại secret riêng
#   POSTGRES_PASSWORD=...                   ← Mật khẩu DB (để mặc định cũng được)
```

---

## 2. Khởi động VectorDB (PostgreSQL + pgvector)

### Chỉ chạy Database (dev local — khuyên dùng)

```bash
docker compose up postgres_vector -d
```

Lệnh này sẽ:
- Pull image `pgvector/pgvector:pg16` (~150MB)
- Tạo database `ufm_admission_db` + user `ufm_admin`
- Chạy `database/init.sql` tự động (tạo bảng, index, extensions)
- Expose port `5432` ra localhost

### Kiểm tra DB đã sẵn sàng

```bash
# Health check
docker compose ps

# Kết nối test
docker exec -it ufm_pgvector psql -U ufm_admin -d ufm_admission_db -c "SELECT 1;"

# Kiểm tra pgvector extension
docker exec -it ufm_pgvector psql -U ufm_admin -d ufm_admission_db \
  -c "SELECT * FROM pg_extension WHERE extname = 'vector';"
```

---

## 3. Chạy toàn bộ hệ thống (DB + App)

```bash
docker compose up -d
```

Sau khi chạy xong:

| Service | URL | Mô tả |
|---------|-----|--------|
| **App** | http://localhost:8000 | FastAPI Server |
| **Chat UI** | http://localhost:8000/chat | Demo Chatbot |
| **Admin UI** | http://localhost:8000/admin | Upload dữ liệu |
| **Swagger** | http://localhost:8000/docs | API Documentation |
| **PostgreSQL** | localhost:5432 | VectorDB (pgvector) |

---

## 4. pgAdmin (Tùy chọn — Quản trị DB bằng giao diện)

```bash
# Chạy cùng pgAdmin
docker compose --profile admin up -d
```

Truy cập: http://localhost:5050
- Email: `admin@ufm.edu.vn` (hoặc giá trị `PGADMIN_EMAIL` trong `.env`)
- Password: `admin` (hoặc giá trị `PGADMIN_PASSWORD`)

Khi thêm Server trong pgAdmin:
- Host: `postgres_vector` (tên container, KHÔNG phải `localhost`)
- Port: `5432`
- Database: `ufm_admission_db`
- Username: `ufm_admin`
- Password: (giá trị `POSTGRES_PASSWORD` trong `.env`)

---

## 5. Chạy App local (KHÔNG Docker) + DB Docker

Cách phổ biến nhất khi dev:

```bash
# Bước 1: Chỉ chạy DB bằng Docker
docker compose up postgres_vector -d

# Bước 2: Đảm bảo .env có POSTGRES_HOST=localhost
# (Docker expose port 5432 ra localhost)

# Bước 3: Cài dependencies Python
pip install -r requirements.txt

# Bước 4: Chạy app local
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

---

## 6. Các lệnh thường dùng

```bash
# Xem logs DB
docker compose logs postgres_vector -f

# Xem logs App
docker compose logs app -f

# Dừng tất cả
docker compose down

# Dừng + XÓA DATA (reset hoàn toàn)
docker compose down -v

# Rebuild App (sau khi sửa code)
docker compose up app --build -d

# Backup database
docker exec ufm_pgvector pg_dump -U ufm_admin ufm_admission_db > backup.sql

# Restore database
docker exec -i ufm_pgvector psql -U ufm_admin ufm_admission_db < backup.sql
```

---

## 7. Cấu trúc Docker

```
docker-compose.yml
├── postgres_vector     ← pgvector/pgvector:pg16
│   ├── Volume: pgvector_data (persistent)
│   ├── Init: database/init.sql (chạy 1 lần)
│   └── Tuning: shared_buffers=256MB, work_mem=64MB
│
├── app                 ← Build từ Dockerfile (Python 3.11-slim)
│   ├── Depends on: postgres_vector (healthy)
│   └── Env: load từ .env
│
└── pgadmin [optional]  ← dpage/pgadmin4
    └── Profile: admin (chạy khi --profile admin)
```

---

## Troubleshooting

| Lỗi | Nguyên nhân | Fix |
|------|------------|-----|
| `port 5432 already in use` | PostgreSQL local đang chạy | `net stop postgresql` hoặc đổi `POSTGRES_PORT` |
| `FATAL: password authentication failed` | Sai password trong `.env` | Xóa volume: `docker compose down -v` rồi `up` lại |
| `connection refused` | DB chưa khởi động xong | Chờ health check pass: `docker compose ps` |
| `pgvector extension not found` | Dùng image PostgreSQL thường | Đổi sang `pgvector/pgvector:pg16` |

# API Commands - Quick Reference

Simple commands to use the Crypto Price API endpoints.

---

## 🚀 Start Server

```powershell
uvicorn main:app --reload --port 8000
```

---

## 📋 Standard Endpoint (Small CSV with addresses)

### Submit Request

```powershell
curl.exe -X POST "http://localhost:8000/prices" `
  -F "csv_file=@coins.csv" `
  -F "target_date=2025-10-31" `
  --output prices_2025-10-31.csv
```

### Debug Mode

```powershell
curl.exe -X POST "http://localhost:8000/prices/debug" `
  -F "csv_file=@coins.csv" `
  -F "target_date=2025-10-31"
```

---

## 🔄 Batch Endpoint (Large CSV with denominations)

### 1. Submit Job

```powershell
curl.exe -X POST "http://localhost:8000/prices/batch" `
  -F "csv_file=@large_file.csv" `
  -F "target_date=2025-10-31"
```

**Response:**

```json
{
  "job_id": "abc-123-xyz",
  "status": "queued"
}
```

### 2. Check Status

```powershell
curl.exe "http://localhost:8000/prices/batch/abc-123-xyz"
```

### 3. Download Result (when completed)

```powershell
curl.exe "http://localhost:8000/prices/batch/abc-123-xyz/download" `
  --output result.csv
```

### 4. List All Jobs

```powershell
curl.exe "http://localhost:8000/prices/batch"
```

### 5. Delete/Cancel Job

```powershell
curl.exe -X DELETE "http://localhost:8000/prices/batch/abc-123-xyz"
```

_Useful for cleaning up stuck or failed jobs_

---

## 🔍 Utility Endpoints

### Health Check

```powershell
curl.exe "http://localhost:8000/healthz"
```

### Get Config

```powershell
curl.exe "http://localhost:8000/config"
```

### Test Price Endpoint

```powershell
curl.exe "http://localhost:8000/test/price/1?target_date=2025-10-31"
```

_Replace `1` with CMC ID (1=BTC, 1027=ETH, etc)_

---

## 📝 CSV Formats

### Standard Endpoint Input (coins.csv)

```csv
coin,address
BTC,0x1234567890abcdef...
ETH,0xabcdef1234567890...
```

### Batch Endpoint Input (large_file.csv)

```csv
denomination
BTC
ETH
SOL
XRP
```

### Output Format

```csv
coin,price_usd_2025-10-31
BTC,67234.50
ETH,2456.78
SOL,145.23
```

---

## 🌐 Access Swagger UI

Open in browser:

```
http://localhost:8000/docs
```

---

## 💡 Common Use Cases

### Quick price check (small file)

```powershell
curl.exe -X POST "http://localhost:8000/prices" `
  -F "csv_file=@coins.csv" `
  -F "target_date=2025-10-31" `
  --output result.csv
```

### Large file processing

```powershell
# Submit
curl.exe -X POST "http://localhost:8000/prices/batch" `
  -F "csv_file=@huge_file.csv" `
  -F "target_date=2025-10-31"

# Wait and check (replace JOB_ID)
curl.exe "http://localhost:8000/prices/batch/JOB_ID"

# Download when ready
curl.exe "http://localhost:8000/prices/batch/JOB_ID/download" `
  --output final_result.csv
```

---

## 🔧 Troubleshooting

### View logs in terminal where server is running

### Check if server is running

```powershell
curl.exe "http://localhost:8000/healthz"
```

### Verify API key is set

```powershell
curl.exe "http://localhost:8000/config"
```

Look for `"api_key_configured": true`

---

## 📌 Notes

- Use `curl.exe` in PowerShell (not just `curl`)
- Replace `localhost` with server IP if remote
- Default port is `8000`
- Job IDs are UUIDs (e.g., `550e8400-e29b-41d4-a716-446655440000`)
- Date format: `YYYY-MM-DD`

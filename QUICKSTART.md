# Quick Start Guide

Get the Coin Price Extractor running in 5 minutes!

## 🚀 Step 1: Install Dependencies

```bash
# Activate your virtual environment (if using one)
.venv\Scripts\activate  # Windows
# source .venv/bin/activate  # macOS/Linux

# Install packages
pip install -r requirements.txt
```

## 🔑 Step 2: Configure API Key

Create a `.env` file in the project root:

```env
CMC_API_KEY=your_coinmarketcap_api_key_here
```

Get your free API key at: https://coinmarketcap.com/api/

## ▶️ Step 3: Start the Server

```bash
uvicorn main:app --reload --port 8000
```

You should see:

```
INFO:     Uvicorn running on http://127.0.0.1:8000
INFO:     Application startup complete.
```

## ✅ Step 4: Test It

### Choose Your Endpoint

**Two endpoints available:**

1. **`/prices`** - Standard endpoint for small CSVs with addresses (up to ~10k rows)
2. **`/prices/batch`** - Batch endpoint for large CSVs with symbols only (up to 1.5M rows)

---

### Method 1: Web Interface (Easiest)

1. Open your browser to: http://127.0.0.1:8000/docs
2. Click on `POST /prices` (standard) or `POST /prices/batch` (large files)
3. Click "Try it out"
4. Upload your CSV file (or use the sample `coins.csv` or `test_batch.csv`)
5. Click "Execute"
6. Download the result CSV (batch jobs: use the job_id to check status and download)

### Method 2: cURL (Command Line)

#### Standard Endpoint (Small Files with Addresses)

**Windows PowerShell:**

```powershell
# Standard endpoint - Test with the included sample file
curl.exe -X POST "http://localhost:8000/prices" `
  -F "csv_file=@coins.csv" `
  -F "target_date=2025-10-31" `
  --output prices_2025-10-31.csv

# File is also saved to output folder
Get-Content output\prices_2025-10-31.csv
```

#### Batch Endpoint (Large Files with Symbols)

**Windows PowerShell:**

```powershell
# Submit batch job
curl.exe -X POST "http://localhost:8000/prices/batch" `
  -F "csv_file=@test_batch.csv" `
  -F "target_date=2025-10-31"

# Response will contain job_id
# {"job_id":"abc-123-xyz","status":"queued"...}

# Check status (replace with your job_id)
curl.exe "http://localhost:8000/prices/batch/abc-123-xyz"

# Download when completed
curl.exe "http://localhost:8000/prices/batch/abc-123-xyz/download" `
  --output result_batch.csv
```

**macOS/Linux:**

```bash
# Standard endpoint
curl -X POST "http://localhost:8000/prices" \
  -F "csv_file=@coins.csv" \
  -F "target_date=2025-10-31" \
  --output prices_2025-10-31.csv

# Batch endpoint
curl -X POST "http://localhost:8000/prices/batch" \
  -F "csv_file=@test_batch.csv" \
  -F "target_date=2025-10-31"

# Check status and download (replace job_id)
curl "http://localhost:8000/prices/batch/abc-123-xyz"
curl "http://localhost:8000/prices/batch/abc-123-xyz/download" \
  --output result_batch.csv

# File is also saved to output folder
cat output/prices_2025-10-31.csv
```

### Method 3: Python Requests

```python
import requests

with open('coins.csv', 'rb') as f:
    response = requests.post(
        'http://localhost:8000/prices',
        files={'csv_file': f},
        data={'target_date': '2025-10-31'}
    )

# Save result
with open('prices_2025-10-31.csv', 'wb') as out:
    out.write(response.content)

print("✅ Prices saved to prices_2025-10-31.csv")
```

## 📝 CSV Format

### Standard Endpoint (`/prices`)

Your CSV must have two columns: `coin` and `address`

**Example (coins.csv):**

```csv
coin,address
USDC,0xa0b86991c6218b36c1d19d4a2e9eb0ce3606eb48
WETH,0xc02aaa39b223fe8d0a0e5c4f27ead9083c756cc2
AURA,CMkj12qHC9RjAUs1MED38Bt7gfyP3TbEpa1mcBno3RUY
BTC,
```

- `coin`: Name/symbol of cryptocurrency
- `address`: Contract address (optional - leave empty to use symbol lookup)
- Supports **Ethereum** (0x...), **Solana** (base58), **Stellar** (G...)

**Output Format:**

```csv
coin,address,price_usd_2025-10-31
USDC,0xa0b86991c6218b36c1d19d4a2e9eb0ce3606eb48,1.0000636021547014
WETH,0xc02aaa39b223fe8d0a0e5c4f27ead9083c756cc2,3577.6414664448266
AURA,CMkj12qHC9RjAUs1MED38Bt7gfyP3TbEpa1mcBno3RUY,0.000036263665228
BTC,,68234.12
```

---

### Batch Endpoint (`/prices/batch`)

Your CSV must have one column: `denomination`

**Example (test_batch.csv):**

```csv
denomination
BTC
ETH
SOL
XRP
USDT
USDC
```

- `denomination`: Symbol of cryptocurrency (BTC, ETH, SOL, etc.)
- For large files (up to 1.5M rows)
- Automatic deduplication of symbols

**Output Format:**

```csv
coin,price_usd_2025-10-31
BTC,109556.16
ETH,3847.08
SOL,187.21
XRP,2.51
USDT,0.9996
USDC,0.9997
```

**Output Format:**

The API returns a downloadable CSV **and** saves it to the `output/` folder.

**Files are saved to:**

- Standard: `output/prices_YYYY-MM-DD.csv`
- Batch: `output/batch_{job_id}_YYYY-MM-DD.csv`

## 🔍 Troubleshooting

### Problem: "CMC_API_KEY not found"

**Solution:** Make sure `.env` file exists and contains:

```env
CMC_API_KEY=your_actual_key_here
```

### Problem: All prices are NULL

**Solutions:**

1. Check date is not in the future (use past dates only)
2. Use debug endpoint: http://127.0.0.1:8000/docs → `/prices/debug`
3. Verify addresses are valid contract addresses

### Problem: "Module not found"

**Solution:** Install dependencies:

```bash
pip install -r requirements.txt
```

### Problem: Connection timeout

**Solution:** Increase timeout in `.env`:

```env
REQUEST_TIMEOUT=60.0
```

## 🎯 Common Tasks

### Check Batch Job Status

```powershell
# List all jobs
curl.exe "http://localhost:8000/prices/batch"

# Check specific job
curl.exe "http://localhost:8000/prices/batch/abc-123-xyz"

# Delete stuck/failed job
curl.exe -X DELETE "http://localhost:8000/prices/batch/abc-123-xyz"
```

### Get Debug Information

Use the debug endpoint to see detailed mapping info:

**Windows PowerShell:**

```powershell
curl.exe -X POST "http://localhost:8000/prices/debug" `
  -F "csv_file=@coins.csv" `
  -F "target_date=2025-10-31" | ConvertFrom-Json | ConvertTo-Json -Depth 10
```

**macOS/Linux:**

```bash
curl -X POST "http://localhost:8000/prices/debug" \
  -F "csv_file=@coins.csv" \
  -F "target_date=2025-10-31" | jq
```

### Check Configuration

```bash
curl http://localhost:8000/config
```

### Health Check

```bash
curl http://localhost:8000/healthz
```

### Enable Debug Logging

Add to `.env`:

```env
LOG_LEVEL=DEBUG
```

Restart the server to see detailed logs.

## 📚 Next Steps

- **Read COMMANDS.md** for quick command reference
- **Read BATCH_API.md** for detailed batch processing docs
- **Read the full README.md** for detailed documentation
- **Check OPTIMIZATIONS.md** to understand all features
- **Customize .env** for your needs
- **Try different dates** to get historical prices

## 🆘 Still Having Issues?

1. Enable debug logging: `LOG_LEVEL=DEBUG` in `.env`
2. Check the console output for error messages
3. Use the `/prices/debug` endpoint to see what's happening
4. Verify your CMC API key is valid and has quota remaining

## 💡 Pro Tips

- **Use PowerShell on Windows:** `curl.exe` instead of `curl` (avoids alias issues)
- **Choose the right endpoint:** `/prices` for addresses, `/prices/batch` for large symbol lists
- **Use batch for large files:** Processes up to 1.5M rows efficiently (chunked reading)
- **Monitor batch jobs:** Check status with `/prices/batch/{job_id}`
- **Delete stuck jobs:** Use DELETE endpoint to clean up failed jobs
- **Use the debug endpoint first** to understand which coins CMC recognizes
- **Keep dates in the past** (historical data only - current date is Nov 19, 2025)
- **Batch process** by including all coins in one CSV (more efficient)
- **Check `/config`** to verify your settings
- **Monitor logs** to track API usage and rate limiting
- **HTTP 400 errors are normal** - means coin isn't in CMC database
- **HTTP 403 errors on DEX are expected** - DEX API requires higher plan
- **Success rate of 70-90% is normal** for mixed coin lists

## ⚡ Performance & Rate Limits

### Standard Endpoint (`/prices`)

With the **Hobbyist Plan** (30 requests/minute):

- **27 coins**: ~37 seconds, ~21 requests
- **60 coins**: ~70 seconds, ~35 requests
- **100+ coins**: Automatically stays under 30 req/min limit

### Batch Endpoint (`/prices/batch`)

With the **Enterprise Plan** (120 requests/minute):

- **1.5M rows with 10 unique symbols**: ~1-2 minutes, ~3 requests
- **1.5M rows with 50k unique symbols**: ~8-22 hours, ~1,500 requests
- Processes in background - no timeout issues!

The API includes:

- ✅ Automatic 2.1s delay between requests
- ✅ Global rate limiter with async lock
- ✅ Exponential backoff on errors
- ✅ Chunked reading for large files (100k rows at a time)
- ✅ No risk of hitting rate limits!

---

**Ready to go!** 🎉

Start with the sample `coins.csv` file, then replace with your own data.

For advanced usage, see the full README.md.

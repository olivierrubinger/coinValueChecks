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

### Method 1: Web Interface (Easiest)

1. Open your browser to: http://127.0.0.1:8000/docs
2. Click on `POST /prices`
3. Click "Try it out"
4. Upload your CSV file (or use the sample `coins.csv`)
5. Click "Execute"
6. Download the result CSV

### Method 2: cURL (Command Line)

**Windows PowerShell:**

```powershell
# Test with the included sample file
curl.exe -X POST "http://localhost:8000/prices" `
  -F "csv_file=@coins.csv" `
  -F "target_date=2025-10-31" `
  --output prices_2025-10-31.csv

# File is also saved to output folder
Get-Content output\prices_2025-10-31.csv
```

**macOS/Linux:**

```bash
curl -X POST "http://localhost:8000/prices" \
  -F "csv_file=@coins.csv" \
  -F "target_date=2025-10-31" \
  --output prices_2025-10-31.csv

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

The API returns a downloadable CSV **and** saves it to the `output/` folder:

```csv
coin,address,price_usd_2025-10-31
USDC,0xa0b86991c6218b36c1d19d4a2e9eb0ce3606eb48,1.0000636021547014
WETH,0xc02aaa39b223fe8d0a0e5c4f27ead9083c756cc2,3577.6414664448266
AURA,CMkj12qHC9RjAUs1MED38Bt7gfyP3TbEpa1mcBno3RUY,0.000036263665228
BTC,,68234.12
```

**Files are saved to:** `output/prices_YYYY-MM-DD.csv`

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
- **Use the debug endpoint first** to understand which coins CMC recognizes
- **Keep dates in the past** (historical data only - current date is Nov 12, 2025)
- **Batch process** by including all coins in one CSV (more efficient)
- **Check `/config`** to verify your settings
- **Monitor logs** to track API usage and rate limiting
- **HTTP 400 errors are normal** - means coin isn't in CMC database
- **HTTP 403 errors on DEX are expected** - DEX API requires higher plan
- **Success rate of 70-90% is normal** for mixed coin lists

## ⚡ Performance & Rate Limits

With the **Hobbyist Plan** (30 requests/minute):

- **27 coins**: ~37 seconds, ~21 requests
- **60 coins**: ~70 seconds, ~35 requests
- **100+ coins**: Automatically stays under 30 req/min limit

The API includes:

- ✅ Automatic 2.1s delay between requests
- ✅ Global rate limiter with async lock
- ✅ Exponential backoff on errors
- ✅ No risk of hitting rate limits!

---

**Ready to go!** 🎉

Start with the sample `coins.csv` file, then replace with your own data.

For advanced usage, see the full README.md.

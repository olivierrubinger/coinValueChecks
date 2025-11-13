# Crypto Asset Price Extractor

**Version 2.0.0** - FastAPI application to fetch historical cryptocurrency prices from CoinMarketCap.

This version uses only standard library `csv` (no pandas dependency) and includes comprehensive error handling, retry logic, and logging.

## Features

- ✅ **Single CSV input** with `coin` and `address` columns
- ✅ **Multi-chain support**: Ethereum (0x addresses), Solana (base58), Stellar (G...)
- ✅ **Smart fallback**: Maps by contract address first, then falls back to symbol lookup
- ✅ **Dual-endpoint price fetching**: Uses `/v2/quotes/historical` and `/v1/ohlcv/historical`
- ✅ **Rate limit protection**: Global rate limiter ensures 30 req/min compliance
- ✅ **Retry logic**: Exponential backoff with configurable retries for API failures
- ✅ **Comprehensive logging**: Track all operations with configurable log levels
- ✅ **Input validation**: Date validation, CSV format checking, and error messages
- ✅ **Debug endpoint**: Detailed JSON output showing mapping and pricing details
- ✅ **Configuration endpoint**: View current settings and API status

## Installation

```bash
# Create virtual environment (recommended)
python -m venv .venv
.venv\Scripts\activate  # Windows PowerShell
# source .venv/bin/activate  # macOS/Linux

# Install dependencies
pip install -r requirements.txt
```

## Configuration

Create a `.env` file in the project root:

```env
# Required
CMC_API_KEY=your_coinmarketcap_api_key_here

# Optional - Customize behavior
MAX_RETRIES=3
RETRY_DELAY=1.0
BATCH_SIZE=20
REQUEST_TIMEOUT=40.0
RATE_LIMIT_DELAY=2.1
LOG_LEVEL=INFO
```

### Environment Variables

| Variable           | Default    | Description                                    |
| ------------------ | ---------- | ---------------------------------------------- |
| `CMC_API_KEY`      | _required_ | Your CoinMarketCap API key                     |
| `MAX_RETRIES`      | `3`        | Number of retry attempts for failed API calls  |
| `RETRY_DELAY`      | `1.0`      | Initial delay in seconds (exponential backoff) |
| `BATCH_SIZE`       | `20`       | Number of addresses/symbols per batch request  |
| `REQUEST_TIMEOUT`  | `40.0`     | HTTP request timeout in seconds                |
| `RATE_LIMIT_DELAY` | `2.1`      | Seconds between API requests (for 30 req/min)  |
| `LOG_LEVEL`        | `INFO`     | Logging level (DEBUG, INFO, WARNING, ERROR)    |

## Usage

### Start the Server

```bash
uvicorn main:app --reload --port 8000
```

The API will be available at `http://127.0.0.1:8000`

### API Documentation

Open `http://127.0.0.1:8000/docs` for interactive Swagger UI documentation.

### Endpoints

#### 1. `GET /healthz`

Health check endpoint.

**Response:**

```json
{
  "status": "ok",
  "version": "2.0.0"
}
```

#### 2. `GET /config`

Get current configuration (without sensitive data).

**Response:**

```json
{
  "version": "2.0.0",
  "cmc_base_url": "https://pro-api.coinmarketcap.com",
  "max_retries": 3,
  "retry_delay_seconds": 1.0,
  "batch_size": 20,
  "request_timeout_seconds": 40.0,
  "log_level": "INFO",
  "api_key_configured": true
}
```

#### 3. `POST /prices`

Main endpoint - returns CSV with prices.

**Parameters:**

- `csv_file` (required): CSV file with columns `coin` and `address`
- `target_date` (optional, default: `2025-10-31`): Target date in YYYY-MM-DD format
- `api_key` (optional): Override API key from environment

**CSV Input Format:**

```csv
coin,address
USDC,0xa0b86991c6218b36c1d19d4a2e9eb0ce3606eb48
WETH,0xc02aaa39b223fe8d0a0e5c4f27ead9083c756cc2
AURA,CMkj12qHC9RjAUs1MED38Bt7gfyP3TbEpa1mcBno3RUY
BTC,
```

**Response:** CSV file download with format:

```csv
coin,address,price_usd_2025-10-31
USDC,0xa0b86991c6218b36c1d19d4a2e9eb0ce3606eb48,1.0000636021547014
WETH,0xc02aaa39b223fe8d0a0e5c4f27ead9083c756cc2,3577.6414664448266
AURA,CMkj12qHC9RjAUs1MED38Bt7gfyP3TbEpa1mcBno3RUY,0.000036263665228
BTC,,68234.12
```

#### 4. `POST /prices/debug`

Debug endpoint - returns detailed JSON with mapping information.

**Parameters:** Same as `/prices`

**Response:**

```json
{
  "summary": {
    "total_rows": 27,
    "address_mapped": 13,
    "symbol_mapped": 2,
    "with_price": 15,
    "symbols_needed": ["BTC", "ETH"],
    "sym_to_id_sample": { "btc": 1, "eth": 1027 }
  },
  "rows": [
    {
      "coin": "USDC",
      "address": "0xa0b86991c6218b36c1d19d4a2e9eb0ce3606eb48",
      "address_decorated": "0xa0b86991c6218b36c1d19d4a2e9eb0ce3606eb48",
      "address_valid": true,
      "mapped_via_address": true,
      "mapped_via_symbol": false,
      "cmc_id": 3408,
      "price_usd": 1.0,
      "reason": null
    }
  ]
}
```

## CSV Input Format

Your CSV must have two columns: `coin` and `address`.

**Example:**

```csv
coin,address
USDC,0xa0b86991c6218b36c1d19d4a2e9eb0ce3606eb48
AURA,CMkj12qHC9RjAUs1MED38Bt7gfyP3TbEpa1mcBno3RUY
USA,GAYCVRGZH2tHms1c5sCprE2JEbuz8tJ9ZxCNUX1cKwWR
BTC,
WETH,0xC02aaA39b223FE8D0A0e5C4F27eAD9083C756Cc2
```

- **coin**: Name or symbol of the cryptocurrency
- **address**: Contract address (optional - will use symbol lookup if empty)
  - Ethereum: 0x-prefixed addresses (42 characters)
  - Solana: Base58 addresses (44 characters, no prefix needed)
  - Stellar: G-prefixed addresses (56 characters)

## Features & Optimizations

### 1. **Retry Logic with Exponential Backoff**

All API calls include automatic retry with exponential backoff:

- Retries on HTTP 429 (rate limit) and 5xx errors
- Configurable max retries and initial delay
- Exponential backoff: 1s → 2s → 4s

### 2. **Comprehensive Error Handling**

- Input validation (CSV format, date format, column presence)
- Date validation (rejects future dates)
- Detailed error messages for troubleshooting
- Try-catch blocks around all external API calls

### 3. **Logging**

All operations are logged with timestamps:

```
2024-01-15 10:30:45 - __main__ - INFO - Processing price request for date: 2024-11-01
2024-01-15 10:30:45 - __main__ - INFO - Processing 27 rows from CSV
2024-01-15 10:30:45 - __main__ - INFO - Mapping 15 addresses to CMC IDs
2024-01-15 10:30:46 - __main__ - INFO - Successfully mapped 13 addresses to CMC IDs
```

Set `LOG_LEVEL=DEBUG` for detailed debugging.

### 4. **Batch Processing**

Efficiently processes multiple addresses/symbols in batches:

- Configurable batch size (default: 20)
- Individual retry for items missing from batch response
- Handles CMC API quirks (200 response with missing items)

### 5. **Multi-Chain Address Support**

Automatically detects and decorates addresses for CMC API:

- **Ethereum**: 0x addresses (used as-is)
- **Solana**: Base58 addresses (decorated with `SOL:` prefix internally)
- **Stellar**: G-prefixed addresses (used as-is)
- Extensible for other chains

**Note:** You provide raw addresses in CSV, decoration is automatic.

### 6. **Dual Price Endpoint Strategy**

Tries two CMC endpoints for maximum coverage:

1. `/v2/cryptocurrency/quotes/historical` (primary)
2. `/v1/cryptocurrency/ohlcv/historical` (fallback - correct endpoint)

### 7. **Rate Limit Protection**

Global rate limiter ensures compliance with API limits:

- Configurable delay between requests (default: 2.1s)
- Thread-safe async lock mechanism
- Ensures < 30 requests per minute for Hobbyist plan
- No risk of HTTP 429 rate limit errors

### 8. **Auto-Detection**

- CSV encoding (UTF-8 or Latin-1)
- CSV delimiter (comma, semicolon, tab)
- Network type (Ethereum, Solana, Stellar)
- No need to specify format

## Troubleshooting

### All prices return NULL

- **Check date**: Ensure `target_date` is not in the future
- **Run debug endpoint**: Use `/prices/debug` to see detailed mapping info
- **Check logs**: Look for API errors or mapping failures

### Address not recognized (HTTP 400 errors)

- **This is normal** - Not all tokens are in CoinMarketCap's database
- **Ethereum**: Ensure address starts with `0x` and is 42 characters
- **Solana**: Base58 address, 44 characters (no prefix needed in CSV)
- **Small/new tokens**: May not be tracked by CMC yet
- **Success rate**: 70-90% is typical for mixed coin lists

### Rate limit errors (HTTP 429)

- **Should not happen** - Global rate limiter prevents this
- If it does occur, increase `RATE_LIMIT_DELAY` (e.g., to `2.5`)
- Check your CMC API plan limits (default assumes 30 req/min)
- Monitor logs to verify rate limiting is working

### Connection timeouts

- Increase `REQUEST_TIMEOUT` for slow connections
- Check network connectivity
- Verify CMC API is accessible

## Development

### Run Tests

```bash
# Install dev dependencies
pip install pytest pytest-asyncio httpx

# Run tests (when available)
pytest
```

### Enable Debug Logging

```bash
# In .env file
LOG_LEVEL=DEBUG
```

### Code Structure

- `main.py`: FastAPI application
  - Utility functions: `_sniff_and_read`, `_valid_address`, `_normalize`
  - Mapping functions: `map_address_to_cmc_ids`, `map_symbol_to_cmc_ids`
  - Price fetching: `fetch_prices_for_ids`
  - Endpoints: `/healthz`, `/config`, `/prices`, `/prices/debug`

## Performance & Limits

With **CoinMarketCap Hobbyist Plan** (30 requests/minute):

| Coins | Time (approx) | Requests | Rate            |
| ----- | ------------- | -------- | --------------- |
| 27    | ~37 seconds   | ~21      | 34/min (safe)   |
| 60    | ~73 seconds   | ~35      | 28/min (safe)   |
| 100+  | Auto-adjusts  | Varies   | Always < 30/min |

**Expected results:**

- ✅ 70-90% success rate for mixed coin lists
- ✅ HTTP 400 errors are normal (coin not in CMC)
- ✅ HTTP 403 on DEX API is expected (requires higher plan)
- ✅ No HTTP 429 rate limit errors (protected by rate limiter)

## Changelog

### Version 2.0.0 (Current - November 2025)

- ✅ **Global rate limiter** with async lock (2.1s delay between requests)
- ✅ **3-column CSV output**: coin, address, price_usd_YYYY-MM-DD
- ✅ **Multi-chain support**: Ethereum, Solana, Stellar
- ✅ **Fixed Solana handling**: Removed SOL: prefix for /v2/info endpoint
- ✅ **Corrected OHLCV endpoint**: v2 → v1
- ✅ **DEX API v4 support** (requires higher plan)
- ✅ Default date changed to 2025-10-31
- ✅ Comprehensive error handling and retry logic
- ✅ Logging with configurable levels (INFO/DEBUG)
- ✅ Input validation (dates, CSV format)
- ✅ Environment-based configuration

### Version 1.0.0

- Initial version with 2-CSV architecture
- Basic CMC API integration

## License

MIT License - Feel free to use and modify.

## Support

For issues or questions:

1. Check the `/prices/debug` endpoint for detailed diagnostics
2. Enable `LOG_LEVEL=DEBUG` to see detailed logs
3. Verify `.env` configuration is correct
4. Ensure CMC API key has sufficient quota

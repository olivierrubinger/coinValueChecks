"""
Crypto Price Extractor

FastAPI application for fetching historical cryptocurrency prices from CoinMarketCap.

Features:
- Single CSV input with 'coin' and 'address' columns
- Multi-chain support (Ethereum, Solana, etc.)
- Retry logic with exponential backoff
- Comprehensive error handling and logging
- Batch processing with individual fallback
- Date validation and input checking
- Debug endpoint for troubleshooting


Configuration via environment variables:
- CMC_API_KEY (required)
- MAX_RETRIES (default: 3)
- RETRY_DELAY (default: 1.0)
- BATCH_SIZE (default: 20)
- REQUEST_TIMEOUT (default: 40.0)
- LOG_LEVEL (default: INFO)
"""

from fastapi import FastAPI, UploadFile, File, HTTPException, Form
from fastapi.responses import Response, JSONResponse
import httpx, os, csv, logging, asyncio
from io import StringIO
from typing import List, Dict, Optional, Tuple
from datetime import datetime, timezone, timedelta
from dotenv import load_dotenv
from pathlib import Path
import pandas as pd
import uuid
import json

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Crypto Price Extractor",
    version="2.0.0",
    description="Reads CSV with 'coin' and 'address' columns, fetches prices from CMC and returns CSV."
)

load_dotenv(dotenv_path=Path(__file__).resolve().parent / ".env")
CMC_API_KEY_ENV = "CMC_API_KEY"
CMC_BASE = "https://pro-api.coinmarketcap.com"

# Configuration constants - can be overridden by environment variables
MAX_RETRIES = int(os.getenv("MAX_RETRIES", "3"))
RETRY_DELAY = float(os.getenv("RETRY_DELAY", "1.0"))  # seconds
BATCH_SIZE = int(os.getenv("BATCH_SIZE", "20"))
REQUEST_TIMEOUT = float(os.getenv("REQUEST_TIMEOUT", "40.0"))  # seconds
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()

# Rate limit protection (30 requests per minute = 2 seconds per request)
RATE_LIMIT_DELAY = float(os.getenv("RATE_LIMIT_DELAY", "2.1"))  # seconds between requests
MIN_DELAY_BETWEEN_REQUESTS = 0.5  # minimum delay to avoid bursts

# Update logging level from env
logging.getLogger(__name__).setLevel(getattr(logging, LOG_LEVEL, logging.INFO))

# Global rate limiter
import time
_last_request_time = 0
_request_lock = asyncio.Lock()

async def _rate_limit():
    """Global rate limiter to respect API limits (30 req/min)"""
    global _last_request_time
    async with _request_lock:
        now = time.time()
        time_since_last = now - _last_request_time
        if time_since_last < RATE_LIMIT_DELAY:
            sleep_time = RATE_LIMIT_DELAY - time_since_last
            logger.debug(f"Rate limiting: sleeping {sleep_time:.2f}s")
            await asyncio.sleep(sleep_time)
        _last_request_time = time.time()


def _env_api_key() -> str:
    api_key = os.getenv(CMC_API_KEY_ENV, "").strip()
    if not api_key:
        raise RuntimeError(f"Please set the {CMC_API_KEY_ENV} environment variable with your CoinMarketCap API key.")
    return api_key


def _normalize(s: Optional[str]) -> str:
    return ("" if s is None else str(s)).strip().lower()


def _sniff_and_read(file_obj) -> List[Dict[str, str]]:
    raw = file_obj.read()
    if isinstance(raw, bytes):
        try:
            text = raw.decode("utf-8")
        except UnicodeDecodeError:
            text = raw.decode("latin-1", errors="ignore")
    else:
        text = raw

    sample = text[:4096]
    try:
        dialect = csv.Sniffer().sniff(sample, delimiters=",;|\t")
        delimiter = dialect.delimiter
    except Exception:
        delimiter = ","

    reader = csv.DictReader(StringIO(text), delimiter=delimiter)
    rows = [dict(r) for r in reader]
    return rows


def _valid_address(addr: Optional[str]) -> bool:
    if not addr:
        return False
    s = addr.strip()
    if s in ("-", "nan", "None", "none", ""):
        return False
    return len(s) >= 8


def _decorate_address_for_cmc(addr: str) -> str:
    """EVM (0x) or Solana (base58) with SOL: prefix"""
    a = addr.strip()
    if a.startswith("0x") and len(a) == 42:
        return a
    if not a.startswith("0x") and 30 <= len(a) <= 50:
        if not a.upper().startswith("SOL:"):
            return f"SOL:{a}"
    return a


def _detect_network_slug(addr: str) -> str:
    """Detects network based on address format"""
    a = addr.strip()
    
    # Ethereum and EVM chains (0x + 40 hex chars)
    if a.startswith("0x") and len(a) == 42:
        return "ethereum"
    
    # Solana (base58, ~32-44 chars, no 0x)
    if not a.startswith("0x") and 32 <= len(a) <= 44:
        # Solana uses base58 alphabet
        return "solana"
    
    # Stellar (starts with G, ~56 chars)
    if a.startswith("G") and len(a) == 56:
        return "stellar"
    
    # Default to ethereum if network cannot be identified
    logger.warning(f"Could not detect network for address {addr[:10]}..., defaulting to ethereum")
    return "ethereum"


async def _make_request_with_retry(
    client: httpx.AsyncClient,
    method: str,
    url: str,
    headers: dict,
    params: Optional[dict] = None,
    max_retries: int = MAX_RETRIES,
    retry_delay: float = RETRY_DELAY
) -> Optional[dict]:
    """Makes HTTP request with exponential backoff retry logic and rate limiting"""
    for attempt in range(max_retries):
        try:
            # Apply global rate limiting
            await _rate_limit()
            
            if method.upper() == "GET":
                response = await client.get(url, headers=headers, params=params)
            else:
                response = await client.post(url, headers=headers, json=params)
            
            response.raise_for_status()
            return response.json()
            
        except httpx.HTTPStatusError as e:
            logger.warning(f"HTTP error {e.response.status_code} on attempt {attempt + 1}/{max_retries}: {url}")
            if e.response.status_code == 429:  # Rate limit
                wait_time = retry_delay * (2 ** attempt)
                logger.info(f"Rate limited. Waiting {wait_time}s before retry...")
                await asyncio.sleep(wait_time)
            elif e.response.status_code >= 500:  # Server error
                wait_time = retry_delay * (2 ** attempt)
                logger.info(f"Server error. Waiting {wait_time}s before retry...")
                await asyncio.sleep(wait_time)
            else:
                logger.error(f"Non-retryable HTTP error: {e.response.status_code}")
                return None
                
        except httpx.RequestError as e:
            logger.warning(f"Request error on attempt {attempt + 1}/{max_retries}: {str(e)}")
            if attempt < max_retries - 1:
                wait_time = retry_delay * (2 ** attempt)
                await asyncio.sleep(wait_time)
        except Exception as e:
            logger.error(f"Unexpected error during request: {str(e)}")
            return None
    
    logger.error(f"Failed after {max_retries} attempts: {url}")
    return None


async def map_address_to_cmc_ids(addresses: List[str], api_key: str) -> Dict[str, int]:
    """Maps addresses to CMC IDs with retry logic (without SOL: prefix)"""
    logger.info(f"Mapping {len(addresses)} addresses to CMC IDs")
    mapping: Dict[str, int] = {}
    headers = {"X-CMC_PRO_API_KEY": api_key}
    addrs = [a for a in {a.strip(): None for a in addresses if _valid_address(a)}.keys()]
    # Do NOT decorate with SOL: for the /info endpoint - use raw addresses
    pairs: List[Tuple[str, str]] = list(zip(addrs, addrs))
    
    async with httpx.AsyncClient(timeout=REQUEST_TIMEOUT) as client:
        for i in range(0, len(pairs), BATCH_SIZE):
            sub = pairs[i:i + BATCH_SIZE]
            chunk = [d for (_, d) in sub]
            if not chunk:
                continue
            
            logger.debug(f"Processing batch {i//BATCH_SIZE + 1}: {len(chunk)} addresses")
            
            # Try batch request with retry
            batch_response = await _make_request_with_retry(
                client,
                "GET",
                f"{CMC_BASE}/v2/cryptocurrency/info",
                headers=headers,
                params={"address": ",".join(chunk)}
            )
            
            if batch_response:
                _extract_ids_from_response(batch_response, mapping)
                logger.debug(f"Batch succeeded, mapped {len(mapping)} total addresses")
            else:
                logger.warning(f"Batch request failed, falling back to individual requests")
            
            # Check which addresses from batch are missing and retry individually
            for orig, single in sub:
                if _normalize(orig) not in mapping:
                    logger.debug(f"Retrying individual address: {orig}")
                    single_response = await _make_request_with_retry(
                        client,
                        "GET",
                        f"{CMC_BASE}/v2/cryptocurrency/info",
                        headers=headers,
                        params={"address": single}
                    )
                    if single_response:
                        _extract_ids_from_response(single_response, mapping)
    
    logger.info(f"Successfully mapped {len(mapping)} addresses to CMC IDs")
    return mapping


def _extract_ids_from_response(json_data: dict, mapping: Dict[str, int]):
    """Extracts contract IDs from CMC response payload"""
    data = json_data.get("data", {})
    for payload in data.values():
        try:
            cid = int(payload.get("id"))
        except Exception:
            continue
        contracts = payload.get("contract_address") or payload.get("contract_addresses") or []
        if isinstance(contracts, dict):
            contracts = [contracts]
        for c in contracts:
            addr = (c.get("contract_address") or c.get("address") or "").strip()
            if _valid_address(addr):
                mapping[_normalize(addr)] = cid


async def map_symbol_to_cmc_ids(symbols: List[str], api_key: str) -> Dict[str, int]:
    """Maps symbol to CMC ID (chooses best rank) with retry logic"""
    logger.info(f"Mapping {len(symbols)} symbols to CMC IDs")
    mapping: Dict[str, int] = {}
    best_rank: Dict[str, int] = {}
    headers = {"X-CMC_PRO_API_KEY": api_key}
    
    uniq = []
    seen = set()
    for s in symbols:
        if not s:
            continue
        u = s.upper().strip()
        if not u or u in seen:
            continue
        seen.add(u)
        uniq.append(u)

    async with httpx.AsyncClient(timeout=REQUEST_TIMEOUT) as client:
        for i in range(0, len(uniq), BATCH_SIZE):
            chunk = uniq[i:i + BATCH_SIZE]
            if not chunk:
                continue
            
            logger.debug(f"Processing symbol batch {i//BATCH_SIZE + 1}: {len(chunk)} symbols")
            
            # Try batch request with retry
            batch_response = await _make_request_with_retry(
                client,
                "GET",
                f"{CMC_BASE}/v1/cryptocurrency/map",
                headers=headers,
                params={"symbol": ",".join(chunk)}
            )
            
            if batch_response:
                _extract_symbols_from_response(batch_response, mapping, best_rank)
                logger.debug(f"Symbol batch succeeded, mapped {len(mapping)} total symbols")
            else:
                logger.warning(f"Symbol batch failed, falling back to individual requests")
                # Fallback individual (rate limiter global controla o timing)
                for s in chunk:
                    single_response = await _make_request_with_retry(
                        client,
                        "GET",
                        f"{CMC_BASE}/v1/cryptocurrency/map",
                        headers=headers,
                        params={"symbol": s}
                    )
                    if single_response:
                        _extract_symbols_from_response(single_response, mapping, best_rank)
    
    logger.info(f"Successfully mapped {len(mapping)} symbols to CMC IDs")
    return mapping


def _extract_symbols_from_response(json_data: dict, mapping: Dict[str, int], best_rank: Dict[str, int]):
    """Extracts symbols from CMC response payload"""
    data = json_data.get("data")
    if data and isinstance(data, list):
        for item in data:
            try:
                sym = (item.get("symbol") or "").upper()
                cmc_id = int(item.get("id"))
                rank = int(item.get("rank") or 10**9)
                key = _normalize(sym)
                if key not in mapping or rank < best_rank.get(key, 10**9):
                    mapping[key] = cmc_id
                    best_rank[key] = rank
            except Exception:
                pass


async def fetch_prices_for_ids(ids: List[int], date_end_utc: datetime, api_key: str) -> Dict[int, Optional[float]]:
    """Fetches historical prices with fallback quotes → ohlcv and retry logic"""
    logger.info(f"Fetching prices for {len(ids)} CMC IDs on date: {date_end_utc.date()}")
    headers = {"X-CMC_PRO_API_KEY": api_key}
    results: Dict[int, Optional[float]] = {}
    if not ids:
        return results

    PRICE_BATCH = 100
    # CMC API quirk: OHLCV endpoint needs the PREVIOUS day to get correct data
    # When you request 2025-10-31, it returns 2025-11-01 data
    # So we need to request one day earlier
    adjusted_date = date_end_utc - timedelta(days=1)
    
    time_start = datetime(adjusted_date.year, adjusted_date.month, adjusted_date.day, 0, 0, 0, tzinfo=timezone.utc)
    time_end = datetime(adjusted_date.year, adjusted_date.month, adjusted_date.day, 23, 59, 59, tzinfo=timezone.utc)
    
    logger.info(f"Requesting API data for adjusted date: {adjusted_date.date()} (will return data for {date_end_utc.date()})")

    async with httpx.AsyncClient(timeout=REQUEST_TIMEOUT) as client:
        for i in range(0, len(ids), PRICE_BATCH):
            chunk = ids[i:i + PRICE_BATCH]
            logger.debug(f"Fetching prices for batch {i//PRICE_BATCH + 1}: {len(chunk)} IDs")
            
            # Use OHLCV endpoint directly for accurate close prices
            # The quotes/historical endpoint only returns snapshot prices, not close prices
            params_ohlcv = {
                "id": ",".join(str(x) for x in chunk),
                "time_start": time_start.date().isoformat(),  # YYYY-MM-DD format
                "time_end": time_end.date().isoformat(),
                "convert": "USD",
                "interval": "daily",
            }
            
            ohlcv_response = await _make_request_with_retry(
                client,
                "GET",
                f"{CMC_BASE}/v1/cryptocurrency/ohlcv/historical",
                headers=headers,
                params=params_ohlcv
            )
            
            if ohlcv_response:
                d2 = ohlcv_response.get("data", {})
                logger.debug(f"OHLCV response data keys: {list(d2.keys())}")
                # OHLCV v1 returns data at top level for single ID, or nested for multiple IDs
                # Check if it's a single ID response (has 'quotes' at top level)
                if "quotes" in d2:
                    # Single ID response: {"data": {"id": 1, "quotes": [...]}}
                    for cid in chunk:
                        series = d2.get("quotes") or []
                        if series:
                            logger.info(f"OHLCV: CMC ID {cid}, got {len(series)} quotes, first date: {series[0].get('time_open', 'N/A')[:10]}, looking for: {date_end_utc.date()}")
                        close_val = _extract_close_from_ohlcv(series, date_end_utc.date())
                        results[cid] = close_val
                        if close_val:
                            logger.info(f"OHLCV: CMC ID {cid} extracted price: ${close_val}")
                        else:
                            logger.warning(f"OHLCV: CMC ID {cid} - no price extracted from {len(series)} quotes")
                else:
                    # Multiple ID response: {"data": {"1": {"quotes": [...]}, "1027": {...}}}
                    for cid in chunk:
                        cid_data = d2.get(str(cid), {})
                        series = cid_data.get("quotes") or []
                        if series:
                            logger.info(f"OHLCV: CMC ID {cid}, got {len(series)} quotes, first date: {series[0].get('time_open', 'N/A')[:10]}, looking for: {date_end_utc.date()}")
                        close_val = _extract_close_from_ohlcv(series, date_end_utc.date())
                        results[cid] = close_val
                        if close_val:
                            logger.info(f"OHLCV: CMC ID {cid} extracted price: ${close_val}")
                        else:
                            logger.warning(f"OHLCV: CMC ID {cid} - no price extracted from {len(series)} quotes")
            else:
                logger.warning(f"OHLCV request failed for {len(chunk)} IDs")
                for cid in chunk:
                    results[cid] = None

    logger.info(f"Price fetch complete: {sum(1 for v in results.values() if v is not None)}/{len(results)} with prices")
    return results


async def fetch_dex_prices_for_addresses(
    addresses: List[str], 
    date_end_utc: datetime, 
    api_key: str
) -> Dict[str, Optional[float]]:
    """
    Attempts to fetch DEX prices for addresses not found as cryptoassets.
    Uses CoinMarketCap DEX API v4.
    """
    logger.info(f"Trying DEX API for {len(addresses)} addresses")
    headers = {"X-CMC_PRO_API_KEY": api_key}
    results: Dict[str, Optional[float]] = {}
    
    if not addresses:
        return results
    
    # For DEX, we need YYYY-MM-DD format
    target_date_str = date_end_utc.date().isoformat()
    
    async with httpx.AsyncClient(timeout=REQUEST_TIMEOUT) as client:
        for addr in addresses:
            network_slug = _detect_network_slug(addr)
            addr_clean = addr.strip()
            
            # Remove SOL: prefix if present
            if addr_clean.upper().startswith("SOL:"):
                addr_clean = addr_clean[4:]
            
            logger.debug(f"Trying DEX for address {addr_clean[:10]}... on network {network_slug}")
            
            # Attempt to fetch historical OHLCV for the DEX pair
            params_dex = {
                "contract_address": addr_clean,
                "network_slug": network_slug,
                "time_start": target_date_str,
                "time_end": target_date_str,
                "interval": "daily",
                "convert_id": "2781",  # 2781 = USD na CMC
            }
            
            dex_response = await _make_request_with_retry(
                client,
                "GET",
                f"{CMC_BASE}/v4/dex/pairs/ohlcv/historical",
                headers=headers,
                params=params_dex
            )
            
            if dex_response and dex_response.get("data"):
                # Extract close price from the first (and only) day
                data = dex_response.get("data", {})
                quotes = data.get("quotes", [])
                if quotes:
                    for quote in quotes:
                        try:
                            close_price = quote.get("quote", {}).get("2781", {}).get("close")
                            if close_price:
                                results[_normalize(addr)] = float(close_price)
                                logger.info(f"Found DEX price for {addr[:10]}...: ${close_price}")
                                break
                        except Exception as e:
                            logger.debug(f"Error extracting DEX price: {e}")
            else:
                logger.debug(f"No DEX data found for {addr_clean[:10]}... on {network_slug}")
                results[_normalize(addr)] = None
    
    logger.info(f"DEX API returned {sum(1 for v in results.values() if v is not None)}/{len(addresses)} prices")
    return results


def _extract_price_from_quotes(quotes: list, target_date) -> Optional[float]:
    """Extracts price from quotes, preferring exact day and using close price"""
    price_val = None
    matching_quotes = []
    
    # Find all quotes that match the target date
    for q in quotes:
        try:
            ts = q.get("timestamp") or q.get("time_close") or q.get("time_open")
            if ts:
                d = datetime.fromisoformat(str(ts).replace("Z", "+00:00")).date()
                if d == target_date:
                    matching_quotes.append(q)
        except Exception:
            pass
    
    # If we found quotes for the target date, use the LAST one (end of day)
    if matching_quotes:
        last_quote = matching_quotes[-1]
        # Try to get close price, fallback to regular price
        price_val = (last_quote.get("quote", {}).get("USD", {}).get("close") or 
                     last_quote.get("quote", {}).get("USD", {}).get("price"))
    elif quotes:
        # Fallback to last quote in the entire array
        last_quote = quotes[-1]
        price_val = (last_quote.get("quote", {}).get("USD", {}).get("close") or 
                     last_quote.get("quote", {}).get("USD", {}).get("price"))
    
    try:
        return float(price_val) if price_val is not None else None
    except Exception:
        return None


def _extract_close_from_ohlcv(series: list, target_date) -> Optional[float]:
    """Extracts close from OHLCV, preferring exact day"""
    close_val = None
    for q in series:
        try:
            to = q.get("time_open")
            if to:
                d = datetime.fromisoformat(str(to).replace("Z", "+00:00")).date()
                if d == target_date:
                    close_val = q.get("quote", {}).get("USD", {}).get("close")
        except Exception:
            pass
    if close_val is None and series:
        close_val = series[-1].get("quote", {}).get("USD", {}).get("close")
    try:
        return float(close_val) if close_val is not None else None
    except Exception:
        return None


def _build_output_csv(
    data: List[Tuple[str, str, Optional[float]]], 
    target_date: str
) -> StringIO:
    """
    Builds CSV with columns: coin, address, price_usd_YYYY-MM-DD
    Args:
        data: List of tuples (coin, address, price)
        target_date: Date string in YYYY-MM-DD format
    """
    buf = StringIO()
    writer = csv.writer(buf)
    
    # Column header with date
    price_column = f"price_usd_{target_date}"
    writer.writerow(["coin", "address", price_column])
    
    for coin, address, price in data:
        price_str = "N/A" if price is None else f"{price:.10f}".rstrip("0").rstrip(".")
        writer.writerow([coin, address or "", price_str])
    
    buf.seek(0)
    return buf


@app.get("/healthz")
async def healthz():
    """Health check endpoint"""
    return JSONResponse({"status": "ok", "version": "2.0.0"})


# ============================================================================
# BATCH PROCESSING - Background jobs for large CSV files
# ============================================================================

# In-memory job storage (in production, use Redis or database)
batch_jobs: Dict[str, dict] = {}
JOBS_DIR = Path("jobs")
JOBS_DIR.mkdir(exist_ok=True)


def _save_job_status(job_id: str, status: dict):
    """Saves job status to disk for persistence"""
    job_file = JOBS_DIR / f"{job_id}.json"
    with open(job_file, 'w') as f:
        json.dump(status, f, indent=2, default=str)
    batch_jobs[job_id] = status


def _load_job_status(job_id: str) -> Optional[dict]:
    """Loads job status from disk"""
    if job_id in batch_jobs:
        return batch_jobs[job_id]
    
    job_file = JOBS_DIR / f"{job_id}.json"
    if job_file.exists():
        with open(job_file, 'r') as f:
            status = json.load(f)
            batch_jobs[job_id] = status
            return status
    return None


async def _process_batch_symbols(
    job_id: str,
    csv_path: Path,
    target_date: str,
    api_key: str,
    chunk_size: int = 100000
):
    """
    Processes large CSV file in chunks to extract symbols and fetch prices.
    Updates job status as it progresses.
    """
    try:
        _save_job_status(job_id, {
            "status": "processing",
            "progress": 0,
            "total_rows": 0,
            "unique_symbols": 0,
            "symbols_with_prices": 0,
            "current_step": "Reading CSV and extracting symbols",
            "started_at": datetime.now(timezone.utc).isoformat(),
            "error": None
        })
        
        # Step 1: Read CSV in chunks and extract unique symbols
        logger.info(f"[Job {job_id}] Starting to read CSV in chunks of {chunk_size}")
        unique_symbols = set()
        total_rows = 0
        
        try:
            # Read CSV in chunks to avoid loading everything in memory
            for chunk_num, chunk in enumerate(pd.read_csv(csv_path, chunksize=chunk_size)):
                if 'denomination' not in chunk.columns:
                    raise ValueError("CSV must have 'denomination' column")
                
                # Extract symbols from this chunk
                symbols_in_chunk = chunk['denomination'].dropna().astype(str).str.strip().str.upper()
                unique_symbols.update(symbols_in_chunk.unique())
                total_rows += len(chunk)
                
                # Update progress
                _save_job_status(job_id, {
                    **batch_jobs[job_id],
                    "progress": min(50, (chunk_num + 1) * 10),  # First 50% is reading
                    "total_rows": total_rows,
                    "unique_symbols": len(unique_symbols),
                    "current_step": f"Reading chunk {chunk_num + 1} ({total_rows:,} rows processed)"
                })
                
                logger.info(f"[Job {job_id}] Processed chunk {chunk_num + 1}: {total_rows:,} total rows, {len(unique_symbols)} unique symbols")
        
        except Exception as e:
            logger.error(f"[Job {job_id}] Error reading CSV: {str(e)}")
            raise ValueError(f"Error reading CSV: {str(e)}")
        
        # Remove empty symbols
        unique_symbols = {s for s in unique_symbols if s and s not in ('', 'NAN', 'NONE', 'NULL')}
        symbols_list = sorted(unique_symbols)
        
        logger.info(f"[Job {job_id}] Extracted {len(symbols_list)} unique symbols from {total_rows:,} rows")
        
        _save_job_status(job_id, {
            **batch_jobs[job_id],
            "progress": 50,
            "total_rows": total_rows,
            "unique_symbols": len(symbols_list),
            "current_step": "Mapping symbols to CMC IDs"
        })
        
        # Step 2: Map symbols to CMC IDs
        logger.info(f"[Job {job_id}] Mapping {len(symbols_list)} symbols to CMC IDs")
        
        try:
            dt_target = datetime.fromisoformat(target_date).replace(tzinfo=timezone.utc)
        except ValueError:
            raise ValueError("Invalid target_date. Use YYYY-MM-DD format.")
        
        sym_to_id = await map_symbol_to_cmc_ids(symbols_list, api_key=api_key)
        
        _save_job_status(job_id, {
            **batch_jobs[job_id],
            "progress": 70,
            "current_step": "Fetching historical prices"
        })
        
        # Step 3: Fetch prices for all CMC IDs
        unique_ids = sorted(set(sym_to_id.values()))
        logger.info(f"[Job {job_id}] Fetching prices for {len(unique_ids)} CMC IDs")
        
        id_to_price = await fetch_prices_for_ids(unique_ids, date_end_utc=dt_target, api_key=api_key)
        
        _save_job_status(job_id, {
            **batch_jobs[job_id],
            "progress": 90,
            "current_step": "Generating output CSV"
        })
        
        # Step 4: Build output CSV
        output_data: List[Tuple[str, Optional[float]]] = []
        symbols_with_prices = 0
        
        for symbol in symbols_list:
            cmc_id = sym_to_id.get(_normalize(symbol.upper()))
            price = id_to_price.get(cmc_id) if cmc_id else None
            output_data.append((symbol, price))
            if price is not None:
                symbols_with_prices += 1
        
        # Generate CSV
        output_dir = Path("output")
        output_dir.mkdir(exist_ok=True)
        output_file = output_dir / f"batch_{job_id}_{target_date}.csv"
        
        buf = StringIO()
        writer = csv.writer(buf, lineterminator='\n')
        
        # Header
        price_column = f"price_usd_{target_date}"
        writer.writerow(["coin", price_column])
        
        # Data
        for symbol, price in output_data:
            price_str = "N/A" if price is None else f"{price:.10f}".rstrip("0").rstrip(".")
            writer.writerow([symbol, price_str])
        
        # Save to file
        output_file.write_text(buf.getvalue(), encoding='utf-8')
        
        logger.info(f"[Job {job_id}] Completed! Output saved to {output_file}")
        
        # Mark as completed
        _save_job_status(job_id, {
            "status": "completed",
            "progress": 100,
            "total_rows": total_rows,
            "unique_symbols": len(symbols_list),
            "symbols_with_prices": symbols_with_prices,
            "current_step": "Completed",
            "started_at": batch_jobs[job_id]["started_at"],
            "completed_at": datetime.now(timezone.utc).isoformat(),
            "output_file": str(output_file),
            "error": None
        })
        
    except Exception as e:
        logger.error(f"[Job {job_id}] Failed with error: {str(e)}")
        _save_job_status(job_id, {
            **batch_jobs.get(job_id, {}),
            "status": "failed",
            "current_step": "Failed",
            "completed_at": datetime.now(timezone.utc).isoformat(),
            "error": str(e)
        })


@app.get("/config")
async def get_config():
    """Returns current configuration (without sensitive data)"""
    return JSONResponse({
        "version": "2.0.0",
        "cmc_base_url": CMC_BASE,
        "max_retries": MAX_RETRIES,
        "retry_delay_seconds": RETRY_DELAY,
        "batch_size": BATCH_SIZE,
        "request_timeout_seconds": REQUEST_TIMEOUT,
        "log_level": LOG_LEVEL,
        "api_key_configured": bool(os.getenv(CMC_API_KEY_ENV))
    })


@app.post("/prices/batch")
async def submit_batch_job(
    csv_file: UploadFile = File(..., description="Large CSV with 'denomination' column"),
    target_date: str = Form("2025-10-31", description="Target date (YYYY-MM-DD)"),
):
    """
    Submit a batch job to process large CSV files (up to 1.5M rows).
    Returns a job_id to track progress.
    
    The CSV must have a 'denomination' column with coin symbols (ETH, SOL, XRP, etc).
    Processing happens in background. Use GET /prices/batch/{job_id} to check status.
    """
    logger.info(f"Received batch job request for date: {target_date}")
    
    try:
        api_key = _env_api_key()
    except RuntimeError as e:
        logger.error(f"API key error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
    
    # Validate date
    try:
        dt_target = datetime.fromisoformat(target_date).replace(tzinfo=timezone.utc)
        now = datetime.now(timezone.utc)
        if dt_target > now:
            raise HTTPException(
                status_code=400,
                detail=f"Date {target_date} is in the future. Use a past date."
            )
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid date. Use YYYY-MM-DD format.")
    
    # Validate file
    if not csv_file.filename or not csv_file.filename.lower().endswith('.csv'):
        raise HTTPException(status_code=400, detail="Please upload a CSV file.")
    
    # Generate job ID
    job_id = str(uuid.uuid4())
    
    # Save uploaded file temporarily
    temp_dir = Path("temp")
    temp_dir.mkdir(exist_ok=True)
    temp_file = temp_dir / f"{job_id}.csv"
    
    try:
        # Save uploaded file
        content = await csv_file.read()
        temp_file.write_bytes(content)
        logger.info(f"[Job {job_id}] Saved uploaded file: {len(content):,} bytes")
        
        # Initialize job status
        _save_job_status(job_id, {
            "status": "queued",
            "progress": 0,
            "total_rows": 0,
            "unique_symbols": 0,
            "symbols_with_prices": 0,
            "current_step": "Queued for processing",
            "created_at": datetime.now(timezone.utc).isoformat(),
            "target_date": target_date,
            "original_filename": csv_file.filename,
            "error": None
        })
        
        # Start background processing
        asyncio.create_task(_process_batch_symbols(job_id, temp_file, target_date, api_key))
        
        logger.info(f"[Job {job_id}] Started background processing")
        
        return JSONResponse({
            "job_id": job_id,
            "status": "queued",
            "message": "Batch job submitted successfully. Use GET /prices/batch/{job_id} to check progress.",
            "status_url": f"/prices/batch/{job_id}"
        })
        
    except Exception as e:
        logger.error(f"Error submitting batch job: {str(e)}")
        if temp_file.exists():
            temp_file.unlink()
        raise HTTPException(status_code=500, detail=f"Error processing file: {str(e)}")


@app.get("/prices/batch/{job_id}")
async def get_batch_job_status(job_id: str):
    """
    Get the status and progress of a batch job.
    
    Returns:
    - status: queued, processing, completed, or failed
    - progress: 0-100%
    - current_step: description of current operation
    - output_file: download path when completed
    """
    status = _load_job_status(job_id)
    
    if not status:
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found")
    
    return JSONResponse(status)


@app.get("/prices/batch/{job_id}/download")
async def download_batch_result(job_id: str):
    """
    Download the result CSV file for a completed batch job.
    """
    status = _load_job_status(job_id)
    
    if not status:
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found")
    
    if status["status"] != "completed":
        raise HTTPException(
            status_code=400, 
            detail=f"Job is not completed yet. Current status: {status['status']}"
        )
    
    output_file = Path(status["output_file"])
    
    if not output_file.exists():
        raise HTTPException(status_code=404, detail="Output file not found")
    
    csv_content = output_file.read_bytes()
    
    return Response(
        content=csv_content,
        media_type="text/csv",
        headers={
            "Content-Disposition": f'attachment; filename="{output_file.name}"'
        }
    )


@app.get("/prices/batch")
async def list_batch_jobs():
    """
    List all batch jobs with their current status.
    """
    all_jobs = []
    
    # Load all job files
    for job_file in JOBS_DIR.glob("*.json"):
        job_id = job_file.stem
        status = _load_job_status(job_id)
        if status:
            all_jobs.append({
                "job_id": job_id,
                "status": status.get("status"),
                "progress": status.get("progress", 0),
                "total_rows": status.get("total_rows", 0),
                "unique_symbols": status.get("unique_symbols", 0),
                "symbols_with_prices": status.get("symbols_with_prices", 0),
                "created_at": status.get("created_at"),
                "completed_at": status.get("completed_at"),
                "target_date": status.get("target_date"),
                "original_filename": status.get("original_filename"),
            })
    
    # Sort by creation date (newest first)
    all_jobs.sort(key=lambda x: x.get("created_at") or x.get("started_at") or "", reverse=True)
    
    return JSONResponse({
        "total_jobs": len(all_jobs),
        "jobs": all_jobs
    })


@app.delete("/prices/batch/{job_id}")
async def delete_batch_job(job_id: str):
    """
    Delete/cancel a batch job and its associated files.
    Useful for cleaning up stuck or failed jobs.
    """
    status = _load_job_status(job_id)
    
    if not status:
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found")
    
    # Delete job status file
    job_file = JOBS_DIR / f"{job_id}.json"
    if job_file.exists():
        job_file.unlink()
        logger.info(f"Deleted job file: {job_file}")
    
    # Delete temp file
    temp_file = Path("temp") / f"{job_id}.csv"
    if temp_file.exists():
        temp_file.unlink()
        logger.info(f"Deleted temp file: {temp_file}")
    
    # Delete output file if exists
    if status.get("output_file"):
        output_file = Path(status["output_file"])
        if output_file.exists():
            output_file.unlink()
            logger.info(f"Deleted output file: {output_file}")
    
    # Remove from memory
    if job_id in batch_jobs:
        del batch_jobs[job_id]
    
    return JSONResponse({
        "message": f"Job {job_id} deleted successfully",
        "job_id": job_id,
        "previous_status": status.get("status")
    })


@app.post("/prices")
async def get_prices(
    csv_file: UploadFile = File(..., description="CSV with 'coin' and 'address' columns"),
    target_date: str = Form("2025-10-31", description="Target date (YYYY-MM-DD)"),
):
    """Main endpoint: returns CSV with prices"""
    logger.info(f"Processing price request for date: {target_date}")
    
    try:
        api_key = _env_api_key()
    except RuntimeError as e:
        logger.error(f"API key error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

    # Validate and parse date first
    try:
        dt_target = datetime.fromisoformat(target_date).replace(tzinfo=timezone.utc)
        
        # Validate not in future
        now = datetime.now(timezone.utc)
        if dt_target > now:
            raise HTTPException(
                status_code=400,
                detail=f"Date {target_date} is in the future. Use a past date (maximum: {now.date().isoformat()})."
            )
    except ValueError:
        logger.error(f"Invalid date format: {target_date}")
        raise HTTPException(status_code=400, detail="Invalid target_date parameter. Use YYYY-MM-DD format.")

    # Validate file upload
    if not csv_file.filename:
        logger.error("No filename in uploaded file")
        raise HTTPException(status_code=400, detail="CSV file not provided.")
    
    if not csv_file.filename.lower().endswith('.csv'):
        logger.warning(f"Uploaded file doesn't have .csv extension: {csv_file.filename}")

    # Read CSV with error handling
    try:
        rows = _sniff_and_read(csv_file.file)
    except Exception as e:
        logger.error(f"Error reading CSV file: {str(e)}")
        raise HTTPException(status_code=400, detail=f"Error reading CSV: {str(e)}")
    
    if not rows:
        logger.error("CSV file is empty")
        raise HTTPException(status_code=400, detail="Empty or invalid CSV file.")

    # Validate columns
    fields = list(rows[0].keys())
    coin_col = None
    address_col = None
    
    for f in fields:
        fl = f.lower().strip()
        if fl == "coin":
            coin_col = f
        elif fl == "address":
            address_col = f
    
    if not coin_col or not address_col:
        logger.error(f"Required columns not found. Available: {fields}")
        raise HTTPException(
            status_code=400, 
            detail=f"CSV must have 'coin' and 'address' columns. Found: {fields}"
        )

    logger.info(f"Processing {len(rows)} rows from CSV")

    # Extract data
    coins: List[str] = []
    addresses: List[Optional[str]] = []
    
    for row in rows:
        coins.append(row.get(coin_col, ""))
        addresses.append(row.get(address_col))

    # Map addresses → CMC IDs
    addrs_present = [a for a in addresses if _valid_address(a)]
    logger.info(f"Found {len(addrs_present)} valid addresses")
    
    try:
        addr_to_id = await map_address_to_cmc_ids(addrs_present, api_key=api_key)
    except Exception as e:
        logger.error(f"Error mapping addresses to CMC IDs: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error mapping addresses: {str(e)}")

    # Fallback by symbol (use coin name when no address)
    symbols_needed: List[str] = []
    idx_to_symbol: Dict[int, str] = {}
    
    for idx, (coin, addr) in enumerate(zip(coins, addresses)):
        if not addr or _normalize(addr) not in addr_to_id:
            if coin.strip():
                symbols_needed.append(coin.strip())
                idx_to_symbol[idx] = coin.strip()
    
    logger.info(f"Need to map {len(symbols_needed)} symbols")
    
    try:
        sym_to_id = await map_symbol_to_cmc_ids(symbols_needed, api_key=api_key)
    except Exception as e:
        logger.error(f"Error mapping symbols to CMC IDs: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error mapping symbols: {str(e)}")

    # Resolve CMC IDs
    cmc_ids: List[Optional[int]] = []
    for idx, (coin, addr) in enumerate(zip(coins, addresses)):
        cmc_id = None
        if addr and _normalize(addr) in addr_to_id:
            cmc_id = addr_to_id[_normalize(addr)]
        else:
            sym = idx_to_symbol.get(idx)
            if sym and _normalize(sym.upper()) in sym_to_id:
                cmc_id = sym_to_id[_normalize(sym.upper())]
        cmc_ids.append(cmc_id)

    # Fetch prices for cryptoassets (via CMC ID)
    unique_ids = sorted({i for i in cmc_ids if i is not None})
    logger.info(f"Fetching prices for {len(unique_ids)} unique CMC IDs")
    
    try:
        id_to_price = await fetch_prices_for_ids(unique_ids, date_end_utc=dt_target, api_key=api_key)
    except Exception as e:
        logger.error(f"Error fetching prices: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error fetching prices: {str(e)}")

    # Identify addresses that didn't get a price (may be DEX pairs)
    addresses_without_price: List[str] = []
    addr_to_idx: Dict[str, int] = {}
    
    for idx, (coin, addr, cmc_id) in enumerate(zip(coins, addresses, cmc_ids)):
        if addr and _valid_address(addr):
            # If no CMC ID OR has ID but no price, try DEX
            if cmc_id is None or id_to_price.get(cmc_id) is None:
                addresses_without_price.append(addr)
                addr_to_idx[_normalize(addr)] = idx
    
    # Try to fetch via DEX API for addresses without price
    dex_prices: Dict[str, Optional[float]] = {}
    if addresses_without_price:
        logger.info(f"Trying DEX API for {len(addresses_without_price)} addresses without price")
        try:
            dex_prices = await fetch_dex_prices_for_addresses(
                addresses_without_price, 
                date_end_utc=dt_target, 
                api_key=api_key
            )
        except Exception as e:
            logger.error(f"Error fetching DEX prices: {str(e)}")
            # Continua mesmo se DEX falhar

    # Monta resultado combinando cryptoasset + DEX prices
    # New format: (coin, address, price)
    result_data: List[Tuple[str, str, Optional[float]]] = []
    for coin, addr, cmc_id in zip(coins, addresses, cmc_ids):
        # Priority: 1) Price via CMC ID, 2) Price via DEX, 3) None
        price = None
        if cmc_id is not None:
            price = id_to_price.get(cmc_id)
        
        # If no price via ID, try DEX
        if price is None and addr and _normalize(addr) in dex_prices:
            price = dex_prices.get(_normalize(addr))
        
        result_data.append((coin, addr or "", price))

    # Generate CSV with new format: coin, address, price_usd_YYYY-MM-DD
    buf = _build_output_csv(result_data, target_date=target_date)
    
    # Save to output folder
    output_dir = Path("output")
    output_dir.mkdir(exist_ok=True)
    output_file = output_dir / f"prices_{target_date}.csv"
    
    csv_content = buf.getvalue()
    csv_bytes = csv_content.encode("utf-8")
    output_file.write_bytes(csv_bytes)
    logger.info(f"Saved output to {output_file}")
    
    # Return complete Response (non-streaming) for curl to show correct progress
    return Response(
        content=csv_bytes,
        media_type="text/csv",
        headers={"Content-Disposition": f'attachment; filename="prices_{target_date}.csv"'}
    )


@app.post("/prices/debug")
async def get_prices_debug(
    csv_file: UploadFile = File(...),
    target_date: str = Form("2025-10-31"),
):
    """Debug endpoint: returns JSON with complete processing details"""
    logger.info(f"Processing debug request for date: {target_date}")
    
    try:
        api_key = _env_api_key()
    except RuntimeError as e:
        logger.error(f"API key error in debug endpoint: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

    # Validate date first
    try:
        dt_target = datetime.fromisoformat(target_date).replace(tzinfo=timezone.utc)
        
        # Validate not in future
        now = datetime.now(timezone.utc)
        if dt_target > now:
            raise HTTPException(
                status_code=400,
                detail=f"Date {target_date} is in the future. Use a past date (maximum: {now.date().isoformat()})."
            )
    except ValueError:
        logger.error(f"Invalid date format in debug: {target_date}")
        raise HTTPException(status_code=400, detail="Invalid date. Use YYYY-MM-DD format.")

    # Read and validate CSV
    try:
        rows = _sniff_and_read(csv_file.file)
    except Exception as e:
        logger.error(f"Error reading CSV in debug: {str(e)}")
        raise HTTPException(status_code=400, detail=f"Error reading CSV: {str(e)}")
    
    if not rows:
        logger.error("CSV is empty in debug endpoint")
        raise HTTPException(status_code=400, detail="CSV file is empty.")

    fields = list(rows[0].keys())
    coin_col = None
    address_col = None
    
    for f in fields:
        fl = f.lower().strip()
        if fl == "coin":
            coin_col = f
        elif fl == "address":
            address_col = f
    
    if not coin_col or not address_col:
        logger.error(f"Required columns missing in debug. Found: {fields}")
        raise HTTPException(status_code=400, detail=f"Expected columns: 'coin' and 'address'. Found: {fields}")

    coins: List[str] = []
    addresses: List[Optional[str]] = []
    
    for row in rows:
        coins.append(row.get(coin_col, ""))
        addresses.append(row.get(address_col))

    logger.info(f"Debug endpoint processing {len(rows)} rows")

    addrs_present = [a for a in addresses if _valid_address(a)]
    
    try:
        addr_to_id = await map_address_to_cmc_ids(addrs_present, api_key=api_key)
    except Exception as e:
        logger.error(f"Error in address mapping during debug: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error mapping addresses: {str(e)}")

    symbols_needed: List[str] = []
    idx_to_symbol: Dict[int, str] = {}
    for idx, (coin, addr) in enumerate(zip(coins, addresses)):
        if not addr or _normalize(addr) not in addr_to_id:
            if coin.strip():
                symbols_needed.append(coin.strip())
                idx_to_symbol[idx] = coin.strip()
    
    try:
        sym_to_id = await map_symbol_to_cmc_ids(symbols_needed, api_key=api_key)
    except Exception as e:
        logger.error(f"Error in symbol mapping during debug: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error mapping symbols: {str(e)}")

    cmc_ids: List[Optional[int]] = []
    for idx, (coin, addr) in enumerate(zip(coins, addresses)):
        cid = None
        if addr and _normalize(addr) in addr_to_id:
            cid = addr_to_id[_normalize(addr)]
        else:
            sym = idx_to_symbol.get(idx)
            if sym and _normalize(sym.upper()) in sym_to_id:
                cid = sym_to_id[_normalize(sym.upper())]
        cmc_ids.append(cid)

    unique_ids = sorted({i for i in cmc_ids if i is not None})
    
    try:
        id_to_price = await fetch_prices_for_ids(unique_ids, date_end_utc=dt_target, api_key=api_key)
    except Exception as e:
        logger.error(f"Error fetching prices in debug: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error fetching prices: {str(e)}")

    # Try DEX for addresses without price
    addresses_without_price = []
    for addr, cid in zip(addresses, cmc_ids):
        if addr and _valid_address(addr):
            if cid is None or id_to_price.get(cid) is None:
                addresses_without_price.append(addr)
    
    dex_prices = {}
    if addresses_without_price:
        try:
            dex_prices = await fetch_dex_prices_for_addresses(
                addresses_without_price,
                date_end_utc=dt_target,
                api_key=api_key
            )
        except Exception as e:
            logger.error(f"Error fetching DEX prices in debug: {str(e)}")

    debug_rows = []
    for coin, addr, cid in zip(coins, addresses, cmc_ids):
        via_address = bool(addr and _normalize(addr) in addr_to_id)
        via_symbol = bool(cid and not via_address)
        
        # Price: try CMC ID first, then DEX
        price_val = id_to_price.get(cid) if cid is not None else None
        via_dex = False
        
        if price_val is None and addr and _normalize(addr) in dex_prices:
            price_val = dex_prices.get(_normalize(addr))
            via_dex = bool(price_val is not None)
        
        reason = None
        if price_val is None:
            if cid is None:
                if not _valid_address(addr):
                    reason = "no_valid_address_symbol_not_found"
                else:
                    reason = "address_not_recognized_by_cmc_or_dex"
            else:
                reason = "id_found_no_historical_price"
        
        debug_rows.append({
            "coin": coin,
            "address": addr,
            "address_decorated": _decorate_address_for_cmc(addr) if addr else None,
            "address_valid": _valid_address(addr),
            "detected_network": _detect_network_slug(addr) if addr and _valid_address(addr) else None,
            "mapped_via_address": via_address,
            "mapped_via_symbol": via_symbol,
            "mapped_via_dex": via_dex,
            "cmc_id": cid,
            "price_usd": price_val,
            "reason": reason
        })

    summary = {
        "total_rows": len(coins),
        "address_mapped": sum(1 for r in debug_rows if r["mapped_via_address"]),
        "symbol_mapped": sum(1 for r in debug_rows if r["mapped_via_symbol"]),
        "dex_mapped": sum(1 for r in debug_rows if r["mapped_via_dex"]),
        "with_price": sum(1 for r in debug_rows if r["price_usd"] is not None),
        "symbols_needed": sorted(set(symbols_needed)),
        "sym_to_id_sample": {k: sym_to_id[k] for k in list(sym_to_id.keys())[:20]},
        "addresses_tried_dex": len(addresses_without_price),
    }
    
    return JSONResponse({"summary": summary, "rows": debug_rows[:200]})


@app.get("/test/price/{cmc_id}")
async def test_price_endpoint(
    cmc_id: int,
    target_date: str = "2025-10-31",
    api_key: Optional[str] = None,
):
    """
    Test endpoint to check what exact data we're getting from CMC API
    Returns raw API responses for debugging date/price issues
    """
    logger.info(f"Testing price fetch for CMC ID {cmc_id} on date {target_date}")
    
    try:
        api_key = api_key or _env_api_key()
    except RuntimeError as e:
        raise HTTPException(status_code=500, detail=str(e))
    
    # Parse target date
    try:
        dt_target = datetime.fromisoformat(target_date).replace(tzinfo=timezone.utc)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid date format. Use YYYY-MM-DD.")
    
    # Prepare date range (entire day in UTC)
    time_start = datetime(dt_target.year, dt_target.month, dt_target.day, 0, 0, 0, tzinfo=timezone.utc)
    time_end = datetime(dt_target.year, dt_target.month, dt_target.day, 23, 59, 59, tzinfo=timezone.utc)
    
    headers = {"X-CMC_PRO_API_KEY": api_key}
    test_results = {
        "cmc_id": cmc_id,
        "target_date": target_date,
        "target_date_parsed": dt_target.isoformat(),
        "time_start": time_start.isoformat(),
        "time_end": time_end.isoformat(),
        "quotes_endpoint": {},
        "ohlcv_endpoint": {},
    }
    
    async with httpx.AsyncClient(timeout=REQUEST_TIMEOUT) as client:
        # Test 1: Quotes/historical endpoint
        params_quotes = {
            "id": str(cmc_id),
            "time_start": time_start.isoformat(),
            "time_end": time_end.isoformat(),
            "interval": "daily",
            "convert": "USD",
        }
        
        logger.info(f"Testing quotes endpoint with params: {params_quotes}")
        
        try:
            await _rate_limit()
            resp_quotes = await client.get(
                f"{CMC_BASE}/v2/cryptocurrency/quotes/historical",
                headers=headers,
                params=params_quotes
            )
            resp_quotes.raise_for_status()
            quotes_data = resp_quotes.json()
            
            test_results["quotes_endpoint"] = {
                "url": str(resp_quotes.url),
                "status": resp_quotes.status_code,
                "data": quotes_data,
                "extracted_price": None
            }
            
            # Try to extract price
            quotes = quotes_data.get("data", {}).get(str(cmc_id), {}).get("quotes") or []
            extracted = _extract_price_from_quotes(quotes, dt_target.date())
            test_results["quotes_endpoint"]["extracted_price"] = extracted
            
        except Exception as e:
            test_results["quotes_endpoint"]["error"] = str(e)
            logger.error(f"Quotes endpoint error: {e}")
        
        # Test 2: OHLCV/historical endpoint
        params_ohlcv = {
            "id": str(cmc_id),
            "time_start": time_start.date().isoformat(),
            "time_end": time_end.date().isoformat(),
            "convert": "USD",
            "interval": "daily",
        }
        
        logger.info(f"Testing OHLCV endpoint with params: {params_ohlcv}")
        
        try:
            await _rate_limit()
            resp_ohlcv = await client.get(
                f"{CMC_BASE}/v1/cryptocurrency/ohlcv/historical",
                headers=headers,
                params=params_ohlcv
            )
            resp_ohlcv.raise_for_status()
            ohlcv_data = resp_ohlcv.json()
            
            test_results["ohlcv_endpoint"] = {
                "url": str(resp_ohlcv.url),
                "status": resp_ohlcv.status_code,
                "data": ohlcv_data,
                "extracted_price": None
            }
            
            # Try to extract price
            series = ohlcv_data.get("data", {}).get(str(cmc_id), {}).get("quotes") or []
            extracted = _extract_close_from_ohlcv(series, dt_target.date())
            test_results["ohlcv_endpoint"]["extracted_price"] = extracted
            
        except Exception as e:
            test_results["ohlcv_endpoint"]["error"] = str(e)
            logger.error(f"OHLCV endpoint error: {e}")
    
    return JSONResponse(test_results)
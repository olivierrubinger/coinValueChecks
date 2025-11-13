"""
Coin Price Extractor - Simplified & Optimized v2.0.0

FastAPI application for fetching historical cryptocurrency prices from CoinMarketCap.

Features:
- Single CSV input with 'coin' and 'address' columns
- Multi-chain support (Ethereum, Solana, etc.)
- Retry logic with exponential backoff
- Comprehensive error handling and logging
- Batch processing with individual fallback
- Date validation and input checking
- Debug endpoint for troubleshooting

Optimizations applied:
1. Retry logic: Exponential backoff for failed API calls (HTTP 429, 5xx errors)
2. Error handling: Try-catch blocks around all API calls with detailed error messages
3. Logging: Comprehensive logging at INFO/DEBUG levels for monitoring and troubleshooting
4. Input validation: CSV format, date format, future date checking
5. Batch processing: Configurable batch sizes with individual retry for missing items
6. Environment-based config: All constants configurable via .env file
7. Multi-chain address decoration: Auto-prefixing for Solana and other chains
8. Dual price endpoint strategy: Quotes/historical → OHLCV/historical fallback
9. Request timeouts: Configurable timeouts to prevent hanging
10. Health & config endpoints: Monitor application status and settings

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
from datetime import datetime, timezone
from dotenv import load_dotenv
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Coin Price Extractor - Simplified",
    version="2.0.0",
    description="Lê 1 CSV com 'coin' e 'address', busca preços na CMC e retorna CSV."
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
        raise RuntimeError(f"Defina a variável de ambiente {CMC_API_KEY_ENV} com a sua API key da CoinMarketCap.")
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
    """EVM (0x) ou Solana (base58) com prefixo SOL:"""
    a = addr.strip()
    if a.startswith("0x") and len(a) == 42:
        return a
    if not a.startswith("0x") and 30 <= len(a) <= 50:
        if not a.upper().startswith("SOL:"):
            return f"SOL:{a}"
    return a


def _detect_network_slug(addr: str) -> str:
    """Detecta a rede baseado no formato do endereço"""
    a = addr.strip()
    
    # Ethereum e EVM chains (0x + 40 hex chars)
    if a.startswith("0x") and len(a) == 42:
        return "ethereum"
    
    # Solana (base58, ~32-44 chars, sem 0x)
    if not a.startswith("0x") and 32 <= len(a) <= 44:
        # Solana usa base58 alphabet
        return "solana"
    
    # Stellar (starts with G, ~56 chars)
    if a.startswith("G") and len(a) == 56:
        return "stellar"
    
    # Default para ethereum se não identificado
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
    """Mapeia endereços -> CMC ID com retry logic (sem prefixo SOL:)"""
    logger.info(f"Mapping {len(addresses)} addresses to CMC IDs")
    mapping: Dict[str, int] = {}
    headers = {"X-CMC_PRO_API_KEY": api_key}
    addrs = [a for a in {a.strip(): None for a in addresses if _valid_address(a)}.keys()]
    # NÃO decorar com SOL: para o endpoint /info - usar endereços raw
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
    """Extrai IDs dos contratos do payload CMC"""
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
    """Mapeia símbolo -> CMC ID (escolhe melhor rank) com retry logic"""
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
    """Extrai símbolos do payload CMC"""
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
    """Busca preços históricos com fallback quotes → ohlcv e retry logic"""
    logger.info(f"Fetching prices for {len(ids)} CMC IDs")
    headers = {"X-CMC_PRO_API_KEY": api_key}
    results: Dict[int, Optional[float]] = {}
    if not ids:
        return results

    PRICE_BATCH = 100
    time_start = datetime(date_end_utc.year, date_end_utc.month, date_end_utc.day, 0, 0, 0, tzinfo=timezone.utc)
    time_end = datetime(date_end_utc.year, date_end_utc.month, date_end_utc.day, 23, 59, 59, tzinfo=timezone.utc)

    async with httpx.AsyncClient(timeout=REQUEST_TIMEOUT) as client:
        for i in range(0, len(ids), PRICE_BATCH):
            chunk = ids[i:i + PRICE_BATCH]
            logger.debug(f"Fetching prices for batch {i//PRICE_BATCH + 1}: {len(chunk)} IDs")
            
            # 1) Try quotes/historical with retry
            params = {
                "id": ",".join(str(x) for x in chunk),
                "time_start": time_start.isoformat(),
                "time_end": time_end.isoformat(),
                "interval": "daily",
                "convert": "USD",
            }
            
            quotes_response = await _make_request_with_retry(
                client,
                "GET",
                f"{CMC_BASE}/v2/cryptocurrency/quotes/historical",
                headers=headers,
                params=params
            )

            missing_ids: List[int] = []
            if quotes_response:
                for cid in chunk:
                    quotes = quotes_response.get("data", {}).get(str(cid), {}).get("quotes") or []
                    price_val = _extract_price_from_quotes(quotes, date_end_utc.date())
                    
                    if price_val is None:
                        missing_ids.append(cid)
                    else:
                        results[cid] = price_val
                logger.debug(f"Quotes endpoint returned {len(chunk) - len(missing_ids)} prices, {len(missing_ids)} missing")
            else:
                logger.warning("Quotes endpoint failed, trying OHLCV for all IDs in chunk")
                missing_ids = list(chunk)

            # 2) Fallback OHLCV with retry (v1 endpoint - correto conforme docs CMC)
            if missing_ids:
                params_ohlcv = {
                    "id": ",".join(str(x) for x in missing_ids),
                    "time_start": time_start.date().isoformat(),  # YYYY-MM-DD format
                    "time_end": time_end.date().isoformat(),
                    "convert": "USD",
                    "interval": "daily",
                }
                
                ohlcv_response = await _make_request_with_retry(
                    client,
                    "GET",
                    f"{CMC_BASE}/v1/cryptocurrency/ohlcv/historical",  # v1, não v2!
                    headers=headers,
                    params=params_ohlcv
                )
                
                if ohlcv_response:
                    d2 = ohlcv_response.get("data", {})
                    for cid in missing_ids:
                        series = d2.get(str(cid), {}).get("quotes") or []
                        close_val = _extract_close_from_ohlcv(series, date_end_utc.date())
                        results[cid] = close_val
                    logger.debug(f"OHLCV fallback completed for {len(missing_ids)} IDs")
                else:
                    logger.warning(f"OHLCV fallback failed for {len(missing_ids)} IDs")
                    for cid in missing_ids:
                        results[cid] = None

    logger.info(f"Price fetch complete: {sum(1 for v in results.values() if v is not None)}/{len(results)} with prices")
    return results


async def fetch_dex_prices_for_addresses(
    addresses: List[str], 
    date_end_utc: datetime, 
    api_key: str
) -> Dict[str, Optional[float]]:
    """
    Tenta buscar preços de DEX para endereços que não foram encontrados como cryptoassets.
    Usa a DEX API v4 da CoinMarketCap.
    """
    logger.info(f"Trying DEX API for {len(addresses)} addresses")
    headers = {"X-CMC_PRO_API_KEY": api_key}
    results: Dict[str, Optional[float]] = {}
    
    if not addresses:
        return results
    
    # Para DEX, precisamos do formato YYYY-MM-DD
    target_date_str = date_end_utc.date().isoformat()
    
    async with httpx.AsyncClient(timeout=REQUEST_TIMEOUT) as client:
        for addr in addresses:
            network_slug = _detect_network_slug(addr)
            addr_clean = addr.strip()
            
            # Remove prefixo SOL: se tiver
            if addr_clean.upper().startswith("SOL:"):
                addr_clean = addr_clean[4:]
            
            logger.debug(f"Trying DEX for address {addr_clean[:10]}... on network {network_slug}")
            
            # Tenta buscar OHLCV histórico do par DEX
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
                # Extrai o close price do primeiro (e único) dia
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
    """Extrai preço das quotes, preferindo dia exato"""
    price_val = None
    for q in quotes:
        try:
            ts = q.get("timestamp") or q.get("time_close") or q.get("time_open")
            if ts:
                d = datetime.fromisoformat(str(ts).replace("Z", "+00:00")).date()
                if d == target_date:
                    price_val = q.get("quote", {}).get("USD", {}).get("price")
        except Exception:
            pass
    if price_val is None and quotes:
        price_val = quotes[-1].get("quote", {}).get("USD", {}).get("price")
    try:
        return float(price_val) if price_val is not None else None
    except Exception:
        return None


def _extract_close_from_ohlcv(series: list, target_date) -> Optional[float]:
    """Extrai close do OHLCV, preferindo dia exato"""
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


@app.post("/prices")
async def get_prices(
    csv_file: UploadFile = File(..., description="CSV com colunas 'coin' e 'address'"),
    target_date: str = Form("2025-10-31", description="Data alvo (YYYY-MM-DD)"),
    api_key: Optional[str] = Form(None, description="(Opcional) CMC API key"),
):
    """Endpoint principal: retorna CSV com preços"""
    logger.info(f"Processing price request for date: {target_date}")
    
    try:
        api_key = api_key or _env_api_key()
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
                detail=f"Data {target_date} está no futuro. Use uma data passada (máximo: {now.date().isoformat()})."
            )
    except ValueError:
        logger.error(f"Invalid date format: {target_date}")
        raise HTTPException(status_code=400, detail="Parâmetro target_date inválido. Use YYYY-MM-DD.")

    # Validate file upload
    if not csv_file.filename:
        logger.error("No filename in uploaded file")
        raise HTTPException(status_code=400, detail="Arquivo CSV não fornecido.")
    
    if not csv_file.filename.lower().endswith('.csv'):
        logger.warning(f"Uploaded file doesn't have .csv extension: {csv_file.filename}")

    # Read CSV with error handling
    try:
        rows = _sniff_and_read(csv_file.file)
    except Exception as e:
        logger.error(f"Error reading CSV file: {str(e)}")
        raise HTTPException(status_code=400, detail=f"Erro ao ler CSV: {str(e)}")
    
    if not rows:
        logger.error("CSV file is empty")
        raise HTTPException(status_code=400, detail="CSV vazio ou inválido.")

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
            detail=f"CSV deve ter colunas 'coin' e 'address'. Encontradas: {fields}"
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
        raise HTTPException(status_code=500, detail=f"Erro ao mapear endereços: {str(e)}")

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
        raise HTTPException(status_code=500, detail=f"Erro ao mapear símbolos: {str(e)}")

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
        raise HTTPException(status_code=500, detail=f"Erro ao buscar preços: {str(e)}")

    # Identifica endereços que não obtiveram preço (podem ser DEX pairs)
    addresses_without_price: List[str] = []
    addr_to_idx: Dict[str, int] = {}
    
    for idx, (coin, addr, cmc_id) in enumerate(zip(coins, addresses, cmc_ids)):
        if addr and _valid_address(addr):
            # Se não tem CMC ID OU tem ID mas não tem preço, tenta DEX
            if cmc_id is None or id_to_price.get(cmc_id) is None:
                addresses_without_price.append(addr)
                addr_to_idx[_normalize(addr)] = idx
    
    # Tenta buscar via DEX API para endereços sem preço
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
    # Novo formato: (coin, address, price)
    result_data: List[Tuple[str, str, Optional[float]]] = []
    for coin, addr, cmc_id in zip(coins, addresses, cmc_ids):
        # Prioridade: 1) Preço via CMC ID, 2) Preço via DEX, 3) None
        price = None
        if cmc_id is not None:
            price = id_to_price.get(cmc_id)
        
        # Se não tem preço via ID, tenta DEX
        if price is None and addr and _normalize(addr) in dex_prices:
            price = dex_prices.get(_normalize(addr))
        
        result_data.append((coin, addr or "", price))

    # Gera CSV com novo formato: coin, address, price_usd_YYYY-MM-DD
    buf = _build_output_csv(result_data, target_date=target_date)
    
    # Save to output folder
    output_dir = Path("output")
    output_dir.mkdir(exist_ok=True)
    output_file = output_dir / f"prices_{target_date}.csv"
    
    csv_content = buf.getvalue()
    csv_bytes = csv_content.encode("utf-8")
    output_file.write_bytes(csv_bytes)
    logger.info(f"Saved output to {output_file}")
    
    # Retorna Response completa (não streaming) para curl mostrar progresso correto
    return Response(
        content=csv_bytes,
        media_type="text/csv",
        headers={"Content-Disposition": f'attachment; filename="prices_{target_date}.csv"'}
    )


@app.post("/prices/debug")
async def get_prices_debug(
    csv_file: UploadFile = File(...),
    target_date: str = Form("2025-10-31"),
    api_key: Optional[str] = Form(None),
):
    """Endpoint de debug: retorna JSON com detalhes completos de processamento"""
    logger.info(f"Processing debug request for date: {target_date}")
    
    try:
        api_key = api_key or _env_api_key()
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
                detail=f"Data {target_date} está no futuro. Use uma data passada (máximo: {now.date().isoformat()})."
            )
    except ValueError:
        logger.error(f"Invalid date format in debug: {target_date}")
        raise HTTPException(status_code=400, detail="Data inválida. Use YYYY-MM-DD.")

    # Read and validate CSV
    try:
        rows = _sniff_and_read(csv_file.file)
    except Exception as e:
        logger.error(f"Error reading CSV in debug: {str(e)}")
        raise HTTPException(status_code=400, detail=f"Erro ao ler CSV: {str(e)}")
    
    if not rows:
        logger.error("CSV is empty in debug endpoint")
        raise HTTPException(status_code=400, detail="CSV vazio.")

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
        raise HTTPException(status_code=400, detail=f"Colunas esperadas: 'coin' e 'address'. Encontradas: {fields}")

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
        raise HTTPException(status_code=500, detail=f"Erro ao mapear endereços: {str(e)}")

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
        raise HTTPException(status_code=500, detail=f"Erro ao mapear símbolos: {str(e)}")

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
        raise HTTPException(status_code=500, detail=f"Erro ao buscar preços: {str(e)}")

    # Tenta DEX para endereços sem preço
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
        
        # Preço: tenta CMC ID primeiro, depois DEX
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
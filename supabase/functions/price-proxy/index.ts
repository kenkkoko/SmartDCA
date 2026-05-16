// Price proxy — Yahoo Finance (stocks) + Binance (crypto) unified endpoint
// Single source per asset class. No API keys required (anon Supabase key still
// guards the function, but no third-party secrets are needed).
//
// Usage:
//   Current price (single symbol or batch):
//     GET /functions/v1/price-proxy?symbols=0050.TW,AAPL&type=stock
//     GET /functions/v1/price-proxy?symbols=bitcoin,ethereum&type=crypto
//
//   Historical series (daily OHLC + close):
//     GET /functions/v1/price-proxy?symbols=0050.TW&type=stock&history=1y
//     GET /functions/v1/price-proxy?symbols=bitcoin&type=crypto&history=2y
//     range: 1mo | 3mo | 6mo | 1y | 2y | 5y | max

import { serve } from "https://deno.land/std@0.168.0/http/server.ts";

const corsHeaders = {
  "Access-Control-Allow-Origin": "*",
  "Access-Control-Allow-Headers": "authorization, x-client-info, apikey, content-type",
};

interface PriceResult {
  symbol: string;
  price: number | null;
  previousClose: number | null;
  change: number | null;
  changePercent: number | null;
  currency: string | null;
  history?: { date: string; price: number; open?: number; high?: number; low?: number }[];
  error?: string;
}

// ─────────────────────────────────────────────
// Yahoo Finance v8 chart API
// ─────────────────────────────────────────────
async function getStockPrice(
  symbol: string,
  historyRange?: string
): Promise<PriceResult> {
  const base = {
    symbol,
    price: null,
    previousClose: null,
    change: null,
    changePercent: null,
    currency: null,
  } as PriceResult;
  try {
    // When fetching history, use wider range; otherwise just 5d for current price
    const range = historyRange || "5d";
    const url = `https://query1.finance.yahoo.com/v8/finance/chart/${encodeURIComponent(symbol)}?interval=1d&range=${range}`;
    const r = await fetch(url, {
      headers: {
        "User-Agent":
          "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0 Safari/537.36",
        Accept: "application/json",
      },
    });

    if (!r.ok) {
      return { ...base, error: `Yahoo returned ${r.status}` };
    }

    const data = await r.json();
    const result = data?.chart?.result?.[0];
    if (!result) {
      return {
        ...base,
        error: data?.chart?.error?.description || "Symbol not found",
      };
    }

    const meta = result.meta || {};
    const price = meta.regularMarketPrice ?? null;
    const prev = meta.previousClose ?? meta.chartPreviousClose ?? null;
    const change = price !== null && prev !== null ? price - prev : null;
    const changePercent = change !== null && prev ? (change / prev) * 100 : null;

    const out: PriceResult = {
      symbol,
      price,
      previousClose: prev,
      change,
      changePercent,
      currency: meta.currency ?? null,
    };

    // Include historical OHLC series if requested
    if (historyRange) {
      const timestamps: number[] = result.timestamp || [];
      const q = result.indicators?.quote?.[0] || {};
      const opens: (number | null)[] = q.open || [];
      const highs: (number | null)[] = q.high || [];
      const lows: (number | null)[] = q.low || [];
      const closes: (number | null)[] = q.close || [];
      const history: { date: string; price: number; open?: number; high?: number; low?: number }[] = [];
      for (let i = 0; i < timestamps.length; i++) {
        const close = closes[i];
        if (close !== null && close !== undefined && !isNaN(close)) {
          const date = new Date(timestamps[i] * 1000).toISOString().slice(0, 10);
          const entry: any = { date, price: close };
          if (opens[i] !== null && opens[i] !== undefined && !isNaN(opens[i])) entry.open = opens[i];
          if (highs[i] !== null && highs[i] !== undefined && !isNaN(highs[i])) entry.high = highs[i];
          if (lows[i] !== null && lows[i] !== undefined && !isNaN(lows[i])) entry.low = lows[i];
          history.push(entry);
        }
      }
      out.history = history;
    }
    return out;
  } catch (e) {
    return { ...base, error: String(e) };
  }
}

// ─────────────────────────────────────────────
// Crypto — Binance public API
// • Klines for historical OHLC (true daily, no key, CORS-friendly)
// • ticker/24hr for current price + 24h change
// CoinGecko slug → Binance USDT pair. Keep this in sync with index.html.
// ─────────────────────────────────────────────
const COIN_TO_BINANCE: Record<string, string> = {
  bitcoin: "BTCUSDT", ethereum: "ETHUSDT", solana: "SOLUSDT", binancecoin: "BNBUSDT",
  ripple: "XRPUSDT", cardano: "ADAUSDT", dogecoin: "DOGEUSDT", polkadot: "DOTUSDT",
  "avalanche-2": "AVAXUSDT", tron: "TRXUSDT", chainlink: "LINKUSDT",
  "matic-network": "MATICUSDT", litecoin: "LTCUSDT", "shiba-inu": "SHIBUSDT",
  cosmos: "ATOMUSDT", uniswap: "UNIUSDT", near: "NEARUSDT", aptos: "APTUSDT",
  sui: "SUIUSDT", arbitrum: "ARBUSDT", optimism: "OPUSDT", filecoin: "FILUSDT",
  "internet-computer": "ICPUSDT", stellar: "XLMUSDT", "bitcoin-cash": "BCHUSDT",
  algorand: "ALGOUSDT", vechain: "VETUSDT", "the-graph": "GRTUSDT",
  aave: "AAVEUSDT", maker: "MKRUSDT", tezos: "XTZUSDT", monero: "XMRUSDT",
  pepe: "PEPEUSDT", floki: "FLOKIUSDT", bonk: "BONKUSDT",
  ondo: "ONDOUSDT", injective: "INJUSDT", sei: "SEIUSDT", kaspa: "KASUSDT",
  "render-token": "RNDRUSDT", "fetch-ai": "FETUSDT", worldcoin: "WLDUSDT",
  "ethereum-classic": "ETCUSDT", "hedera-hashgraph": "HBARUSDT",
};

function getBinanceSymbol(slug: string): string | null {
  const lower = slug.toLowerCase();
  if (COIN_TO_BINANCE[lower]) return COIN_TO_BINANCE[lower];
  // Heuristic: short alphanumeric (likely a ticker like 'pepe') → try LOWERUSDT
  if (/^[a-z0-9]{2,7}$/.test(lower)) return lower.toUpperCase() + "USDT";
  return null;
}

async function getCryptoPrice(
  symbol: string,
  historyRange?: string
): Promise<PriceResult> {
  const base = {
    symbol,
    price: null,
    previousClose: null,
    change: null,
    changePercent: null,
    currency: "USD",
  } as PriceResult;

  const binSym = getBinanceSymbol(symbol);
  if (!binSym) {
    return { ...base, error: `Unsupported symbol on Binance: ${symbol}` };
  }

  try {
    if (historyRange) {
      const days =
        historyRange === "1mo" ? 30 :
        historyRange === "3mo" ? 90 :
        historyRange === "6mo" ? 180 :
        historyRange === "1y" ? 365 :
        historyRange === "2y" ? 730 :
        historyRange === "5y" ? 1825 :
        historyRange === "max" ? 1825 : 365;
      const limit = Math.min(1000, days + 5);
      const url = `https://api.binance.com/api/v3/klines?symbol=${binSym}&interval=1d&limit=${limit}`;
      const r = await fetch(url, { headers: { Accept: "application/json" } });
      if (!r.ok) return { ...base, error: `Binance returned ${r.status}` };
      const klines: any[] = await r.json();
      const history = klines.map((k: any[]) => ({
        date: new Date(k[0]).toISOString().slice(0, 10),
        price: +k[4],
        open: +k[1],
        high: +k[2],
        low: +k[3],
      }));
      const last = klines[klines.length - 1];
      const prev = klines[klines.length - 2];
      return {
        ...base,
        symbol: symbol.toLowerCase(),
        price: last ? +last[4] : null,
        previousClose: prev ? +prev[4] : null,
        change: last && prev ? +last[4] - +prev[4] : null,
        changePercent: last && prev ? ((+last[4] - +prev[4]) / +prev[4]) * 100 : null,
        currency: "USD",
        history,
      };
    }

    // Current quote — Binance 24h ticker
    const r = await fetch(
      `https://api.binance.com/api/v3/ticker/24hr?symbol=${binSym}`,
      { headers: { Accept: "application/json" } }
    );
    if (!r.ok) return { ...base, error: `Binance returned ${r.status}` };
    const d = await r.json();
    const price = +d.lastPrice;
    const change = +d.priceChange;
    const changePercent = +d.priceChangePercent;
    const prev = price - change;
    return {
      symbol: symbol.toLowerCase(),
      price,
      previousClose: prev,
      change,
      changePercent,
      currency: "USD",
    };
  } catch (e) {
    return { ...base, error: String(e) };
  }
}

serve(async (req) => {
  if (req.method === "OPTIONS")
    return new Response("ok", { headers: corsHeaders });

  try {
    const url = new URL(req.url);
    const symbolsRaw = url.searchParams.get("symbols") || "";
    const type = url.searchParams.get("type") || "stock";
    const historyRange = url.searchParams.get("history") || undefined;
    const symbols = symbolsRaw
      .split(",")
      .map((s) => s.trim())
      .filter(Boolean);

    if (symbols.length === 0) {
      return new Response(
        JSON.stringify({ error: "symbols query param is required" }),
        {
          status: 400,
          headers: { ...corsHeaders, "Content-Type": "application/json" },
        }
      );
    }
    if (symbols.length > 50) {
      return new Response(
        JSON.stringify({ error: "Too many symbols (max 50)" }),
        {
          status: 400,
          headers: { ...corsHeaders, "Content-Type": "application/json" },
        }
      );
    }

    const fetcher = type === "crypto" ? getCryptoPrice : getStockPrice;
    const results = await Promise.all(
      symbols.map((s) => fetcher(s, historyRange))
    );

    return new Response(JSON.stringify({ data: results }), {
      headers: { ...corsHeaders, "Content-Type": "application/json" },
    });
  } catch (e) {
    console.error("price-proxy error", String(e));
    return new Response(JSON.stringify({ error: String(e) }), {
      status: 500,
      headers: { ...corsHeaders, "Content-Type": "application/json" },
    });
  }
});

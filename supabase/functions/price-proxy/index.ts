// Price proxy — Yahoo Finance + CoinMarketCap unified endpoint
// Returns latest price + previous close + change% for one or many symbols.
//
// Usage:
//   GET /functions/v1/price-proxy?symbols=0050.TW,AAPL&type=stock
//   GET /functions/v1/price-proxy?symbols=bitcoin,ethereum&type=crypto

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
  error?: string;
}

// ─────────────────────────────────────────────
// Yahoo Finance v8 chart API — works for TW & US stocks, ETFs
// (yfinance Python library uses this same endpoint underneath)
// ─────────────────────────────────────────────
async function getStockPrice(symbol: string): Promise<PriceResult> {
  const base = {
    symbol,
    price: null,
    previousClose: null,
    change: null,
    changePercent: null,
    currency: null,
  } as PriceResult;
  try {
    const url = `https://query1.finance.yahoo.com/v8/finance/chart/${encodeURIComponent(symbol)}?interval=1d&range=5d`;
    const r = await fetch(url, {
      headers: {
        "User-Agent":
          "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0 Safari/537.36",
        "Accept": "application/json",
      },
    });

    if (!r.ok) {
      return { ...base, error: `Yahoo returned ${r.status}` };
    }

    const data = await r.json();
    const result = data?.chart?.result?.[0];
    if (!result) {
      return { ...base, error: data?.chart?.error?.description || "Symbol not found" };
    }

    const meta = result.meta || {};
    const price = meta.regularMarketPrice ?? null;
    const prev = meta.previousClose ?? meta.chartPreviousClose ?? null;
    const change = price !== null && prev !== null ? price - prev : null;
    const changePercent = change !== null && prev ? (change / prev) * 100 : null;

    return {
      symbol,
      price,
      previousClose: prev,
      change,
      changePercent,
      currency: meta.currency ?? null,
    };
  } catch (e) {
    return { ...base, error: String(e) };
  }
}

// ─────────────────────────────────────────────
// CoinMarketCap — for crypto
// ─────────────────────────────────────────────
async function getCryptoPrice(symbol: string): Promise<PriceResult> {
  const CMC_API_KEY = Deno.env.get("CMC_API_KEY");
  const base = {
    symbol,
    price: null,
    previousClose: null,
    change: null,
    changePercent: null,
    currency: "USD",
  } as PriceResult;
  if (!CMC_API_KEY) {
    return { ...base, error: "CMC_API_KEY not configured" };
  }
  try {
    const r = await fetch(
      `https://pro-api.coinmarketcap.com/v1/cryptocurrency/quotes/latest?slug=${encodeURIComponent(symbol.toLowerCase())}`,
      {
        headers: { "X-CMC_PRO_API_KEY": CMC_API_KEY, Accept: "application/json" },
      }
    );
    if (!r.ok) return { ...base, error: `CMC returned ${r.status}` };
    const data = await r.json();
    const coin = data?.data && Object.values(data.data)[0] as any;
    if (!coin) return { ...base, error: "Symbol not found" };
    const price = coin.quote?.USD?.price ?? null;
    const changePercent = coin.quote?.USD?.percent_change_24h ?? null;
    const prev = price !== null && changePercent !== null ? price / (1 + changePercent / 100) : null;
    const change = price !== null && prev !== null ? price - prev : null;
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
  if (req.method === "OPTIONS") return new Response("ok", { headers: corsHeaders });

  try {
    const url = new URL(req.url);
    const symbolsRaw = url.searchParams.get("symbols") || "";
    const type = url.searchParams.get("type") || "stock"; // "stock" | "crypto"
    const symbols = symbolsRaw.split(",").map(s => s.trim()).filter(Boolean);

    if (symbols.length === 0) {
      return new Response(JSON.stringify({ error: "symbols query param is required" }), {
        status: 400,
        headers: { ...corsHeaders, "Content-Type": "application/json" },
      });
    }
    if (symbols.length > 50) {
      return new Response(JSON.stringify({ error: "Too many symbols (max 50)" }), {
        status: 400,
        headers: { ...corsHeaders, "Content-Type": "application/json" },
      });
    }

    const fetcher = type === "crypto" ? getCryptoPrice : getStockPrice;
    const results = await Promise.all(symbols.map(fetcher));

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

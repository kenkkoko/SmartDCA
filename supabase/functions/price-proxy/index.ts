// Price proxy — Yahoo Finance + CoinMarketCap unified endpoint
// Supports current quote OR historical price series.
//
// Usage:
//   Current price (single symbol or batch):
//     GET /functions/v1/price-proxy?symbols=0050.TW,AAPL&type=stock
//     GET /functions/v1/price-proxy?symbols=bitcoin,ethereum&type=crypto
//
//   Historical series:
//     GET /functions/v1/price-proxy?symbols=0050.TW&type=stock&history=1y
//     range: 1mo | 3mo | 6mo | 1y | 2y | 5y | max
//     interval: 1d (default), 1wk, 1mo

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
  history?: { date: string; price: number }[];
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

    // Include historical series if requested
    if (historyRange) {
      const timestamps: number[] = result.timestamp || [];
      const closes: (number | null)[] =
        result.indicators?.quote?.[0]?.close || [];
      const history: { date: string; price: number }[] = [];
      for (let i = 0; i < timestamps.length; i++) {
        const close = closes[i];
        if (close !== null && close !== undefined && !isNaN(close)) {
          const date = new Date(timestamps[i] * 1000).toISOString().slice(0, 10);
          history.push({ date, price: close });
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
// CoinMarketCap — for crypto (current only for now)
// Historical needs CMC paid plan; fallback to CoinGecko free historical
// ─────────────────────────────────────────────
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

  // For history, use CoinGecko free public API
  if (historyRange) {
    try {
      const days =
        historyRange === "1mo"
          ? 30
          : historyRange === "3mo"
          ? 90
          : historyRange === "6mo"
          ? 180
          : historyRange === "1y"
          ? 365
          : historyRange === "2y"
          ? 730
          : historyRange === "5y"
          ? 1825
          : 365;
      const cgUrl = `https://api.coingecko.com/api/v3/coins/${encodeURIComponent(
        symbol.toLowerCase()
      )}/market_chart?vs_currency=usd&days=${days}&interval=daily`;
      const r = await fetch(cgUrl, {
        headers: { Accept: "application/json" },
      });
      if (!r.ok) {
        return { ...base, error: `CoinGecko returned ${r.status}` };
      }
      const data = await r.json();
      const prices: [number, number][] = data?.prices || [];
      const history = prices.map(([ts, price]) => ({
        date: new Date(ts).toISOString().slice(0, 10),
        price,
      }));
      const last = prices[prices.length - 1];
      const prev = prices[prices.length - 2];
      return {
        ...base,
        symbol: symbol.toLowerCase(),
        price: last ? last[1] : null,
        previousClose: prev ? prev[1] : null,
        change: last && prev ? last[1] - prev[1] : null,
        changePercent: last && prev ? ((last[1] - prev[1]) / prev[1]) * 100 : null,
        currency: "USD",
        history,
      };
    } catch (e) {
      return { ...base, error: String(e) };
    }
  }

  // Current quote via CMC
  const CMC_API_KEY = Deno.env.get("CMC_API_KEY");
  if (!CMC_API_KEY) {
    return { ...base, error: "CMC_API_KEY not configured" };
  }
  try {
    const r = await fetch(
      `https://pro-api.coinmarketcap.com/v1/cryptocurrency/quotes/latest?slug=${encodeURIComponent(
        symbol.toLowerCase()
      )}`,
      {
        headers: { "X-CMC_PRO_API_KEY": CMC_API_KEY, Accept: "application/json" },
      }
    );
    if (!r.ok) return { ...base, error: `CMC returned ${r.status}` };
    const data = await r.json();
    const coin = data?.data && (Object.values(data.data)[0] as any);
    if (!coin) return { ...base, error: "Symbol not found" };
    const price = coin.quote?.USD?.price ?? null;
    const changePercent = coin.quote?.USD?.percent_change_24h ?? null;
    const prev =
      price !== null && changePercent !== null
        ? price / (1 + changePercent / 100)
        : null;
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

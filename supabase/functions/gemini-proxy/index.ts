import { serve } from "https://deno.land/std@0.168.0/http/server.ts";
import { createClient } from "https://esm.sh/@supabase/supabase-js@2";

const corsHeaders = {
  "Access-Control-Allow-Origin": "*",
  "Access-Control-Allow-Headers": "authorization, x-client-info, apikey, content-type",
};

// SUPABASE_URL / SUPABASE_SERVICE_ROLE_KEY are injected automatically — the
// service-role client bypasses RLS.
function adminClient() {
  return createClient(
    Deno.env.get("SUPABASE_URL")!,
    Deno.env.get("SUPABASE_SERVICE_ROLE_KEY")!,
    { auth: { persistSession: false, autoRefreshToken: false } },
  );
}

// Resolves the user behind the request's Bearer token and confirms they are
// premium (or admin). Returns true on success, false otherwise.
// Note: when the frontend has no session it sends the public anon key as the
// token; getUser() rejects it (no `sub`), so anonymous callers resolve to false.
async function isPremiumCaller(req: Request): Promise<boolean> {
  const token = (req.headers.get("Authorization") || "").replace(/^Bearer\s+/i, "").trim();
  if (!token) return false;
  const admin = adminClient();
  const { data: { user }, error } = await admin.auth.getUser(token);
  if (error || !user) return false;
  const { data: profile } = await admin
    .from("user_profiles")
    .select("is_premium, is_admin")
    .eq("id", user.id)
    .single();
  return !!(profile && (profile.is_premium || profile.is_admin));
}

// --- Macro context -----------------------------------------------------------
// main.py writes one macro_snapshot row per day (Alpha Vantage data + upcoming
// economic events). When the caller asks for `macro: true`, the latest row is
// appended to the prompt server-side. Events are filtered against the request
// time so a release that already happened is never described as upcoming.
type MacroEvent = { at: string; type: string; title: string; impact: string };

const TW_FMT = new Intl.DateTimeFormat("zh-TW", {
  timeZone: "Asia/Taipei", month: "numeric", day: "numeric", weekday: "short",
  hour: "2-digit", minute: "2-digit", hour12: false,
});
const IMPACT_LABEL: Record<string, string> = { high: "高影響", medium: "中影響", low: "低影響" };
const STALE_DAYS = 3;

async function buildMacroBlock(): Promise<string> {
  const { data, error } = await adminClient()
    .from("macro_snapshot")
    .select("date, macro_context, upcoming_events")
    .order("date", { ascending: false })
    .limit(1)
    .maybeSingle();
  if (error || !data) {
    if (error) console.error("macro_snapshot read failed", error.message);
    return "";
  }

  const now = Date.now();
  const events = ((data.upcoming_events || []) as MacroEvent[])
    .map((e) => ({ ...e, t: Date.parse(e.at) }))
    .filter((e) => e.t > now && e.t < now + 7 * 86400_000)
    .sort((a, b) => a.t - b.t);
  const eventLines = events.map((e) => {
    const hours = Math.round((e.t - now) / 3600_000);
    const eta = hours < 24 ? `約 ${hours} 小時後` : `約 ${Math.round(hours / 24)} 天後`;
    return `- ${TW_FMT.format(e.t)}(${eta}):${e.title}(${IMPACT_LABEL[e.impact] || e.impact})`;
  });
  const hasHighSoon = events.some((e) => e.impact === "high" && e.t - now < 24 * 3600_000);

  const ageDays = Math.floor((now - Date.parse(`${data.date}T00:00:00+08:00`)) / 86400_000);
  const stale = ageDays > STALE_DAYS ? `(注意:總經資料已 ${ageDays} 天未更新,僅供參考)` : "";

  return [
    "---",
    `【伺服器提供的總經背景】${stale}`,
    data.macro_context || "(無總經數值)",
    "",
    `未來 7 天美國經濟事件(台灣時間,現在是 ${TW_FMT.format(now)}):`,
    eventLines.length ? eventLines.join("\n") : "- 無重大事件",
    "",
    "【總經使用規則】",
    "1. 情緒指數與價格位置仍是主要依據,總經只用來調整建議的力道與時機,不要推翻上面的行動邏輯。",
    "2. 殖利率一個月內明顯上升、或 CPI 連續升溫時,語氣偏保守;反之可略為積極。",
    hasHighSoon
      ? "3. 24 小時內有高影響事件,請明確提醒將單筆投入拆分到事件公布之後。"
      : "3. 若 7 天內有高影響事件,可簡短提及即可。",
    "4. 建議中用一句話點出最關鍵的總經因素,字數上限可放寬到 80 字。",
  ].join("\n");
}

serve(async (req) => {
  if (req.method === "OPTIONS") {
    return new Response("ok", { headers: corsHeaders });
  }
  try {
    const { prompt, apiKey: userApiKey, macro } = await req.json();
    if (!prompt || typeof prompt !== "string") {
      return new Response(JSON.stringify({ error: "prompt is required" }), {
        status: 400, headers: { ...corsHeaders, "Content-Type": "application/json" },
      });
    }
    // Two paths:
    //  • BYOK (user supplies their own key) — they pay, so no Premium gate.
    //  • Server key — a paid resource, so it's Premium-only. Verify the caller
    //    server-side; the frontend isPremium flag is not trustworthy.
    const hasBYOK = typeof userApiKey === "string" && userApiKey.trim().length > 10;
    if (!hasBYOK && !(await isPremiumCaller(req))) {
      return new Response(JSON.stringify({ error: "Premium membership required for AI analysis" }), {
        status: 403, headers: { ...corsHeaders, "Content-Type": "application/json" },
      });
    }

    const GEMINI_API_KEY = hasBYOK
      ? userApiKey.trim()
      : Deno.env.get("GEMINI_API_KEY");
    if (!GEMINI_API_KEY) {
      console.error("No Gemini key (user did not supply, env var empty)");
      return new Response(JSON.stringify({ error: "No Gemini API key available" }), {
        status: 500, headers: { ...corsHeaders, "Content-Type": "application/json" },
      });
    }

    let finalPrompt = prompt;
    if (macro === true) {
      try {
        const block = await buildMacroBlock();
        if (block) finalPrompt = `${prompt}\n\n${block}`;
      } catch (err) {
        console.error("buildMacroBlock failed", String(err)); // AI still answers without macro
      }
    }

    const geminiUrl = `https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash-lite:generateContent?key=${GEMINI_API_KEY}`;
    const response = await fetch(geminiUrl, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ contents: [{ parts: [{ text: finalPrompt }] }] }),
    });

    const data = await response.json();

    // Log full response when Gemini failed
    if (!response.ok) {
      console.error("Gemini API error", response.status, JSON.stringify(data));
      return new Response(JSON.stringify({ error: "Gemini API error", status: response.status, details: data }), {
        status: response.status, headers: { ...corsHeaders, "Content-Type": "application/json" },
      });
    }

    // Even with 200, check if response has expected structure
    if (!data.candidates || !data.candidates[0]) {
      console.error("Unexpected Gemini response", JSON.stringify(data));
      return new Response(JSON.stringify({ error: "Unexpected Gemini response", details: data }), {
        status: 502, headers: { ...corsHeaders, "Content-Type": "application/json" },
      });
    }

    return new Response(JSON.stringify(data), {
      headers: { ...corsHeaders, "Content-Type": "application/json" },
    });
  } catch (err) {
    console.error("Function error", String(err));
    return new Response(JSON.stringify({ error: String(err) }), {
      status: 500, headers: { ...corsHeaders, "Content-Type": "application/json" },
    });
  }
});

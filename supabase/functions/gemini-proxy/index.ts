import { serve } from "https://deno.land/std@0.168.0/http/server.ts";
import { createClient } from "https://esm.sh/@supabase/supabase-js@2";

const corsHeaders = {
  "Access-Control-Allow-Origin": "*",
  "Access-Control-Allow-Headers": "authorization, x-client-info, apikey, content-type",
};

// Resolves the user behind the request's Bearer token and confirms they are
// premium (or admin). Returns true on success, false otherwise.
// SUPABASE_URL / SUPABASE_SERVICE_ROLE_KEY are injected automatically — the
// service-role client bypasses RLS so the lookup is trustworthy. Note: when the
// frontend has no session it sends the public anon key as the token; getUser()
// rejects it (no `sub`), so anonymous callers correctly resolve to false.
async function isPremiumCaller(req: Request): Promise<boolean> {
  const token = (req.headers.get("Authorization") || "").replace(/^Bearer\s+/i, "").trim();
  if (!token) return false;
  const admin = createClient(
    Deno.env.get("SUPABASE_URL")!,
    Deno.env.get("SUPABASE_SERVICE_ROLE_KEY")!,
    { auth: { persistSession: false, autoRefreshToken: false } },
  );
  const { data: { user }, error } = await admin.auth.getUser(token);
  if (error || !user) return false;
  const { data: profile } = await admin
    .from("user_profiles")
    .select("is_premium, is_admin")
    .eq("id", user.id)
    .single();
  return !!(profile && (profile.is_premium || profile.is_admin));
}

serve(async (req) => {
  if (req.method === "OPTIONS") {
    return new Response("ok", { headers: corsHeaders });
  }
  try {
    const { prompt, apiKey: userApiKey } = await req.json();
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

    const geminiUrl = `https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash-lite:generateContent?key=${GEMINI_API_KEY}`;
    const response = await fetch(geminiUrl, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ contents: [{ parts: [{ text: prompt }] }] }),
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

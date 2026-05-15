import { serve } from "https://deno.land/std@0.168.0/http/server.ts";

const corsHeaders = {
  "Access-Control-Allow-Origin": "*",
  "Access-Control-Allow-Headers": "authorization, x-client-info, apikey, content-type",
};

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
    // Prefer user-provided key (BYOK); fallback to server-side env var
    const GEMINI_API_KEY = (typeof userApiKey === "string" && userApiKey.trim().length > 10)
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

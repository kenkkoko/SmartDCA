import { serve } from "https://deno.land/std@0.168.0/http/server.ts";
import { createClient } from "https://esm.sh/@supabase/supabase-js@2";

const corsHeaders = {
  "Access-Control-Allow-Origin": "*",
  "Access-Control-Allow-Headers": "authorization, x-client-info, apikey, content-type",
};

// See gemini-proxy for notes. Verifies the Bearer token belongs to a premium
// (or admin) user via a service-role lookup; anonymous/anon-key callers resolve
// to false.
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
    // Uses a paid CMC key — Premium-only. Verified server-side.
    if (!(await isPremiumCaller(req))) {
      return new Response(JSON.stringify({ error: "Premium membership required" }), {
        status: 403, headers: { ...corsHeaders, "Content-Type": "application/json" },
      });
    }

    const url = new URL(req.url);
    const slugs = url.searchParams.get("slugs") || "bitcoin,ethereum";
    const CMC_API_KEY = Deno.env.get("CMC_API_KEY");
    if (!CMC_API_KEY) {
      return new Response(JSON.stringify({ error: "Server misconfiguration" }), {
        status: 500, headers: { ...corsHeaders, "Content-Type": "application/json" },
      });
    }
    const response = await fetch(
      `https://pro-api.coinmarketcap.com/v1/cryptocurrency/quotes/latest?slug=${slugs}`,
      { headers: { "X-CMC_PRO_API_KEY": CMC_API_KEY, Accept: "application/json" } }
    );
    const data = await response.json();
    return new Response(JSON.stringify(data), {
      headers: { ...corsHeaders, "Content-Type": "application/json" },
    });
  } catch (err) {
    return new Response(JSON.stringify({ error: String(err) }), {
      status: 500, headers: { ...corsHeaders, "Content-Type": "application/json" },
    });
  }
});

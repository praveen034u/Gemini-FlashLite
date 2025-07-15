// import { serve } from "https://deno.land/std/http/server.ts";
// import { createClient } from "https://esm.sh/@supabase/supabase-js@2";

// serve(async (req) => {
//   const supabase = createClient(
//     Deno.env.get("SUPABASE_URL")!,
//     Deno.env.get("SUPABASE_SERVICE_ROLE_KEY")!
//   );

//   const { error } = await supabase
//     .from("message_history")
//     .delete()
//     .lt("timestamp", new Date(Date.now() - 1000 * 60 * 60 * 24 * 30)); // 30 days

//   if (error) {
//     return new Response("Cleanup failed: " + error.message, { status: 500 });
//   }

//   return new Response("Old sessions cleaned up.", { status: 200 });
// });

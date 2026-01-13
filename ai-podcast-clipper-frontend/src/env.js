import { createEnv } from "@t3-oss/env-nextjs";
import { z } from "zod";

export const env = createEnv({
  /**
   * Specify your server-side environment variables schema here. This way you can ensure the app
   * isn't built with invalid env vars.
   */
  server: {
    AUTH_SECRET:
      process.env.NODE_ENV === "production"
        ? z.string()
        : z.string().optional(),
    DATABASE_URL: z.string().url(),
    NODE_ENV: z
      .enum(["development", "test", "production"])
      .default("development"),
    OSS_ACCESS_KEY_ID: z.string(),
    OSS_ACCESS_KEY_SECRET: z.string(),
    OSS_REGION: z.string(),
    OSS_ENDPOINT: z.string(),
    OSS_BUCKET: z.string(),
    PROCESS_VIDEO_ENDPOINT: z.string(),
    PROCESS_VIDEO_ENDPOINT_AUTH: z.string(),
    STRIPE_SECRET_KEY: z.string(),
    STRIPE_SMALL_CREDIT_PACK: z.string(),
    STRIPE_MEDIUM_CREDIT_PACK: z.string(),
    STRIPE_LARGE_CREDIT_PACK: z.string(),
    BASE_URL: z.string(),
    STRIPE_WEBHOOK_SECRET: z.string(),
  },

  /**
   * Specify your client-side environment variables schema here. This way you can ensure the app
   * isn't built with invalid env vars. To expose them to the client, prefix them with
   * `NEXT_PUBLIC_`.
   */
  client: {
    // NEXT_PUBLIC_CLIENTVAR: z.string(),
    NEXT_PUBLIC_STRIPE_PUBLISHABLE_KEY: z.string(),
  },

  /**
   * You can't destruct `process.env` as a regular object in the Next.js edge runtimes (e.g.
   * middlewares) or client-side so we need to destruct manually.
   */
  runtimeEnv: {
    AUTH_SECRET: process.env.AUTH_SECRET,
    DATABASE_URL: process.env.DATABASE_URL,
    NODE_ENV: process.env.NODE_ENV,
    OSS_ACCESS_KEY_ID: process.env.OSS_ACCESS_KEY_ID,
    OSS_ACCESS_KEY_SECRET: process.env.OSS_ACCESS_KEY_SECRET,
    OSS_REGION: process.env.OSS_REGION,
    OSS_ENDPOINT: process.env.OSS_ENDPOINT,
    OSS_BUCKET: process.env.OSS_BUCKET,
    PROCESS_VIDEO_ENDPOINT: process.env.PROCESS_VIDEO_ENDPOINT,
    PROCESS_VIDEO_ENDPOINT_AUTH: process.env.PROCESS_VIDEO_ENDPOINT_AUTH,
    NEXT_PUBLIC_STRIPE_PUBLISHABLE_KEY:
      process.env.NEXT_PUBLIC_STRIPE_PUBLISHABLE_KEY,
    STRIPE_SECRET_KEY: process.env.STRIPE_SECRET_KEY,
    STRIPE_SMALL_CREDIT_PACK: process.env.STRIPE_SMALL_CREDIT_PACK,
    STRIPE_MEDIUM_CREDIT_PACK: process.env.STRIPE_MEDIUM_CREDIT_PACK,
    STRIPE_LARGE_CREDIT_PACK: process.env.STRIPE_LARGE_CREDIT_PACK,
    BASE_URL: process.env.BASE_URL,
    STRIPE_WEBHOOK_SECRET: process.env.STRIPE_WEBHOOK_SECRET,
  },
  /**
   * Run `build` or `dev` with `SKIP_ENV_VALIDATION` to skip env validation. This is especially
   * useful for Docker builds.
   */
  skipValidation: !!process.env.SKIP_ENV_VALIDATION,
  /**
   * Makes it so that empty strings are treated as undefined. `SOME_VAR: z.string()` and
   * `SOME_VAR=''` will throw an error.
   */
  emptyStringAsUndefined: true,
});

import { defineConfig } from "vitest/config";

export default defineConfig({
  test: {
    include: ["src/chart/**/*.test.ts", "src/chart/**/*.test.tsx"],
    environment: "jsdom",
    globals: true,
  },
});

/// <reference types="vitest/config" />
import tailwindcss from '@tailwindcss/vite';
import react from '@vitejs/plugin-react';
import { defineConfig } from 'vite';

// Dev: `vite` serves the UI on :5173 and forwards /api/* to the FastAPI app (same origin, so no
// CORS; changeOrigin rewrites the Host header so TrustedHostMiddleware accepts proxied calls).
// Prod: `vite build` writes web/dist, which `mimoe-agent serve` mounts at "/".
export default defineConfig({
  plugins: [react(), tailwindcss()],
  server: {
    proxy: { '/api': { target: process.env.API_URL ?? 'http://127.0.0.1:8000', changeOrigin: true } },
  },
  // One bundle (about 190 kB gzipped) served from localhost: splitting would only add files to web/dist.
  build: { outDir: 'dist', emptyOutDir: true, chunkSizeWarningLimit: 800 },
  test: { environment: 'node', include: ['src/**/*.test.ts'] },
});

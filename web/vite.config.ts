/// <reference types="vitest/config" />
import react from '@vitejs/plugin-react';
import { defineConfig } from 'vite';

// Dev: `vite` serves the UI on :5173 and forwards /api/* to the FastAPI app (same origin, so no
// CORS; changeOrigin rewrites the Host header so TrustedHostMiddleware accepts proxied calls).
// Prod: `vite build` writes web/dist, which `mimoe-agent serve` mounts at "/".
export default defineConfig({
  plugins: [react()],
  server: {
    proxy: { '/api': { target: process.env.API_URL ?? 'http://127.0.0.1:8000', changeOrigin: true } },
  },
  build: { outDir: 'dist', emptyOutDir: true },
  test: { environment: 'node', include: ['src/**/*.test.ts'] },
});

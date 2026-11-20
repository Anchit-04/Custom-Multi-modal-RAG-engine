import { defineConfig } from 'vitest/config';
import react from '@vitejs/plugin-react';

// Cross-origin isolation enables SharedArrayBuffer, which onnxruntime-web needs
// for multithreaded WASM (the CPU fallback). "credentialless" rather than
// "require-corp" so model files can still load from the Hugging Face CDN.
const isolation = {
  'Cross-Origin-Opener-Policy': 'same-origin',
  'Cross-Origin-Embedder-Policy': 'credentialless',
};

export default defineConfig({
  plugins: [react()],
  server: { headers: isolation },
  preview: { headers: isolation },
  worker: { format: 'es' },
  test: {
    include: ['src/**/*.test.ts', 'tests/**/*.test.ts'],
    environment: 'node',
  },
});

import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';

export default defineConfig({
  plugins: [react()],
  // 自訂網域(dca.hellokai07.com)掛在根路徑
  base: '/',
  build: {
    outDir: 'dist',
    // app.jsx 是整站單一模組(~430KB source),bundle 大是預期的
    chunkSizeWarningLimit: 1500,
  },
});

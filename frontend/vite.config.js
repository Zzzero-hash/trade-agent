import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// https://vitejs.dev/config/
export default defineConfig({
  base: './',
  plugins: [react()],
  server: {
    port: 5173,
    host: '0.0.0.0', // Bind to all interfaces instead of localhost only
    open: true
  },
  build: {
    outDir: 'dist',
    sourcemap: true
  }
})

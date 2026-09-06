import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// Vite config — two things worth noting:
// 1. proxy: forwards /api calls to FastAPI during development
//    Without this, the browser would get CORS errors because React runs on :5173
//    and FastAPI on :8000. The proxy makes it look like they're the same origin.
// 2. In production (Vercel), you set the real API URL in an env variable instead.
export default defineConfig({
  plugins: [react()],
  server: {
    proxy: {
      '/api': {
        target: 'http://localhost:8000',
        changeOrigin: true,
      },
    },
  },
})

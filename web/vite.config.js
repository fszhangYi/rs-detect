import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';
import path from 'node:path';

/** Dev proxy to FastAPI (uvicorn). Production: build → dist, served by FastAPI. */
export default defineConfig({
  plugins: [
    react(),
    {
      name: 'async-entry-module',
      transformIndexHtml(html) {
        // type=module already defers; async lets the browser fetch/parse without
        // blocking HTML completion — splash paints first.
        return html.replace(
          /<script type="module"( crossorigin)? src="(\/static\/[^"]+\.js)"><\/script>/g,
          '<script type="module"$1 async src="$2"></script>',
        );
      },
    },
  ],
  resolve: {
    alias: {
      '@': path.resolve(__dirname, 'src'),
    },
  },
  server: {
    host: '0.0.0.0',
    port: 5174,
    strictPort: true,
    proxy: {
      '/api': {
        target: 'http://127.0.0.1:8787',
        changeOrigin: true,
      },
      '/assets': {
        target: 'http://127.0.0.1:8787',
        changeOrigin: true,
      },
    },
  },
  preview: {
    host: '0.0.0.0',
    port: 5174,
    strictPort: true,
  },
  build: {
    outDir: 'dist',
    emptyOutDir: true,
    // Avoid clashing with FastAPI mount `/assets` (URDF/STL models under web/assets)
    assetsDir: 'static',
    cssCodeSplit: true,
    modulePreload: {
      // Only preload polyfills / direct entry deps — not lazy three/antd panel chunks
      resolveDependencies(filename, deps) {
        return deps.filter((d) => !/three|urdf|ArmPanel|DetectPanel|TrainPanel|ExtrinsicsPanel|GraspPanel/i.test(d));
      },
    },
    rollupOptions: {
      output: {
        manualChunks(id) {
          if (!id.includes('node_modules')) return undefined;
          if (id.includes('three') || id.includes('urdf-loader')) return 'three';
          if (id.includes('antd') || id.includes('@ant-design') || id.includes('rc-')) return 'antd';
          if (id.includes('react-dom') || id.includes('/react/') || id.includes('\\react\\') || id.includes('scheduler')) {
            return 'react';
          }
          return 'vendor';
        },
      },
    },
    chunkSizeWarningLimit: 900,
  },
  // Served under FastAPI at `/` (not Electron file://)
  base: '/',
});

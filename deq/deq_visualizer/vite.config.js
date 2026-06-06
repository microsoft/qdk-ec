import { defineConfig } from 'vite'
import { fileURLToPath, URL } from 'node:url'
import { resolve } from 'path'
import anywidget from '@anywidget/vite'
import vue from '@vitejs/plugin-vue'
import cssInjectedByJsPlugin from 'vite-plugin-css-injected-by-js'
import { compress_js } from './compress_js'

const outDir = '../deq/visual/static'

export default defineConfig({
  plugins: [vue(), cssInjectedByJsPlugin(), anywidget(), compress_js({ folder: outDir, js_filename: 'lib.js', gzip_filename: 'lib.js.b64' })],
  resolve: {
    alias: {
      '@': fileURLToPath(new URL('./src', import.meta.url)),
    },
  },
  build: {
    chunkSizeWarningLimit: 2000, // 2MB chunk limit to remove warning (we will use compression)
    outDir,
    rollupOptions: {
      input: {
        lib: resolve(__dirname, 'src/lib.ts'),
      },
      preserveEntrySignatures: 'strict',
      output: {
        entryFileNames: (chunkInfo) => (chunkInfo.name === 'main' ? 'index.js' : 'lib.js'),
        format: 'es',
        chunkFileNames: `[name].js`,
        assetFileNames: `[name].[ext]`,
        inlineDynamicImports: true, // to make sure the output is a single file
        // disable chunks to ensure a single js file
        manualChunks: undefined,
      },
    },
    minify: true,
    cssCodeSplit: false,
  },
})

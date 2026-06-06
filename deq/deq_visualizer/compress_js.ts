import path from 'path'
import fs from 'fs'
import { PluginOption } from 'vite'
import { compress_content, decompress_content, assert_buffer_equal } from './src/compress'

export interface PluginConfig {
  folder: string
  js_filename: string
  gzip_filename: string
}

async function do_compress(folder: string, js_filename: string, gzip_filename: string) {
  const js_filepath = path.join(folder, js_filename)
  const js_content = fs.readFileSync(js_filepath)
  const base64_string = await compress_content(js_content)
  // now test decompress (to be used in browser)
  const decompressed = await decompress_content(base64_string)
  assert_buffer_equal(js_content, decompressed)
  // write to file
  const zip_filepath = path.join(folder, gzip_filename)
  if (fs.existsSync(zip_filepath)) {
    fs.unlinkSync(zip_filepath)
  }
  fs.writeFileSync(zip_filepath, base64_string)
}

export function compress_js(user_config: PluginConfig): PluginOption {
  const { folder, js_filename, gzip_filename } = user_config
  return {
    name: 'vite-plugin-zip-file',
    apply: 'build',
    async closeBundle() {
      await do_compress(folder, js_filename, gzip_filename)
    },
  }
}

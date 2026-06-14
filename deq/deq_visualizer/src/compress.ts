export function uint8_to_array_buffer(array: Uint8Array): ArrayBuffer {
  return array.buffer.slice(array.byteOffset, array.byteLength + array.byteOffset) as ArrayBuffer
}

export function array_buffer_to_base64(buffer: ArrayBuffer): string {
  return btoa(new Uint8Array(buffer).reduce((data, byte) => data + String.fromCharCode(byte), ''))
}

export function base64_to_array_buffer(base64_str: string): ArrayBuffer {
  return uint8_to_array_buffer(Uint8Array.from(atob(base64_str), (c) => c.charCodeAt(0)))
}

export async function compress_content(buffer: ArrayBuffer): Promise<string> {
  const blob = new Blob([buffer])
  const stream = blob.stream().pipeThrough(new CompressionStream('gzip'))
  const compressed = await new Response(stream).arrayBuffer()
  const base64_string = array_buffer_to_base64(compressed)
  return base64_string
}

export async function decompress_content(base64_str: string): Promise<ArrayBuffer> {
  const base64_binary = base64_to_array_buffer(base64_str)
  const blob = new Blob([base64_binary])
  const decompressed_stream = blob.stream().pipeThrough(new DecompressionStream('gzip'))
  const decompressed = await new Response(decompressed_stream).arrayBuffer()
  return decompressed
}

export function assert_buffer_equal(buf1: ArrayBuffer, buf2: ArrayBuffer) {
  const error = new Error('decompressed buffer not equal to the original buffer')
  if (buf1.byteLength != buf2.byteLength) {
    throw error
  }
  const dv1 = new Int8Array(buf1)
  const dv2 = new Int8Array(buf2)
  for (let i = 0; i != buf1.byteLength; i++) {
    if (dv1[i] != dv2[i]) {
      throw error
    }
  }
}

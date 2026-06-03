#!/usr/bin/env node
/**
 * Parse scratchblocks pseudocode using the storyboard VM parser.
 *
 * Usage:
 *   node vm_pseudocode_parser.mjs < input.txt
 *
 * Outputs JSON:
 *   { "valid": true/false, "error": "...", "scripts": <parsed or null> }
 */

import path from "path"
import { fileURLToPath } from "url"
import { createRequire } from "module"

const __dirname = path.dirname(fileURLToPath(import.meta.url))
const require = createRequire(import.meta.url)

const parserPath = path.resolve(__dirname, "../../scratch-storyboard/scratch-storyboard-vm/src/engine/pseudocode-parser.js")
let parsePseudoCode
try {
  parsePseudoCode = require(parserPath)
} catch (err) {
  console.error(JSON.stringify({ valid: false, error: String(err), scripts: null }))
  process.exit(1)
}

async function readStdin() {
  return await new Promise(resolve => {
    let data = ""
    process.stdin.setEncoding("utf8")
    process.stdin.on("data", chunk => (data += chunk))
    process.stdin.on("end", () => resolve(data))
  })
}

async function main() {
  const input = (await readStdin()).trim()
  if (!input) {
    console.log(JSON.stringify({ valid: false, error: "empty input", scripts: null }))
    process.exit(1)
  }
  try {
    const originalLog = console.log
    console.log = () => {}
    const scripts = parsePseudoCode(input, [], [], [], [])
    console.log = originalLog
    const valid = Array.isArray(scripts) && scripts.length > 0
    console.log(JSON.stringify({ valid, error: null, scripts }))
    process.exit(valid ? 0 : 1)
  } catch (err) {
    console.log(JSON.stringify({ valid: false, error: String(err), scripts: null }))
    process.exit(1)
  }
}

main()

import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import { createRequire } from "node:module";
import { after, test } from "node:test";
import oniguruma from "vscode-oniguruma";
import textmate from "vscode-textmate";

const require = createRequire(import.meta.url);
const wasm = await readFile(require.resolve("vscode-oniguruma/release/onig.wasm"));
await oniguruma.loadWASM(wasm.buffer.slice(wasm.byteOffset, wasm.byteOffset + wasm.byteLength));
const source = await readFile(new URL("../syntaxes/deq.tmLanguage.json", import.meta.url), "utf8");
const registry = new textmate.Registry({
  onigLib: Promise.resolve({
    createOnigScanner: (patterns) => new oniguruma.OnigScanner(patterns),
    createOnigString: (value) => new oniguruma.OnigString(value),
  }),
  loadGrammar: async (scope) => scope === "source.deq"
    ? textmate.parseRawGrammar(source, "deq.tmLanguage.json")
    : null,
});
const grammar = await registry.loadGrammar("source.deq");
assert.ok(grammar);
const rootStack = grammar.tokenizeLine("", textmate.INITIAL).ruleStack;
after(() => registry.dispose());

function tokenize(source) {
  let stack = rootStack;
  const tokens = [];
  for (const line of source.split("\n")) {
    const result = grammar.tokenizeLine(line, stack);
    tokens.push(...result.tokens.map((token) => ({
      text: line.slice(token.startIndex, token.endIndex),
      scopes: token.scopes,
    })));
    stack = result.ruleStack;
  }
  assert.ok(stack.equals(rootStack), "unterminated syntax context");
  return tokens;
}

function scoped(tokens, text, scope) {
  const matches = tokens.filter((token) => token.text === text);
  assert.ok(matches.length > 0, `missing token ${JSON.stringify(text)}`);
  assert.ok(matches.every((token) => token.scopes.includes(scope)),
    `${JSON.stringify(text)} must have scope ${scope}: ${JSON.stringify(matches)}`);
}

test("nested multiline metadata has keys, values, and scientific numbers", () => {
  const tokens = tokenize(`@METADATA("mytool", {
  "enabled": true,
  "nothing": null,
  "items": [
    1e-3, -.5, .25, 2., 1E+6,
    {"name": "ancilla", "flag": false}
  ]
})
GADGET Read {
  M 0
}`);
  for (const key of ["enabled", "nothing", "items", "name", "flag"]) {
    scoped(tokens, `"${key}"`, "support.type.property-name.deq");
  }
  for (const value of ["mytool", "ancilla"]) {
    scoped(tokens, `"${value}"`, "string.quoted.double.deq");
  }
  for (const number of ["1e-3", "-.5", ".25", "2.", "1E+6"]) {
    scoped(tokens, number, "constant.numeric.deq");
  }
  for (const value of ["true", "false"]) {
    scoped(tokens, value, "constant.language.boolean.deq");
  }
  scoped(tokens, "null", "constant.language.null.deq");
  scoped(tokens, ":", "punctuation.separator.key-value.deq");
  scoped(tokens, "GADGET", "keyword.control.deq");
  assert.ok(tokens.filter((token) => token.text === "GADGET")
    .every((token) => !token.scopes.includes("meta.annotation.deq")));
});

test("quoted delimiters and escaped quotes do not end metadata early", () => {
  const value = String.raw`") } ] # true null @START \"quoted\""`;
  const tokens = tokenize(`@METADATA("tool", {"message": ${value}, "ok": true})
PROGRAM Main {
  Prepare 0
}`);
  scoped(tokens, value, "string.quoted.double.deq");
  scoped(tokens, '"ok"', "support.type.property-name.deq");
  scoped(tokens, "true", "constant.language.boolean.deq");
  scoped(tokens, "PROGRAM", "keyword.control.deq");
  assert.ok(!tokens.some((token) => token.scopes.includes("comment.line.number-sign.deq")));
});

test("comments preserve multiline argument and array contexts", () => {
  const tokens = tokenize(`@METADATA( # annotation
  "tool",
  [true, # array
   null]
)
GADGET G {
  R 0
}`);
  scoped(tokens, "# annotation", "comment.line.number-sign.deq");
  scoped(tokens, "# array", "comment.line.number-sign.deq");
  scoped(tokens, '"tool"', "string.quoted.double.deq");
  scoped(tokens, "true", "constant.language.boolean.deq");
  scoped(tokens, "null", "constant.language.null.deq");
});

test("object keys remain distinct from values across lines and comments", () => {
  const tokens = tokenize(`@METADATA("tool", {
  "key" # key description
    :
    "value",
  "nested": {"empty": [], "object": {}}
})`);
  for (const key of ["key", "nested", "empty", "object"]) {
    scoped(tokens, `"${key}"`, "support.type.property-name.deq");
  }
  scoped(tokens, '"value"', "string.quoted.double.deq");
  scoped(tokens, "# key description", "comment.line.number-sign.deq");
});

test("metadata works inside CODE, GADGET, COMPOSE, and PROGRAM blocks", () => {
  for (const [header, statement] of [
    ["CODE C [[1,1]]", "LOGICAL X0 Z0"],
    ["GADGET G", "OUTPUT C 0"],
    ["COMPOSE Wrapped", "INPUT C 0"],
    ["PROGRAM Main", "Prepare 0"],
  ]) {
    const tokens = tokenize(`${header} {
  @METADATA("record", {"tags": ["X", "Z"]})
  ${statement}
}`);
    scoped(tokens, "@METADATA", "entity.name.function.decorator.deq");
    scoped(tokens, '"tags"', "support.type.property-name.deq");
    scoped(tokens, '"X"', "string.quoted.double.deq");
  }
});

test("bare decorators and existing numeric keyword arguments still work", () => {
  const tokens = tokenize(`@PRIVATE GADGET Hidden {
  R 0
}
@CHECKS("manual", verify=0)
GADGET Read {
  @DECODE_ONLY
  M 0
}
PROGRAM Main {
  @START(at=1.2e-3, duration=.07)
  Read 0
}`);
  scoped(tokens, "@PRIVATE", "entity.name.function.decorator.deq");
  scoped(tokens, "GADGET", "keyword.control.deq");
  scoped(tokens, "verify", "variable.parameter.deq");
  scoped(tokens, "at", "variable.parameter.deq");
  scoped(tokens, "duration", "variable.parameter.deq");
  scoped(tokens, "1.2e-3", "constant.numeric.deq");
  scoped(tokens, ".07", "constant.numeric.deq");
});

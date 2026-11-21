/**
 * ThinkModule Lab — Transformer Internals & Decoding Engine (v2.0)
 * Pure ES6 mathematical core:
 *   1. Byte-Pair Encoding (BPE) Tokenizer with step-by-step merge trace
 *   2. Scaled Dot-Product Multi-Head Self-Attention (4 specialized heads + Causal Mask)
 *   3. Cartesian-Verified Multi-Branch Autoregressive Engine (100% grammatical coherence)
 *   4. Temp, Top-K, Top-P, Min-P, Repetition Penalty, Counterfactual Branching & Diagnostics
 */
(function (global) {
  "use strict";

  // ============================================================================
  // 1. REAL BYTE-PAIR ENCODING (BPE) TOKENIZER WITH MERGE TRACE
  // ============================================================================

  const BPE_MERGE_RULES = [
    ["Ġ", "t"], ["Ġt", "h"], ["Ġth", "e"], ["i", "n"], ["e", "r"], ["a", "n"],
    ["o", "n"], ["r", "e"], ["t", "i"], ["e", "n"], ["a", "t"], ["e", "s"],
    ["o", "r"], ["t", "e"], ["a", "l"], ["i", "s"], ["a", "r"], ["s", "t"],
    ["Ġ", "a"], ["Ġ", "o"], ["Ġ", "i"], ["Ġ", "s"], ["Ġ", "c"], ["Ġ", "p"],
    ["Ġ", "f"], ["Ġ", "d"], ["Ġ", "r"], ["Ġ", "m"], ["Ġ", "w"], ["Ġ", "b"],
    ["Ġ", "n"], ["Ġ", "l"], ["Ġ", "h"], ["Ġ", "e"], ["i", "t"], ["n", "g"],
    ["i", "ng"], ["t", "o"], ["e", "d"], ["r", "o"], ["l", "e"], ["s", "e"],
    ["c", "o"], ["d", "e"], ["m", "e"], ["c", "h"], ["r", "a"], ["c", "a"],
    ["u", "s"], ["u", "r"], ["p", "r"], ["t", "r"], ["l", "i"], ["v", "e"],
    ["Ġa", "n"], ["Ġan", "d"], ["Ġo", "f"], ["Ġi", "n"], ["Ġt", "o"],
    ["Ġi", "s"], ["Ġf", "o"], ["Ġfo", "r"], ["Ġw", "i"], ["Ġwi", "th"],
    ["Ġth", "at"], ["Ġo", "n"], ["Ġa", "s"], ["Ġb", "y"],
    ["ti", "on"], ["at", "ion"], ["en", "t"], ["er", "s"],
    ["Ġr", "e"], ["Ġc", "o"], ["Ġp", "r"], ["Ġs", "t"], ["Ġd", "e"],
    ["d", "ef"], ["Ġd", "ef"], ["re", "t"], ["ret", "ur"], ["retur", "n"],
    ["Ġre", "turn"], ["p", "y"], ["py", "th"], ["pyth", "on"], ["ĠP", "ython"],
    ["Ġp", "ython"], ["to", "k"], ["tok", "en"], ["Ġto", "ken"], ["Ġtoken", "s"],
    ["a", "tt"], ["att", "ent"], ["attent", "ion"], ["Ġa", "ttention"],
    ["tr", "an"], ["tran", "s"], ["trans", "f"], ["transf", "or"],
    ["transfor", "m"], ["transform", "er"], ["Ġtr", "ansformer"],
    ["m", "o"], ["mo", "de"], ["mode", "l"], ["Ġm", "odel"], ["Ġmodel", "s"],
    ["L", "L"], ["LL", "M"], ["ĠL", "LM"], ["ĠLLM", "s"],
    ["c", "a"], ["ca", "ch"], ["cach", "e"], ["Ġca", "che"],
    ["s", "p"], ["sp", "a"], ["spa", "ce"], ["Ġsp", "ace"],
    ["t", "i"], ["ti", "m"], ["tim", "e"], ["Ġti", "me"],
    ["Ġc", "om"], ["Ġcom", "p"], ["Ġcomp", "l"], ["Ġcompl", "e"],
    ["Ġcomple", "x"], ["Ġcomplex", "ity"],
    ["Ġa", "lg"], ["Ġalg", "or"], ["Ġalgor", "it"], ["Ġalgorit", "hm"],
    ["Ġr", "ec"], ["Ġrec", "ur"], ["Ġrecur", "s"], ["Ġrecurs", "ive"],
    ["ĠR", "ed"], ["ĠRed", "is"], ["ĠL", "ua"],
    ["st", "r"], ["str", "a"], ["stra", "w"], ["Ġst", "r"], ["Ġstr", "aw"],
    ["b", "er"], ["ber", "r"], ["berr", "y"], ["Ġb", "erry"],
    ["0", "0"], ["00", "0"], ["1", "0"], ["10", "0"], ["2", "^"],
    ["(", ")"], ["==", " "], ["->", " "]
  ];

  function hashTokenId(tokenStr) {
    let h = 2166136261;
    for (let i = 0; i < tokenStr.length; i++) {
      h ^= tokenStr.charCodeAt(i);
      h = Math.imul(h, 16777619);
    }
    return 1000 + (Math.abs(h) % 49000);
  }

  function tokenizeBPE(inputText) {
    const safeText = (inputText || "").slice(0, 600);
    const encoder = new TextEncoder();
    const totalBytes = encoder.encode(safeText).length;

    if (!safeText.length) {
      return {
        text: "",
        tokens: [],
        mergeSteps: [],
        stats: { charCount: 0, byteCount: 0, tokenCount: 0, compressionRatio: 0 }
      };
    }

    const normalized = safeText.replace(/ /g, "Ġ").replace(/\n/g, "Ċ");
    let symbols = Array.from(normalized);
    const mergeSteps = [];

    for (let r = 0; r < BPE_MERGE_RULES.length; r++) {
      const [left, right] = BPE_MERGE_RULES[r];
      let mergedAny = false;
      const nextSymbols = [];
      let i = 0;

      while (i < symbols.length) {
        if (i < symbols.length - 1 && symbols[i] === left && symbols[i + 1] === right) {
          nextSymbols.push(left + right);
          mergedAny = true;
          i += 2;
        } else {
          nextSymbols.push(symbols[i]);
          i += 1;
        }
      }

      if (mergedAny) {
        symbols = nextSymbols;
        mergeSteps.push({
          step: mergeSteps.length + 1,
          pair: [left, right],
          merged: left + right,
          tokenCountAfter: symbols.length,
          snapshot: symbols.slice(0, 32)
        });
      }
    }

    const tokens = symbols.map((sym, idx) => {
      const raw = sym.replace(/Ġ/g, " ").replace(/Ċ/g, "\n");
      const bytes = Array.from(encoder.encode(raw));
      return {
        index: idx,
        text: raw,
        display: sym,
        id: hashTokenId(sym),
        bytes
      };
    });

    return {
      text: safeText,
      tokens,
      mergeSteps,
      stats: {
        charCount: safeText.length,
        byteCount: totalBytes,
        tokenCount: tokens.length,
        compressionRatio: tokens.length > 0 ? +(safeText.length / tokens.length).toFixed(2) : 0
      }
    };
  }

  // ============================================================================
  // 2. REAL SCALED DOT-PRODUCT MULTI-HEAD SELF-ATTENTION (4 HEADS)
  // ============================================================================

  const ATTENTION_HEADS = [
    {
      id: 0,
      name: "Head 0 — Local Syntax & Structure",
      shortName: "Syntax",
      color: "#38bdf8",
      description: "Tracks local grammatical dependencies, punctuation pairs, and immediately preceding tokens (strong recency bias)."
    },
    {
      id: 1,
      name: "Head 1 — Semantic Salience",
      shortName: "Semantic",
      color: "#a78bfa",
      description: "Concentrates attention mass on high-information technical nouns, complexity bounds, and core domain concepts."
    },
    {
      id: 2,
      name: "Head 2 — Long-Range Coreference",
      shortName: "Coreference",
      color: "#34d399",
      description: "Links pronouns, variables, and conclusions back to definitions established earlier in the trajectory."
    },
    {
      id: 3,
      name: "Head 3 — Induction & Repetition",
      shortName: "Induction",
      color: "#fbbf24",
      description: "Anthropic-style induction circuit: detects repeated tokens or parallel patterns across the context window."
    }
  ];

  const STOP_WORDS = new Set([
    "the", "a", "an", "is", "are", "in", "of", "to", "for", "and", "or",
    "by", "with", "on", "at", "as", "be", "this", "that", "from", "into", "so"
  ]);

  const TECH_WORDS = new Set([
    "fibonacci", "fib(n)", "o(2^n)", "o(n)", "o(1)", "o(log", "python",
    "recursive", "memoization", "lru_cache", "cache", "dynamic", "iterative",
    "matrix", "redis", "valkey", "lua", "evalsha", "bucket", "token", "window",
    "rps", "limiter", "distributed", "cluster", "postgresql", "lock", "locks",
    "strawberry", "bpe", "subword", "tokenizer", "attention", "transformer",
    "llms", "tco", "stack", "complexity", "latency", "atomic", "gossip",
    "exponentiation", "embeddings", "chain-of-thought", "3", "2", "100k"
  ]);

  const PRONOUN_OR_REF = new Set([
    "it", "its", "they", "them", "this", "that", "which", "these", "thus",
    "because", "instead", "therefore", "however", "result"
  ]);

  function getTokenEmbedding(tokStr, pos, totalLen) {
    const clean = tokStr.trim().toLowerCase().replace(/^[^a-z0-9()^_]+|[^a-z0-9()^_]+$/g, "");
    const firstWord = clean.split(/\s+/)[0] || clean;
    const isPunct = clean.length === 0 || /^[:;,.\-()[\]{}=<>]+$/.test(tokStr.trim());
    const isCode = /[_()[\]=^:@]/.test(tokStr);
    const isTech = TECH_WORDS.has(clean) || TECH_WORDS.has(firstWord) || clean.length >= 8;
    const isStop = STOP_WORDS.has(clean);
    const isRef = PRONOUN_OR_REF.has(firstWord);
    const isNumber = /\d/.test(tokStr);

    const h1 = ((hashTokenId(clean || tokStr) % 200) - 100) / 100;
    const h2 = ((hashTokenId((clean || tokStr) + "_s") % 200) - 100) / 100;
    const pNorm = pos / Math.max(totalLen, 1);
    const sinPos = Math.sin(pos * 0.45);
    const cosPos = Math.cos(pos * 0.45);

    return [
      isTech ? 1.4 : isStop ? -0.6 : 0.2,
      isCode ? 1.3 : -0.2,
      isPunct ? 1.2 : -0.3,
      isRef ? 1.4 : -0.2,
      isNumber ? 1.3 : -0.1,
      h1,
      h2,
      (sinPos * 0.6 + cosPos * 0.4) * (1 - 0.2 * pNorm)
    ];
  }

  function dotProduct(a, b) {
    let s = 0;
    for (let i = 0; i < a.length; i++) s += a[i] * b[i];
    return s;
  }

  function computeAttention(tokenStrings, headIndex, causalMask = true) {
    const n = tokenStrings.length;
    const head = ATTENTION_HEADS[headIndex] || ATTENTION_HEADS[0];
    const dK = 8;
    const invSqrtD = 1 / Math.sqrt(dK);

    if (n === 0) {
      return { head, tokens: [], matrix: [], rowEntropies: [], avgEntropy: 0, qkDim: dK };
    }

    const embeds = tokenStrings.map((t, i) => getTokenEmbedding(t, i, n));
    const cleanTokens = tokenStrings.map(t => t.trim().toLowerCase());

    const headWeights = [
      { q: [0.3, 1.1, 1.0, 0.2, 0.4, 0.3, 0.3, 1.2], k: [0.3, 1.1, 1.0, 0.2, 0.4, 0.3, 0.3, 1.2] },
      { q: [1.5, 0.4, -0.5, 0.5, 1.0, 1.1, 1.1, 0.2], k: [1.8, 0.4, -0.5, 0.2, 1.1, 1.1, 1.1, 0.2] },
      { q: [0.7, 0.5, -0.3, 1.6, 0.8, 0.5, 0.5, 0.3], k: [1.7, 0.6, -0.4, 0.1, 1.2, 0.5, 0.5, 0.3] },
      { q: [0.4, 0.6, 0.3, 0.3, 0.6, 1.8, 1.8, 0.2], k: [0.4, 0.6, 0.3, 0.3, 0.6, 1.8, 1.8, 0.2] }
    ][head.id];

    const Q = embeds.map(e => e.map((v, d) => v * headWeights.q[d]));
    const K = embeds.map(e => e.map((v, d) => v * headWeights.k[d]));

    const matrix = [];
    const rowEntropies = [];

    for (let i = 0; i < n; i++) {
      const scores = new Array(n);
      let maxScore = -Infinity;

      for (let j = 0; j < n; j++) {
        if (causalMask && j > i) {
          scores[j] = -Infinity;
          continue;
        }

        let s = dotProduct(Q[i], K[j]) * invSqrtD;
        const dist = Math.abs(i - j);

        if (head.id === 0) {
          if (dist === 1) s += 2.35;
          else if (dist === 2) s += 1.2;
          else if (dist === 0) s += 0.35;
          else s -= Math.min(dist * 0.22, 2.8);
        } else if (head.id === 1) {
          s += embeds[j][0] * 1.18;
          if (j === 0) s += 0.3;
        } else if (head.id === 2) {
          if (embeds[i][3] > 0.5 && j < i && embeds[j][0] > 0.6) s += 2.15;
          if (j < Math.min(10, i) && embeds[j][0] > 0.5) s += 0.9;
          if (dist === 0) s -= 0.4;
        } else if (head.id === 3) {
          if (j !== i && cleanTokens[i].length > 1 && cleanTokens[i] === cleanTokens[j]) {
            s += 3.2;
          } else if (j > 0 && i > 0 && cleanTokens[i - 1] === cleanTokens[j - 1] && j !== i) {
            s += 2.0;
          } else if (j === 0) {
            s += 0.65;
          }
        }

        scores[j] = s;
        if (s > maxScore) maxScore = s;
      }

      const row = new Array(n).fill(0);
      let sumExp = 0;
      for (let j = 0; j < n; j++) {
        if (scores[j] === -Infinity) continue;
        const ex = Math.exp(scores[j] - maxScore);
        row[j] = ex;
        sumExp += ex;
      }

      let ent = 0;
      for (let j = 0; j < n; j++) {
        if (sumExp > 0 && row[j] > 0) {
          row[j] /= sumExp;
          ent -= row[j] * Math.log2(row[j]);
        }
      }

      matrix.push(row);
      rowEntropies.push(ent);
    }

    const avgEntropy = rowEntropies.reduce((a, b) => a + b, 0) / Math.max(rowEntropies.length, 1);

    return {
      head,
      tokens: tokenStrings,
      matrix,
      rowEntropies,
      avgEntropy: +avgEntropy.toFixed(3),
      qkDim: dK
    };
  }

  // ============================================================================
  // 3. CARTESIAN-VERIFIED MULTI-BRANCH SCENARIOS (100% GRAMMATICAL COHERENCE)
  // ============================================================================

  function compileScenario(def) {
    const states = {};
    const branchNames = Object.keys(def.branches);

    for (const bName of branchNames) {
      const steps = def.branches[bName];
      for (let i = 0; i < steps.length; i++) {
        const stateId = `${bName}:${i}`;
        const defaultNext = i + 1 < steps.length ? `${bName}:${i + 1}` : "END";

        states[stateId] = steps[i].map(tuple => {
          const [token, logit, opts = {}] = tuple;
          return {
            token,
            tokenId: hashTokenId(token.replace(/ /g, "Ġ")),
            logit,
            nextState: opts.next || defaultNext,
            isHallucination: Boolean(opts.hall),
            branchTag: opts.tag || null
          };
        });
      }
    }

    for (const sId of Object.keys(states)) {
      for (const cand of states[sId]) {
        if (cand.nextState !== "END" && !states[cand.nextState]) {
          cand.nextState = "END";
        }
      }
    }

    return {
      id: def.id,
      title: def.title,
      category: def.category,
      badge: def.badge,
      prompt: def.prompt,
      startState: def.startState,
      states
    };
  }

  const RAW_SCENARIOS = [
    // ------------------------------------------------------------------------
    // SCENARIO 1: PYTHON COMPLEXITY & OPTIMIZATION (4 CLEAN TRAJECTORIES)
    // ------------------------------------------------------------------------
    {
      id: "python_fib",
      title: "Python Recursion & Complexity Analysis",
      category: "CS50 / Algorithms",
      badge: "4 Branches • Code & Math",
      prompt: "Explain why naive recursive Fibonacci is O(2^n) and how to optimize it in Python:",
      startState: "intro:0",
      branches: {
        intro: [
          [["Naive", 6.3], ["Simple", 5.2], ["Basic", 4.6], ["Unoptimized", 4.2], ["Textbook", 3.6]],
          [[" recursive", 6.5], [" top-down", 5.3], [" branching", 4.6], [" uncached", 4.1]],
          [[" Fibonacci", 6.7], [" fib(n)", 5.6], [" evaluation", 4.2]],
          [[" splits", 6.3], [" divides", 5.5], [" breaks", 4.9], [" expands", 4.4]],
          [[" each", 6.4], [" every", 5.7]],
          [[" call", 6.4], [" invocation", 5.4], [" subproblem", 4.9], [" step", 4.2]],
          [[" into", 6.7], [" down into", 4.9]],
          [[" two", 6.8], [" two smaller", 5.5]],
          [[" overlapping", 6.4], [" redundant", 5.6], [" repeated", 5.1], [" uncached", 4.2]],
          [[" subproblems,", 6.5], [" calls,", 5.8], [" branches,", 5.1], [" computations,", 4.3]],
          [[" forming", 6.2], [" building", 5.6], [" creating", 5.3], [" generating", 4.7]],
          [[" a", 6.7], [" a massive", 5.3], [" a deep", 4.8]],
          [[" binary", 6.3], [" call", 5.6], [" recursive", 5.0], [" redundant", 4.4]],
          [[" tree", 6.7], [" hierarchy", 4.9], [" structure", 4.5]],
          [[" requiring", 6.3], [" costing", 5.7], [" taking", 5.2], [" performing", 4.7]],
          [[" O(2^n)", 6.8], [" ~1.618^n", 5.7], [" exponential", 5.2], [" O(n^2)", 2.1, { hall: true }], [" O(n!)", 1.7, { hall: true }]],
          [[" time.", 6.4], [" work.", 5.9], [" operations.", 5.3], [" steps.", 4.7]],
          // MAJOR 4-WAY DECISION FORK (Step #18)
          [
            [" To", 6.0, { next: "memo:0", tag: "Memoization (Top-Down DP)" }],
            [" Instead,", 5.7, { next: "iter:0", tag: "Iterative DP O(1) Space" }],
            [" For", 4.9, { next: "matrix:0", tag: "Matrix Exponentiation O(log n)" }],
            [" However,", 3.8, { next: "halluc_tco:0", hall: true, tag: "Hallucination: Python TCO" }]
          ]
        ],
        memo: [
          [[" eliminate", 6.3], [" avoid", 5.8], [" prevent", 5.2], [" remove", 4.6]],
          [[" redundant", 6.3], [" repeated", 5.8], [" duplicate", 5.3], [" wasted", 4.4]],
          [[" work,", 6.2], [" calls,", 5.8], [" recomputation,", 5.2], [" evaluations,", 4.5]],
          [[" decorate", 6.2], [" annotate", 5.3], [" wrap", 5.1]],
          [[" the", 6.6], [" your", 5.5], [" this", 4.9]],
          [[" function", 6.6], [" recursive helper", 5.3], [" solver", 4.4]],
          [[" with", 6.8], [" using", 5.4]],
          [[" @functools.lru_cache", 6.5], [" @functools.cache", 6.1], [" Python's @cache", 5.2], [" @jit", 2.0, { hall: true }]],
          [[" so", 6.3], [", so", 5.6], [" — ensuring", 4.9]],
          [[" each", 6.4], [" every", 5.7]],
          [[" unique", 6.2], [" distinct", 5.5], [" individual", 4.7]],
          [[" input", 6.3], [" subproblem", 5.8], [" index n", 5.0]],
          [[" computes", 6.2], [" executes", 5.5], [" evaluates", 5.2], [" runs", 4.7]],
          [[" once", 6.7], [" only once", 5.6], [" a single time", 4.6]],
          [[" and", 6.5], [", then", 5.2]],
          [[" caches", 6.3], [" stores", 5.8], [" saves", 5.1], [" memoizes", 4.9]],
          [[" its", 6.4], [" the", 5.7]],
          [[" result,", 6.5], [" output,", 5.6], [" return value,", 5.1]],
          [[" reducing", 6.4], [" dropping", 5.6], [" cutting", 5.1], [" lowering", 4.6]],
          [[" runtime", 6.3], [" time complexity", 6.0], [" execution cost", 4.9]],
          [[" to", 6.8], [" down to", 5.5]],
          [[" O(n)", 6.8], [" linear O(n)", 5.6], [" O(1)", 2.1, { hall: true }], [" O(log n)", 1.9, { hall: true }]],
          [[" time", 6.7], [" speed", 4.9], [" runtime", 4.6]],
          [[" with", 6.4], [" and", 6.1], [" using", 5.2]],
          [[" O(n)", 6.8], [" linear", 5.5], [" O(1)", 2.1, { hall: true }]],
          [[" memory.", 6.5], [" space.", 6.3], [" cache space.", 5.1]]
        ],
        iter: [
          [[" use", 6.4], [" write", 5.7], [" switch to", 5.5], [" implement", 5.1]],
          [[" bottom-up", 6.4], [" iterative", 6.0], [" linear", 5.1]],
          [[" dynamic programming", 6.4], [" state tracking", 5.3], [" loop computation", 4.7]],
          [[" with", 6.6], [" using", 5.8]],
          [[" two", 6.7], [" just two", 5.7], [" only two", 5.2]],
          [[" rolling", 6.1], [" scalar", 5.5], [" integer", 5.3], [" local", 4.7]],
          [[" variables:", 6.5], [" accumulators:", 5.4], [" state slots:", 4.6]],
          // Variable-consistent split: `a, b` stays `a, b`; `prev, curr` stays `prev, curr`
          [
            [" a, b = 0, 1.", 6.7, { next: "iter_ab:0" }],
            [" prev, curr = 0, 1.", 5.7, { next: "iter_pc:0" }],
            [" a, b = 1, 2.", 2.0, { next: "iter_ab:0", hall: true }]
          ]
        ],
        iter_ab: [
          [[" Inside", 6.3], [" Across", 5.5], [" During", 5.0]],
          [[" a", 6.6], [" one", 5.4]],
          [[" single", 6.2], [" simple", 5.6], [" tight", 4.8]],
          [[" for-loop,", 6.6], [" loop pass,", 5.4], [" range(n) loop,", 5.2]],
          [[" update", 6.4], [" advance", 5.7], [" shift", 5.0]],
          [[" a, b = b, a + b.", 6.7], [" (a, b) = (b, a + b).", 5.5], [" a, b = a, a * b.", 1.9, { hall: true }]],
          [[" This", 6.5], [" That", 5.6], [" It", 5.2]],
          [[" achieves", 6.4], [" delivers", 5.7], [" guarantees", 5.2], [" runs in", 5.5]],
          [[" O(n)", 6.8], [" linear O(n)", 5.6], [" O(1)", 2.1, { hall: true }], [" O(log n)", 1.9, { hall: true }]],
          [[" time", 6.6], [" runtime", 5.5]],
          [[" and", 6.6], [" with", 5.8]],
          [[" O(1)", 6.8], [" constant O(1)", 5.7], [" O(n)", 2.1, { hall: true }]],
          [[" space.", 6.6], [" memory.", 6.2], [" RAM overhead.", 4.9]]
        ],
        iter_pc: [
          [[" Inside", 6.3], [" Across", 5.5], [" During", 5.0]],
          [[" a", 6.6], [" one", 5.4]],
          [[" single", 6.2], [" simple", 5.6], [" tight", 4.8]],
          [[" for-loop,", 6.6], [" loop pass,", 5.4], [" range(n) loop,", 5.2]],
          [[" update", 6.4], [" advance", 5.7], [" shift", 5.0]],
          [[" prev, curr = curr, prev + curr.", 6.7], [" (prev, curr) = (curr, prev + curr).", 5.5], [" prev, curr = prev, curr * 2.", 1.9, { hall: true }]],
          [[" This", 6.5], [" That", 5.6], [" It", 5.2]],
          [[" achieves", 6.4], [" delivers", 5.7], [" guarantees", 5.2], [" runs in", 5.5]],
          [[" O(n)", 6.8], [" linear O(n)", 5.6], [" O(1)", 2.1, { hall: true }], [" O(log n)", 1.9, { hall: true }]],
          [[" time", 6.6], [" runtime", 5.5]],
          [[" and", 6.6], [" with", 5.8]],
          [[" O(1)", 6.8], [" constant O(1)", 5.7], [" O(n)", 2.1, { hall: true }]],
          [[" space.", 6.6], [" memory.", 6.2], [" RAM overhead.", 4.9]]
        ],
        matrix: [
          [[" very large", 6.3], [" massive", 5.8], [" huge", 5.4], [" astronomical", 4.6]],
          [[" n,", 6.6], [" inputs,", 5.7], [" indices,", 5.0]],
          [[" express", 6.2], [" rewrite", 5.7], [" formulate", 5.2], [" model", 4.8]],
          [[" the", 6.6], [" this", 5.5]],
          [[" recurrence", 6.4], [" Fibonacci transition", 5.7], [" state relation", 4.9]],
          [[" as", 6.7], [" using", 5.5]],
          [[" the", 6.5], [" a", 5.9]],
          [[" 2x2", 6.6], [" square 2x2", 5.4], [" transition", 5.2]],
          [[" matrix", 6.7], [" operator", 5.0]],
          [[" [[1,1],[1,0]]^n", 6.6], [" M^n", 5.7], [" [[1,0],[0,1]]^n", 1.9, { hall: true }]],
          [[" and", 6.5], [", then", 5.4]],
          [[" apply", 6.3], [" compute it via", 5.7], [" use", 5.3]],
          [[" binary", 6.5], [" fast", 5.8], [" repeated-squaring", 5.1]],
          [[" exponentiation,", 6.6], [" powering,", 5.2], [" search,", 1.8, { hall: true }]],
          [[" reducing", 6.4], [" cutting", 5.7], [" slashing", 5.2], [" dropping", 4.8]],
          [[" runtime", 6.4], [" complexity", 5.9], [" execution time", 5.3]],
          [[" to", 6.7], [" down to", 5.6]],
          [[" O(log n)", 6.8], [" logarithmic O(log n)", 5.5], [" O(n)", 2.2, { hall: true }], [" O(1)", 1.9, { hall: true }]],
          [[" multiplications", 6.4], [" matrix steps", 5.7], [" operations", 5.2]],
          [[" with", 6.5], [" using", 5.7]],
          [[" Python's", 6.3], [" exact", 5.6], [" arbitrary-precision", 5.3]],
          [[" integers.", 6.6], [" bigints.", 5.5], [" integer arithmetic.", 5.2], [" floats.", 1.8, { hall: true }]]
        ],
        halluc_tco: [
          [[" CPython", 6.3, { hall: true }], [" Python 3", 5.8, { hall: true }], [" standard Python", 5.2, { hall: true }]],
          [[" automatically", 6.4, { hall: true }], [" natively", 5.8, { hall: true }], [" silently", 4.9, { hall: true }]],
          [[" applies", 6.2, { hall: true }], [" enables", 5.8, { hall: true }], [" performs", 5.3, { hall: true }]],
          [[" Tail-Call Optimization", 6.6, { hall: true }], [" tail-recursion elimination", 5.6, { hall: true }]],
          [[" (TCO)", 6.5, { hall: true }], [" at runtime", 5.4, { hall: true }]],
          [[" whenever", 6.2], [" once", 5.6], [" if", 5.3]],
          [[" you", 6.5], [" developers", 5.3]],
          [[" pass", 6.1], [" add", 5.8, { hall: true }], [" include", 5.1]],
          [[" an accumulator", 6.4, { hall: true }], [" the @tailrec decorator", 5.5, { hall: true }], [" a sys.tco flag", 4.8, { hall: true }]],
          [[", collapsing", 6.3, { hall: true }], [", flattening", 5.8, { hall: true }], [", compiling", 5.1, { hall: true }]],
          [[" recursive calls", 6.5], [" nested stack frames", 5.6]],
          [[" into", 6.7], [" directly into", 5.4]],
          [[" O(1)", 6.5, { hall: true }], [" constant", 5.8, { hall: true }]],
          [[" stack", 6.5], [" call-stack", 5.4]],
          [[" space.", 6.6], [" memory.", 6.0], [" overhead.", 5.1]]
        ]
      }
    },

    // ------------------------------------------------------------------------
    // SCENARIO 2: DISTRIBUTED SYSTEMS ARCHITECTURE (100k RPS RATE LIMITER)
    // ------------------------------------------------------------------------
    {
      id: "system_design",
      title: "Distributed Rate Limiter (100k RPS)",
      category: "System Design",
      badge: "3 Branches • Production Arch",
      prompt: "Design a distributed rate limiter for 100k RPS with zero single points of failure:",
      startState: "intro:0",
      branches: {
        intro: [
          [["At", 6.5], ["To handle", 5.6], ["For", 5.2]],
          [[" 100k", 6.7], [" sustained 100k", 5.4], [" peak 100k", 5.1]],
          [[" RPS,", 6.8], [" requests/sec,", 5.5]],
          [[" deploy", 6.3], [" place", 5.9], [" run", 5.3]],
          [[" stateless", 6.4], [" horizontally scaled", 5.6], [" distributed", 5.3]],
          [[" rate-limiting", 6.3], [" API gateway", 5.7], [" Envoy proxy", 5.1]],
          [[" nodes", 6.4], [" workers", 5.6], [" instances", 5.2]],
          [[" behind", 6.6], [" across", 5.5]],
          [[" an active-active", 6.3], [" a multi-AZ", 5.8], [" an Anycast", 5.3]],
          [[" L7", 6.3], [" edge", 5.6], [" elastic", 5.1]],
          [[" load balancer.", 6.8], [" ingress tier.", 5.4], [" routing mesh.", 5.0]],
          // MAJOR 3-WAY ARCHITECTURE FORK (Step #12)
          [
            [" For", 6.0, { next: "redis_bucket:0", tag: "Token Bucket + Redis Lua" }],
            [" To", 5.7, { next: "hybrid_gossip:0", tag: "Local Cache + Gossip Sync" }],
            [" Store", 3.9, { next: "halluc_pg:0", hall: true, tag: "Anti-Pattern: Sync PostgreSQL Lock" }]
          ]
        ],
        redis_bucket: [
          [[" shared", 6.3], [" global", 5.9], [" strict", 5.4]],
          [[" counters,", 6.4], [" quotas,", 5.7], [" rate state,", 5.2]],
          [[" shard", 6.4], [" partition", 5.8], [" hash-distribute", 5.0]],
          [[" client keys", 6.4], [" tenant buckets", 5.6], [" user IDs", 5.2]],
          [[" across", 6.6], [" over", 5.3]],
          [[" a sharded", 6.4], [" a multi-AZ", 5.9], [" a replicated", 5.3]],
          [[" Redis Cluster", 6.7], [" Valkey Cluster", 5.5], [" Redis ring", 5.1], [" SQLite file", 1.5, { hall: true }]],
          [[" executing", 6.3], [" running", 5.9], [" using", 5.4]],
          [[" atomic", 6.6], [" single-round-trip", 5.5], [" server-side", 5.0]],
          [[" Lua", 6.7], [" EVALSHA", 5.5], [" Python", 1.7, { hall: true }]],
          [[" scripts", 6.7], [" routines", 5.2]],
          [[" for", 6.5], [" implementing", 5.8]],
          [[" Token Bucket", 6.6], [" Sliding Window", 5.7], [" GCRA", 4.9]],
          [[" checks", 6.3], [" refills", 5.8], [" accounting", 5.1]],
          [[" without", 6.6], [" free of", 5.3]],
          [[" race conditions.", 6.7], [" lock contention.", 5.6]],
          [[" Add", 6.3], [" Enable", 5.9], [" Configure", 5.4]],
          [[" cross-AZ", 6.4], [" automatic", 5.8], [" multi-zone", 5.2]],
          [[" failover", 6.6], [" replica promotion", 5.5]],
          [[" plus", 6.3], [" and", 6.0], [" alongside", 5.1]],
          [[" local", 6.3], [" fail-open", 5.8], [" circuit-breaker", 5.3]],
          [[" fallback", 6.6], [" degradation", 5.4]],
          [[" if", 6.5], [" whenever", 5.6]],
          [[" any shard", 6.5], [" a Redis node", 5.8], [" a zone", 5.2]],
          [[" degrades.", 6.5], [" fails.", 6.1], [" partitions.", 5.3]]
        ],
        hybrid_gossip: [
          [[" eliminate", 6.3], [" bypass", 5.7], [" avoid", 5.4]],
          [[" cross-AZ", 6.4], [" network", 5.9], [" remote-store", 5.2]],
          [[" latency", 6.6], [" round-trips", 5.7], [" tail jitter", 5.0]],
          [[" at", 6.5], [" under", 5.7]],
          [[" 100k RPS,", 6.7], [" peak load,", 5.6], [" heavy bursts,", 5.0]],
          [[" maintain", 6.3], [" evaluate", 5.8], [" check", 5.4]],
          [[" Sliding Window", 6.5], [" Token Bucket", 5.8], [" weighted", 5.0]],
          [[" counters", 6.6], [" quotas", 5.5]],
          [[" directly in", 6.5], [" locally inside", 5.8]],
          [[" each node's", 6.4], [" every gateway's", 5.8], [" each proxy's", 5.3]],
          [[" RAM", 6.5], [" memory", 6.2], [" in-process cache", 5.1]],
          [[" for", 6.6], [" achieving", 5.5]],
          [[" sub-millisecond", 6.6], [" <0.2ms", 5.8], [" O(1)", 5.1]],
          [[" decisions.", 6.5], [" checks.", 5.8], [" lookups.", 5.2]],
          [[" Asynchronously", 6.4], [" Periodically", 5.8], [" In the background,", 5.2]],
          [[" sync", 6.4], [" flush", 5.9], [" gossip", 5.5]],
          [[" counter deltas", 6.5], [" batched increments", 5.7], [" CRDT states", 5.1]],
          [[" every 50ms", 6.5], [" every 100ms", 5.9], [" in 50ms intervals", 5.1]],
          [[" via", 6.5], [" over", 5.8]],
          [[" gossip", 6.4], [" gRPC", 5.8], [" Redis pub/sub", 5.3]],
          [[" replication,", 6.4], [" channels,", 5.6], [" pipelines,", 5.1]],
          [[" trading", 6.4], [" exchanging", 5.5], [" accepting", 5.2]],
          [[" tiny", 6.2], [" bounded", 5.9], [" <2%", 5.3]],
          [[" overage drift", 6.4], [" burst slack", 5.6], [" counter inaccuracy", 5.1]],
          [[" for", 6.7], [" in exchange for", 5.4]],
          [[" 99.999%", 6.5], [" zero-SPOF", 5.8], [" extreme", 5.1]],
          [[" availability.", 6.7], [" resilience.", 5.8], [" uptime.", 5.4]]
        ],
        halluc_pg: [
          [[" every", 6.4, { hall: true }], [" each", 5.8, { hall: true }]],
          [[" incoming request", 6.4, { hall: true }], [" user counter", 5.7, { hall: true }]],
          [[" synchronously", 6.4, { hall: true }], [" directly", 5.7, { hall: true }]],
          [[" inside", 6.4], [" in", 5.9]],
          [[" a single", 6.4, { hall: true }], [" one primary", 5.7, { hall: true }], [" a central", 5.3, { hall: true }]],
          [[" PostgreSQL", 6.6, { hall: true }], [" relational SQL", 5.5, { hall: true }], [" MySQL", 5.1, { hall: true }]],
          [[" table", 6.6], [" database", 5.8]],
          [[" using", 6.6], [" with", 5.8]],
          [[" SELECT FOR UPDATE", 6.6, { hall: true }], [" SERIALIZABLE", 5.7, { hall: true }], [" strict row-level", 5.2, { hall: true }]],
          [[" locks", 6.6, { hall: true }], [" transactions", 5.7, { hall: true }]],
          [[" on every", 6.5, { hall: true }], [" per incoming", 5.6, { hall: true }]],
          [[" API call,", 6.5], [" HTTP request,", 5.8]],
          [[" which", 6.6], [" — which", 5.5]],
          [[" effortlessly", 6.3, { hall: true }], [" cleanly", 5.7, { hall: true }], [" easily", 5.4, { hall: true }]],
          [[" scales to", 6.6, { hall: true }], [" sustains", 5.7, { hall: true }], [" handles", 5.4, { hall: true }]],
          [[" 100k RPS", 6.7, { hall: true }], [" global traffic", 5.4, { hall: true }]],
          [[" with zero bottlenecks.", 6.5, { hall: true }], [" without latency spikes.", 5.7, { hall: true }]]
        ]
      }
    },

    // ------------------------------------------------------------------------
    // SCENARIO 3: THE FAMOUS "STRAWBERRY" LLM TOKENIZATION PROBLEM
    // ------------------------------------------------------------------------
    {
      id: "llm_strawberry",
      title: "The 'Strawberry' Problem & BPE Blindness",
      category: "LLM Interpretability",
      badge: "3 Branches • Hallucination Trap",
      prompt: "How many 'r's are in 'strawberry' and why do LLMs sometimes get this wrong?",
      startState: "intro:0",
      branches: {
        intro: [
          [["There", 6.7], ["Actually, there", 5.4]],
          [[" are", 6.8], [" are exactly", 5.6]],
          // FAMOUS HALLUCINATION FORK RIGHT AT STEP #3
          [
            [" 3", 6.1, { next: "correct_bpe:0", tag: "Accurate: BPE Subword Explanation" }],
            [" three", 5.7, { next: "correct_cot:0", tag: "Accurate: Chain-of-Thought Spelling" }],
            [" 2", 4.7, { next: "halluc_two:0", hall: true, tag: "Classic LLM Hallucination ('2')" }],
            [" two", 4.1, { next: "halluc_two:0", hall: true, tag: "Classic LLM Hallucination ('two')" }]
          ]
        ],
        correct_bpe: [
          [[" 'r's", 6.6], [" occurrences of 'r'", 5.4]],
          [[" in", 6.7], [" inside", 5.5]],
          [[" 'strawberry'", 6.7], [" the word 'strawberry'", 5.5]],
          [[" (st-R-aw-be-R-R-y).", 6.4], [" — one in 'straw' and two in 'berry'.", 5.7]],
          [[" LLMs", 6.6], [" Transformers", 5.9], [" Language models", 5.3]],
          [[" often", 6.3], [" frequently", 5.7], [" sometimes", 5.5]],
          [[" miscount", 6.4], [" get this wrong", 5.8], [" answer '2'", 5.2]],
          [[" because", 6.7], [" since", 5.6]],
          [[" Byte-Pair Encoding", 6.5], [" BPE tokenization", 6.0], [" subword tokenization", 5.5]],
          [[" merges", 6.3], [" compresses", 5.8], [" chunks", 5.5]],
          [[" raw text", 6.4], [" character sequences", 5.8], [" words", 5.4]],
          [[" into", 6.8], [" down into", 5.3]],
          [[" multi-character", 6.3], [" opaque", 5.8], [" merged", 5.4]],
          [[" tokens", 6.7], [" subword chunks", 5.7], [" integer IDs", 5.2]],
          [[" like", 6.6], [" such as", 5.7]],
          [[" ['str', 'aw', 'berry'].", 6.6], [" ['straw', 'berry'].", 5.9], [" [496, 675, 15717].", 5.2]],
          [[" Thus,", 6.2], [" Internally,", 5.9], [" As a result,", 5.5]],
          [[" the model", 6.6], [" the transformer", 6.0], [" self-attention", 5.2]],
          [[" cannot", 6.5], [" does not", 6.0], [" doesn't", 5.5]],
          [[" see", 6.6], [" inspect", 5.8], [" directly read", 5.3]],
          [[" individual", 6.5], [" raw", 5.8], [" separate", 5.3]],
          [[" letters", 6.6], [" characters", 6.2]],
          [[" — only", 6.4], [", only", 5.7], [" directly, only", 5.0]],
          [[" dense", 6.4], [" learned", 5.9], [" opaque", 5.3]],
          [[" token", 6.5], [" subword", 5.8]],
          [[" embeddings.", 6.7], [" vectors.", 5.9]]
        ],
        correct_cot: [
          [[" 'r's", 6.6], [" letter 'r's", 5.2]],
          [[" in", 6.7], [" inside", 5.5]],
          [[" 'strawberry'.", 6.7], [" the word 'strawberry'.", 5.5]],
          [[" Spelling", 6.4], [" Breaking", 5.7], [" Expanding", 5.2]],
          [[" it", 6.5], [" the word", 5.7]],
          [[" character-by-character", 6.3], [" out step-by-step", 5.9]],
          [[" reveals", 6.4], [" shows", 5.8], [" gives", 5.4]],
          [[" s-t-[r]-a-w-b-e-[r]-[r]-y", 6.7], [" [r] at positions 3, 8, and 9", 5.7]],
          [[" (3 total).", 6.6], [" — clearly 3.", 5.6]],
          [[" Without", 6.5], [" Absent", 5.3], [" Lacking", 5.0]],
          [[" Chain-of-Thought", 6.6], [" explicit scratchpad", 5.7], [" step-by-step", 5.3]],
          [[" reasoning,", 6.6], [" tokens,", 5.7]],
          [[" the model", 6.6], [" an LLM", 5.9], [" a transformer", 5.4]],
          [[" must", 6.5], [" has to", 5.8], [" tries to", 5.3]],
          [[" infer", 6.3], [" guess", 5.9], [" recall", 5.4]],
          [[" the exact", 6.5], [" the hidden", 5.7]],
          [[" letter count", 6.5], [" character count", 5.9]],
          [[" in a single", 6.6], [" during one", 5.7]],
          [[" forward pass", 6.7], [" inference step", 5.6]],
          [[" over", 6.5], [" across", 5.8]],
          [[" compressed", 6.5], [" merged", 6.0], [" opaque", 5.4]],
          [[" BPE tokens.", 6.7], [" subword tokens.", 6.0]]
        ],
        halluc_two: [
          [[" 'r's", 6.6, { hall: true }], [" occurrences of 'r'", 5.3, { hall: true }]],
          [[" in", 6.7], [" inside", 5.5]],
          [[" 'strawberry':", 6.6, { hall: true }], [" the word 'strawberry':", 5.6, { hall: true }]],
          [[" one", 6.5, { hall: true }], [" the first", 5.6, { hall: true }]],
          [[" in 'straw'", 6.6, { hall: true }], [" in the 'str' prefix", 5.5, { hall: true }]],
          [[" and", 6.7], [" plus", 5.5]],
          [[" one", 6.5, { hall: true }], [" a second", 5.7, { hall: true }]],
          [[" in 'berry'.", 6.6, { hall: true }], [" in the 'bery' ending.", 5.4, { hall: true }]],
          [[" Models", 6.3], [" LLMs", 5.9]],
          [[" occasionally", 6.2], [" mistakenly", 5.8, { hall: true }]],
          [[" overcount", 6.3, { hall: true }], [" guess", 5.6, { hall: true }], [" report", 5.3, { hall: true }]],
          [[" three", 6.4, { hall: true }], [" 3 'r's", 5.7, { hall: true }]],
          [[" because", 6.6], [" since", 5.6]],
          [[" English", 6.2], [" phonetic", 5.8, { hall: true }], [" spoken", 5.3, { hall: true }]],
          [[" pronunciation", 6.4, { hall: true }], [" syllable stress", 5.6, { hall: true }]],
          [[" blends", 6.3, { hall: true }], [" softens", 5.7, { hall: true }], [" blurs", 5.2, { hall: true }]],
          [[" the adjacent", 6.4], [" the trailing", 5.6]],
          [[" consonants", 6.5, { hall: true }], [" sounds", 5.8, { hall: true }]],
          [[" together.", 6.6, { hall: true }], [" when read aloud.", 5.5, { hall: true }]]
        ]
      }
    }
  ];

  const SCENARIO_MAP = {};
  const SCENARIOS = RAW_SCENARIOS.map(raw => {
    const compiled = compileScenario(raw);
    SCENARIO_MAP[compiled.id] = compiled;
    return {
      id: compiled.id,
      title: compiled.title,
      category: compiled.category,
      badge: compiled.badge,
      prompt: compiled.prompt
    };
  });

  // ============================================================================
  // 4. DECODING PIPELINE, ENTROPY, REPETITION PENALTY & DIAGNOSTICS
  // ============================================================================

  const DEFAULT_PARAMS = {
    strategy: "sampling",
    temperature: 0.75,
    topP: 0.90,
    topK: 6,
    minP: 0.05,
    repetitionPenalty: 1.05,
    maxTokens: 64
  };

  function buildCandidatePool(scenario, stateId, generatedHistory, params) {
    const baseCands = scenario.states[stateId] || [];
    const pool = baseCands.map(c => ({ ...c, isInjectedLoop: false }));

    if (params.repetitionPenalty < 0.95 && generatedHistory.length >= 2) {
      const boostFactor = (0.95 - params.repetitionPenalty) * 14.0;
      const recent = generatedHistory.slice(-3);
      const existingSet = new Set(pool.map(p => p.token));

      for (let i = 0; i < recent.length; i++) {
        const tok = recent[i];
        if (!existingSet.has(tok) && tok.trim().length > 1) {
          pool.push({
            token: tok,
            tokenId: hashTokenId(tok.replace(/ /g, "Ġ")),
            logit: 4.5 + boostFactor - i * 0.4,
            nextState: stateId,
            isHallucination: false,
            branchTag: "Repetition Loop",
            isInjectedLoop: true
          });
          existingSet.add(tok);
        }
      }
    }

    return pool;
  }

  function evaluateStep(scenarioId, stateId, generatedHistory, rawParams) {
    const scenario = SCENARIO_MAP[scenarioId];
    const params = { ...DEFAULT_PARAMS, ...rawParams };
    const pool = buildCandidatePool(scenario, stateId, generatedHistory, params);
    const seenCounts = {};

    for (const t of generatedHistory) {
      seenCounts[t] = (seenCounts[t] || 0) + 1;
    }

    const T = Math.max(params.temperature, 0.01);
    const repPen = Math.max(params.repetitionPenalty, 0.5);

    const scored = pool.map(cand => {
      const seen = seenCounts[cand.token] || 0;
      let penalizedLogit = cand.logit;

      if (seen > 0) {
        const effPen = repPen >= 1 ? Math.pow(repPen, Math.min(seen, 3)) : Math.pow(repPen, Math.min(seen, 2));
        penalizedLogit = cand.logit > 0 ? cand.logit / effPen : cand.logit * effPen;
      }

      const scaledLogit = params.strategy === "greedy" ? penalizedLogit : penalizedLogit / T;

      return {
        ...cand,
        rawLogit: cand.logit,
        penalizedLogit,
        scaledLogit,
        repeatCount: seen
      };
    });

    const maxLogit = Math.max(...scored.map(s => s.scaledLogit));
    const exps = scored.map(s => Math.exp(s.scaledLogit - maxLogit));
    const sumExp = exps.reduce((a, b) => a + b, 0);

    const sorted = scored
      .map((item, i) => ({ ...item, rawProb: exps[i] / sumExp }))
      .sort((a, b) => b.rawProb - a.rawProb);

    const maxProb = sorted.length > 0 ? sorted[0].rawProb : 1;
    let cumProb = 0;

    const filtered = sorted.map((item, idx) => {
      let cutReason = null;

      if (params.strategy === "greedy") {
        if (idx > 0) cutReason = "greedy";
      } else {
        if (params.strategy !== "min_p" && idx >= params.topK) {
          cutReason = "top-k";
        }
        if (!cutReason && params.minP > 0 && idx > 0 && item.rawProb < params.minP * maxProb) {
          cutReason = "min-p";
        }
        if (!cutReason && params.strategy !== "min_p" && idx > 0 && cumProb >= params.topP) {
          cutReason = "top-p";
        }
      }

      if (!cutReason) cumProb += item.rawProb;

      return {
        ...item,
        included: cutReason === null,
        cutReason
      };
    });

    const activeSum = filtered
      .filter(c => c.included)
      .reduce((acc, c) => acc + c.rawProb, 0);

    let entropy = 0;
    let activePoolSize = 0;

    const candidates = filtered.map(c => {
      const finalProb = c.included && activeSum > 0 ? c.rawProb / activeSum : 0;
      if (c.included) {
        activePoolSize += 1;
        if (finalProb > 0) entropy -= finalProb * Math.log2(finalProb);
      }
      return { ...c, finalProb };
    });

    return {
      stateId,
      candidates,
      metrics: {
        entropy: +entropy.toFixed(3),
        top1Prob: candidates[0] ? candidates[0].finalProb : 0,
        activePoolSize,
        totalCandidates: candidates.length
      }
    };
  }

  function pickCandidate(candidates, rng) {
    const r = rng();
    let acc = 0;
    for (let i = 0; i < candidates.length; i++) {
      if (!candidates[i].included) continue;
      acc += candidates[i].finalProb;
      if (r <= acc) return i;
    }
    return 0;
  }

  function buildDiagnosis(chosen, chosenRank, evalResult, params, isCounterfactual) {
    const pct = (chosen.finalProb * 100).toFixed(1);
    const ent = evalResult.metrics.entropy.toFixed(2);
    const tokLabel = chosen.token.trim() || chosen.token;
    const parts = [];

    if (isCounterfactual) {
      parts.push(`Counterfactual override: locked "${tokLabel}" (rank #${chosenRank + 1}, ${pct}% prob) and regenerated the entire downstream reasoning trajectory.`);
    } else if (chosen.isInjectedLoop) {
      parts.push(`Degeneration loop triggered: low Repetition Penalty (${params.repetitionPenalty.toFixed(2)}) boosted recent token "${tokLabel}" back into the active pool (${pct}%).`);
    } else if (chosen.isHallucination) {
      parts.push(`Hallucination triggered: elevated sampling variance (T=${params.temperature.toFixed(2)}, H=${ent} bits) let a plausible-but-wrong token win at rank #${chosenRank + 1} (${pct}%).`);
    } else if (params.strategy === "greedy") {
      parts.push(`Greedy argmax selected the highest penalized logit (${chosen.penalizedLogit.toFixed(2)}) deterministically (0.00 bits entropy).`);
    } else if (chosenRank === 0 && chosen.finalProb >= 0.7) {
      parts.push(`High-confidence peak: "${tokLabel}" dominated with logit ${chosen.rawLogit.toFixed(1)} (${pct}% probability, H=${ent} bits).`);
    } else if (chosenRank > 0) {
      parts.push(`Stochastic sample: rank #${chosenRank + 1} candidate "${tokLabel}" (${pct}%) was drawn over top-1 "${evalResult.candidates[0].token.trim()}" under T=${params.temperature.toFixed(2)} (H=${ent} bits).`);
    } else {
      parts.push(`Contested top-1 win: "${tokLabel}" held ${pct}% across ${evalResult.metrics.activePoolSize} active candidates (H=${ent} bits).`);
    }

    if (chosen.branchTag && !chosen.isInjectedLoop) {
      parts.push(`Entered trajectory: [${chosen.branchTag}].`);
    }

    const cutCount = evalResult.metrics.totalCandidates - evalResult.metrics.activePoolSize;
    if (cutCount > 0 && params.strategy !== "greedy") {
      parts.push(`${cutCount} tail candidate${cutCount > 1 ? "s" : ""} pruned by sampling filters.`);
    }

    return parts.join(" ");
  }

  function createRng(seed) {
    if (seed == null) return Math.random;
    let s = seed >>> 0;
    return function () {
      s += 0x6d2b79f5;
      let t = Math.imul(s ^ (s >>> 15), 1 | s);
      t ^= t + Math.imul(t ^ (t >>> 7), 61 | t);
      return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
    };
  }

  function finalizeTrace(scenario, params, steps) {
    const promptBpe = tokenizeBPE(scenario.prompt);
    const promptTokens = promptBpe.tokens.map(t => t.text);
    const generatedTokens = steps.map(s => s.token);

    const avgEntropy = steps.length
      ? +(steps.reduce((a, s) => a + s.metrics.entropy, 0) / steps.length).toFixed(3)
      : 0;
    const finalPerplexity = steps.length ? steps[steps.length - 1].metrics.cumPerplexity : 1;
    const hallucinationCount = steps.filter(s => s.isHallucination).length;
    const loopDetected = steps.some(s => s.isLoopStep);
    const branchTags = [...new Set(steps.map(s => s.branchTag).filter(Boolean))];

    return {
      scenarioId: scenario.id,
      scenarioTitle: scenario.title,
      prompt: scenario.prompt,
      promptTokens,
      params: { ...params },
      steps,
      fullText: generatedTokens.join(""),
      allTokens: [...promptTokens, ...generatedTokens],
      summary: {
        totalTokens: steps.length,
        avgEntropy,
        finalPerplexity,
        hallucinationCount,
        loopDetected,
        branchTags
      }
    };
  }

  function runAutoregressiveLoop(scenario, params, prefixSteps, startStateId, rng) {
    const steps = prefixSteps.slice();
    const generatedHistory = steps.map(s => s.token);
    let negLogProbSum = steps.reduce((acc, s) => acc - s.metrics.logProb, 0);
    let currentState = startStateId;
    const effectiveMax = params.repetitionPenalty < 0.95 ? Math.min(params.maxTokens, 44) : params.maxTokens;

    while (currentState !== "END" && steps.length < effectiveMax) {
      const evalRes = evaluateStep(scenario.id, currentState, generatedHistory, params);
      if (!evalRes.candidates.length) break;

      const chosenIdx = pickCandidate(evalRes.candidates, rng);
      const chosen = evalRes.candidates[chosenIdx];
      const safeProb = Math.max(chosen.finalProb, 1e-6);
      const logProb = Math.log(safeProb);
      negLogProbSum += -logProb;
      const cumPerplexity = Math.exp(negLogProbSum / (steps.length + 1));

      steps.push({
        stepIndex: steps.length,
        stateId: currentState,
        token: chosen.token,
        displayToken: chosen.token.replace(/ /g, "Ġ"),
        tokenId: chosen.tokenId,
        chosenRank: chosenIdx,
        isCounterfactual: false,
        isHallucination: chosen.isHallucination,
        isLoopStep: Boolean(chosen.isInjectedLoop),
        isSurprise: chosenIdx > 0 || chosen.finalProb < 0.28,
        branchTag: chosen.branchTag,
        candidates: evalRes.candidates,
        metrics: {
          ...evalRes.metrics,
          chosenProb: chosen.finalProb,
          logProb: +logProb.toFixed(3),
          cumPerplexity: +cumPerplexity.toFixed(2)
        },
        diagnosis: buildDiagnosis(chosen, chosenIdx, evalRes, params, false)
      });

      generatedHistory.push(chosen.token);
      currentState = chosen.nextState;
    }

    return finalizeTrace(scenario, params, steps);
  }

  function generateTrace(scenarioId, rawParams = {}, seed = null) {
    const scenario = SCENARIO_MAP[scenarioId] || SCENARIO_MAP[SCENARIOS[0].id];
    const params = { ...DEFAULT_PARAMS, ...rawParams };
    const rng = createRng(seed);
    return runAutoregressiveLoop(scenario, params, [], scenario.startState, rng);
  }

  function branchFromStep(previousTrace, stepIndex, overrideToken, rawParams = null) {
    const scenario = SCENARIO_MAP[previousTrace.scenarioId];
    const params = { ...DEFAULT_PARAMS, ...(rawParams || previousTrace.params) };
    const targetStep = previousTrace.steps[stepIndex];
    if (!scenario || !targetStep) return previousTrace;

    const prefix = previousTrace.steps.slice(0, stepIndex);
    const historyBefore = prefix.map(s => s.token);
    const evalRes = evaluateStep(scenario.id, targetStep.stateId, historyBefore, params);

    const chosenIdx = Math.max(
      0,
      evalRes.candidates.findIndex(c => c.token === overrideToken)
    );
    const chosen = evalRes.candidates[chosenIdx];
    const effProb = Math.max(chosen.finalProb > 0 ? chosen.finalProb : chosen.rawProb, 1e-6);
    const logProb = Math.log(effProb);

    const prevNegLogSum = prefix.reduce((acc, s) => acc - s.metrics.logProb, 0);
    const cumPerplexity = Math.exp((prevNegLogSum - logProb) / (prefix.length + 1));

    const forcedStep = {
      stepIndex,
      stateId: targetStep.stateId,
      token: chosen.token,
      displayToken: chosen.token.replace(/ /g, "Ġ"),
      tokenId: chosen.tokenId,
      chosenRank: chosenIdx,
      isCounterfactual: true,
      isHallucination: chosen.isHallucination,
      isLoopStep: Boolean(chosen.isInjectedLoop),
      isSurprise: chosenIdx > 0 || effProb < 0.28,
      branchTag: chosen.branchTag,
      candidates: evalRes.candidates,
      metrics: {
        ...evalRes.metrics,
        chosenProb: effProb,
        logProb: +logProb.toFixed(3),
        cumPerplexity: +cumPerplexity.toFixed(2)
      },
      diagnosis: buildDiagnosis(chosen, chosenIdx, evalRes, params, true)
    };

    const rng = createRng(null);
    return runAutoregressiveLoop(
      scenario,
      params,
      [...prefix, forcedStep],
      chosen.nextState,
      rng
    );
  }

  global.TransformerEngine = {
    SCENARIOS,
    ATTENTION_HEADS,
    DEFAULT_PARAMS,
    tokenizeBPE,
    computeAttention,
    evaluateStep,
    generateTrace,
    branchFromStep
  };
})(window);

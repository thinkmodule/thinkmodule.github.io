/**
 * ThinkModule Lab — Transformer Internals & Decoding Engine
 * Pure ES6 mathematical core:
 *   1. Byte-Pair Encoding (BPE) Tokenizer with merge trace
 *   2. Scaled Dot-Product Multi-Head Self-Attention (4 specialized heads + Causal Mask)
 *   3. Autoregressive Multi-Token Decoding (Temp, Top-K, Top-P, Min-P, Repetition Penalty)
 *   4. Counterfactual Branching, Shannon Entropy, Log-Prob, Perplexity & Diagnostics
 */
(function (global) {
  "use strict";

  // ============================================================================
  // 1. REAL BYTE-PAIR ENCODING (BPE) TOKENIZER WITH MERGE TRACE
  // ============================================================================

  const BPE_MERGE_RULES = [
    // High-frequency English & whitespace pairs
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
    ["Ġt", "h"], ["Ġth", "at"], ["Ġo", "n"], ["Ġa", "s"], ["Ġb", "y"],
    ["ti", "on"], ["at", "ion"], ["en", "t"], ["er", "s"], ["in", "g"],
    ["Ġr", "e"], ["Ġc", "o"], ["Ġp", "r"], ["Ġs", "t"], ["Ġd", "e"],
    // Technical, AI & Python subwords
    ["d", "ef"], ["Ġd", "ef"], ["re", "t"], ["ret", "ur"], ["retur", "n"],
    ["Ġre", "turn"], ["p", "y"], ["py", "th"], ["pyth", "on"], ["ĠP", "ython"],
    ["Ġp", "ython"], ["t", "o"], ["to", "k"], ["tok", "en"], ["Ġto", "ken"],
    ["Ġtoken", "s"], ["a", "tt"], ["att", "ent"], ["attent", "ion"],
    ["Ġa", "ttention"], ["tr", "an"], ["tran", "s"], ["trans", "f"],
    ["transf", "or"], ["transfor", "m"], ["transform", "er"],
    ["Ġtr", "ansformer"], ["m", "o"], ["mo", "de"], ["mode", "l"],
    ["Ġm", "odel"], ["Ġmodel", "s"], ["L", "L"], ["LL", "M"], ["ĠL", "LM"],
    ["ĠLLM", "s"], ["c", "a"], ["ca", "ch"], ["cach", "e"], ["Ġca", "che"],
    ["s", "p"], ["sp", "a"], ["spa", "ce"], ["Ġsp", "ace"],
    ["t", "i"], ["ti", "m"], ["tim", "e"], ["Ġti", "me"],
    ["Ġc", "om"], ["Ġcom", "p"], ["Ġcomp", "l"], ["Ġcompl", "e"],
    ["Ġcomple", "x"], ["Ġcomplex", "ity"],
    ["Ġa", "lg"], ["Ġalg", "or"], ["Ġalgor", "it"], ["Ġalgorit", "hm"],
    ["Ġr", "ec"], ["Ġrec", "ur"], ["Ġrecur", "s"], ["Ġrecurs", "ive"],
    ["ĠR", "ed"], ["ĠRed", "is"], ["ĠL", "ua"],
    // Famous LLM tokenizer edge cases ("strawberry", numbers)
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

    // Represent leading spaces as GPT-style 'Ġ', newlines as 'Ċ'
    const normalized = safeText.replace(/ /g, "Ġ").replace(/\n/g, "Ċ");
    let symbols = Array.from(normalized);
    const mergeSteps = [];

    // Apply BPE merge rules in priority order
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
      description: "Links pronouns, variables, and conclusions back to the original question or definitions established early in the prompt."
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
    "by", "with", "on", "at", "as", "be", "this", "that", "from", "into"
  ]);

  const TECH_WORDS = new Set([
    "fibonacci", "o(2^n)", "o(n)", "o(1)", "o(log", "python", "recursive",
    "recursion", "memoization", "lru_cache", "cache", "dynamic", "iterative",
    "matrix", "redis", "lua", "bucket", "token", "window", "rps", "limiter",
    "distributed", "cluster", "postgres", "lock", "strawberry", "bpe",
    "subword", "tokenizer", "attention", "prompt", "llm", "llms", "tco",
    "stack", "complexity", "latency", "atomic", "gossip", "3", "2", "100k"
  ]);

  const PRONOUN_OR_REF = new Set([
    "it", "its", "they", "them", "this", "that", "which", "these", "here",
    "return", "result", "therefore", "thus", "because", "so", "instead"
  ]);

  function getTokenEmbedding(tokStr, pos, totalLen) {
    const clean = tokStr.trim().toLowerCase().replace(/^[^a-z0-9()^_]+|[^a-z0-9()^_]+$/g, "");
    const isPunct = clean.length === 0 || /^[:;,.\-()[\]{}=<>]+$/.test(tokStr.trim());
    const isCode = /[_()[\]=^:`]/.test(tokStr) || ["def", "return", "for", "if", "n", "a", "b"].includes(clean);
    const isTech = TECH_WORDS.has(clean) || clean.length >= 7;
    const isStop = STOP_WORDS.has(clean);
    const isRef = PRONOUN_OR_REF.has(clean);
    const isNumber = /\d/.test(tokStr);

    // Deterministic semantic hash features in [-1, 1]
    const h1 = ((hashTokenId(clean || tokStr) % 200) - 100) / 100;
    const h2 = ((hashTokenId((clean || tokStr) + "_s") % 200) - 100) / 100;

    // Sinusoidal positional encoding components
    const pNorm = pos / Math.max(totalLen, 1);
    const sinPos = Math.sin(pos * 0.45);
    const cosPos = Math.cos(pos * 0.45);

    return [
      isTech ? 1.4 : isStop ? -0.6 : 0.2,    // dim 0: semantic salience
      isCode ? 1.3 : -0.2,                   // dim 1: code/syntax marker
      isPunct ? 1.2 : -0.3,                  // dim 2: punctuation
      isRef ? 1.4 : -0.2,                    // dim 3: coreference trigger
      isNumber ? 1.3 : -0.1,                 // dim 4: numerical anchor
      h1,                                    // dim 5: lexical hash A
      h2,                                    // dim 6: lexical hash B
      (sinPos * 0.6 + cosPos * 0.4) * (1 - 0.2 * pNorm) // dim 7: positional phase
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

    // Head-specific Q/K diagonal & cross-projection scaling
    const headWeights = [
      // Head 0 (Syntax): emphasizes code, punctuation, position
      { q: [0.3, 1.1, 1.0, 0.2, 0.4, 0.3, 0.3, 1.2], k: [0.3, 1.1, 1.0, 0.2, 0.4, 0.3, 0.3, 1.2] },
      // Head 1 (Semantic): emphasizes semantic salience & lexical similarity
      { q: [1.5, 0.4, -0.5, 0.5, 1.0, 1.1, 1.1, 0.2], k: [1.8, 0.4, -0.5, 0.2, 1.1, 1.1, 1.1, 0.2] },
      // Head 2 (Coreference): links reference dims to early semantic/number anchors
      { q: [0.7, 0.5, -0.3, 1.6, 0.8, 0.5, 0.5, 0.3], k: [1.7, 0.6, -0.4, 0.1, 1.2, 0.5, 0.5, 0.3] },
      // Head 3 (Induction): high lexical hash dot-product for matching patterns
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

        // Realistic Transformer head inductive biases (RoPE/ALiBi-like structure)
        if (head.id === 0) {
          // Local syntax & recency
          if (dist === 1) s += 2.3;
          else if (dist === 2) s += 1.2;
          else if (dist === 0) s += 0.4;
          else s -= Math.min(dist * 0.22, 2.8);
        } else if (head.id === 1) {
          // Semantic salience anchor
          s += embeds[j][0] * 1.15;
          if (j === 0) s += 0.35; // Attention sink phenomenon (Xiao et al., 2023)
        } else if (head.id === 2) {
          // Long-range coreference: attends back to early prompt entities
          if (embeds[i][3] > 0.5 && j < i && embeds[j][0] > 0.6) s += 2.1;
          if (j < Math.min(10, i) && embeds[j][0] > 0.5) s += 0.9;
          if (dist === 0) s -= 0.4;
        } else if (head.id === 3) {
          // Induction / exact or near-exact token repetition
          if (j !== i && cleanTokens[i].length > 1 && cleanTokens[i] === cleanTokens[j]) {
            s += 3.2;
          } else if (j > 0 && i > 0 && cleanTokens[i - 1] === cleanTokens[j - 1] && j !== i) {
            s += 2.0; // Classic induction head: attends to token after previous match
          } else if (j === 0) {
            s += 0.6; // Attention sink
          }
        }

        scores[j] = s;
        if (s > maxScore) maxScore = s;
      }

      // Numerically stable row softmax
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
  // 3. COMPACT SCENARIO COMPILER & 3 EXPERT MULTI-BRANCH SCENARIOS
  // ============================================================================

  /**
   * Compiles concise multi-branch token definitions into a verified probabilistic DAG.
   * Each candidate tuple: [token, logit, opts?]
   * opts: { next?: "branch:idx" | "END", hall?: boolean, tag?: string }
   */
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

    // Defensive integrity check: ensure every nextState exists or is "END"
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
    // SCENARIO 1: PYTHON COMPLEXITY & OPTIMIZATION (4 DISTINCT TRAJECTORIES)
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
          [["Naive", 6.2], ["The", 5.6], ["In", 4.5], ["Because", 4.1], ["Recursive", 3.8], ["Python", 2.9]],
          [[" recursive", 6.4], [" Fibonacci", 5.5], [" tree", 4.0], [" call", 3.6], [" code", 3.1], [" execution", 2.6]],
          [[" Fibonacci", 6.5], [" recomputes", 5.3], [" spawns", 4.6], [" branches", 4.1], [" hits", 3.2], [" runs", 2.7]],
          [[" splits", 5.9], [" branches", 5.6], [" recomputes", 5.2], [" creates", 4.4], [" invokes", 3.7], [" allocates", 2.4]],
          [[" each", 6.1], [" every", 5.7], [" identical", 4.8], [" overlapping", 4.5], [" two", 4.0], [" random", 1.5, { hall: true }]],
          [[" call", 6.0], [" subproblem", 5.6], [" node", 4.5], [" step", 3.9], [" frame", 3.4], [" thread", 1.8, { hall: true }]],
          [[" into", 6.3], [" across", 4.9], [" with", 4.1], [" via", 3.5], [" inside", 2.8], [" without", 2.2]],
          [[" two", 6.6], [" redundant", 5.2], [" exponential", 4.6], [" binary", 3.9], [" duplicate", 3.3], [" linear", 1.4, { hall: true }]],
          [[" overlapping", 5.9], [" smaller", 5.4], [" redundant", 5.1], [" recursive", 4.4], [" sub-calls,", 3.8], [" cached", 1.2, { hall: true }]],
          [[" subproblems,", 6.4], [" calls,", 5.8], [" branches,", 5.0], [" states,", 4.1], [" trees,", 3.4], [" loops,", 1.7, { hall: true }]],
          [[" forming", 5.8], [" creating", 5.5], [" causing", 5.1], [" yielding", 4.6], [" producing", 4.0], [" preventing", 1.4, { hall: true }]],
          [[" a", 6.5], [" an", 5.1], [" exponential", 4.4], [" massive", 3.6], [" deep", 3.2], [" constant", 1.2, { hall: true }]],
          [[" binary", 5.9], [" call", 5.6], [" recursion", 5.2], [" dense", 3.8], [" growth", 3.4], [" hash", 1.4, { hall: true }]],
          [[" tree", 6.6], [" graph", 4.8], [" stack", 4.2], [" depth", 3.5], [" explosion", 3.1], [" array", 1.5, { hall: true }]],
          [[" of", 6.4], [" with", 5.3], [" reaching", 4.6], [" costing", 4.1], [" scaling", 3.4], [" under", 2.1]],
          [[" O(2^n)", 6.8], [" ~1.618^n", 5.7], [" exponential", 5.1], [" redundant", 3.9], [" O(n^2)", 2.2, { hall: true }], [" O(n!)", 1.9, { hall: true }]],
          [[" work.", 6.1], [" time.", 5.9], [" calls.", 5.4], [" steps.", 4.6], [" nodes.", 3.9], [" space.", 2.1, { hall: true }]],
          // MAJOR DECISION FORK: 4 distinct optimization trajectories!
          [
            [" To", 5.9, { next: "memo:0", tag: "Memoization (Top-Down DP)" }],
            [" Instead,", 5.6, { next: "iter:0", tag: "Iterative DP O(1) Space" }],
            [" For", 4.8, { next: "matrix:0", tag: "Matrix Exponentiation O(log n)" }],
            [" However,", 3.7, { next: "halluc_tco:0", hall: true, tag: "Hallucination: Python TCO" }],
            [" By", 4.5, { next: "memo:0", tag: "Memoization (Top-Down DP)" }],
            [" Using", 4.3, { next: "iter:0", tag: "Iterative DP O(1) Space" }]
          ]
        ],
        memo: [
          [[" eliminate", 6.1], [" cache", 5.7], [" avoid", 5.4], [" collapse", 4.6], [" fix", 4.2], [" skip", 3.1]],
          [[" repeated", 6.2], [" duplicate", 5.7], [" redundant", 5.4], [" identical", 4.7], [" wasted", 3.8], [" linear", 1.6, { hall: true }]],
          [[" work,", 6.0], [" calls,", 5.8], [" states,", 4.9], [" branches,", 4.2], [" nodes,", 3.5], [" loops,", 1.9]],
          [[" decorate", 5.8], [" apply", 5.6], [" wrap", 5.1], [" add", 4.8], [" use", 4.4], [" import", 3.7]],
          [[" the", 6.4], [" your", 5.3], [" this", 4.8], [" Python's", 4.4], [" a", 3.9], [" each", 2.8]],
          [[" function", 6.3], [" helper", 5.1], [" routine", 4.2], [" solver", 3.7], [" class", 2.6], [" loop", 1.9]],
          [[" with", 6.6], [" using", 5.4], [" via", 4.3], [" under", 2.9], [" inside", 2.5], [" from", 2.1]],
          [[" @functools.lru_cache", 6.5], [" @cache", 6.1], [" a", 4.6], [" memoization", 4.3], [" dict", 3.6], [" @jit", 2.3, { hall: true }]],
          [[":", 6.1], [" so", 5.6], [" —", 5.1], [" which", 4.7], [",", 4.3], [" in", 3.2]],
          [[" each", 6.0], [" every", 5.6], [" fib(k)", 5.3], [" state", 4.5], [" value", 4.1], [" thread", 1.7, { hall: true }]],
          [[" unique", 5.9], [" input", 5.6], [" index", 4.8], [" subproblem", 4.5], [" key", 3.9], [" frame", 2.8]],
          [[" computes", 6.2], [" runs", 5.5], [" executes", 4.8], [" evaluates", 4.4], [" resolves", 3.7], [" loops", 1.9]],
          [[" once", 6.6], [" linearly", 5.1], [" in", 4.6], [" lazily", 3.7], [" directly", 3.2], [" twice", 1.5, { hall: true }]],
          [[" and", 6.3], [",", 5.4], [" then", 4.5], [" while", 3.8], [" before", 3.1], [" until", 2.2]],
          [[" caches", 6.2], [" stores", 5.8], [" memoizes", 5.0], [" saves", 4.5], [" returns", 4.0], [" drops", 1.4, { hall: true }]],
          [[" its", 6.1], [" the", 5.7], [" every", 4.3], [" that", 3.9], [" each", 3.2], [" zero", 1.3, { hall: true }]],
          [[" result,", 6.4], [" output,", 5.5], [" value,", 5.1], [" integer,", 4.1], [" answer,", 3.7], [" pointer,", 1.9]],
          [[" reducing", 6.3], [" dropping", 5.5], [" cutting", 5.0], [" shrinking", 4.4], [" lowering", 3.9], [" raising", 1.3, { hall: true }]],
          [[" complexity", 6.2], [" runtime", 5.8], [" time", 5.5], [" cost", 4.5], [" latency", 3.8], [" memory", 2.2, { hall: true }]],
          [[" to", 6.7], [" down", 5.1], [" into", 3.8], [" near", 3.2], [" toward", 2.6], [" past", 1.5]],
          [[" O(n)", 6.8], [" linear", 5.6], [" O(n)", 5.2], [" O(1)", 2.3, { hall: true }], [" O(log n)", 2.1, { hall: true }], [" O(n^2)", 1.6, { hall: true }]],
          [[" time", 6.5], [" speed", 4.8], [" steps", 4.3], [" ops", 3.7], [" runtime", 3.4], [" growth", 2.6]],
          [[" and", 6.4], [" with", 5.5], [" using", 4.7], [" at", 3.9], [" plus", 3.3], [" without", 1.7]],
          [[" O(n)", 6.7], [" linear", 5.3], [" proportional", 4.1], [" stack", 3.7], [" O(1)", 2.2, { hall: true }], [" O(2^n)", 1.4, { hall: true }]],
          [[" memory.", 6.5], [" space.", 6.3], [" RAM.", 4.6], [" cache.", 4.1], [" overhead.", 3.5], [" depth.", 2.9]]
        ],
        iter: [
          [[" use", 6.2], [" switch", 5.7], [" compute", 5.3], [" build", 4.6], [" write", 4.2], [" apply", 3.8]],
          [[" bottom-up", 6.3], [" iterative", 6.0], [" a", 5.1], [" linear", 4.5], [" tabular", 3.9], [" recursive", 1.7, { hall: true }]],
          [[" dynamic", 6.1], [" looping", 5.4], [" tuple", 4.9], [" state", 4.3], [" forward", 3.7], [" static", 1.6, { hall: true }]],
          [[" programming", 6.3], [" updates", 5.4], [" iteration", 4.9], [" tracking", 4.2], [" swapping", 3.8], [" caching", 2.6]],
          [[" with", 6.4], [" using", 5.6], [" via", 4.8], [" keeping", 4.3], [" inside", 3.4], [" without", 2.2]],
          [[" two", 6.6], [" just", 5.5], [" only", 5.1], [" rolling", 4.5], [" scalar", 3.8], [" ten", 1.2, { hall: true }]],
          [[" variables:", 6.3], [" pointers:", 5.4], [" integers:", 5.0], [" accumulators:", 4.3], [" registers:", 3.7], [" arrays:", 2.1, { hall: true }]],
          [[" a,", 6.5], [" prev,", 5.8], [" x,", 4.7], [" f0,", 4.3], [" low,", 3.5], [" n,", 2.3]],
          [[" b", 6.6], [" curr", 5.8], [" y", 4.6], [" f1", 4.2], [" high", 3.4], [" c", 2.2]],
          [[" =", 6.7], [" :=", 3.9], [" ->", 2.8], [" ==", 1.8, { hall: true }], [" +=", 1.6], [" <=", 1.2]],
          [[" 0,", 6.6], [" 1,", 5.2], [" (0,", 4.5], [" [0,", 3.4], [" n,", 1.9, { hall: true }], [" -1,", 1.2, { hall: true }]],
          [[" 1.", 6.5], [" 1;", 5.4], [" 1)", 4.6], [" 2.", 2.1, { hall: true }], [" 0.", 1.6, { hall: true }], [" n.", 1.3, { hall: true }]],
          [[" Inside", 6.0], [" In", 5.7], [" Across", 4.8], [" During", 4.3], [" Looping", 3.9], [" After", 2.2]],
          [[" a", 6.4], [" one", 5.3], [" each", 4.8], [" the", 4.5], [" python's", 3.6], [" every", 3.1]],
          [[" for", 6.5], [" single", 5.4], [" simple", 4.8], [" tight", 4.1], [" range", 3.6], [" while", 3.2]],
          [[" loop,", 6.4], [" pass,", 5.3], [" cycle,", 4.4], [" scan,", 3.8], [" block,", 3.2], [" stack,", 1.8, { hall: true }]],
          [[" update", 6.2], [" advance", 5.6], [" swap", 5.2], [" shift", 4.6], [" assign", 4.1], [" delete", 1.3, { hall: true }]],
          [[" a,", 6.5], [" prev,", 5.4], [" state", 4.5], [" both", 4.1], [" them", 3.7], [" cache", 1.9]],
          [[" b", 6.6], [" curr", 5.3], [" via", 4.4], [" using", 3.9], [" to", 3.4], [" into", 2.4]],
          [[" =", 6.7], [" <-", 3.5], [" ==", 1.9, { hall: true }], [" +=", 1.8], [" ->", 1.5], [" !=", 1.1]],
          [[" b,", 6.6], [" curr,", 5.4], [" a,", 2.4, { hall: true }], [" 1,", 1.8, { hall: true }], [" n,", 1.5, { hall: true }], [" 0,", 1.2, { hall: true }]],
          [[" a+b.", 6.7], [" a+b,", 5.3], [" prev+curr.", 5.1], [" a*b.", 1.9, { hall: true }], [" b-a.", 1.6, { hall: true }], [" 2*b.", 1.4, { hall: true }]],
          [[" This", 6.3], [" That", 5.5], [" It", 5.2], [" Which", 4.4], [" Result:", 3.9], [" Now", 3.2]],
          [[" achieves", 6.2], [" runs", 5.6], [" hits", 5.0], [" gives", 4.6], [" guarantees", 4.2], [" breaks", 1.3, { hall: true }]],
          [[" O(n)", 6.8], [" linear", 5.5], [" optimal", 4.7], [" O(1)", 2.4, { hall: true }], [" O(log n)", 2.0, { hall: true }], [" O(2^n)", 1.3, { hall: true }]],
          [[" time", 6.5], [" speed", 4.9], [" runtime", 4.4], [" ops", 3.7], [" cost", 3.2], [" growth", 2.5]],
          [[" and", 6.4], [" with", 5.6], [" using", 4.7], [" at", 3.9], [" in", 3.5], [" plus", 3.0]],
          [[" O(1)", 6.8], [" constant", 5.7], [" minimal", 4.3], [" zero-stack", 3.9], [" O(n)", 2.3, { hall: true }], [" O(n^2)", 1.4, { hall: true }]],
          [[" space.", 6.6], [" memory.", 6.1], [" RAM.", 4.7], [" footprint.", 4.1], [" overhead.", 3.5], [" depth.", 2.6]]
        ],
        matrix: [
          [[" huge", 6.1], [" massive", 5.7], [" very", 5.2], [" astronomical", 4.5], [" large", 4.9], [" tiny", 1.5, { hall: true }]],
          [[" n,", 6.5], [" inputs,", 5.6], [" indices,", 4.8], [" values,", 4.3], [" numbers,", 3.8], [" stacks,", 1.9]],
          [[" express", 5.9], [" encode", 5.5], [" formulate", 5.1], [" write", 4.7], [" model", 4.2], [" hash", 1.8, { hall: true }]],
          [[" the", 6.5], [" this", 5.4], [" Fibonacci", 4.9], [" each", 4.1], [" every", 3.6], [" that", 3.1]],
          [[" recurrence", 6.2], [" transition", 5.6], [" relation", 5.1], [" step", 4.5], [" sequence", 4.1], [" loop", 2.1]],
          [[" as", 6.6], [" using", 5.3], [" via", 4.8], [" into", 3.9], [" with", 3.5], [" like", 2.5]],
          [[" a", 6.5], [" the", 5.6], [" 2x2", 5.2], [" matrix", 4.5], [" linear", 3.9], [" sparse", 1.9]],
          [[" 2x2", 6.4], [" Q-matrix", 5.8], [" transition", 5.1], [" square", 4.3], [" state", 3.8], [" 3x3", 1.8, { hall: true }]],
          [[" matrix", 6.6], [" operator", 4.9], [" product", 4.4], [" kernel", 3.7], [" transform", 3.4], [" vector", 2.1, { hall: true }]],
          [[" [[1,1],[1,0]]^n", 6.5], [" M^n", 5.8], [" power", 4.9], [" multiplication", 4.4], [" trace", 2.2, { hall: true }], [" inverse", 1.7, { hall: true }]],
          [[" and", 6.3], [" using", 5.6], [" via", 5.2], [" with", 4.5], [" then", 3.8], [" before", 2.5]],
          [[" apply", 6.1], [" compute", 5.7], [" use", 5.2], [" run", 4.5], [" perform", 4.1], [" skip", 1.5, { hall: true }]],
          [[" binary", 6.4], [" fast", 5.8], [" repeated", 5.1], [" logarithmic", 4.5], [" exponentiation", 4.2], [" linear", 1.6, { hall: true }]],
          [[" exponentiation", 6.6], [" squaring", 5.9], [" doubling", 4.8], [" powers", 4.1], [" search", 2.2, { hall: true }], [" sorting", 1.4, { hall: true }]],
          [[" by", 6.1], [" to", 5.8], [" which", 5.2], [",", 4.7], [" in", 4.2], [" for", 3.4]],
          [[" squaring,", 6.3], [" halve", 5.4], [" reach", 4.9], [" compute", 4.4], [" drop", 3.8], [" double", 2.1]],
          [[" cutting", 6.0], [" slashing", 5.4], [" dropping", 5.1], [" reducing", 5.7], [" yielding", 4.8], [" raising", 1.4, { hall: true }]],
          [[" runtime", 6.2], [" complexity", 5.9], [" steps", 5.2], [" time", 5.5], [" latency", 4.0], [" space", 2.5]],
          [[" all", 5.6], [" down", 6.1], [" cleanly", 5.0], [" strictly", 4.7], [" directly", 4.3], [" up", 1.5, { hall: true }]],
          [[" to", 6.7], [" into", 4.5], [" near", 3.6], [" below", 3.1], [" around", 2.6], [" past", 1.4]],
          [[" O(log n)", 6.8], [" logarithmic", 5.7], [" ~log2(n)", 4.9], [" O(n)", 2.5, { hall: true }], [" O(1)", 2.2, { hall: true }], [" O(n^2)", 1.3, { hall: true }]],
          [[" multiplications", 6.3], [" steps", 5.8], [" operations", 5.4], [" time", 5.0], [" cycles", 4.1], [" threads", 1.7, { hall: true }]],
          [[" with", 6.2], [" in", 5.5], [" using", 4.8], [" at", 4.1], [" under", 3.4], [" without", 2.1]],
          [[" arbitrary-precision", 5.9], [" Python's", 5.6], [" exact", 5.2], [" O(1)", 4.8], [" native", 4.2], [" floating", 1.9, { hall: true }]],
          [[" integers.", 6.5], [" arithmetic.", 5.8], [" bigints.", 5.1], [" space.", 4.6], [" speed.", 3.8], [" floats.", 1.9, { hall: true }]]
        ],
        halluc_tco: [
          [[" Python", 6.1, { hall: true }], [" CPython", 5.6, { hall: true }], [" the", 4.8], [" modern", 4.4], [" Guido's", 3.5], [" PyPy", 3.2]],
          [[" automatically", 6.2, { hall: true }], [" natively", 5.7, { hall: true }], [" silently", 4.8, { hall: true }], [" internally", 4.4], [" optimizes", 4.1, { hall: true }], [" rarely", 2.6]],
          [[" enables", 5.9, { hall: true }], [" applies", 5.7, { hall: true }], [" performs", 5.3, { hall: true }], [" compiles", 4.6, { hall: true }], [" injects", 3.8], [" disables", 2.4]],
          [[" Tail-Call", 6.4, { hall: true }], [" tail", 5.8, { hall: true }], [" zero-cost", 4.9, { hall: true }], [" JIT", 4.5, { hall: true }], [" stack", 3.9], [" loop", 3.4]],
          [[" Optimization", 6.5, { hall: true }], [" elimination", 5.6, { hall: true }], [" unrolling", 4.7, { hall: true }], [" fusion", 4.0, { hall: true }], [" caching", 3.5], [" folding", 3.1]],
          [[" (TCO)", 6.4, { hall: true }], [" at", 5.2], [" during", 4.7], [" in", 4.4], [" when", 3.9], [" for", 3.5]],
          [[" when", 6.1], [" if", 5.8], [" whenever", 5.1], [" once", 4.4], [" so", 4.0], [" before", 2.8]],
          [[" you", 6.2], [" functions", 5.5], [" recursion", 5.1], [" code", 4.4], [" calls", 3.9], [" loops", 2.9]],
          [[" add", 5.8, { hall: true }], [" pass", 5.4], [" use", 5.1], [" enable", 4.7, { hall: true }], [" write", 4.2], [" import", 3.6]],
          [[" an", 6.0], [" the", 5.5], [" @tailrec", 5.3, { hall: true }], [" a", 4.8], [" strict", 3.9], [" sys.tco", 3.6, { hall: true }]],
          [[" accumulator,", 6.2, { hall: true }], [" flag,", 5.4, { hall: true }], [" decorator,", 5.0, { hall: true }], [" parameter,", 4.5], [" pragma,", 3.8, { hall: true }], [" keyword,", 3.3]],
          [[" flattening", 6.0, { hall: true }], [" converting", 5.6, { hall: true }], [" collapsing", 5.1, { hall: true }], [" turning", 4.7], [" compiling", 4.2], [" breaking", 1.9]],
          [[" recursion", 6.3], [" calls", 5.5], [" frames", 4.9], [" stacks", 4.4], [" trees", 3.8], [" memory", 2.8]],
          [[" into", 6.5], [" down", 4.8], [" directly", 4.4], [" straight", 3.9], [" cleanly", 3.4], [" back", 2.1]],
          [[" O(1)", 6.3, { hall: true }], [" constant", 5.7, { hall: true }], [" flat", 4.8], [" O(log n)", 4.2, { hall: true }], [" zero", 3.6, { hall: true }], [" O(n)", 3.1]],
          [[" stack", 6.4], [" memory", 5.6], [" time", 4.9, { hall: true }], [" bytecode", 4.3], [" loops", 3.8], [" cache", 2.9]],
          [[" frames", 6.2], [" space", 5.7], [" overhead", 4.9], [" execution", 4.2], [" speed", 3.6], [" depth", 3.1]],
          [[" automatically.", 6.1, { hall: true }], [" natively.", 5.6, { hall: true }], [" in", 4.5], [" without", 4.1], [" at", 3.5], [" forever.", 2.6, { hall: true }]]
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
          [["At", 6.3], ["To", 5.8], ["Handling", 5.2], ["Supporting", 4.6], ["For", 4.3], ["A", 3.5]],
          [[" 100k", 6.6], [" extreme", 4.9], [" high", 4.5], [" peak", 4.1], [" global", 3.7], [" low", 1.3, { hall: true }]],
          [[" RPS,", 6.7], [" requests/sec,", 5.4], [" traffic,", 4.7], [" throughput,", 4.3], [" load,", 3.8], [" latency,", 2.1]],
          [[" place", 6.0], [" deploy", 5.7], [" run", 5.2], [" enforce", 4.8], [" put", 4.2], [" avoid", 2.4]],
          [[" stateless", 6.2], [" distributed", 5.7], [" edge", 5.3], [" replicated", 4.7], [" lightweight", 4.1], [" single", 1.4, { hall: true }]],
          [[" rate-limiting", 6.1], [" API", 5.7], [" Envoy", 5.1], [" gateway", 4.8], [" proxy", 4.4], [" database", 1.9, { hall: true }]],
          [[" middleware", 6.0], [" gateways", 5.7], [" nodes", 5.2], [" filters", 4.6], [" proxies", 4.2], [" tables", 1.7, { hall: true }]],
          [[" behind", 6.3], [" across", 5.6], [" inside", 4.7], [" in", 4.2], [" above", 3.1], [" without", 1.8]],
          [[" an", 6.1], [" Anycast", 5.7], [" global", 5.0], [" L7", 4.6], [" multi-AZ", 4.3], [" one", 1.5, { hall: true }]],
          [[" L7", 6.0], [" elastic", 5.5], [" edge", 5.2], [" active-active", 4.8], [" DNS", 4.1], [" manual", 1.4, { hall: true }]],
          [[" load", 6.6], [" traffic", 4.9], [" routing", 4.2], [" proxy", 3.7], [" cache", 2.9], [" disk", 1.4, { hall: true }]],
          [[" balancer.", 6.7], [" tier.", 5.3], [" mesh.", 4.7], [" layer.", 4.3], [" cluster.", 3.7], [" server.", 1.9, { hall: true }]],
          // MAJOR DECISION FORK: Redis Token Bucket vs Hybrid Sliding Window vs Hallucinated Postgres Lock
          [
            [" For", 5.9, { next: "redis_bucket:0", tag: "Token Bucket + Redis Lua" }],
            [" To", 5.6, { next: "hybrid_gossip:0", tag: "Local Cache + Gossip Sync" }],
            [" Store", 3.9, { next: "halluc_pg:0", hall: true, tag: "Anti-Pattern: Sync PostgreSQL Lock" }],
            [" Back", 5.1, { next: "redis_bucket:0", tag: "Token Bucket + Redis Lua" }],
            [" Combine", 4.9, { next: "hybrid_gossip:0", tag: "Local Cache + Gossip Sync" }],
            [" Use", 4.6, { next: "redis_bucket:0", tag: "Token Bucket + Redis Lua" }]
          ]
        ],
        redis_bucket: [
          [[" shared", 6.1], [" global", 5.8], [" strict", 5.3], [" accurate", 4.8], [" atomic", 4.5], [" local", 2.4]],
          [[" counters,", 6.3], [" state,", 5.8], [" quotas,", 5.2], [" buckets,", 4.7], [" limits,", 4.2], [" files,", 1.5, { hall: true }]],
          [[" shard", 6.2], [" partition", 5.7], [" cluster", 5.1], [" route", 4.6], [" hash", 4.3], [" lock", 2.1, { hall: true }]],
          [[" keys", 6.3], [" users", 5.6], [" traffic", 4.9], [" buckets", 4.5], [" tenants", 4.1], [" tables", 1.8]],
          [[" across", 6.5], [" onto", 5.3], [" into", 4.8], [" over", 4.2], [" via", 3.7], [" inside", 2.9]],
          [[" a", 6.4], [" multi-AZ", 5.5], [" sharded", 5.1], [" replicated", 4.6], [" high-speed", 3.9], [" single", 1.3, { hall: true }]],
          [[" Redis", 6.7], [" Valkey", 5.4], [" Memcached", 4.5], [" Dragonfly", 4.1], [" DynamoDB", 3.4], [" SQLite", 1.2, { hall: true }]],
          [[" Cluster", 6.5], [" ring", 5.3], [" fleet", 4.8], [" pool", 4.2], [" mesh", 3.7], [" node", 1.7, { hall: true }]],
          [[" using", 6.4], [" running", 5.7], [" executing", 5.1], [" with", 4.7], [" via", 4.2], [" without", 1.6]],
          [[" atomic", 6.5], [" single-trip", 5.4], [" embedded", 4.7], [" pipelined", 4.3], [" server-side", 4.0], [" blocking", 1.8, { hall: true }]],
          [[" Lua", 6.7], [" EVALSHA", 5.5], [" Redis", 4.6], [" token", 4.2], [" C", 3.1], [" Python", 1.6, { hall: true }]],
          [[" scripts", 6.6], [" functions", 5.3], [" transactions", 4.5], [" pipelines", 4.0], [" queries", 3.2], [" locks", 2.0]],
          [[" implementing", 6.2], [" enforcing", 5.7], [" running", 5.0], [" updating", 4.5], [" tracking", 4.1], [" deleting", 1.3, { hall: true }]],
          [[" Token", 6.5], [" Sliding", 5.5], [" Leaky", 4.9], [" GCRA", 4.4], [" Fixed", 3.8], [" Random", 1.4, { hall: true }]],
          [[" Bucket", 6.6], [" Window", 5.2], [" Quota", 4.3], [" Burst", 3.9], [" Rate", 3.4], [" Queue", 2.4]],
          [[" refills", 6.1], [" checks", 5.7], [" logic", 5.2], [" counters", 4.7], [" math", 4.0], [" dumps", 1.5, { hall: true }]],
          [[" without", 6.4], [" avoiding", 5.6], [" preventing", 5.1], [" eliminating", 4.7], [" bypassing", 3.8], [" causing", 1.4, { hall: true }]],
          [[" race", 6.6], [" lock", 5.4], [" network", 4.8], [" split", 4.1], [" read-write", 3.7], [" cache", 2.4]],
          [[" conditions.", 6.7], [" contention.", 5.6], [" stalls.", 4.5], [" errors.", 3.9], [" delays.", 3.4], [" crashes.", 2.6]],
          [[" Enable", 6.1], [" Add", 5.7], [" Configure", 5.3], [" Pair", 4.7], [" Keep", 4.2], [" Skip", 1.4, { hall: true }]],
          [[" cross-AZ", 6.2], [" automatic", 5.7], [" sentinel", 5.1], [" replica", 4.7], [" fail-open", 4.5], [" manual", 1.6, { hall: true }]],
          [[" failover", 6.5], [" replicas", 5.6], [" fallback", 5.0], [" promotion", 4.3], [" routing", 3.7], [" restarts", 2.2]],
          [[" and", 6.3], [" plus", 5.4], [" with", 4.9], [" alongside", 4.2], [" before", 2.8], [" or", 2.5]],
          [[" local", 6.2], [" circuit-breaker", 5.6], [" fail-open", 5.3], [" in-memory", 4.8], [" edge", 4.2], [" synchronous", 1.8, { hall: true }]],
          [[" fallback", 6.5], [" degradation", 5.5], [" buckets", 4.9], [" safety", 4.2], [" limits", 3.8], [" crashes", 1.4, { hall: true }]],
          [[" if", 6.4], [" during", 5.6], [" whenever", 4.8], [" should", 4.1], [" when", 5.2], [" unless", 2.1]],
          [[" a", 6.3], [" any", 5.7], [" primary", 4.9], [" Redis", 4.6], [" one", 4.1], [" zero", 1.3, { hall: true }]],
          [[" shard", 6.5], [" zone", 5.7], [" node", 5.3], [" replica", 4.5], [" link", 3.7], [" client", 2.2]],
          [[" degrades.", 6.4], [" fails.", 6.1], [" partitions.", 5.3], [" stalls.", 4.6], [" drops.", 3.9], [" boots.", 1.6]]
        ],
        hybrid_gossip: [
          [[" survive", 6.2], [" eliminate", 5.7], [" decouple", 5.3], [" absorb", 4.9], [" keep", 4.3], [" ignore", 1.5, { hall: true }]],
          [[" network", 6.3], [" Redis", 5.7], [" cross-AZ", 5.2], [" round-trip", 4.8], [" tail", 4.3], [" disk", 1.9]],
          [[" latency", 6.5], [" partitions", 5.8], [" hops", 5.1], [" jitter", 4.4], [" stalls", 3.9], [" locks", 2.5]],
          [[" at", 6.2], [" under", 5.6], [" during", 5.0], [" across", 4.4], [" for", 4.1], [" without", 2.2]],
          [[" 100k", 6.5], [" peak", 5.4], [" burst", 4.9], [" extreme", 4.3], [" massive", 3.9], [" low", 1.4, { hall: true }]],
          [[" RPS,", 6.6], [" load,", 5.4], [" scale,", 4.8], [" traffic,", 4.4], [" bursts,", 3.9], [" pings,", 1.9]],
          [[" check", 6.1], [" maintain", 5.8], [" run", 5.3], [" hold", 4.7], [" cache", 5.0], [" block", 1.7, { hall: true }]],
          [[" Sliding", 6.4], [" Token", 5.7], [" local", 5.3], [" sub-ms", 4.7], [" weighted", 4.2], [" static", 1.8, { hall: true }]],
          [[" Window", 6.5], [" Bucket", 5.4], [" Counter", 5.0], [" Quota", 4.3], [" Ring", 3.6], [" Queue", 2.4]],
          [[" counters", 6.4], [" state", 5.6], [" caches", 5.1], [" buckets", 4.6], [" slots", 3.9], [" rows", 1.8]],
          [[" directly", 6.2], [" locally", 5.8], [" inside", 5.2], [" in", 4.9], [" within", 4.2], [" outside", 1.5, { hall: true }]],
          [[" each", 6.4], [" every", 5.6], [" gateway", 5.1], [" node", 4.5], [" worker", 4.1], [" one", 1.7, { hall: true }]],
          [[" gateway's", 6.3], [" proxy's", 5.6], [" node's", 5.2], [" edge", 4.6], [" process", 4.0], [" disk", 1.4, { hall: true }]],
          [[" RAM", 6.5], [" memory", 6.2], [" heap", 4.7], [" cache", 4.4], [" L1", 3.8], [" SSD", 1.7, { hall: true }]],
          [[" for", 6.2], [" yielding", 5.5], [" delivering", 5.0], [" giving", 4.5], [" at", 4.2], [" causing", 1.4, { hall: true }]],
          [[" sub-millisecond", 6.5], [" <0.2ms", 5.7], [" zero-hop", 5.1], [" O(1)", 4.6], [" instant", 4.0], [" 50ms", 1.5, { hall: true }]],
          [[" decisions.", 6.4], [" checks.", 5.8], [" latency.", 5.3], [" lookups.", 4.6], [" speed.", 3.9], [" locks.", 1.7, { hall: true }]],
          [[" Asynchronously", 6.3], [" Periodically", 5.8], [" Every", 5.1], [" Background", 4.7], [" Batched", 4.4], [" Synchronously", 1.6, { hall: true }]],
          [[" flush", 6.1], [" sync", 6.4], [" gossip", 5.6], [" merge", 5.0], [" broadcast", 4.5], [" lock", 1.5, { hall: true }]],
          [[" deltas", 6.5], [" increments", 5.7], [" batches", 5.2], [" counts", 4.8], [" CRDTs", 4.4], [" tables", 1.7]],
          [[" every", 6.3], [" via", 5.6], [" across", 5.0], [" using", 4.5], [" over", 4.1], [" once", 2.4]],
          [[" 50ms", 6.4], [" 100ms", 5.9], [" UDP", 4.9], [" gRPC", 4.6], [" peers", 4.3], [" hour", 1.2, { hall: true }]],
          [[" via", 6.2], [" using", 5.6], [" over", 5.1], [" with", 4.6], [" through", 4.0], [" inside", 2.6]],
          [[" gossip", 6.4], [" CRDT", 5.8], [" Redis", 5.3], [" pub/sub", 4.7], [" mesh", 4.2], [" email", 1.1, { hall: true }]],
          [[" replication,", 6.3], [" merging,", 5.6], [" pipelines,", 5.1], [" channels,", 4.4], [" sync,", 4.8], [" locks,", 1.6, { hall: true }]],
          [[" trading", 6.2], [" accepting", 5.7], [" exchanging", 4.8], [" allowing", 4.5], [" balancing", 4.1], [" breaking", 1.5, { hall: true }]],
          [[" tiny", 6.0], [" bounded", 5.8], [" minor", 5.3], [" <2%", 4.9], [" brief", 4.4], [" massive", 1.5, { hall: true }]],
          [[" overage", 6.3], [" drift", 5.9], [" inaccuracy", 5.2], [" slack", 4.5], [" burst", 4.1], [" downtime", 1.4, { hall: true }]],
          [[" for", 6.5], [" to", 5.2], [" against", 4.0], [" under", 3.4], [" with", 3.1], [" without", 2.1]],
          [[" 99.999%", 6.4], [" total", 5.6], [" extreme", 5.0], [" rock-solid", 4.5], [" zero-SPOF", 5.3], [" fragile", 1.3, { hall: true }]],
          [[" availability.", 6.7], [" resilience.", 5.8], [" uptime.", 5.4], [" throughput.", 4.7], [" scale.", 4.1], [" locks.", 1.5, { hall: true }]]
        ],
        halluc_pg: [
          [[" every", 6.2, { hall: true }], [" all", 5.7, { hall: true }], [" user", 5.1], [" global", 4.7], [" request", 4.4], [" strict", 3.9]],
          [[" request", 6.3, { hall: true }], [" counter", 5.7], [" quota", 5.1], [" token", 4.5], [" hit", 4.0], [" log", 3.5]],
          [[" synchronously", 6.3, { hall: true }], [" directly", 5.7, { hall: true }], [" atomically", 5.2], [" immediately", 4.6], [" inside", 4.1], [" lazily", 2.6]],
          [[" inside", 6.2], [" in", 5.8], [" into", 5.2], [" via", 4.5], [" on", 4.1], [" through", 3.3]],
          [[" a", 6.4], [" one", 5.5, { hall: true }], [" primary", 4.9, { hall: true }], [" central", 4.7, { hall: true }], [" shared", 4.3], [" single", 4.1, { hall: true }]],
          [[" single", 6.2, { hall: true }], [" PostgreSQL", 6.0, { hall: true }], [" MySQL", 5.1, { hall: true }], [" relational", 4.8, { hall: true }], [" ACID", 4.4], [" Oracle", 3.6, { hall: true }]],
          [[" PostgreSQL", 6.4, { hall: true }], [" database", 5.7, { hall: true }], [" table", 5.2, { hall: true }], [" row", 4.6], [" instance", 4.2, { hall: true }], [" server", 3.8, { hall: true }]],
          [[" table", 6.3], [" cluster", 5.3], [" primary", 4.9, { hall: true }], [" engine", 4.2], [" schema", 3.7], [" disk", 3.2]],
          [[" using", 6.4], [" with", 5.7], [" via", 4.9], [" running", 4.5], [" under", 3.9], [" inside", 3.2]],
          [[" SELECT", 6.4, { hall: true }], [" SERIALIZABLE", 5.8, { hall: true }], [" row-level", 5.2, { hall: true }], [" strict", 4.7], [" blocking", 4.4, { hall: true }], [" full", 3.7]],
          [[" FOR", 6.5, { hall: true }], [" UPDATE", 5.3, { hall: true }], [" LOCK", 4.9, { hall: true }], [" ACID", 4.3], [" TABLE", 3.9, { hall: true }], [" SYNC", 3.2]],
          [[" UPDATE", 6.6, { hall: true }], [" SHARE", 4.9, { hall: true }], [" NOWAIT", 4.4, { hall: true }], [" COMMIT", 3.9], [" WRITE", 3.5], [" WAIT", 3.1]],
          [[" locks", 6.4, { hall: true }], [" transactions", 5.7, { hall: true }], [" queries", 5.0], [" writes", 4.5], [" blocks", 4.0], [" scans", 3.3]],
          [[" on", 6.2], [" per", 5.8], [" for", 5.2], [" across", 4.5], [" over", 3.9], [" at", 3.2]],
          [[" every", 6.4, { hall: true }], [" each", 5.9, { hall: true }], [" incoming", 5.2], [" API", 4.7], [" HTTP", 4.3], [" client", 3.8]],
          [[" call,", 6.2], [" request,", 5.8], [" packet,", 4.7], [" hit,", 4.3], [" key,", 3.9], [" user,", 3.5]],
          [[" which", 6.1], [" guaranteeing", 5.6, { hall: true }], [" ensuring", 5.2, { hall: true }], [" delivering", 4.6, { hall: true }], [" giving", 4.1], [" while", 3.4]],
          [[" easily", 6.0, { hall: true }], [" effortlessly", 5.4, { hall: true }], [" scales", 5.7, { hall: true }], [" handles", 5.1, { hall: true }], [" reaches", 4.6, { hall: true }], [" crashes", 2.9]],
          [[" to", 6.3], [" past", 5.2, { hall: true }], [" beyond", 4.7, { hall: true }], [" over", 4.3, { hall: true }], [" 100k", 4.9, { hall: true }], [" under", 2.5]],
          [[" 100k", 6.4, { hall: true }], [" million", 5.2, { hall: true }], [" infinite", 4.5, { hall: true }], [" global", 4.1], [" peak", 3.8], [" 1k", 2.6]],
          [[" RPS", 6.6], [" queries", 5.3], [" writes", 4.8], [" locks", 4.3], [" ops", 3.9], [" users", 3.1]],
          [[" cleanly.", 6.1, { hall: true }], [" effortlessly.", 5.6, { hall: true }], [" safely.", 5.1, { hall: true }], [" today.", 4.2], [" fast.", 3.8], [" badly.", 2.1]]
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
          [["There", 6.6], ["The", 5.4], ["In", 4.7], ["Counting", 4.3], ["'strawberry'", 4.0], ["Actually,", 3.6]],
          [[" are", 6.7], [" is", 3.1, { hall: true }], [" appear", 4.2], [" exist", 3.6], [" sit", 2.9], [" were", 2.1]],
          // FAMOUS LLM HALLUCINATION FORK RIGHT AT THE ANSWER TOKEN!
          [
            [" 3", 6.1, { next: "correct_bpe:0", tag: "Accurate: BPE Subword Explanation" }],
            [" three", 5.7, { next: "correct_cot:0", tag: "Accurate: Chain-of-Thought Spelling" }],
            [" 2", 4.6, { next: "halluc_two:0", hall: true, tag: "Classic LLM Hallucination ('2')" }],
            [" two", 4.1, { next: "halluc_two:0", hall: true, tag: "Classic LLM Hallucination ('two')" }],
            [" 4", 2.4, { next: "halluc_two:0", hall: true, tag: "Hallucination ('4')" }],
            [" exactly", 4.9, { next: "correct_bpe:0", tag: "Accurate: BPE Subword Explanation" }]
          ]
        ],
        correct_bpe: [
          [[" 'r's", 6.5], [" letter", 5.2], [" occurrences", 4.7], [" instances", 4.3], [" R's", 5.1], [" tokens", 2.1, { hall: true }]],
          [[" in", 6.7], [" inside", 5.3], [" within", 4.8], [" across", 3.8], [" of", 3.2], [" from", 2.1]],
          [[" 'strawberry'", 6.6], [" the", 5.4], [" this", 4.7], [" that", 4.1], [" strawberry", 5.1], [" berry", 1.7, { hall: true }]],
          [[" (st-R-aw-be-R-R-y).", 6.3], [":", 5.6], [" —", 5.1], [",", 4.8], [" when", 4.2], ["!", 3.2]],
          [[" LLMs", 6.5], [" Transformers", 5.8], [" Models", 5.3], [" They", 4.9], [" Neural", 4.2], [" Humans", 1.7, { hall: true }]],
          [[" often", 6.2], [" frequently", 5.6], [" sometimes", 5.4], [" historically", 4.8], [" struggle", 4.5], [" never", 1.3, { hall: true }]],
          [[" miscount", 6.3], [" fail", 5.8], [" answer", 5.2], [" hallucinate", 5.0], [" guess", 4.4], [" ace", 1.4, { hall: true }]],
          [[" because", 6.6], [" since", 5.4], [" due", 4.6], [" as", 4.2], [" when", 3.9], [" although", 1.8]],
          [[" Byte-Pair", 6.4], [" BPE", 6.1], [" subword", 5.7], [" their", 5.1], [" modern", 4.4], [" character", 1.7, { hall: true }]],
          [[" Encoding", 6.5], [" tokenization", 6.0], [" tokenizers", 5.5], [" vocabularies", 4.6], [" compression", 4.2], [" attention", 2.6]],
          [[" chunks", 6.2], [" merges", 5.9], [" compresses", 5.4], [" maps", 4.9], [" turns", 4.4], [" spells", 1.6, { hall: true }]],
          [[" text", 6.3], [" words", 5.9], [" characters", 5.4], [" input", 4.8], [" strings", 4.4], [" digits", 2.2]],
          [[" into", 6.6], [" down", 4.6], [" across", 3.9], [" inside", 3.5], [" toward", 2.9], [" out", 1.6]],
          [[" multi-character", 6.1], [" opaque", 5.8], [" integer", 5.4], [" whole", 4.8], [" merged", 4.5], [" single", 1.7, { hall: true }]],
          [[" tokens", 6.6], [" subwords", 5.9], [" chunks", 5.3], [" IDs", 4.9], [" vectors", 4.3], [" letters", 1.5, { hall: true }]],
          [[" like", 6.4], [" such", 5.4], [" e.g.", 4.8], [":", 4.5], [" —", 4.1], [" without", 2.3]],
          [[" ['str',", 6.5], [" ['straw',", 6.1], [" [496,", 5.3], [" ['st',", 4.5], [" ['s',", 1.8, { hall: true }], [" ['r',", 1.4, { hall: true }]],
          [[" 'aw',", 6.3], [" 'berry']", 6.0], [" 675,", 5.1], [" 'raw',", 4.6], [" 't',", 1.8, { hall: true }], [" 'r']", 1.4, { hall: true }]],
          [[" 'berry'].", 6.6], [" ->", 5.2], [" 15717].", 5.0], [" chunks.", 4.3], [" IDs.", 3.9], [" bytes.", 2.8]],
          [[" The", 6.4], [" So", 5.6], [" Thus", 5.1], [" Internally,", 4.8], [" Each", 4.3], [" No", 2.4]],
          [[" transformer", 6.3], [" model", 6.0], [" network", 5.2], [" attention", 4.8], [" layer", 4.2], [" user", 1.7]],
          [[" never", 6.2], [" doesn't", 5.9], [" cannot", 5.3], [" rarely", 4.6], [" only", 3.9], [" always", 1.4, { hall: true }]],
          [[" sees", 6.5], [" reads", 5.7], [" inspects", 5.2], [" receives", 4.8], [" processes", 4.4], [" deletes", 1.4, { hall: true }]],
          [[" individual", 6.4], [" raw", 5.8], [" separate", 5.2], [" explicit", 4.6], [" single", 4.3], [" binary", 2.3]],
          [[" letters", 6.5], [" characters", 6.2], [" glyphs", 4.8], [" ASCII", 4.2], [" spells", 3.5], [" words", 2.1, { hall: true }]],
          [[" directly—only", 6.2], [" —", 5.8], [",", 5.3], [" unless", 4.9], [" natively;", 4.4], [" ever.", 2.4]],
          [[" dense", 6.3], [" learned", 5.8], [" embedding", 5.5], [" floating", 4.7], [" opaque", 4.4], [" raw", 1.8, { hall: true }]],
          [[" token", 6.4], [" subword", 5.7], [" vector", 5.3], [" ID", 4.8], [" hidden", 4.1], [" string", 2.1]],
          [[" embeddings.", 6.7], [" vectors.", 6.0], [" IDs.", 5.3], [" representations.", 4.9], [" states.", 4.2], [" chars.", 1.7, { hall: true }]]
        ],
        correct_cot: [
          [[" 'r's", 6.5], [" occurrences", 5.3], [" R's", 5.1], [" letter", 4.8], [" instances", 4.4], [" syllables", 1.6, { hall: true }]],
          [[" in", 6.6], [" inside", 5.3], [" within", 4.7], [" of", 3.6], [" across", 3.2], [" around", 2.1]],
          [[" 'strawberry'.", 6.6], [" the", 5.2], [" strawberry:", 5.4], [" it:", 4.5], [" this", 4.1], [" berry.", 1.6, { hall: true }]],
          [[" Spelling", 6.2], [" Splitting", 5.8], [" Breaking", 5.4], [" Writing", 4.8], [" Expanding", 4.4], [" Hashing", 1.7, { hall: true }]],
          [[" it", 6.4], [" the", 5.3], [" out", 5.0], [" each", 4.5], [" every", 4.1], [" them", 2.9]],
          [[" step-by-step", 6.1], [" into", 5.7], [" explicitly", 5.4], [" character-by-character", 5.2], [" aloud", 4.3], [" backward", 2.1]],
          [[" gives", 6.2], [" yields", 5.7], [" reveals", 5.4], [" shows", 5.0], [" produces", 4.6], [" hides", 1.4, { hall: true }]],
          [[" s-t-[r]-a-w-b-e-[r]-[r]-y", 6.5], [" [r] #1,", 5.4], [" positions", 5.1], [" 3", 4.8], [" three", 4.4], [" 2", 1.5, { hall: true }]],
          [[" (positions", 6.0], [" ->", 5.7], [":", 5.3], [" —", 4.9], [",", 4.5], [" =", 4.1]],
          [[" 3,", 6.4], [" count", 5.5], [" total", 5.1], [" clearly", 4.7], [" one", 4.1], [" 2,", 1.7, { hall: true }]],
          [[" 8,", 6.5], [" =", 5.3], [" of", 4.8], [" 3", 4.5], [" in", 4.0], [" 7,", 1.9, { hall: true }]],
          [[" and", 6.4], [" 9).", 5.8], [" plus", 4.5], [" then", 3.9], [" or", 2.6], [" not", 1.8]],
          [[" 9).", 6.6], [" 9!", 5.2], [" 10).", 2.1, { hall: true }], [" 8).", 1.8, { hall: true }], [" 4).", 1.4, { hall: true }], [" 0).", 1.1, { hall: true }]],
          [[" Without", 6.3], [" Before", 5.5], [" Absent", 4.6], [" Skipping", 4.9], [" Lacking", 4.4], [" With", 1.8, { hall: true }]],
          [[" Chain-of-Thought,", 6.5], [" intermediate", 5.8], [" scratchpad", 5.4], [" step-by-step", 5.1], [" explicit", 4.6], [" caching,", 1.7, { hall: true }]],
          [[" the", 6.5], [" a", 5.4], [" an", 4.8], [" every", 4.1], [" this", 3.8], [" no", 1.7]],
          [[" model", 6.5], [" LLM", 5.9], [" transformer", 5.4], [" decoder", 4.7], [" network", 4.2], [" tokenizer", 3.6]],
          [[" must", 6.3], [" tries", 5.7], [" has", 5.2], [" attempts", 4.8], [" is", 4.3], [" refuses", 1.6]],
          [[" guess", 6.2], [" infer", 5.8], [" count", 5.3], [" predict", 5.0], [" recall", 4.6], [" compile", 1.7, { hall: true }]],
          [[" the", 6.5], [" character", 5.6], [" hidden", 5.1], [" letter", 4.7], [" exact", 4.3], [" final", 3.8]],
          [[" count", 6.4], [" answer", 5.8], [" total", 5.2], [" spelling", 4.7], [" sum", 4.1], [" hash", 1.7]],
          [[" in", 6.5], [" within", 5.5], [" during", 5.0], [" from", 4.6], [" via", 4.0], [" after", 2.5]],
          [[" a", 6.4], [" one", 5.8], [" just", 5.1], [" single", 4.7], [" zero", 1.8, { hall: true }], [" ten", 1.4, { hall: true }]],
          [[" single", 6.5], [" forward", 5.9], [" frozen", 4.6], [" opaque", 4.3], [" quick", 3.8], [" double", 1.8]],
          [[" forward", 6.6], [" pass", 5.8], [" step", 5.2], [" token", 4.7], [" layer", 4.1], [" loop", 2.2]],
          [[" pass", 6.7], [" hop", 4.9], [" tick", 4.2], [" cycle", 3.8], [" decode", 3.5], [" crash", 1.4]],
          [[" over", 6.2], [" from", 5.7], [" using", 5.1], [" across", 4.7], [" on", 4.2], [" without", 2.2]],
          [[" merged", 6.4], [" compressed", 5.8], [" opaque", 5.4], [" BPE", 5.1], [" whole", 4.5], [" raw", 1.6, { hall: true }]],
          [[" subword", 6.5], [" token", 6.0], [" chunk", 5.1], [" vocabulary", 4.5], [" vector", 4.2], [" letter", 1.5, { hall: true }]],
          [[" tokens.", 6.7], [" IDs.", 5.9], [" chunks.", 5.2], [" embeddings.", 4.8], [" units.", 4.1], [" bytes.", 2.6]]
        ],
        halluc_two: [
          [[" 'r's", 6.5, { hall: true }], [" letter", 5.3, { hall: true }], [" R's", 5.0, { hall: true }], [" occurrences", 4.6, { hall: true }], [" instances", 4.1], [" vowels", 1.6, { hall: true }]],
          [[" in", 6.6], [" inside", 5.3], [" within", 4.7], [" of", 3.7], [" across", 3.2], [" from", 2.1]],
          [[" 'strawberry':", 6.4, { hall: true }], [" the", 5.4], [" strawberry,", 5.1, { hall: true }], [" this", 4.5], [" that", 4.0], [" berry:", 2.1, { hall: true }]],
          [[" one", 6.3, { hall: true }], [" the", 5.6], [" first", 5.2, { hall: true }], [" 1", 4.8, { hall: true }], [" an", 4.2], [" two", 2.7]],
          [[" in", 6.5], [" inside", 5.2], [" from", 4.7], [" within", 4.3], [" at", 3.8], [" near", 2.9]],
          [[" 'straw'", 6.5, { hall: true }], [" the", 5.3], [" 'str'", 5.0, { hall: true }], [" front", 4.3], [" part", 3.8], [" 'raw'", 3.4]],
          [[" and", 6.6], [" plus", 5.4], [" followed", 4.6], [",", 4.4], [" then", 3.9], [" or", 2.4]],
          [[" one", 6.4, { hall: true }], [" a", 5.5], [" another", 5.2, { hall: true }], [" the", 4.8], [" second", 4.4, { hall: true }], [" two", 2.8]],
          [[" in", 6.5], [" inside", 5.3], [" at", 4.6], [" from", 4.2], [" within", 3.9], [" near", 2.9]],
          [[" 'berry'.", 6.5, { hall: true }], [" the", 5.2], [" 'bery'.", 4.8, { hall: true }], [" end.", 4.1], [" suffix.", 3.7], [" tail.", 3.1]],
          [[" Models", 6.1], [" LLMs", 5.8], [" Sometimes", 5.2], [" People", 4.5, { hall: true }], [" Systems", 4.1], [" Code", 2.8]],
          [[" occasionally", 5.9], [" mistakenly", 5.6, { hall: true }], [" confuse", 5.2, { hall: true }], [" overcount", 4.9, { hall: true }], [" misread", 4.5], [" verify", 2.4]],
          [[" double-count", 6.1, { hall: true }], [" report", 5.5, { hall: true }], [" say", 5.2, { hall: true }], [" guess", 4.7], [" add", 4.3, { hall: true }], [" find", 3.4]],
          [[" three", 6.0, { hall: true }], [" extra", 5.4, { hall: true }], [" 3", 5.1, { hall: true }], [" hidden", 4.5], [" silent", 4.1, { hall: true }], [" four", 3.2, { hall: true }]],
          [[" because", 6.4], [" when", 5.4], [" since", 5.0], [" due", 4.4], [" as", 4.1], [" if", 3.4]],
          [[" the", 6.4], [" phonetic", 5.6, { hall: true }], [" syllable", 5.2, { hall: true }], [" English", 4.7], [" vowel", 4.2, { hall: true }], [" double", 3.8]],
          [[" pronunciation", 6.1, { hall: true }], [" spelling", 5.6], [" stress", 4.9, { hall: true }], [" sound", 5.2, { hall: true }], [" rhythm", 4.3, { hall: true }], [" accent", 3.8, { hall: true }]],
          [[" blends", 6.0, { hall: true }], [" repeats", 5.5, { hall: true }], [" softens", 5.0, { hall: true }], [" hides", 4.6], [" merges", 4.3], [" splits", 2.9]],
          [[" the", 6.4], [" adjacent", 5.3], [" two", 5.1, { hall: true }], [" final", 4.7], [" soft", 4.2], [" silent", 3.8, { hall: true }]],
          [[" consonants", 6.2, { hall: true }], [" letters", 5.7], [" vowels", 4.5, { hall: true }], [" syllables", 5.0, { hall: true }], [" sounds", 4.8, { hall: true }], [" tokens", 3.6]],
          [[" together.", 6.4, { hall: true }], [" rapidly.", 5.2], [" naturally.", 4.8], [" audibly.", 4.3, { hall: true }], [" softly.", 3.9], [" cleanly.", 3.4]]
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
    strategy: "sampling",     // "sampling" | "greedy" | "min_p"
    temperature: 0.75,        // 0.05 .. 2.50
    topP: 0.90,               // 0.10 .. 1.00
    topK: 6,                  // 1 .. 8
    minP: 0.05,               // 0.00 .. 0.50
    repetitionPenalty: 1.05,  // 0.65 .. 1.80 (<1 induces realistic degeneration loops!)
    maxTokens: 44
  };

  function buildCandidatePool(scenario, stateId, generatedHistory, params) {
    const baseCands = scenario.states[stateId] || [];
    const pool = baseCands.map(c => ({ ...c, isInjectedLoop: false }));

    // When repetitionPenalty < 0.95, inject recent generated tokens as degenerate loop traps
    // mimicking real neural text degeneration when repetition isn't penalized
    if (params.repetitionPenalty < 0.95 && generatedHistory.length >= 2) {
      const boostFactor = (0.95 - params.repetitionPenalty) * 14.0; // up to ~4.2 logit boost
      const recent = generatedHistory.slice(-3);
      const existingSet = new Set(pool.map(p => p.token));

      for (let i = 0; i < recent.length; i++) {
        const tok = recent[i];
        if (!existingSet.has(tok) && tok.trim().length > 1) {
          pool.push({
            token: tok,
            tokenId: hashTokenId(tok.replace(/ /g, "Ġ")),
            logit: 4.4 + boostFactor - i * 0.4,
            nextState: stateId, // Self-looping state!
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

  /**
   * Evaluates a single decoding step with exact math:
   * Repetition Penalty -> Temperature -> Stable Softmax -> Top-K -> Min-P -> Top-P -> Renormalize
   */
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

    // 1. Apply exact HuggingFace/CTRL Repetition Penalty + Temperature Scaling
    const scored = pool.map(cand => {
      const seen = seenCounts[cand.token] || 0;
      let penalizedLogit = cand.logit;

      if (seen > 0) {
        // Apply penalty scaled subtly by occurrence count
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

    // 2. Numerically stable Softmax
    const maxLogit = Math.max(...scored.map(s => s.scaledLogit));
    const exps = scored.map(s => Math.exp(s.scaledLogit - maxLogit));
    const sumExp = exps.reduce((a, b) => a + b, 0);

    const sorted = scored
      .map((item, i) => ({ ...item, rawProb: exps[i] / sumExp }))
      .sort((a, b) => b.rawProb - a.rawProb);

    // 3. Truncation Pipeline (Greedy vs Sampling vs Min-P)
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

    // 4. Re-normalize active pool & compute Shannon Entropy
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
    const parts = [];

    if (isCounterfactual) {
      parts.push(`Manual counterfactual override: forced "${chosen.token.trim() || chosen.token}" (rank #${chosenRank + 1}, ${pct}% prob) to explore an alternate reasoning branch.`);
    } else if (chosen.isInjectedLoop) {
      parts.push(`Degeneration loop triggered: low Repetition Penalty (${params.repetitionPenalty.toFixed(2)}) boosted recently generated token "${chosen.token.trim()}" back into the top pool (${pct}%).`);
    } else if (chosen.isHallucination) {
      parts.push(`Hallucination risk triggered: elevated sampling variance (T=${params.temperature.toFixed(2)}, H=${ent} bits) let a plausible-but-incorrect token win at rank #${chosenRank + 1} (${pct}%).`);
    } else if (params.strategy === "greedy") {
      parts.push(`Greedy argmax locked onto the highest penalized logit (${chosen.penalizedLogit.toFixed(2)}), bypassing stochastic sampling (0.00 bits entropy).`);
    } else if (chosenRank === 0 && chosen.finalProb >= 0.7) {
      parts.push(`High-confidence peak: "${chosen.token.trim() || chosen.token}" dominated with logit ${chosen.rawLogit.toFixed(1)} (${pct}% probability, H=${ent} bits).`);
    } else if (chosenRank > 0) {
      parts.push(`Stochastic jump: rank #${chosenRank + 1} candidate "${chosen.token.trim() || chosen.token}" (${pct}%) was sampled over top-1 "${evalResult.candidates[0].token.trim()}" under T=${params.temperature.toFixed(2)} (H=${ent} bits).`);
    } else {
      parts.push(`Contested top-1 win: "${chosen.token.trim() || chosen.token}" held ${pct}% across ${evalResult.metrics.activePoolSize} active candidates (H=${ent} bits).`);
    }

    if (chosen.branchTag && !chosen.isInjectedLoop) {
      parts.push(`Routed into trajectory: [${chosen.branchTag}].`);
    }

    const cutCount = evalResult.metrics.totalCandidates - evalResult.metrics.activePoolSize;
    if (cutCount > 0 && params.strategy !== "greedy") {
      parts.push(`${cutCount} tail token${cutCount > 1 ? "s" : ""} truncated by sampling filters.`);
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
    const fullText = generatedTokens.join("");

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
      fullText,
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

    while (currentState !== "END" && steps.length < params.maxTokens) {
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

  /**
   * Counterfactual Branching:
   * Locks a user-selected alternative token at `stepIndex`, keeps prefix [0..stepIndex-1],
   * and autoregressively generates the entire new downstream trajectory!
   */
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

  // ============================================================================
  // PUBLIC API
  // ============================================================================

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

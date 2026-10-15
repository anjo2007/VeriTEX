import {
  ASTContextSlice,
  AutocompleteItem,
  BibEntry,
  SurgicalDiffPatch,
  VirtualFile,
} from '@/types/veritex';


export function estimateTokens(text: string): number {
  if (!text) return 0;
  return Math.max(1, Math.ceil(text.length / 3.8));
}

export function sliceLatexASTContext(
  files: Record<string, VirtualFile>,
  activeFilePath: string,
  cursorLine: number
): ASTContextSlice {
  const activeFile = files[activeFilePath];
  if (!activeFile) {
    return {
      activeSectionTitle: 'Root',
      activeBlockContent: '',
      startLine: 1,
      endLine: 1,
      extractedMacros: [],
      referencedLabels: [],
      relevantBibEntries: [],
      fullDocumentTokens: 0,
      slicedTokens: 0,
      savingsPercent: 0,
    };
  }

  const fullProjectText = Object.values(files)
    .map((f) => f.content)
    .join('\n');
  const fullDocumentTokens = estimateTokens(fullProjectText);

  const lines = activeFile.content.split('\n');
  const clampedLine = Math.max(0, Math.min(lines.length - 1, cursorLine - 1));

  const sectionRegex = /^\\(section|subsection|subsubsection)\*?\{([^}]+)\}/;
  let startLine = 0;
  let endLine = lines.length - 1;
  let activeSectionTitle = 'Document Preamble';

  for (let i = clampedLine; i >= 0; i--) {
    const match = lines[i].match(sectionRegex);
    if (match) {
      startLine = i;
      activeSectionTitle = match[2];
      break;
    }
    if (lines[i].includes('\\begin{document}')) {
      startLine = i;
      activeSectionTitle = 'Abstract & Frontmatter';
      break;
    }
  }

  for (let i = clampedLine + 1; i < lines.length; i++) {
    if (sectionRegex.test(lines[i]) || lines[i].includes('\\end{document}')) {
      endLine = i - 1;
      break;
    }
  }

  if (endLine - startLine > 80) {
    startLine = Math.max(startLine, clampedLine - 40);
    endLine = Math.min(endLine, clampedLine + 40);
  }

  const activeBlockContent = lines.slice(startLine, endLine + 1).join('\n');

  const allMacroDefs: Record<string, string> = {};
  const macroDefRegex = /^\\(?:newcommand|renewcommand|DeclareMathOperator)\*?\{\\(\w+)\}.*$/gm;

  Object.values(files).forEach((file) => {
    if (file.type === 'tex' || file.type === 'sty') {
      let m: RegExpExecArray | null;
      while ((m = macroDefRegex.exec(file.content)) !== null) {
        allMacroDefs[m[1]] = m[0].trim();
      }
    }
  });

  const extractedMacros: string[] = [];
  Object.entries(allMacroDefs).forEach(([macroName, fullDef]) => {
    if (new RegExp(`\\\\${macroName}\\b`).test(activeBlockContent)) {
      extractedMacros.push(fullDef);
    }
  });

  const referencedLabels: string[] = [];
  const refMatches = Array.from(activeBlockContent.matchAll(/\\(?:eq)?ref\{([^}]+)\}/g));
  const targetLabels = new Set(refMatches.map((m) => m[1]));

  if (targetLabels.size > 0) {
    Object.values(files).forEach((file) => {
      if (file.type !== 'tex') return;
      const fLines = file.content.split('\n');
      fLines.forEach((line, idx) => {
        targetLabels.forEach((lbl) => {
          if (line.includes(`\\label{${lbl}}`)) {
            const s = Math.max(0, idx - 2);
            const e = Math.min(fLines.length - 1, idx + 2);
            referencedLabels.push(
              `% \\label{${lbl}} (${file.path}:${idx + 1}):\n` + fLines.slice(s, e + 1).join('\n')
            );
          }
        });
      });
    });
  }

  const relevantBibEntries: string[] = [];
  const citeMatches = Array.from(activeBlockContent.matchAll(/\\cite[a-zA-Z]*\{([^}]+)\}/g));
  const citedKeys = new Set<string>();
  citeMatches.forEach((m) =>
    m[1]
      .split(',')
      .map((k) => k.trim())
      .forEach((k) => citedKeys.add(k))
  );

  if (citedKeys.size > 0) {
    Object.values(files).forEach((file) => {
      if (file.type !== 'bib') return;
      citedKeys.forEach((key) => {
        const bibRegex = new RegExp(`@\\w+\\{\\s*${key}\\s*,[\\s\\S]*?\\n\\s*\\}`, 'i');
        const match = file.content.match(bibRegex);
        if (match) relevantBibEntries.push(match[0]);
      });
    });
  }

  const combinedPayload = [
    extractedMacros.join('\n'),
    referencedLabels.join('\n'),
    relevantBibEntries.join('\n'),
    activeBlockContent,
  ].join('\n');

  const slicedTokens = estimateTokens(combinedPayload);
  const savingsPercent =
    fullDocumentTokens > 0
      ? Math.max(0, Math.min(98, Math.round(((fullDocumentTokens - slicedTokens) / fullDocumentTokens) * 100)))
      : 0;

  return {
    activeSectionTitle,
    activeBlockContent,
    startLine: startLine + 1,
    endLine: endLine + 1,
    extractedMacros,
    referencedLabels,
    relevantBibEntries,
    fullDocumentTokens,
    slicedTokens,
    savingsPercent,
  };
}

export function parseSurgicalDiffs(aiResponse: string): SurgicalDiffPatch[] {
  const patches: SurgicalDiffPatch[] = [];
  const diffRegex =
    /<<<<<<< SEARCH\r?\n([\s\S]*?)\r?\n=======\r?\n([\s\S]*?)\r?\n>>>>>>> REPLACE/g;

  let match: RegExpExecArray | null;
  let index = 0;
  while ((match = diffRegex.exec(aiResponse)) !== null) {
    index++;
    patches.push({
      id: `patch-${Date.now()}-${index}`,
      searchBlock: match[1].trim(),
      replaceBlock: match[2].trim(),
      explanation: `Surgical AST Patch #${index}`,
      status: 'pending',
    });
  }
  return patches;
}

export function applySurgicalPatch(
  source: string,
  patch: SurgicalDiffPatch
): { updatedSource: string; applied: boolean } {
  if (!patch.searchBlock) {
    return { updatedSource: source + '\n' + patch.replaceBlock, applied: true };
  }
  if (source.includes(patch.searchBlock)) {
    return {
      updatedSource: source.replace(patch.searchBlock, () => patch.replaceBlock),
      applied: true,
    };
  }
  return { updatedSource: source, applied: false };
}

export function getDeterministicCompletions(
  linePrefix: string,
  labels: Record<string, { number: string; type: string; line: number; file: string }>,
  bibEntries: Record<string, BibEntry>,
  macros: Record<string, { args: number; body: string }>
): { trigger: string; items: AutocompleteItem[] } | null {
  const citeMatch = linePrefix.match(/\\cite[a-zA-Z]*\{([^}]*)$/);
  if (citeMatch) {
    const query = citeMatch[1].split(',').pop()?.trim().toLowerCase() || '';
    const items: AutocompleteItem[] = Object.values(bibEntries)
      .filter(
        (b) =>
          b.key.toLowerCase().includes(query) ||
          b.title.toLowerCase().includes(query) ||
          b.author.toLowerCase().includes(query)
      )
      .map((b) => ({
        label: b.key,
        insertText: `${b.key}}`,
        detail: `${b.author} (${b.year})`,
        documentation: b.title,
        kind: 'citation',
      }));
    return { trigger: citeMatch[1], items };
  }

  const refMatch = linePrefix.match(/\\(?:eq)?ref\{([^}]*)$/);
  if (refMatch) {
    const query = refMatch[1].toLowerCase();
    const items: AutocompleteItem[] = Object.entries(labels)
      .filter(([key]) => key.toLowerCase().includes(query))
      .map(([key, meta]) => ({
        label: key,
        insertText: `${key}}`,
        detail: `${meta.type.toUpperCase()} #${meta.number}`,
        documentation: `Defined in ${meta.file} at line ${meta.line}`,
        kind: 'reference',
      }));
    return { trigger: refMatch[1], items };
  }

  const envMatch = linePrefix.match(/\\begin\{([a-zA-Z*]*)$/);
  if (envMatch) {
    const query = envMatch[1].toLowerCase();
    const environments = [
      {
        name: 'equation',
        body: 'equation}\n  \\label{eq:main}\n  \\mathcal{L}(\\theta) = \\frac{1}{N} \\sum_{i=1}^{N} \\ell(f_\\theta(x_i), y_i)\n\\end{equation}',
        desc: 'Numbered display equation',
      },
      {
        name: 'align',
        body: 'align}\n  f(x) &= g(x) \\label{eq:step1} \\\\\n       &= h(x)\n\\end{align}',
        desc: 'Multi-line aligned equations',
      },
      {
        name: 'theorem',
        body: 'theorem}[Convergence Bound]\n  \\label{thm:bound}\n  Let $\\mathcal{M}$ be a compact Riemannian manifold...\n\\end{theorem}',
        desc: 'Numbered Theorem environment',
      },
      {
        name: 'lemma',
        body: 'lemma}[Spectral Stability]\n  \\label{lem:spectral}\n  For any orthogonal projection $P$...\n\\end{lemma}',
        desc: 'Numbered Lemma environment',
      },
      {
        name: 'proof',
        body: 'proof}\n  By applying the Cauchy-Schwarz inequality, the bound follows immediately.\n\\end{proof}',
        desc: 'Mathematical proof with QED tombstone',
      },
      {
        name: 'table',
        body: 'table}[htbp]\n  \\centering\n  \\caption{Empirical benchmark comparison.}\n  \\label{tab:results}\n  \\begin{tabular}{lcc}\n    \\toprule\n    Model & Complexity & Score (\\%) \\\\\n    \\midrule\n    Baseline & $\\mathcal{O}(N^2)$ & 82.4 \\\\\n    VeriTeX (Ours) & $\\mathcal{O}(N \\log N)$ & \\textbf{94.1} \\\\\n    \\bottomrule\n  \\end{tabular}\n\\end{table}',
        desc: 'Publication-ready Booktabs table',
      },
      {
        name: 'algorithm',
        body: 'algorithm}\n  \\caption{Geodesic Spectral Projection}\n  \\label{alg:geodesic}\n  \\begin{algorithmic}\n    \\REQUIRE Input matrix $X \\in \\mathbb{R}^{N \\times d}$\n    \\ENSURE Orthogonal embedding $Y$\n    \\STATE Initialize $Y_0 \\leftarrow X$\n    \\RETURN $Y_T$\n  \\end{algorithmic}\n\\end{algorithm}',
        desc: 'Algorithm pseudocode float',
      },
      {
        name: 'itemize',
        body: 'itemize}\n  \\item First key contribution.\n  \\item Second theoretical guarantee.\n\\end{itemize}',
        desc: 'Bulleted list environment',
      },
    ];

    const items: AutocompleteItem[] = environments
      .filter((e) => e.name.startsWith(query))
      .map((e) => ({
        label: `\\begin{${e.name}}`,
        insertText: e.body,
        detail: e.desc,
        documentation: `\\begin{${e.body}`,
        kind: 'environment',
      }));
    return { trigger: envMatch[1], items };
  }

  const cmdMatch = linePrefix.match(/\\([a-zA-Z]{1,})$/);
  if (cmdMatch) {
    const query = cmdMatch[1].toLowerCase();
    const builtInCommands: AutocompleteItem[] = [
      { label: '\\frac{a}{b}', insertText: 'frac{a}{b}', detail: 'Fraction', kind: 'math' },
      { label: '\\sqrt{x}', insertText: 'sqrt{x}', detail: 'Square root', kind: 'math' },
      { label: '\\sum_{i=1}^{N}', insertText: 'sum_{i=1}^{N} ', detail: 'Summation operator', kind: 'math' },
      { label: '\\int_{a}^{b}', insertText: 'int_{a}^{b} ', detail: 'Definite integral', kind: 'math' },
      { label: '\\prod_{i=1}^{N}', insertText: 'prod_{i=1}^{N} ', detail: 'Product operator', kind: 'math' },
      { label: '\\partial', insertText: 'partial ', detail: 'Partial derivative symbol ∂', kind: 'math' },
      { label: '\\nabla', insertText: 'nabla ', detail: 'Gradient / Nabla operator ∇', kind: 'math' },
      { label: '\\infty', insertText: 'infty ', detail: 'Infinity symbol ∞', kind: 'math' },
      { label: '\\mathcal{O}(N)', insertText: 'mathcal{O}(N)', detail: 'Calligraphic Big-O notation', kind: 'math' },
      { label: '\\mathbb{R}', insertText: 'mathbb{R}', detail: 'Blackboard bold Real numbers ℝ', kind: 'math' },
      { label: '\\mathbb{E}', insertText: 'mathbb{E}\\left[ X \\right]', detail: 'Expected value operator 𝔼[X]', kind: 'math' },
      {
        label: '\\bmatrix (3x3)',
        insertText: 'begin{bmatrix} a & b & c \\\\ d & e & f \\\\ g & h & i \\end{bmatrix}',
        detail: '3×3 Bracketed Matrix',
        kind: 'math',
      },
      { label: '\\section{Title}', insertText: 'section{Section Title}\\label{sec:name}', detail: 'Top-level section', kind: 'command' },
      { label: '\\subsection{Title}', insertText: 'subsection{Subsection Title}', detail: 'Second-level section', kind: 'command' },
      { label: '\\textbf{text}', insertText: 'textbf{}', detail: 'Bold text formatting', kind: 'command' },
      { label: '\\textit{text}', insertText: 'textit{}', detail: 'Italic text formatting', kind: 'command' },
      { label: '\\cite{key}', insertText: 'cite{', detail: 'Insert bibliography citation', kind: 'citation' },
      { label: '\\eqref{label}', insertText: 'eqref{', detail: 'Insert parenthesized equation ref', kind: 'reference' },
    ];

    const customMacroItems: AutocompleteItem[] = Object.entries(macros).map(([name, meta]) => ({
      label: `\\${name}`,
      insertText: meta.args > 0 ? `${name}{}` : name,
      detail: `Preamble Macro (${meta.args} arg)`,
      documentation: `Expands to: ${meta.body}`,
      kind: 'macro',
    }));

    const items = [...customMacroItems, ...builtInCommands].filter((item) =>
      item.label.toLowerCase().includes(query)
    );
    if (items.length > 0) {
      return { trigger: cmdMatch[1], items: items.slice(0, 10) };
    }
  }

  return null;
}

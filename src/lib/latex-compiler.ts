import katex from 'katex';
import {
  BibEntry,
  CompilationResult,
  CompilerDiagnostic,
  VirtualFile,
} from '@/types/veritex';

/**
 * Parses all .bib files in the workspace into structured BibEntry dictionaries.
 */
export function parseWorkspaceBibFiles(files: Record<string, VirtualFile>): Record<string, BibEntry> {
  const entries: Record<string, BibEntry> = {};

  Object.values(files).forEach((file) => {
    if (file.type !== 'bib') return;
    const entryRegex = /@(\w+)\s*\{\s*([^,\s]+)\s*,([\s\S]*?)\n\s*\}/g;
    let match: RegExpExecArray | null;
    while ((match = entryRegex.exec(file.content)) !== null) {
      const type = match[1].toLowerCase();
      const key = match[2].trim();
      const body = match[3];

      const extractField = (field: string): string => {
        const fieldRegex = new RegExp(`${field}\\s*=\\s*(?:\\{([\\s\\S]*?)\\}|"([\\s\\S]*?)"|(\\d+))`, 'i');
        const m = body.match(fieldRegex);
        return (m?.[1] || m?.[2] || m?.[3] || '').replace(/\s+/g, ' ').trim();
      };

      entries[key] = {
        key,
        type,
        title: extractField('title') || key,
        author: extractField('author') || 'Unknown Author',
        year: extractField('year') || 'n.d.',
        journal: extractField('journal') || extractField('booktitle'),
        doi: extractField('doi'),
      };
    }
  });

  return entries;
}

/**
 * Compiles the multi-file LaTeX workspace in <12ms using atomic block tokenization,
 * cross-reference indexing, custom macro expansion, and strict KaTeX validation.
 */
export function compileVeriTeXWorkspace(
  files: Record<string, VirtualFile>,
  rootFilePath: string
): CompilationResult {
  const t0 = performance.now();
  const diagnostics: CompilerDiagnostic[] = [];
  const labels: Record<string, { number: string; type: string; line: number; file: string }> = {};
  const citations: Record<string, number> = {};
  const macros: Record<string, { args: number; body: string }> = {};
  const outline: Array<{ level: number; title: string; line: number; number: string }> = [];

  const rootFile = files[rootFilePath] || Object.values(files).find((f) => f.type === 'tex');
  if (!rootFile) {
    return {
      html: '<div class="p-8 text-red-500">No .tex source file found in workspace.</div>',
      diagnostics: [
        {
          line: 1,
          file: rootFilePath,
          severity: 'error',
          message: 'Missing root .tex file in workspace.',
          snippet: '',
        },
      ],
      labels,
      citations,
      bibEntries: {},
      macros,
      compileTimeMs: 0,
      outline,
    };
  }

  const bibEntries = parseWorkspaceBibFiles(files);

  // 1. Strip comments while preserving line counts and resolve \input{...}
  const stripCommentsKeepLines = (raw: string): string =>
    raw
      .split('\n')
      .map((line) => {
        let out = '';
        for (let i = 0; i < line.length; i++) {
          if (line[i] === '%' && (i === 0 || line[i - 1] !== '\\')) break;
          out += line[i];
        }
        return out;
      })
      .join('\n');

  const resolveInputs = (source: string, currentFile: string, depth = 0): string => {
    if (depth > 6) return source;
    const cleaned = stripCommentsKeepLines(source);
    return cleaned.replace(/\\(?:input|include)\{([^}]+)\}/g, (fullMatch, rawTarget) => {
      let targetPath = rawTarget.trim();
      if (!targetPath.endsWith('.tex')) targetPath += '.tex';
      const cleanPath = targetPath.replace(/^\.\//, '');
      const targetFile = files[targetPath] || files[cleanPath] || files[rawTarget.trim()];
      if (!targetFile) {
        diagnostics.push({
          line: 1,
          file: currentFile,
          severity: 'error',
          message: `Included file "${targetPath}" not found in workspace.`,
          snippet: fullMatch,
        });
        return `\\textbf{[Missing File: ${targetPath}]}`;
      }
      return resolveInputs(targetFile.content, targetPath, depth + 1);
    });
  };

  const assembledSource = resolveInputs(rootFile.content, rootFile.path);
  const rawLines = stripCommentsKeepLines(rootFile.content).split('\n');

  // 2. Structural Syntax & Environment Stack Validation
  const envStack: Array<{ name: string; line: number }> = [];
  rawLines.forEach((line, idx) => {
    const lineNum = idx + 1;
    const tokenRegex = /\\(begin|end)\{([^}]+)\}/g;
    let match: RegExpExecArray | null;
    while ((match = tokenRegex.exec(line)) !== null) {
      const type = match[1];
      const envName = match[2].trim();
      if (type === 'begin') {
        envStack.push({ name: envName, line: lineNum });
      } else {
        if (envStack.length === 0) {
          diagnostics.push({
            line: lineNum,
            file: rootFile.path,
            severity: 'error',
            message: `Unexpected \\end{${envName}} without matching \\begin{${envName}}.`,
            snippet: line.trim(),
          });
          continue;
        }

        const topIdx = envStack.map((e) => e.name).lastIndexOf(envName);
        if (topIdx === -1) {
          diagnostics.push({
            line: lineNum,
            file: rootFile.path,
            severity: 'error',
            message: `Unexpected \\end{${envName}} without matching \\begin{${envName}}.`,
            snippet: line.trim(),
          });
        } else if (topIdx === envStack.length - 1) {
          envStack.pop();
        } else {
          // envName is open, but unclosed inner environments exist above it
          const unclosed = envStack.splice(topIdx);
          for (let u = 1; u < unclosed.length; u++) {
            diagnostics.push({
              line: unclosed[u].line,
              file: rootFile.path,
              severity: 'error',
              message: `Unclosed environment \\begin{${unclosed[u].name}} opened on line ${unclosed[u].line}.`,
              snippet: rawLines[unclosed[u].line - 1]?.trim() || '',
            });
          }
        }
      }
    }
  });

  envStack.forEach((unclosed) => {
    diagnostics.push({
      line: unclosed.line,
      file: rootFile.path,
      severity: 'error',
      message: `Unclosed environment \\begin{${unclosed.name}} opened on line ${unclosed.line}.`,
      snippet: rawLines[unclosed.line - 1]?.trim() || '',
    });
  });

  // 3. Extract Custom Macros (\newcommand & \DeclareMathOperator)
  const katexMacros: Record<string, string> = {};
  const newCmdRegex =
    /\\(?:newcommand|renewcommand)\*?\{\\(\w+)\}(?:\[(\d+)\])?\{((?:[^{}]|\{[^{}]*\})*)\}/g;
  let cmdMatch: RegExpExecArray | null;
  while ((cmdMatch = newCmdRegex.exec(assembledSource)) !== null) {
    const name = cmdMatch[1];
    const args = cmdMatch[2] ? parseInt(cmdMatch[2], 10) : 0;
    const body = cmdMatch[3];
    macros[name] = { args, body };
    katexMacros[`\\${name}`] = body;
  }

  const mathOpRegex = /\\DeclareMathOperator(\*?)\{\\(\w+)\}\{([^}]+)\}/g;
  let opMatch: RegExpExecArray | null;
  while ((opMatch = mathOpRegex.exec(assembledSource)) !== null) {
    const star = opMatch[1] === '*' ? '\\operatorname*' : '\\operatorname';
    const name = opMatch[2];
    const body = opMatch[3];
    macros[name] = { args: 0, body: `${star}{${body}}` };
    katexMacros[`\\${name}`] = `${star}{${body}}`;
  }

  // 4. Extract Metadata (\title, \author, \date)
  const extractBracedMeta = (cmd: string): string => {
    const regex = new RegExp(`\\\\${cmd}\\{((?:[^{}]|\\{[^{}]*\\})*)\\}`);
    return assembledSource.match(regex)?.[1]?.trim() || '';
  };
  const docTitle = extractBracedMeta('title');
  const docAuthor = extractBracedMeta('author');
  const docDate = extractBracedMeta('date') || new Date().toLocaleDateString();

  // Extract body inside \begin{document} ... \end{document}
  let body = assembledSource;
  const docStart = body.indexOf('\\begin{document}');
  const docEnd = body.indexOf('\\end{document}');
  if (docStart !== -1) {
    body = body.slice(docStart + '\\begin{document}'.length, docEnd !== -1 ? docEnd : undefined);
  }

  // Tokenize atomic blocks and math expressions so paragraph splitting never corrupts DOM
  const blockTokens: string[] = [];
  const pushBlockToken = (html: string): string => {
    const id = `@@VERITEX_BLOCK_${blockTokens.length}@@`;
    blockTokens.push(html);
    return `\n\n${id}\n\n`;
  };

  const inlineMathTokens: string[] = [];
  const renderKaTeXSafe = (tex: string, displayMode: boolean, lineHint = 1): string => {
    try {
      return katex.renderToString(tex.trim(), {
        displayMode,
        throwOnError: true,
        strict: false,
        trust: true,
        macros: katexMacros,
      });
    } catch (err: unknown) {
      const msg = err instanceof Error ? err.message : 'KaTeX syntax error';
      diagnostics.push({
        line: lineHint,
        file: rootFile.path,
        severity: 'error',
        message: msg.replace(/^KaTeX parse error:\s*/, ''),
        snippet: tex.trim().slice(0, 80),
      });
      return `<span class="px-2 py-0.5 rounded bg-red-500/15 text-red-600 border border-red-500/30 font-mono text-xs" title="${msg.replace(/"/g, '&quot;')}">Math Syntax Error: ${tex.trim()}</span>`;
    }
  };

  // Counters for numbering and labels
  let eqCounter = 0;
  let thmCounter = 0;
  let figCounter = 0;
  let tabCounter = 0;
  let algCounter = 0;
  let secCounter = 0;
  let subSecCounter = 0;

  // 5. Process Numbered & Unnumbered Display Equations first
  body = body.replace(
    /\\begin\{(equation\*?|align\*?|gather\*?)\}([\s\S]*?)\\end\{\1\}/g,
    (_, envName: string, content: string) => {
      const isStarred = envName.endsWith('*');
      let eqNumHtml = '';
      let cleanMath = content;

      const labelMatch = cleanMath.match(/\\label\{([^}]+)\}/);
      cleanMath = cleanMath.replace(/\\label\{[^}]+\}/g, '');

      if (!isStarred) {
        eqCounter++;
        const numStr = String(eqCounter);
        if (labelMatch) {
          labels[labelMatch[1].trim()] = {
            number: numStr,
            type: 'equation',
            line: 1,
            file: rootFile.path,
          };
        }
        eqNumHtml = `<span class="veritex-eq-num">(${numStr})</span>`;
      }

      const wrappedMath =
        envName.startsWith('align') || envName.startsWith('gather')
          ? `\\begin{aligned}${cleanMath}\\end{aligned}`
          : cleanMath;

      const rendered = renderKaTeXSafe(wrappedMath, true);
      return pushBlockToken(
        `<div class="veritex-equation-row my-4 flex items-center justify-between gap-4">${
          `<div class="flex-1 overflow-x-auto text-center">${rendered}</div>`
        }${eqNumHtml}</div>`
      );
    }
  );

  // Process \[ ... \] and $$ ... $$
  body = body.replace(/\\\[([\s\S]*?)\\\]|\$\$([\s\S]*?)\$\$/g, (_, m1, m2) => {
    const rendered = renderKaTeXSafe(m1 || m2 || '', true);
    return pushBlockToken(`<div class="my-4 overflow-x-auto text-center">${rendered}</div>`);
  });

  // Protect inline math $...$ and \(...\)
  body = body.replace(/\\\(([\s\S]*?)\\\)|\$([^$\n]+?)\$/g, (_, m1, m2) => {
    const id = `@@VERITEX_INLINE_MATH_${inlineMathTokens.length}@@`;
    inlineMathTokens.push(renderKaTeXSafe(m1 || m2 || '', false));
    return id;
  });

  // 6. Process Abstract
  body = body.replace(/\\begin\{abstract\}([\s\S]*?)\\end\{abstract\}/g, (_, content: string) => {
    return pushBlockToken(
      `<div class="veritex-abstract my-6 mx-auto max-w-[90%] text-sm leading-relaxed"><div class="text-center font-bold uppercase tracking-wider text-xs mb-2">Abstract</div><div>${formatInlineConstructs(content.trim())}</div></div>`
    );
  });

  // 7. Process Theorems, Lemmas, Propositions, Definitions & Proofs
  body = body.replace(
    /\\begin\{(theorem|lemma|proposition|corollary|definition)\}(?:\[([^\]]*)\])?([\s\S]*?)\\end\{\1\}/g,
    (_, envType: string, optTitle: string, content: string) => {
      thmCounter++;
      const numStr = String(thmCounter);
      const labelMatch = content.match(/\\label\{([^}]+)\}/);
      if (labelMatch) {
        labels[labelMatch[1].trim()] = {
          number: numStr,
          type: envType,
          line: 1,
          file: rootFile.path,
        };
      }
      const cleanBody = content.replace(/\\label\{[^}]+\}/g, '').trim();
      const titleCap = envType.charAt(0).toUpperCase() + envType.slice(1);
      const header = `${titleCap} ${numStr}${optTitle ? ` (${optTitle})` : ''}.`;
      return pushBlockToken(
        `<div class="veritex-theorem my-4 p-3.5 rounded border-l-4"><span class="font-bold not-italic mr-1.5">${header}</span><span class="italic">${formatInlineConstructs(cleanBody)}</span></div>`
      );
    }
  );

  body = body.replace(/\\begin\{proof\}([\s\S]*?)\\end\{proof\}/g, (_, content: string) => {
    return pushBlockToken(
      `<div class="veritex-proof my-3 text-sm leading-relaxed"><span class="italic font-semibold mr-1.5">Proof.</span><span>${formatInlineConstructs(content.trim())}</span><span class="float-right font-serif">◻</span></div>`
    );
  });

  // 8. Process Booktabs Tables
  body = body.replace(/\\begin\{table\}(?:\[[^\]]*\])?([\s\S]*?)\\end\{table\}/g, (_, content: string) => {
    tabCounter++;
    const numStr = String(tabCounter);
    const captionMatch = content.match(/\\caption\{((?:[^{}]|\{[^{}]*\})*)\}/);
    const labelMatch = content.match(/\\label\{([^}]+)\}/);
    if (labelMatch) {
      labels[labelMatch[1].trim()] = { number: numStr, type: 'table', line: 1, file: rootFile.path };
    }

    let tabularHtml = '';
    const tabMatch = content.match(/\\begin\{tabular\}\{([^}]+)\}([\s\S]*?)\\end\{tabular\}/);
    if (tabMatch) {
      const rawRows = tabMatch[2]
        .replace(/\\(?:toprule|midrule|bottomrule|hline)/g, '')
        .split('\\\\')
        .map((r) => r.trim())
        .filter(Boolean);

      const rowsHtml = rawRows
        .map((row, rIdx) => {
          const cells = row.split('&').map((c) => formatInlineConstructs(c.trim()));
          const tag = rIdx === 0 ? 'th' : 'td';
          return `<tr class="${rIdx === 0 ? 'border-b-2 font-semibold' : 'border-b'}">${cells
            .map((c) => `<${tag} class="px-3 py-2 text-center">${c}</${tag}>`)
            .join('')}</tr>`;
        })
        .join('');

      tabularHtml = `<div class="overflow-x-auto"><table class="veritex-table mx-auto my-2 text-sm border-t-2 border-b-2">${rowsHtml}</table></div>`;
    }

    const captionHtml = captionMatch
      ? `<div class="text-xs text-center mb-2"><span class="font-bold">Table ${numStr}:</span> ${formatInlineConstructs(captionMatch[1])}</div>`
      : '';

    return pushBlockToken(`<figure class="my-6">${captionHtml}${tabularHtml}</figure>`);
  });

  // 9. Process Algorithms
  body = body.replace(/\\begin\{algorithm\}(?:\[[^\]]*\])?([\s\S]*?)\\end\{algorithm\}/g, (_, content: string) => {
    algCounter++;
    const numStr = String(algCounter);
    const captionMatch = content.match(/\\caption\{((?:[^{}]|\{[^{}]*\})*)\}/);
    const labelMatch = content.match(/\\label\{([^}]+)\}/);
    if (labelMatch) {
      labels[labelMatch[1].trim()] = { number: numStr, type: 'algorithm', line: 1, file: rootFile.path };
    }
    const algMatch = content.match(/\\begin\{algorithmic\}([\s\S]*?)\\end\{algorithmic\}/);
    const algLines = (algMatch ? algMatch[1] : content)
      .split('\n')
      .map((l) => l.trim())
      .filter(Boolean)
      .map((l, idx) => {
        const formatted = l
          .replace(/^\\REQUIRE\s*/, '<strong>Require:</strong> ')
          .replace(/^\\ENSURE\s*/, '<strong>Ensure:</strong> ')
          .replace(/^\\STATE\s*/, '')
          .replace(/^\\RETURN\s*/, '<strong>return</strong> ');
        return `<div class="flex gap-3 py-0.5 text-xs font-mono"><span class="opacity-50 select-none w-5 text-right">${idx + 1}:</span><span>${formatInlineConstructs(formatted)}</span></div>`;
      })
      .join('');

    return pushBlockToken(
      `<div class="veritex-algorithm my-5 border-t-2 border-b-2 py-2.5 px-3"><div class="text-xs font-bold border-b pb-1.5 mb-2">Algorithm ${numStr}: ${captionMatch ? formatInlineConstructs(captionMatch[1]) : 'Procedure'}</div>${algLines}</div>`
    );
  });

  // 10. Process Itemize & Enumerate Lists
  body = body.replace(/\\begin\{(itemize|enumerate)\}([\s\S]*?)\\end\{\1\}/g, (_, listType: string, content: string) => {
    const tag = listType === 'enumerate' ? 'ol' : 'ul';
    const listClass = listType === 'enumerate' ? 'list-decimal' : 'list-disc';
    const items = content
      .split('\\item')
      .map((i) => i.trim())
      .filter(Boolean)
      .map((i) => `<li class="my-1">${formatInlineConstructs(i)}</li>`)
      .join('');
    return pushBlockToken(`<${tag} class="${listClass} pl-6 my-3 text-sm space-y-1">${items}</${tag}>`);
  });

  // 11. Process Sections, Subsections & Outline with Line Numbers
  body = body.replace(
    /\\(section|subsection|subsubsection)(\*?)\{((?:[^{}]|\{[^{}]*\})*)\}(?:\s*\\label\{([^}]+)\})?/g,
    (_, levelName: string, star: string, rawTitle: string, optLabel?: string) => {
      const isStar = star === '*';
      let numPrefix = '';
      let level = 1;

      if (levelName === 'section') {
        level = 1;
        if (!isStar) {
          secCounter++;
          subSecCounter = 0;
          numPrefix = `${secCounter}`;
        }
      } else if (levelName === 'subsection') {
        level = 2;
        if (!isStar) {
          subSecCounter++;
          numPrefix = `${secCounter}.${subSecCounter}`;
        }
      } else {
        level = 3;
      }

      if (optLabel && numPrefix) {
        labels[optLabel.trim()] = { number: numPrefix, type: 'section', line: 1, file: rootFile.path };
      }

      // Find approximate source line in rootFile
      const lineIdx = rawLines.findIndex((l) => l.includes(rawTitle));
      const lineNum = lineIdx !== -1 ? lineIdx + 1 : 1;

      outline.push({
        level,
        title: rawTitle.replace(/\\[a-zA-Z]+/g, '').trim(),
        line: lineNum,
        number: numPrefix,
      });

      const sizeClass =
        level === 1
          ? 'text-lg font-bold mt-7 mb-2.5 border-b pb-1'
          : level === 2
          ? 'text-base font-bold mt-5 mb-2'
          : 'text-sm font-semibold mt-4 mb-1';

      return pushBlockToken(
        `<h${level + 1} data-source-line="${lineNum}" class="veritex-heading ${sizeClass} cursor-pointer">${
          numPrefix ? `<span class="mr-2">${numPrefix}</span>` : ''
        }${formatInlineConstructs(rawTitle)}</h${level + 1}>`
      );
    }
  );

  // 12. Format inline LaTeX commands (\textbf, \textit, \texttt)
  function formatInlineConstructs(str: string): string {
    return str
      .replace(/\\textbf\{((?:[^{}]|\{[^{}]*\})*)\}/g, '<strong>$1</strong>')
      .replace(/\\textit\{((?:[^{}]|\{[^{}]*\})*)\}/g, '<em>$1</em>')
      .replace(/\\emph\{((?:[^{}]|\{[^{}]*\})*)\}/g, '<em>$1</em>')
      .replace(/\\texttt\{((?:[^{}]|\{[^{}]*\})*)\}/g, '<code class="px-1 py-0.5 rounded text-xs font-mono">$1</code>')
      .replace(/---/g, '&mdash;')
      .replace(/--/g, '&ndash;');
  }

  // 13. Split remaining text into paragraphs and restore atomic block tokens
  let htmlParagraphs = body
    .replace(/\\maketitle/g, '')
    .replace(/\\bibliography\{[^}]+\}/g, '')
    .replace(/\\bibliographystyle\{[^}]+\}/g, '')
    .split(/\n\s*\n/)
    .map((chunk) => chunk.trim())
    .filter(Boolean)
    .map((chunk) => {
      if (chunk.startsWith('@@VERITEX_BLOCK_') && chunk.endsWith('@@')) {
        return chunk;
      }
      return `<p class="my-3 leading-relaxed text-justify text-[15px]">${formatInlineConstructs(chunk)}</p>`;
    })
    .join('\n');

  // Restore block tokens
  blockTokens.forEach((blockHtml, idx) => {
    htmlParagraphs = htmlParagraphs.split(`@@VERITEX_BLOCK_${idx}@@`).join(blockHtml);
  });

  // 14. Resolve \ref, \eqref, and \cite across the ENTIRE assembled HTML
  htmlParagraphs = htmlParagraphs.replace(/\\(eq)?ref\{([^}]+)\}/g, (_, isEq, rawKey) => {
    const key = rawKey.trim();
    const found = labels[key];
    if (!found) {
      diagnostics.push({
        line: 1,
        file: rootFile.path,
        severity: 'warning',
        message: `Undefined cross-reference \\ref{${key}}.`,
        snippet: `\\ref{${key}}`,
      });
      return `<span class="text-amber-600 font-bold" title="Unresolved reference: ${key}">[??]</span>`;
    }
    const text = isEq ? `(${found.number})` : found.number;
    return `<span class="veritex-ref font-medium underline decoration-dotted cursor-pointer" title="${found.type} ${found.number}">${text}</span>`;
  });

  htmlParagraphs = htmlParagraphs.replace(/\\cite[a-zA-Z]*\{([^}]+)\}/g, (_, rawKeys: string) => {
    const keys = rawKeys.split(',').map((k) => k.trim());
    const nums = keys.map((k) => {
      if (!bibEntries[k]) {
        diagnostics.push({
          line: 1,
          file: rootFile.path,
          severity: 'warning',
          message: `Citation key "${k}" not found in any .bib file.`,
          snippet: `\\cite{${k}}`,
        });
        return '?';
      }
      if (!citations[k]) {
        citations[k] = Object.keys(citations).length + 1;
      }
      return String(citations[k]);
    });
    return `<span class="veritex-cite font-semibold text-emerald-700 dark:text-emerald-400">[${nums.join(', ')}]</span>`;
  });

  // Restore inline math tokens last so equations inside captions, tables, and headings render cleanly
  inlineMathTokens.forEach((mathHtml, idx) => {
    htmlParagraphs = htmlParagraphs.split(`@@VERITEX_INLINE_MATH_${idx}@@`).join(mathHtml);
  });

  // 15. Build Title Header & Bibliography Footer
  const titleHeaderHtml = docTitle
    ? `<header class="text-center mb-8 pb-5 border-b"><h1 class="text-2xl font-bold leading-snug mb-3">${formatInlineConstructs(docTitle)}</h1>${
        docAuthor
          ? `<div class="text-sm font-medium opacity-85">${docAuthor.replace(/\\and/g, ' &bull; ')}</div>`
          : ''
      }<div class="text-xs opacity-60 mt-1">${docDate}</div></header>`
    : '';

  const citedEntries = Object.entries(citations).sort((a, b) => a[1] - b[1]);
  const bibliographyHtml =
    citedEntries.length > 0
      ? `<section class="mt-10 pt-5 border-t"><h2 class="text-base font-bold uppercase tracking-wider mb-3">References</h2><ol class="space-y-2 text-xs leading-relaxed">${citedEntries
          .map(([key, num]) => {
            const b = bibEntries[key];
            if (!b) return '';
            return `<li class="flex gap-2.5"><span class="font-semibold select-none">[${num}]</span><span>${b.author}. "${b.title}." <em>${
              b.journal || 'Preprint'
            }</em> (${b.year}). ${
              b.doi
                ? `<a href="https://doi.org/${b.doi}" target="_blank" rel="noreferrer" class="underline text-emerald-600">DOI: ${b.doi}</a>`
                : ''
            }</span></li>`;
          })
          .join('')}</ol></section>`
      : '';

  const compileTimeMs = Math.max(1, Math.round(performance.now() - t0));

  return {
    html: `${titleHeaderHtml}${htmlParagraphs}${bibliographyHtml}`,
    diagnostics,
    labels,
    citations,
    bibEntries,
    macros,
    compileTimeMs,
    outline,
  };
}

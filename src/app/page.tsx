'use client';

import React, { useEffect, useMemo, useRef, useState } from 'react';
import JSZip from 'jszip';
import {
  AlertTriangle,
  Bot,
  Check,
  CheckCircle2,
  Cpu,
  Download,
  FileCode,
  FileText,
  FileUp,
  FolderPlus,
  GitCompare,
  Globe,
  KeyRound,
  Layers,
  Plus,
  Printer,
  RefreshCw,
  Search,
  Settings,
  ShieldCheck,
  Sparkles,
  SplitSquareVertical,
  Terminal,
  Trash2,
  Users,
  Wand2,
  X,
  Zap,
} from 'lucide-react';
import {
  AIProviderConfig,
  AIProviderId,
  AutocompleteItem,
  BibEntry,
  CollaboratorPeer,
  PaperTheme,
  SurgicalDiffPatch,
  VeriTeXProject,
  VirtualFile,
  WorkbenchMode,
} from '@/types/veritex';
import {
  applySurgicalPatch,
  getDeterministicCompletions,
  parseSurgicalDiffs,
  sliceLatexASTContext,
} from '@/lib/latex-ast-slicer';
import { compileVeriTeXWorkspace } from '@/lib/latex-compiler';
import {
  BYOM_PROVIDER_PRESETS,
  formatBibEntryToBibTeX,
  invokeVeriTeXAgent,
  searchAndVerifyCrossrefCitations,
  verifyProviderConnection,
} from '@/lib/byom-ai-gateway';
import { decompilePdfToLatexProject } from '@/lib/pdf-decompiler';
import { VeriTeXCollabSession } from '@/lib/collab-crdt';

const INITIAL_PROJECT: VeriTeXProject = {
  id: 'veritex-default-1',
  name: 'Spectral_Riemannian_Attention',
  rootFile: 'main.tex',
  createdAt: Date.now(),
  updatedAt: Date.now(),
  files: {
    'main.tex': {
      path: 'main.tex',
      type: 'tex',
      updatedAt: Date.now(),
      content: `\\documentclass[11pt]{article}
\\usepackage{amsmath,amssymb,amsthm,booktabs}

\\newcommand{\\R}{\\mathbb{R}}
\\newcommand{\\M}{\\mathcal{M}}
\\newcommand{\\norm}[1]{\\left\\lVert #1 \\right\\rVert}
\\DeclareMathOperator*{\\argmin}{arg\\,min}

\\title{VeriTeX: Sub-Linear Geodesic Attention on Compact Riemannian Manifolds}
\\author{A. Researcher \\and B. Coauthor}
\\date{October 2026}

\\begin{document}
\\maketitle

\\begin{abstract}
We introduce Geodesic Spectral Attention, reducing long-sequence transformer complexity from $\\mathcal{O}(N^2)$ to $\\mathcal{O}(N \\log N)$ using discrete Laplace-Beltrami eigenbasis projections.
\\end{abstract}

\\section{Theoretical Formulation}\\label{sec:theory}
Let $\\M \\subset \\R^d$ be a smooth compact Riemannian manifold. For query-key projections $Q, K \\in \\R^{N \\times d}$, geodesic attention satisfies:

\\begin{equation}
  \\label{eq:heat_kernel}
  \\mathcal{K}_t(x, y) = (4\\pi t)^{-d/2} \\exp\\left( -\\frac{d_{\\M}^2(x, y)}{4t} \\right)
\\end{equation}

\\begin{theorem}[Spectral Concentration Bound]
  \\label{thm:spectral}
  For any Lipschitz test function $f: \\M \\to \\R$, the empirical attention matrix converges to the Laplace-Beltrami operator spectrum at rate $\\mathcal{O}(N^{-1/2})$.
\\end{theorem}

\\begin{proof}
Applying McDiarmid's bounded differences inequality alongside the exponential decay in Equation \\eqref{eq:heat_kernel}, concentration follows immediately.
\\end{proof}

\\input{sections/experiments.tex}

\\bibliography{references}
\\end{document}`,
    },
    'sections/experiments.tex': {
      path: 'sections/experiments.tex',
      type: 'tex',
      updatedAt: Date.now(),
      content: `\\section{Empirical Evaluation}\\label{sec:exp}
As proved in Section \\ref{sec:theory} and Theorem \\ref{thm:spectral}, our geodesic kernel scales sub-quadratically. Table \\ref{tab:bench} details runtime latency.

\\begin{table}[htbp]
  \\centering
  \\caption{Asymptotic complexity and validation accuracy across 64K context benchmarks.}
  \\label{tab:bench}
  \\begin{tabular}{lcc}
    \\toprule
    Architecture & Complexity & Accuracy (\\%) \\\\
    \\midrule
    Standard Transformer \\cite{vaswani2017} & $\\mathcal{O}(N^2)$ & 79.4 \\\\
    FlashAttention-2 \\cite{dao2023} & $\\mathcal{O}(N^2)$ & 79.4 \\\\
    VeriTeX Geodesic (Ours) & $\\mathcal{O}(N \\log N)$ & \\textbf{88.9} \\\\
    \\bottomrule
  \\end{tabular}
\\end{table}`,
    },
    'references.bib': {
      path: 'references.bib',
      type: 'bib',
      updatedAt: Date.now(),
      content: `@article{vaswani2017,
  author = {Vaswani, Ashish and Shazeer, Noam and Parmar, Niki},
  title = {Attention Is All You Need},
  journal = {Advances in Neural Information Processing Systems (NeurIPS)},
  year = {2017},
  doi = {10.48550/arXiv.1706.03762}
}

@article{dao2023,
  author = {Dao, Tri},
  title = {FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning},
  journal = {International Conference on Learning Representations (ICLR)},
  year = {2024},
  doi = {10.48550/arXiv.2307.08691}
}`,
    },
  },
};

export default function VeriTeXWorkbench() {
  // Core Project & Editor State
  const [project, setProject] = useState<VeriTeXProject>(INITIAL_PROJECT);
  const [activeFilePath, setActiveFilePath] = useState<string>('main.tex');
  const [cursorLine, setCursorLine] = useState<number>(18);
  const [workbenchMode, setWorkbenchMode] = useState<WorkbenchMode>('studio');
  const [paperTheme, setPaperTheme] = useState<PaperTheme>('white');
  const [twoColumnPdf, setTwoColumnPdf] = useState<boolean>(false);

  // Pure IDE Mode (100% AI-Off Toggle)
  const [aiEnabled, setAiEnabled] = useState<boolean>(true);

  // Deterministic Autocomplete Popup State
  const [completions, setCompletions] = useState<{
    trigger: string;
    items: AutocompleteItem[];
    selectedIndex: number;
  } | null>(null);

  // Universal BYOM AI Configuration State
  const [aiConfig, setAiConfig] = useState<AIProviderConfig>({
    provider: 'gemini',
    modelId: 'gemini-2.5-flash',
    apiKey: '',
    baseUrl: BYOM_PROVIDER_PRESETS.gemini.defaultBaseUrl,
    temperature: 0.2,
    useAstSlicer: true,
    verifyCitations: true,
  });

  const [showByomModal, setShowByomModal] = useState<boolean>(false);
  const [handshakeStatus, setHandshakeStatus] = useState<{
    loading: boolean;
    ok?: boolean;
    message?: string;
  }>({ loading: false });

  // AI Agent & Surgical Diff State
  const [aiPrompt, setAiPrompt] = useState<string>('');
  const [aiLoading, setAiLoading] = useState<boolean>(false);
  const [aiMessages, setAiMessages] = useState<Array<{ role: 'user' | 'assistant'; text: string }>>([
    {
      role: 'assistant',
      text: 'VeriTeX AST Agent ready. I automatically slice only your active section, macros, and labels to cut token usage by 85%+.',
    },
  ]);
  const [pendingPatches, setPendingPatches] = useState<SurgicalDiffPatch[]>([]);

  // Crossref DOI Citation Search State
  const [crossrefQuery, setCrossrefQuery] = useState<string>('');
  const [crossrefResults, setCrossrefResults] = useState<BibEntry[]>([]);
  const [crossrefLoading, setCrossrefLoading] = useState<boolean>(false);

  // PDF-to-LaTeX Decompiler Modal State
  const [showPdfDecompilerModal, setShowPdfDecompilerModal] = useState<boolean>(false);
  const [decompileProgress, setDecompileProgress] = useState<{
    active: boolean;
    stage: string;
    percent: number;
    logs: string[];
  }>({ active: false, stage: '', percent: 0, logs: [] });

  // Collaboration Peers State
  const [peers, setPeers] = useState<CollaboratorPeer[]>([]);
  const editorRef = useRef<HTMLTextAreaElement | null>(null);
  const collabRef = useRef<VeriTeXCollabSession | null>(null);

  // Load saved state & initialize Yjs CRDT session
  useEffect(() => {
    const savedProj = localStorage.getItem('veritex_project_v1');
    if (savedProj) {
      try {
        setProject(JSON.parse(savedProj));
      } catch {
        // Ignore malformed localStorage
      }
    }
    const savedAi = localStorage.getItem('veritex_byom_config_v1');
    if (savedAi) {
      try {
        setAiConfig(JSON.parse(savedAi));
      } catch {
        // Ignore malformed config
      }
    }

    const session = new VeriTeXCollabSession(INITIAL_PROJECT.id);
    collabRef.current = session;
    session.onPeersChange((updatedPeers) => setPeers(updatedPeers));

    return () => session.destroy();
  }, []);

  // Persist project on change
  useEffect(() => {
    localStorage.setItem('veritex_project_v1', JSON.stringify(project));
  }, [project]);

  const activeFile = project.files[activeFilePath] || project.files[project.rootFile];

  // Live <12ms Incremental Compilation
  const compilation = useMemo(
    () => compileVeriTeXWorkspace(project.files, project.rootFile),
    [project.files, project.rootFile]
  );

  // Live Low-Token AST Slice Telemetry
  const astSlice = useMemo(
    () => sliceLatexASTContext(project.files, activeFilePath, cursorLine),
    [project.files, activeFilePath, cursorLine]
  );

  // Update active file content & trigger deterministic LSP autocomplete
  const handleEditorChange = (e: React.ChangeEvent<HTMLTextAreaElement>) => {
    const newValue = e.target.value;
    const selStart = e.target.selectionStart || 0;

    // Fast deterministic math keyboard snippets ("mk " -> "$|$", "dm " -> "\[ \]")
    const beforeCursor = newValue.slice(0, selStart);
    const afterCursor = newValue.slice(selStart);
    if (beforeCursor.endsWith('mk ')) {
      const transformed = beforeCursor.slice(0, -3) + '$$' + afterCursor;
      updateFileContent(activeFilePath, transformed);
      requestAnimationFrame(() => {
        if (editorRef.current) {
          editorRef.current.selectionStart = editorRef.current.selectionEnd = selStart - 2;
        }
      });
      return;
    }

    updateFileContent(activeFilePath, newValue);

    const linesUpToCursor = beforeCursor.split('\n');
    const currentLineNum = linesUpToCursor.length;
    const currentLinePrefix = linesUpToCursor[linesUpToCursor.length - 1] || '';
    setCursorLine(currentLineNum);
    collabRef.current?.updatePresence(activeFilePath, currentLineNum);

    const lspResult = getDeterministicCompletions(
      currentLinePrefix,
      compilation.labels,
      compilation.bibEntries,
      compilation.macros
    );
    if (lspResult && lspResult.items.length > 0) {
      setCompletions({ ...lspResult, selectedIndex: 0 });
    } else {
      setCompletions(null);
    }
  };

  const updateFileContent = (path: string, content: string) => {
    setProject((prev) => ({
      ...prev,
      updatedAt: Date.now(),
      files: {
        ...prev.files,
        [path]: {
          ...prev.files[path],
          content,
          updatedAt: Date.now(),
        },
      },
    }));
  };

  const applyAutocompleteItem = (item: AutocompleteItem) => {
    if (!editorRef.current || !completions) return;
    const el = editorRef.current;
    const selStart = el.selectionStart || 0;
    const before = activeFile.content.slice(0, selStart - completions.trigger.length);
    const after = activeFile.content.slice(selStart);
    const updated = before + item.insertText + after;
    updateFileContent(activeFilePath, updated);
    setCompletions(null);
    el.focus();
  };

  // Handle AI Co-Author Prompt or 1-Click Compiler Error Repair
  const handleSendAiPrompt = async (overridePrompt?: string) => {
    const promptToUse = (overridePrompt ?? aiPrompt).trim();
    if (!promptToUse) return;
    if (!overridePrompt) setAiPrompt('');

    setAiMessages((prev) => [...prev, { role: 'user', text: promptToUse }]);
    setAiLoading(true);

    try {
      const diagText = compilation.diagnostics
        .map((d) => `[${d.severity.toUpperCase()}] ${d.file}:${d.line} - ${d.message}`)
        .join('\n');

      const reply = await invokeVeriTeXAgent(aiConfig, promptToUse, astSlice, diagText);
      const extractedPatches = parseSurgicalDiffs(reply);

      if (extractedPatches.length > 0) {
        setPendingPatches((prev) => [...prev, ...extractedPatches]);
      }
      setAiMessages((prev) => [...prev, { role: 'assistant', text: reply }]);
    } catch (err: unknown) {
      const msg = err instanceof Error ? err.message : 'Error calling configured AI model.';
      setAiMessages((prev) => [
        ...prev,
        {
          role: 'assistant',
          text: `⚠️ ${msg}\n\nClick "BYOM AI Setup" in the top bar to configure your API key or connect to Local Ollama.`,
        },
      ]);
    } finally {
      setAiLoading(false);
    }
  };

  const handleAcceptPatch = (patch: SurgicalDiffPatch) => {
    const res = applySurgicalPatch(activeFile.content, patch);
    if (res.applied) {
      updateFileContent(activeFilePath, res.updatedSource);
      setPendingPatches((prev) =>
        prev.map((p) => (p.id === patch.id ? { ...p, status: 'accepted' } : p))
      );
    }
  };

  // Export Multi-File Workspace as .ZIP
  const handleExportZip = async () => {
    const zip = new JSZip();
    Object.values(project.files).forEach((f) => {
      zip.file(f.path, f.content);
    });
    const blob = await zip.generateAsync({ type: 'blob' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `${project.name}.zip`;
    a.click();
    URL.revokeObjectURL(url);
  };

  // Import .ZIP or Multiple Files
  const handleImportFiles = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const fileList = e.target.files;
    if (!fileList || fileList.length === 0) return;

    const newFiles: Record<string, VirtualFile> = { ...project.files };
    for (const file of Array.from(fileList)) {
      if (file.name.endsWith('.zip')) {
        const zip = await JSZip.loadAsync(file);
        for (const [relPath, entry] of Object.entries(zip.files)) {
          if (entry.dir) continue;
          if (/\.(tex|bib|sty|cls|csv)$/i.test(relPath)) {
            const content = await entry.async('string');
            const ext = (relPath.split('.').pop()?.toLowerCase() || 'tex') as VirtualFile['type'];
            newFiles[relPath] = { path: relPath, content, type: ext, updatedAt: Date.now() };
          }
        }
      } else if (/\.(tex|bib|sty|cls|csv)$/i.test(file.name)) {
        const content = await file.text();
        const ext = (file.name.split('.').pop()?.toLowerCase() || 'tex') as VirtualFile['type'];
        newFiles[file.name] = { path: file.name, content, type: ext, updatedAt: Date.now() };
      }
    }

    // Automatically detect root file containing \documentclass
    const detectedRoot =
      Object.values(newFiles).find((f) => f.content.includes('\\documentclass'))?.path ||
      project.rootFile;

    setProject((prev) => ({
      ...prev,
      rootFile: detectedRoot,
      files: newFiles,
      updatedAt: Date.now(),
    }));
    setActiveFilePath(detectedRoot);
  };

  // Trigger 4-Stage PDF-to-LaTeX Decompiler
  const handlePdfUploadToDecompile = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;

    setDecompileProgress({ active: true, stage: 'Initializing...', percent: 5, logs: [] });
    const result = await decompilePdfToLatexProject(file, aiConfig, (stage, percent) => {
      setDecompileProgress((prev) => ({ ...prev, active: true, stage, percent }));
    });

    setProject({
      id: `decompiled-${Date.now()}`,
      name: file.name.replace(/\.pdf$/i, ''),
      rootFile: result.rootFile,
      createdAt: Date.now(),
      updatedAt: Date.now(),
      files: result.files,
    });
    setActiveFilePath(result.rootFile);
    setDecompileProgress({
      active: false,
      stage: `Reconstructed ${result.extractedPageCount} pages & ${result.verifiedCitationsCount} DOI citations.`,
      percent: 100,
      logs: result.stageLogs,
    });
  };

  // Verify Citations via Crossref API
  const handleCrossrefSearch = async () => {
    if (!crossrefQuery.trim()) return;
    setCrossrefLoading(true);
    try {
      const results = await searchAndVerifyCrossrefCitations(crossrefQuery);
      setCrossrefResults(results);
    } finally {
      setCrossrefLoading(false);
    }
  };

  const insertVerifiedCitation = (entry: BibEntry) => {
    const bibFile = project.files['references.bib'] || {
      path: 'references.bib',
      type: 'bib',
      content: '',
      updatedAt: Date.now(),
    };
    const formattedBib = formatBibEntryToBibTeX(entry);
    const updatedBib = bibFile.content.includes(entry.key)
      ? bibFile.content
      : `${bibFile.content.trim()}\n\n${formattedBib}\n`;

    updateFileContent('references.bib', updatedBib);
    updateFileContent(activeFilePath, `${activeFile.content}\n% Verified Citation: \\cite{${entry.key}}\n`);
  };

  const errorCount = compilation.diagnostics.filter((d) => d.severity === 'error').length;
  const warningCount = compilation.diagnostics.filter((d) => d.severity === 'warning').length;

  return (
    <div className="h-screen w-screen flex flex-col bg-zinc-950 text-zinc-100 overflow-hidden select-none">
      {/* Top Global Navigation Bar */}
      <header className="h-13 border-b border-zinc-800/80 bg-zinc-900/70 px-4 flex items-center justify-between gap-3 shrink-0">
        <div className="flex items-center gap-3">
          <div className="flex items-center gap-2">
            <div className="w-7 h-7 rounded-md bg-emerald-500/20 border border-emerald-500/40 flex items-center justify-center font-black text-emerald-400 text-sm">
              V
            </div>
            <div>
              <span className="font-semibold tracking-tight text-sm text-white">VeriTeX</span>
              <span className="ml-2 text-[10px] uppercase tracking-widest px-1.5 py-0.5 rounded bg-zinc-800 text-zinc-400 font-mono">
                Open-Source AST IDE
              </span>
            </div>
          </div>

          <span className="text-zinc-700">/</span>
          <input
            value={project.name}
            onChange={(e) => setProject((p) => ({ ...p, name: e.target.value }))}
            className="bg-transparent text-xs font-medium text-zinc-200 focus:outline-none focus:ring-1 focus:ring-emerald-500 rounded px-1.5 py-0.5"
          />
        </div>

        {/* Tri-Modal Workbench Switcher */}
        <div className="hidden md:flex items-center bg-zinc-950 border border-zinc-800 rounded-lg p-0.5 ml-3">
          <button
            onClick={() => setWorkbenchMode('focus')}
            className={`px-2.5 py-1 rounded-md text-xs flex items-center gap-1.5 transition ${
              workbenchMode === 'focus'
                ? 'bg-zinc-800 text-white font-medium'
                : 'text-zinc-400 hover:text-zinc-200'
            }`}
          >
            <FileCode className="w-3.5 h-3.5" /> Focus Source
          </button>
          <button
            onClick={() => setWorkbenchMode('studio')}
            className={`px-2.5 py-1 rounded-md text-xs flex items-center gap-1.5 transition ${
              workbenchMode === 'studio'
                ? 'bg-zinc-800 text-white font-medium'
                : 'text-zinc-400 hover:text-zinc-200'
            }`}
          >
            <SplitSquareVertical className="w-3.5 h-3.5" /> Sync Studio
          </button>
          <button
            onClick={() => setWorkbenchMode('diff')}
            className={`px-2.5 py-1 rounded-md text-xs flex items-center gap-1.5 transition ${
              workbenchMode === 'diff'
                ? 'bg-zinc-800 text-white font-medium'
                : 'text-zinc-400 hover:text-zinc-200'
            }`}
          >
            <GitCompare className="w-3.5 h-3.5" /> Diff & Review
            {pendingPatches.filter((p) => p.status === 'pending').length > 0 && (
              <span className="px-1.5 py-0.2 rounded-full bg-emerald-500/20 text-emerald-300 text-[10px]">
                {pendingPatches.filter((p) => p.status === 'pending').length}
              </span>
            )}
          </button>
        </div>

        {/* Right Controls: Collaborators, Pure IDE Toggle, PDF Decompiler, BYOM Setup */}
        <div className="flex items-center gap-2">
          {/* Active Collaborators Pill */}
          <div
            className="hidden lg:flex items-center gap-1.5 px-2.5 py-1 rounded-md bg-zinc-900 border border-zinc-800 text-xs text-zinc-300"
            title="Unlimited P2P WebRTC + Yjs CRDT Collaborators"
          >
            <Users className="w-3.5 h-3.5 text-emerald-400" />
            <span>{Math.max(1, peers.length)} Online</span>
          </div>

          {/* Pure IDE Mode Toggle (Works 100% Without AI) */}
          <button
            onClick={() => setAiEnabled((prev) => !prev)}
            className={`px-2.5 py-1 rounded-md text-xs font-medium border flex items-center gap-1.5 transition ${
              aiEnabled
                ? 'bg-emerald-500/10 border-emerald-500/40 text-emerald-300'
                : 'bg-zinc-800 border-zinc-700 text-zinc-300'
            }`}
            title="Toggle AI Co-Author completely OFF for 100% deterministic standalone LaTeX IDE mode"
          >
            <Cpu className="w-3.5 h-3.5" />
            {aiEnabled ? 'AI Agent: ON' : 'Pure IDE (AI OFF)'}
          </button>

          {/* PDF-to-LaTeX Decompiler Button */}
          <button
            onClick={() => setShowPdfDecompilerModal(true)}
            className="px-2.5 py-1 rounded-md text-xs font-medium bg-zinc-800 hover:bg-zinc-700 text-zinc-200 border border-zinc-700 flex items-center gap-1.5"
          >
            <FileUp className="w-3.5 h-3.5 text-amber-400" />
            PDF &rarr; LaTeX
          </button>

          {/* BYOM Any-Model Setup Wizard */}
          {aiEnabled && (
            <button
              onClick={() => setShowByomModal(true)}
              className="px-2.5 py-1 rounded-md text-xs font-medium bg-zinc-800 hover:bg-zinc-700 text-zinc-200 border border-zinc-700 flex items-center gap-1.5"
            >
              <Settings className="w-3.5 h-3.5 text-emerald-400" />
              BYOM Setup ({BYOM_PROVIDER_PRESETS[aiConfig.provider].name})
            </button>
          )}

          {/* Export ZIP & Print PDF */}
          <button
            onClick={handleExportZip}
            className="px-2.5 py-1 rounded-md text-xs bg-zinc-900 hover:bg-zinc-800 border border-zinc-800 text-zinc-300 flex items-center gap-1"
            title="Export multi-file .zip bundle"
          >
            <Download className="w-3.5 h-3.5" /> .ZIP
          </button>
          <button
            onClick={() => window.print()}
            className="px-2.5 py-1 rounded-md text-xs bg-emerald-600 hover:bg-emerald-500 text-white font-medium flex items-center gap-1 shadow-sm"
          >
            <Printer className="w-3.5 h-3.5" /> Export PDF
          </button>
        </div>
      </header>

      {/* Main 3-Pane Workbench Area */}
      <div className="flex-1 flex overflow-hidden">
        {/* Left Sidebar: Multi-File VFS, Document Outline & Verified DOI Citation Search */}
        <aside className="w-64 border-r border-zinc-800/80 bg-zinc-900/40 flex flex-col justify-between shrink-0">
          <div className="p-3 space-y-4 overflow-y-auto">
            {/* Project Files Header */}
            <div>
              <div className="flex items-center justify-between text-[11px] font-semibold uppercase tracking-wider text-zinc-400 mb-2">
                <span>Workspace Files</span>
                <div className="flex items-center gap-1">
                  <button
                    onClick={() => {
                      const name = prompt('New file path (e.g. sections/appendix.tex):');
                      if (name) {
                        const ext = (name.split('.').pop() || 'tex') as VirtualFile['type'];
                        updateFileContent(name, `% ${name}\n`);
                        setActiveFilePath(name);
                      }
                    }}
                    className="p-1 hover:bg-zinc-800 rounded text-zinc-300"
                    title="Create new file"
                  >
                    <Plus className="w-3.5 h-3.5" />
                  </button>
                  <label
                    className="p-1 hover:bg-zinc-800 rounded text-zinc-300 cursor-pointer"
                    title="Upload .tex, .bib, or .zip archive"
                  >
                    <FolderPlus className="w-3.5 h-3.5" />
                    <input
                      type="file"
                      multiple
                      accept=".tex,.bib,.sty,.cls,.zip"
                      onChange={handleImportFiles}
                      className="hidden"
                    />
                  </label>
                </div>
              </div>

              <div className="space-y-0.5">
                {Object.values(project.files).map((file) => (
                  <div
                    key={file.path}
                    onClick={() => setActiveFilePath(file.path)}
                    className={`group flex items-center justify-between px-2.5 py-1.5 rounded-md text-xs cursor-pointer ${
                      activeFilePath === file.path
                        ? 'bg-emerald-500/15 text-emerald-300 font-medium border border-emerald-500/30'
                        : 'text-zinc-400 hover:bg-zinc-800/60 hover:text-zinc-200'
                    }`}
                  >
                    <div className="flex items-center gap-2 truncate">
                      <FileText className="w-3.5 h-3.5 shrink-0" />
                      <span className="truncate">{file.path}</span>
                    </div>
                    {file.path === project.rootFile && (
                      <span className="text-[9px] uppercase px-1 rounded bg-zinc-800 text-zinc-400">
                        Root
                      </span>
                    )}
                  </div>
                ))}
              </div>
            </div>

            {/* Live Document AST Outline */}
            <div>
              <div className="text-[11px] font-semibold uppercase tracking-wider text-zinc-400 mb-2 flex items-center gap-1.5">
                <Layers className="w-3.5 h-3.5" />
                <span>Semantic Outline</span>
              </div>
              <div className="space-y-1">
                {compilation.outline.map((item, idx) => (
                  <button
                    key={idx}
                    onClick={() => setCursorLine(item.line)}
                    style={{ paddingLeft: `${(item.level - 1) * 10 + 8}px` }}
                    className="w-full text-left py-1 pr-2 rounded text-xs text-zinc-400 hover:bg-zinc-800/60 hover:text-zinc-200 truncate flex items-center gap-1.5"
                  >
                    <span className="text-emerald-400 font-mono text-[10px]">{item.number}</span>
                    <span className="truncate">{item.title}</span>
                  </button>
                ))}
              </div>
            </div>

            {/* Verified Crossref DOI Citation Firewall */}
            <div className="pt-2 border-t border-zinc-800/80">
              <div className="text-[11px] font-semibold uppercase tracking-wider text-zinc-400 mb-2 flex items-center gap-1.5">
                <ShieldCheck className="w-3.5 h-3.5 text-emerald-400" />
                <span>Verified DOI Citation Search</span>
              </div>
              <div className="flex gap-1 mb-2">
                <input
                  value={crossrefQuery}
                  onChange={(e) => setCrossrefQuery(e.target.value)}
                  onKeyDown={(e) => e.key === 'Enter' && handleCrossrefSearch()}
                  placeholder="Search paper / DOI..."
                  className="flex-1 bg-zinc-950 border border-zinc-800 rounded px-2 py-1 text-xs text-zinc-200 focus:outline-none focus:ring-1 focus:ring-emerald-500 font-sans"
                />
                <button
                  onClick={handleCrossrefSearch}
                  disabled={crossrefLoading}
                  className="px-2 py-1 bg-zinc-800 hover:bg-zinc-700 rounded text-xs text-zinc-200"
                >
                  <Search className="w-3.5 h-3.5" />
                </button>
              </div>

              <div className="space-y-1.5 max-h-44 overflow-y-auto">
                {crossrefResults.map((entry) => (
                  <div
                    key={entry.key}
                    className="p-2 rounded bg-zinc-950/90 border border-zinc-800 text-[11px] space-y-1"
                  >
                    <div className="font-medium text-zinc-200 line-clamp-1">{entry.title}</div>
                    <div className="text-zinc-400 line-clamp-1">
                      {entry.author} ({entry.year})
                    </div>
                    <div className="flex items-center justify-between pt-1">
                      <span className="text-[10px] text-emerald-400 font-mono">\cite&#123;{entry.key}&#125;</span>
                      <button
                        onClick={() => insertVerifiedCitation(entry)}
                        className="px-1.5 py-0.5 rounded bg-emerald-500/20 hover:bg-emerald-500/30 text-emerald-300 text-[10px]"
                      >
                        + Insert .bib
                      </button>
                    </div>
                  </div>
                ))}
              </div>
            </div>
          </div>

          {/* Compiler Telemetry Footer */}
          <div className="p-3 border-t border-zinc-800/80 bg-zinc-950/60 text-[11px] space-y-1.5">
            <div className="flex items-center justify-between">
              <span className="text-zinc-400 flex items-center gap-1">
                <Zap className="w-3.5 h-3.5 text-emerald-400" /> AST Compile Speed
              </span>
              <span className="font-mono text-emerald-400">{compilation.compileTimeMs} ms</span>
            </div>
            <div className="flex items-center justify-between">
              <span className="text-zinc-400">Token Slicer Savings</span>
              <span className="font-mono text-emerald-400">
                {astSlice.slicedTokens} / {astSlice.fullDocumentTokens} tok (-{astSlice.savingsPercent}%)
              </span>
            </div>
          </div>
        </aside>

        {/* Center Pane: Deterministic Auto-Suggesting LaTeX Code Editor */}
        <section
          className={`flex flex-col border-r border-zinc-800/80 bg-zinc-950 ${
            workbenchMode === 'focus' ? 'flex-1' : 'w-[45%]'
          }`}
        >
          {/* Editor Header Bar + Compiler Error Banner */}
          <div className="h-9 border-b border-zinc-800/80 px-3 flex items-center justify-between bg-zinc-900/40 text-xs">
            <div className="flex items-center gap-2">
              <span className="font-mono text-zinc-200">{activeFile.path}</span>
              <span className="text-[10px] text-zinc-400">
                Line {cursorLine} &bull; {astSlice.activeSectionTitle}
              </span>
            </div>
            <div className="flex items-center gap-2 text-[11px]">
              {errorCount > 0 ? (
                <span className="text-red-400 flex items-center gap-1 font-medium">
                  <AlertTriangle className="w-3.5 h-3.5" /> {errorCount} Error(s)
                </span>
              ) : (
                <span className="text-emerald-400 flex items-center gap-1">
                  <CheckCircle2 className="w-3.5 h-3.5" /> Verified Clean
                </span>
              )}

              {errorCount > 0 && aiEnabled && (
                <button
                  onClick={() =>
                    handleSendAiPrompt(
                      `Fix the LaTeX compilation errors in ${activeFile.path} using a surgical SEARCH/REPLACE patch.`
                    )
                  }
                  className="px-2 py-0.5 rounded bg-red-500/20 hover:bg-red-500/30 text-red-300 border border-red-500/40 flex items-center gap-1"
                >
                  <Wand2 className="w-3 h-3" /> Fix with AI
                </button>
              )}
            </div>
          </div>

          {/* Pending Surgical Diff Review Ribbon */}
          {pendingPatches.some((p) => p.status === 'pending') && (
            <div className="bg-emerald-950/40 border-b border-emerald-500/30 p-2.5 space-y-2 max-h-48 overflow-y-auto">
              <div className="text-xs font-semibold text-emerald-300 flex items-center justify-between">
                <span>Pending Surgical AI Diffs (Zero Full-File Rewrite)</span>
                <span className="text-[11px] font-mono">
                  Saved {astSlice.savingsPercent}% Context Tokens
                </span>
              </div>
              {pendingPatches
                .filter((p) => p.status === 'pending')
                .map((patch) => (
                  <div
                    key={patch.id}
                    className="p-2 rounded bg-zinc-950 border border-zinc-800 text-xs font-mono space-y-1.5"
                  >
                    <div className="text-red-400 line-through whitespace-pre-wrap bg-red-950/30 px-2 py-1 rounded">
                      {patch.searchBlock}
                    </div>
                    <div className="text-emerald-400 whitespace-pre-wrap bg-emerald-950/30 px-2 py-1 rounded">
                      {patch.replaceBlock}
                    </div>
                    <div className="flex justify-end gap-2 pt-1 font-sans">
                      <button
                        onClick={() => handleAcceptPatch(patch)}
                        className="px-2.5 py-1 rounded bg-emerald-600 hover:bg-emerald-500 text-white text-xs flex items-center gap-1 font-semibold"
                      >
                        <Check className="w-3 h-3" /> Accept Patch
                      </button>
                      <button
                        onClick={() =>
                          setPendingPatches((prev) =>
                            prev.map((p) => (p.id === patch.id ? { ...p, status: 'rejected' } : p))
                          )
                        }
                        className="px-2.5 py-1 rounded bg-zinc-800 hover:bg-zinc-700 text-zinc-300 text-xs flex items-center gap-1"
                      >
                        <X className="w-3 h-3" /> Reject
                      </button>
                    </div>
                  </div>
                ))}
            </div>
          )}

          {/* Code Editor + Deterministic Autocomplete Popup */}
          <div className="relative flex-1 flex overflow-hidden">
            {/* Line Number Gutter */}
            <div className="w-11 py-4 bg-zinc-950 border-r border-zinc-900 text-right pr-2.5 font-mono text-xs text-zinc-600 select-none">
              {activeFile.content.split('\n').map((_, idx) => {
                const lineNum = idx + 1;
                const hasError = compilation.diagnostics.some(
                  (d) => d.line === lineNum && d.severity === 'error'
                );
                return (
                  <div
                    key={lineNum}
                    className={`leading-6 ${
                      hasError ? 'text-red-400 font-bold' : lineNum === cursorLine ? 'text-emerald-400' : ''
                    }`}
                  >
                    {lineNum}
                  </div>
                );
              })}
            </div>

            <textarea
              ref={editorRef}
              value={activeFile.content}
              onChange={handleEditorChange}
              onClick={(e) => {
                const sel = e.currentTarget.selectionStart || 0;
                setCursorLine(e.currentTarget.value.slice(0, sel).split('\n').length);
              }}
              spellCheck={false}
              className="flex-1 p-4 bg-zinc-950 text-zinc-100 font-mono text-xs leading-6 focus:outline-none resize-none selection:bg-emerald-500/30"
              placeholder="Type LaTeX here... Try \cite{, \ref{, \begin{, or type 'mk ' for inline math."
            />

            {/* Deterministic Non-AI Autocomplete Overlay */}
            {completions && completions.items.length > 0 && (
              <div className="absolute left-14 top-12 z-30 w-96 rounded-lg bg-zinc-900 border border-zinc-700 shadow-2xl overflow-hidden">
                <div className="px-3 py-1.5 bg-zinc-950 border-b border-zinc-800 flex items-center justify-between text-xs text-zinc-400">
                  <span>Deterministic TeX LSP Suggestions (0 AI Tokens)</span>
                  <span className="font-mono">Click to insert</span>
                </div>
                <div className="max-h-56 overflow-y-auto divide-y divide-zinc-800/60">
                  {completions.items.map((item, idx) => (
                    <button
                      key={idx}
                      onClick={() => applyAutocompleteItem(item)}
                      className="w-full px-3 py-2 text-left hover:bg-emerald-500/15 flex items-center justify-between gap-2 text-xs transition"
                    >
                      <div>
                        <div className="font-mono text-xs text-emerald-300 font-medium">
                          {item.label}
                        </div>
                        <div className="text-[11px] text-zinc-400 truncate max-w-[260px]">
                          {item.detail}
                        </div>
                      </div>
                      <span className="text-[9px] uppercase px-1.5 py-0.5 rounded bg-zinc-800 text-zinc-300 font-mono">
                        {item.kind}
                      </span>
                    </button>
                  ))}
                </div>
              </div>
            )}
          </div>

          {/* Compiler Diagnostics Log Drawer */}
          {compilation.diagnostics.length > 0 && (
            <div className="h-28 border-t border-zinc-800 bg-zinc-950 px-3 py-2 overflow-y-auto font-mono text-[11px] space-y-1">
              {compilation.diagnostics.map((diag, i) => (
                <div
                  key={i}
                  onClick={() => setCursorLine(diag.line)}
                  className={`cursor-pointer flex items-center gap-2 ${
                    diag.severity === 'error' ? 'text-red-400' : 'text-amber-400'
                  }`}
                >
                  <span>
                    [{diag.severity.toUpperCase()}] {diag.file}:{diag.line}
                  </span>
                  <span className="truncate">{diag.message}</span>
                </div>
              ))}
            </div>
          )}
        </section>

        {/* Right / Center-Right Pane: Live Academic Paper Preview */}
        {workbenchMode !== 'focus' && (
          <section className="flex-1 flex flex-col bg-zinc-900/60 overflow-hidden">
            {/* Paper Toolbar */}
            <div className="h-9 border-b border-zinc-800/80 px-3 flex items-center justify-between bg-zinc-900/40 text-xs">
              <div className="flex items-center gap-2">
                <span className="text-zinc-400">Surface:</span>
                {(['white', 'ivory', 'night'] as PaperTheme[]).map((t) => (
                  <button
                    key={t}
                    onClick={() => setPaperTheme(t)}
                    className={`px-2 py-0.5 rounded capitalize text-[11px] ${
                      paperTheme === t
                        ? 'bg-zinc-700 text-white font-medium'
                        : 'text-zinc-400 hover:text-zinc-200'
                    }`}
                  >
                    {t}
                  </button>
                ))}
              </div>
              <button
                onClick={() => setTwoColumnPdf((prev) => !prev)}
                className="px-2 py-0.5 rounded bg-zinc-800 hover:bg-zinc-700 text-zinc-200 text-[11px]"
              >
                Layout: {twoColumnPdf ? '2-Column IEEE' : '1-Column Standard'}
              </button>
            </div>

            {/* Scrollable Paper Sheet Viewport */}
            <div className="flex-1 overflow-y-auto p-6">
              <article
                id="veritex-printable-paper"
                className={`veritex-paper-sheet paper-theme-${paperTheme} mx-auto max-w-[816px] min-h-[1056px] p-12 rounded-lg shadow-xl ${
                  twoColumnPdf ? 'columns-2 gap-8' : ''
                }`}
                dangerouslySetInnerHTML={{ __html: compilation.html }}
              />
            </div>
          </section>
        )}

        {/* Far-Right Drawer: Low-Token BYOM AI Co-Author (Hidden when Pure IDE Mode is active) */}
        {aiEnabled && (
          <aside className="w-80 border-l border-zinc-800/80 bg-zinc-900/50 flex flex-col shrink-0">
            <div className="p-3 border-b border-zinc-800/80 flex items-center justify-between">
              <div className="flex items-center gap-2">
                <Bot className="w-4 h-4 text-emerald-400" />
                <div>
                  <div className="text-xs font-semibold text-zinc-100">VeriTeX AST Agent</div>
                  <div className="text-[10px] text-zinc-400">
                    {aiConfig.provider.toUpperCase()} &bull; {aiConfig.modelId}
                  </div>
                </div>
              </div>
              <span className="text-[10px] px-1.5 py-0.5 rounded bg-emerald-500/15 text-emerald-300 border border-emerald-500/30 font-mono">
                -{astSlice.savingsPercent}% Tokens
              </span>
            </div>

            {/* Chat History */}
            <div className="flex-1 p-3 overflow-y-auto space-y-3 text-xs">
              {aiMessages.map((m, idx) => (
                <div
                  key={idx}
                  className={`p-2.5 rounded-lg leading-relaxed whitespace-pre-wrap ${
                    m.role === 'user'
                      ? 'bg-emerald-600/20 border border-emerald-500/30 text-emerald-100 ml-4'
                      : 'bg-zinc-950 border border-zinc-800 text-zinc-300 mr-2'
                  }`}
                >
                  {m.text}
                </div>
              ))}
              {aiLoading && (
                <div className="p-2.5 rounded-lg bg-zinc-950 border border-zinc-800 text-zinc-400 flex items-center gap-2">
                  <RefreshCw className="w-3.5 h-3.5 animate-spin text-emerald-400" />
                  Streaming surgical AST diff...
                </div>
              )}
            </div>

            {/* Quick Scientific Actions */}
            <div className="px-3 pt-2 flex flex-wrap gap-1 border-t border-zinc-800/80">
              <button
                onClick={() =>
                  handleSendAiPrompt(
                    'Verify the mathematical rigor of the active theorem and refine the proof using a surgical SEARCH/REPLACE block.'
                  )
                }
                className="px-2 py-1 rounded bg-zinc-800 hover:bg-zinc-700 text-[10px] text-zinc-300"
              >
                Verify Proof
              </button>
              <button
                onClick={() =>
                  handleSendAiPrompt(
                    'Polish the academic prose in this section for NeurIPS/IEEE publication standards using a surgical SEARCH/REPLACE block.'
                  )
                }
                className="px-2 py-1 rounded bg-zinc-800 hover:bg-zinc-700 text-[10px] text-zinc-300"
              >
                Polish Prose
              </button>
            </div>

            {/* Prompt Input */}
            <div className="p-3 space-y-2">
              <textarea
                value={aiPrompt}
                onChange={(e) => setAiPrompt(e.target.value)}
                onKeyDown={(e) => {
                  if (e.key === 'Enter' && !e.shiftKey) {
                    e.preventDefault();
                    handleSendAiPrompt();
                  }
                }}
                rows={3}
                placeholder={`Ask ${aiConfig.modelId} (Slicing only "${astSlice.activeSectionTitle}")...`}
                className="w-full p-2 rounded-lg bg-zinc-950 border border-zinc-800 text-xs text-zinc-200 focus:outline-none focus:ring-1 focus:ring-emerald-500 font-sans resize-none"
              />
              <button
                onClick={() => handleSendAiPrompt()}
                disabled={aiLoading}
                className="w-full py-1.5 rounded-lg bg-emerald-600 hover:bg-emerald-500 text-white font-medium text-xs flex items-center justify-center gap-1.5 shadow-sm disabled:opacity-50"
              >
                <Sparkles className="w-3.5 h-3.5" /> Send Sliced Context ({astSlice.slicedTokens} tokens)
              </button>
            </div>
          </aside>
        )}
      </div>

      {/* Modal 1: 3-Step Guided BYOM (Bring Your Own Model) Setup Wizard */}
      {showByomModal && (
        <div className="fixed inset-0 z-50 bg-black/75 flex items-center justify-center p-4">
          <div className="w-full max-w-2xl rounded-xl bg-zinc-900 border border-zinc-800 shadow-2xl overflow-hidden">
            <div className="px-5 py-4 border-b border-zinc-800 flex items-center justify-between">
              <div className="flex items-center gap-2">
                <KeyRound className="w-4 h-4 text-emerald-400" />
                <h2 className="text-sm font-semibold text-white">
                  Universal BYOM (Bring-Your-Own-Model) Configuration Wizard
                </h2>
              </div>
              <button
                onClick={() => setShowByomModal(false)}
                className="text-zinc-400 hover:text-white"
              >
                <X className="w-4 h-4" />
              </button>
            </div>

            <div className="p-5 space-y-5 max-h-[80vh] overflow-y-auto text-xs">
              {/* Step 1: Select Provider */}
              <div>
                <div className="font-semibold text-zinc-200 mb-2">
                  Step 1: Choose Any Industry AI Provider or Local Engine
                </div>
                <div className="grid grid-cols-3 gap-2">
                  {Object.values(BYOM_PROVIDER_PRESETS).map((preset) => (
                    <button
                      key={preset.id}
                      onClick={() =>
                        setAiConfig((prev) => ({
                          ...prev,
                          provider: preset.id,
                          modelId: preset.defaultModel,
                          baseUrl: preset.defaultBaseUrl,
                        }))
                      }
                      className={`p-2.5 rounded-lg border text-left transition ${
                        aiConfig.provider === preset.id
                          ? 'bg-emerald-500/15 border-emerald-500 text-white'
                          : 'bg-zinc-950 border-zinc-800 text-zinc-400 hover:border-zinc-700'
                      }`}
                    >
                      <div className="font-semibold text-xs">{preset.name}</div>
                      <div className="text-[10px] text-emerald-400 mt-0.5">{preset.badge}</div>
                    </button>
                  ))}
                </div>
              </div>

              {/* Step 2: Step-by-Step Guide & Credentials */}
              <div className="p-3.5 rounded-lg bg-zinc-950 border border-zinc-800 space-y-3">
                <div className="font-semibold text-emerald-400">
                  Step 2: Quick Setup Guide for {BYOM_PROVIDER_PRESETS[aiConfig.provider].name}
                </div>
                <ul className="space-y-1 text-zinc-300">
                  {BYOM_PROVIDER_PRESETS[aiConfig.provider].setupSteps.map((step, i) => (
                    <li key={i}>{step}</li>
                  ))}
                </ul>

                <div className="grid grid-cols-2 gap-3 pt-2">
                  <div>
                    <label className="block text-[11px] text-zinc-400 mb-1">Model ID</label>
                    <input
                      value={aiConfig.modelId}
                      onChange={(e) => setAiConfig((p) => ({ ...p, modelId: e.target.value }))}
                      list="veritex-model-presets"
                      className="w-full bg-zinc-900 border border-zinc-700 rounded px-2.5 py-1.5 text-zinc-100 font-mono"
                    />
                    <datalist id="veritex-model-presets">
                      {BYOM_PROVIDER_PRESETS[aiConfig.provider].models.map((m) => (
                        <option key={m} value={m} />
                      ))}
                    </datalist>
                  </div>
                  <div>
                    <label className="block text-[11px] text-zinc-400 mb-1">Endpoint Base URL</label>
                    <input
                      value={aiConfig.baseUrl}
                      onChange={(e) => setAiConfig((p) => ({ ...p, baseUrl: e.target.value }))}
                      className="w-full bg-zinc-900 border border-zinc-700 rounded px-2.5 py-1.5 text-zinc-100 font-mono"
                    />
                  </div>
                </div>

                {aiConfig.provider !== 'ollama' && (
                  <div>
                    <label className="block text-[11px] text-zinc-400 mb-1">
                      API Key (Stored strictly in your browser localStorage)
                    </label>
                    <input
                      type="password"
                      value={aiConfig.apiKey}
                      onChange={(e) => setAiConfig((p) => ({ ...p, apiKey: e.target.value }))}
                      placeholder="Paste your API key here..."
                      className="w-full bg-zinc-900 border border-zinc-700 rounded px-2.5 py-1.5 text-zinc-100 font-mono"
                    />
                  </div>
                )}
              </div>

              {/* Step 3: Live Diagnostic Handshake */}
              <div className="flex items-center justify-between pt-2">
                <button
                  onClick={async () => {
                    setHandshakeStatus({ loading: true });
                    const res = await verifyProviderConnection(aiConfig);
                    setHandshakeStatus({ loading: false, ok: res.ok, message: res.message });
                  }}
                  disabled={handshakeStatus.loading}
                  className="px-3 py-1.5 rounded-lg bg-zinc-800 hover:bg-zinc-700 text-zinc-100 font-medium flex items-center gap-1.5"
                >
                  <Terminal className="w-3.5 h-3.5 text-emerald-400" />
                  {handshakeStatus.loading ? 'Testing Handshake...' : 'Step 3: Verify & Benchmark Handshake'}
                </button>

                <button
                  onClick={() => {
                    localStorage.setItem('veritex_byom_config_v1', JSON.stringify(aiConfig));
                    setShowByomModal(false);
                  }}
                  className="px-4 py-1.5 rounded-lg bg-emerald-600 hover:bg-emerald-500 text-white font-semibold"
                >
                  Save & Activate Model
                </button>
              </div>

              {handshakeStatus.message && (
                <div
                  className={`p-2.5 rounded-lg border text-xs ${
                    handshakeStatus.ok
                      ? 'bg-emerald-950/40 border-emerald-500/40 text-emerald-300'
                      : 'bg-red-950/40 border-red-500/40 text-red-300'
                  }`}
                >
                  {handshakeStatus.message}
                </div>
              )}
            </div>
          </div>
        </div>
      )}

      {/* Modal 2: 4-Stage PDF-to-LaTeX Decompiler Modal */}
      {showPdfDecompilerModal && (
        <div className="fixed inset-0 z-50 bg-black/75 flex items-center justify-center p-4">
          <div className="w-full max-w-lg rounded-xl bg-zinc-900 border border-zinc-800 shadow-2xl overflow-hidden">
            <div className="px-5 py-4 border-b border-zinc-800 flex items-center justify-between">
              <div className="flex items-center gap-2">
                <FileUp className="w-4 h-4 text-amber-400" />
                <h2 className="text-sm font-semibold text-white">
                  4-Stage PDF-to-LaTeX Decompiler &amp; BibTeX Reconstructor
                </h2>
              </div>
              <button
                onClick={() => setShowPdfDecompilerModal(false)}
                className="text-zinc-400 hover:text-white"
              >
                <X className="w-4 h-4" />
              </button>
            </div>

            <div className="p-5 space-y-4 text-xs">
              <p className="text-zinc-300 leading-relaxed">
                Upload any scientific PDF manuscript. VeriTeX segments the layout, extracts equations and sections into
                compilable multi-file LaTeX, and queries Crossref to verify real DOIs.
              </p>

              <label className="block border-2 border-dashed border-zinc-700 hover:border-emerald-500/60 rounded-xl p-6 text-center cursor-pointer transition">
                <FileUp className="w-7 h-7 text-emerald-400 mx-auto mb-2" />
                <div className="font-semibold text-zinc-200">Click to select a PDF manuscript</div>
                <div className="text-[11px] text-zinc-400 mt-1">
                  Automatically generates main.tex, sections/methodology.tex &amp; references.bib
                </div>
                <input
                  type="file"
                  accept=".pdf"
                  onChange={handlePdfUploadToDecompile}
                  className="hidden"
                />
              </label>

              {decompileProgress.stage && (
                <div className="p-3 rounded-lg bg-zinc-950 border border-zinc-800 space-y-2">
                  <div className="flex justify-between text-emerald-400 font-medium">
                    <span>{decompileProgress.stage}</span>
                    <span>{decompileProgress.percent}%</span>
                  </div>
                  <div className="w-full h-1.5 bg-zinc-800 rounded-full overflow-hidden">
                    <div
                      className="h-full bg-emerald-500 transition-all duration-300"
                      style={{ width: `${decompileProgress.percent}%` }}
                    />
                  </div>
                  {decompileProgress.logs.map((log, i) => (
                    <div key={i} className="text-[11px] text-zinc-400 font-mono">
                      &bull; {log}
                    </div>
                  ))}
                </div>
              )}
            </div>
          </div>
        </div>
      )}
    </div>
  );
}

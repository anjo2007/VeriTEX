'use client';

import React, { useEffect, useMemo, useRef, useState, useCallback } from 'react';
import JSZip from 'jszip';
import katex from 'katex';
import {
  AlertCircle,
  AlertTriangle,
  AlignLeft,
  ArrowRight,
  BookOpen,
  Bot,
  Check,
  CheckCircle2,
  ChevronDown,
  ChevronRight,
  Clock,
  Command,
  Copy,
  Cpu,
  Download,
  ExternalLink,
  Eye,
  FileCode,
  FilePlus2,
  FileText,
  FolderOpen,
  FolderPlus,
  GitCompare,
  Globe,
  Info,
  KeyRound,
  Layers,
  Layout,
  Maximize2,
  Minimize2,
  PanelLeftClose,
  PanelLeftOpen,
  Plus,
  Printer,
  RefreshCw,
  Search,
  Settings,
  Share2,
  ShieldCheck,
  Sigma,
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
  CompilerDiagnostic,
  PaperTheme,
  SidebarTab,
  SurgicalDiffPatch,
  VeriTeXProject,
  VirtualFile,
  WorkbenchMode,
} from '@/types/veritex';
import {
  applySurgicalPatch,
  detectActiveMathAtCursor,
  getDeterministicCompletions,
  highlightLatexSyntax,
  parseSurgicalDiffs,
  sliceLatexASTContext,
} from '@/lib/latex-ide-engine';
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
We introduce Geodesic Spectral Attention, reducing long-sequence transformer complexity from $\\mathcal{O}(N^2)$ to $\\mathcal{O}(N \\log N)$ using discrete Laplace-Beltrami eigenbasis projections on Riemannian manifolds.
\\end{abstract}

\\section{Theoretical Formulation}\\label{sec:theory}
Let $\\M \\subset \\R^d$ be a smooth compact Riemannian manifold without boundary. For query-key projection matrices $Q, K \\in \\R^{N \\times d}$, the classical geodesic heat kernel satisfies:

\\begin{equation}
  \\label{eq:heat_kernel}
  \\mathcal{K}_t(x, y) = (4\\pi t)^{-d/2} \\exp\\left( -\\frac{d_{\\M}^2(x, y)}{4t} \\right)
\\end{equation}

By expanding $\\mathcal{K}_t$ across the orthonormal spectral eigenfunctions $\\{\\phi_k\\}_{k=1}^{\\infty}$ of the Laplace-Beltrami operator $\\Delta_{\\M}$, we decouple non-local spatial interactions into independent harmonic coordinates:

\\begin{align}
  \\mathcal{K}_t(x, y) &= \\sum_{k=1}^{\\infty} e^{-\\lambda_k t} \\phi_k(x) \\phi_k(y) \\label{eq:spectral_decomp} \\\\
  \\mathcal{A}(Q, K)   &= \\operatorname{Softmax}\\left( \\Phi_Q \\Lambda_t \\Phi_K^T \\right)
\\end{align}

\\begin{theorem}[Spectral Convergence Bound]\\label{thm:convergence}
Let $\\Delta_{\\M}$ have discrete eigenvalues $0 = \\lambda_0 < \\lambda_1 \\le \\lambda_2 \\le \\dots$. Truncating the spectral series at $K = \\mathcal{O}(\\log N)$ modes yields uniform approximation error bounded by:
\\begin{equation}
  \\sup_{x, y \\in \\M} \\left| \\mathcal{K}_t(x, y) - \\sum_{k=1}^K e^{-\\lambda_k t} \\phi_k(x) \\phi_k(y) \\right| \\le C \\cdot N^{-2}
\\end{equation}
\\end{theorem}

\\begin{proof}
By applying Weyl's asymptotic formula for compact Riemannian manifolds, the eigenvalue growth obeys $\\lambda_k \\sim c_d (k / \\operatorname{Vol}(\\M))^{2/d}$. Exponential damping of higher harmonics guarantees sub-linear geometric decay, completing the proof.
\\end{proof}

\\section{Empirical Benchmarks}\\label{sec:experiments}
We evaluated VeriTeX on standard scientific sequence benchmarks against traditional quadratic attention mechanisms \\cite{vaswani2017}.

\\begin{table}[htbp]
  \\centering
  \\caption{Computational complexity and empirical throughput comparison.}
  \\label{tab:benchmark}
  \\begin{tabular}{lccc}
    \\toprule
    Attention Architecture & Asymptotic Complexity & 64k Tokens (ms) & Accuracy (\\%) \\\\
    \\midrule
    Standard Softmax Attention & $\\mathcal{O}(N^2)$ & 412.8 & 84.2 \\\\
    Linear FlashAttention-3    & $\\mathcal{O}(N)$   & 48.5  & 81.6 \\\\
    VeriTeX Geodesic (Ours)    & $\\mathcal{O}(N \\log N)$ & \\textbf{36.2} & \\textbf{89.7} \\\\
    \\bottomrule
  \\end{tabular}
\\end{table}

As shown in Eq. \\eqref{eq:spectral_decomp} and Table \\ref{tab:benchmark}, the spectral projection provides superior representation fidelity with minimal overhead.

\\bibliography{references}
\\end{document}`,
    },
    'references.bib': {
      path: 'references.bib',
      type: 'bib',
      updatedAt: Date.now(),
      content: `@article{vaswani2017,
  author = {Vaswani, Ashish and Shazeer, Noam and Parmar, Niki and Uszkoreit, Jakob and Jones, Llion and Gomez, Aidan N and Kaiser, {\\L}ukasz and Polosukhin, Illia},
  title = {Attention Is All You Need},
  journal = {Advances in Neural Information Processing Systems (NeurIPS)},
  year = {2017},
  doi = {10.48550/arXiv.1706.03762}
}

@article{beltrami1868,
  author = {Beltrami, Eugenio},
  title = {Saggio di interpretazione della geometria non-euclidea},
  journal = {Giornale di Matematiche},
  year = {1868}
}`,
    },
  },
};

const TEMPLATES: Record<string, { name: string; desc: string; files: Record<string, string> }> = {
  nature: {
    name: 'Nature / Science Journal',
    desc: 'High-impact publication template with abstract, methods, and references',
    files: {
      'main.tex': `\\documentclass[10pt,twocolumn]{article}
\\usepackage{amsmath,amssymb,booktabs,graphicx}

\\title{Scalable Machine Learning on Manifold Geometries}
\\author{Jane Doe$^{1,*}$, John Smith$^{2}$ \\\\
\\small $^{1}$Department of Computer Science, University of Oxford \\\\
\\small $^{2}$Institute for Advanced Study, Princeton \\\\
\\small $^*$Corresponding author: j.doe@ox.ac.uk}
\\date{\\today}

\\begin{document}
\\maketitle

\\begin{abstract}
Geometric deep learning expands modern AI to non-Euclidean domains. Here we present an exact polynomial-time algorithm for geodesic embeddings, achieving order-of-magnitude accelerations.
\\end{abstract}

\\section{Introduction}
Non-Euclidean representations enable structural modeling of molecular topologies and complex graphs.

\\section{Methods}
We employ discretized Riemannian gradient flows over discrete triangulated manifolds.

\\section{Results}
Benchmarking demonstrates robust convergence across all parameter regimes.

\\bibliography{references}
\\end{document}`,
      'references.bib': `@article{nature2026,
  author = {Doe, Jane and Smith, John},
  title = {Geometric Deep Learning Foundations},
  journal = {Nature Machine Intelligence},
  year = {2026}
}`,
    },
  },
  ieee: {
    name: 'IEEE Transactions',
    desc: 'Two-column engineering & computer science standard layout',
    files: {
      'main.tex': `\\documentclass[conference]{IEEEtran}
\\usepackage{amsmath,amssymb,booktabs}

\\title{VeriTeX: Real-Time Deterministic Compilation for Collaborative LaTeX}
\\author{\\IEEEauthorblockN{First Author}
\\IEEEauthorblockA{School of Electrical Engineering\\\\
University of Cambridge\\\\
Cambridge, UK}
\\and
\\IEEEauthorblockN{Second Author}
\\IEEEauthorblockA{Department of Computer Science\\\\
MIT\\\\
Cambridge, USA}}

\\begin{document}
\\maketitle

\\begin{abstract}
This paper formulates an in-browser incremental AST compilation pipeline with zero server compute overhead.
\\end{abstract}

\\section{Introduction}
Modern cloud LaTeX platforms suffer from severe latency bottlenecks during peak load.

\\section{System Architecture}
Our architecture offloads all compilation to WebAssembly and KaTeX client workers.

\\end{document}`,
      'references.bib': `@article{ieee2025,
  author = {Author, First and Author, Second},
  title = {Distributed Web-Native Scientific Toolchains},
  journal = {IEEE Transactions on Software Engineering},
  year = {2025}
}`,
    },
  },
  neurips: {
    name: 'NeurIPS / ICML Conference',
    desc: 'Premier machine learning format with math environments & booktabs',
    files: {
      'main.tex': `\\documentclass{article}
\\usepackage{amsmath,amssymb,amsthm,booktabs}

\\title{Sub-Quadratic Transformers via Manifold Projections}
\\author{Anonymous Authors \\\\ Paper under double-blind review}
\\date{}

\\begin{document}
\\maketitle

\\begin{abstract}
Attention mechanisms typically require $\\mathcal{O}(N^2)$ space and time. We prove that harmonic spectral decomposition reduces complexity to sub-quadratic regimes without loss of expressivity.
\\end{abstract}

\\section{Introduction}
Long-sequence context windows represent a core challenge in large language modeling.

\\section{Harmonic Attention Formulation}
\\begin{equation}
  \\mathcal{H}(X) = \\sum_{k=1}^K \\sigma(\\lambda_k) \\cdot \\psi_k(X)
\\end{equation}

\\end{document}`,
      'references.bib': `@inproceedings{neurips2024,
  author = {Anonymous, A.},
  title = {Spectral Attention in High Dimensions},
  booktitle = {NeurIPS},
  year = {2024}
}`,
    },
  },
  thesis: {
    name: 'Academic Thesis / Multi-File',
    desc: 'Modular dissertation structure with \\input{} and relative paths',
    files: {
      'main.tex': `\\documentclass[12pt]{report}
\\usepackage{amsmath,amssymb,booktabs}

\\title{Harmonic Analysis and Spectral Geometry in Deep Learning}
\\author{Candidate for Doctor of Philosophy}
\\date{2026}

\\begin{document}
\\maketitle

\\begin{abstract}
This dissertation investigates the intersection of differential geometry and deep neural representation learning.
\\end{abstract}

\\input{./sections/intro.tex}
\\input{./sections/methods.tex}

\\bibliography{references}
\\end{document}`,
      'sections/intro.tex': `\\section{Introduction to Spectral Geometry}
The study of Laplace-Beltrami operators on Riemannian manifolds provides fundamental geometric invariants.`,
      'sections/methods.tex': `\\section{Numerical Discretization}
We adopt finite element triangulations to compute empirical spectrum matrices with high numeric precision.`,
      'references.bib': `@book{doCarmo1992,
  author = {do Carmo, Manfredo P.},
  title = {Riemannian Geometry},
  publisher = {Birkh\\"{a}user},
  year = {1992}
}`,
    },
  },
};

const MATH_SYMBOLS: Array<{ category: string; symbols: Array<{ label: string; insert: string; desc: string }> }> = [
  {
    category: 'Greek Letters',
    symbols: [
      { label: 'α', insert: '\\alpha', desc: 'alpha' },
      { label: 'β', insert: '\\beta', desc: 'beta' },
      { label: 'γ', insert: '\\gamma', desc: 'gamma' },
      { label: 'θ', insert: '\\theta', desc: 'theta' },
      { label: 'λ', insert: '\\lambda', desc: 'lambda' },
      { label: 'μ', insert: '\\mu', desc: 'mu' },
      { label: 'σ', insert: '\\sigma', desc: 'sigma' },
      { label: 'ω', insert: '\\omega', desc: 'omega' },
      { label: 'Δ', insert: '\\Delta', desc: 'Delta' },
      { label: 'Ω', insert: '\\Omega', desc: 'Omega' },
      { label: '∇', insert: '\\nabla', desc: 'nabla' },
    ],
  },
  {
    category: 'Operators & Calculus',
    symbols: [
      { label: '∑', insert: '\\sum_{i=1}^{N} ', desc: 'summation' },
      { label: '∫', insert: '\\int_{a}^{b} ', desc: 'integral' },
      { label: '∏', insert: '\\prod_{i=1}^{N} ', desc: 'product' },
      { label: '∂', insert: '\\partial ', desc: 'partial derivative' },
      { label: 'a/b', insert: '\\frac{a}{b}', desc: 'fraction' },
      { label: '√x', insert: '\\sqrt{x}', desc: 'square root' },
      { label: '∞', insert: '\\infty', desc: 'infinity' },
      { label: 'lim', insert: '\\lim_{x \\to 0} ', desc: 'limit' },
    ],
  },
  {
    category: 'Relations & Logic',
    symbols: [
      { label: '≤', insert: '\\le ', desc: 'less or equal' },
      { label: '≥', insert: '\\ge ', desc: 'greater or equal' },
      { label: '≠', insert: '\\ne ', desc: 'not equal' },
      { label: '≈', insert: '\\approx ', desc: 'approx' },
      { label: '∈', insert: '\\in ', desc: 'element of' },
      { label: '⊂', insert: '\\subset ', desc: 'subset' },
      { label: '∀', insert: '\\forall ', desc: 'for all' },
      { label: '∃', insert: '\\exists ', desc: 'exists' },
      { label: '→', insert: '\\to ', desc: 'to arrow' },
      { label: '⇒', insert: '\\implies ', desc: 'implies' },
      { label: '⇔', insert: '\\iff ', desc: 'if and only if' },
    ],
  },
  {
    category: 'Blackboard & Sets',
    symbols: [
      { label: 'ℝ', insert: '\\mathbb{R}', desc: 'real numbers' },
      { label: 'ℂ', insert: '\\mathbb{C}', desc: 'complex numbers' },
      { label: 'ℤ', insert: '\\mathbb{Z}', desc: 'integers' },
      { label: '𝔼', insert: '\\mathbb{E}\\left[ X \\right]', desc: 'expectation' },
      { label: '𝒪', insert: '\\mathcal{O}(N)', desc: 'big O' },
      { label: 'ℳ', insert: '\\mathcal{M}', desc: 'manifold' },
      { label: '‖·‖', insert: '\\left\\lVert x \\right\\rVert', desc: 'norm' },
    ],
  },
];

export default function VeriTeXStudioPage() {
  // Project & File State
  const [project, setProject] = useState<VeriTeXProject>(INITIAL_PROJECT);
  const [activeFilePath, setActiveFilePath] = useState<string>('main.tex');
  const [openTabs, setOpenTabs] = useState<string[]>(['main.tex', 'references.bib']);
  const [workbenchMode, setWorkbenchMode] = useState<WorkbenchMode>('studio');
  const [paperTheme, setPaperTheme] = useState<PaperTheme>('white');
  const [paperZoom, setPaperZoom] = useState<number>(100);

  // Editor Navigation & Cursor State
  const [cursorLine, setCursorLine] = useState<number>(1);
  const [cursorOffset, setCursorOffset] = useState<number>(0);
  const [activeMathFormula, setActiveMathFormula] = useState<string | null>(null);

  // Layout & Panes
  const [sidebarTab, setSidebarTab] = useState<SidebarTab>('explorer');
  const [sidebarCollapsed, setSidebarCollapsed] = useState<boolean>(false);
  const [splitWidth, setSplitWidth] = useState<number>(50); // percentage (25 to 75)
  const isDraggingSplitter = useRef<boolean>(false);

  // LSP Autocomplete State
  const [completions, setCompletions] = useState<{ trigger: string; items: AutocompleteItem[] } | null>(null);
  const [selectedCompletionIdx, setSelectedCompletionIdx] = useState<number>(0);

  // AI & BYOM State
  const [aiConfig, setAiConfig] = useState<AIProviderConfig>({
    provider: 'gemini',
    modelId: 'gemini-2.5-flash',
    apiKey: '',
    baseUrl: 'https://generativelanguage.googleapis.com/v1beta',
    temperature: 0.2,
    useAstSlicer: true,
    verifyCitations: true,
  });
  const [byomModalOpen, setByomModalOpen] = useState<boolean>(false);
  const [handshakeStatus, setHandshakeStatus] = useState<{ testing: boolean; message: string; ok?: boolean } | null>(null);
  const [aiPromptInput, setAiPromptInput] = useState<string>('');
  const [aiGenerating, setAiGenerating] = useState<boolean>(false);
  const [pendingPatches, setPendingPatches] = useState<SurgicalDiffPatch[]>([]);

  // Inline Ctrl+K Command Bar
  const [inlineAiOpen, setInlineAiOpen] = useState<boolean>(false);
  const [inlineAiPrompt, setInlineAiPrompt] = useState<string>('');

  // Global Command Palette (Ctrl+P)
  const [commandPaletteOpen, setCommandPaletteOpen] = useState<boolean>(false);
  const [paletteSearch, setPaletteSearch] = useState<string>('');
  const [paletteSelectedIdx, setPaletteSelectedIdx] = useState<number>(0);

  // Crossref Citation Search
  const [crossrefQuery, setCrossrefQuery] = useState<string>('');
  const [crossrefLoading, setCrossrefLoading] = useState<boolean>(false);
  const [crossrefResults, setCrossrefResults] = useState<BibEntry[]>([]);

  // P2P Collaboration Session (WebRTC + CRDT)
  const [collabRoomId, setCollabRoomId] = useState<string>('veritex-mesh');
  const [collabPeers, setCollabPeers] = useState<CollaboratorPeer[]>([]);
  const [collabModalOpen, setCollabModalOpen] = useState<boolean>(false);
  const collabSessionRef = useRef<VeriTeXCollabSession | null>(null);
  const [collabCopied, setCollabCopied] = useState<boolean>(false);

  // Decompiler State
  const [decompileProgress, setDecompileProgress] = useState<{
    active: boolean;
    stage: string;
    percent: number;
  }>({ active: false, stage: '', percent: 0 });

  // DOM Refs
  const editorRef = useRef<HTMLTextAreaElement>(null);
  const backdropRef = useRef<HTMLDivElement>(null);
  const gutterRef = useRef<HTMLDivElement>(null);
  const mathInspectorRef = useRef<HTMLDivElement>(null);
  const previewScrollRef = useRef<HTMLDivElement>(null);

  // Active File Reference
  const activeFile = useMemo(() => {
    return project.files[activeFilePath] || project.files[project.rootFile] || Object.values(project.files)[0];
  }, [project.files, activeFilePath, project.rootFile]);

  // Fast Incremental LaTeX Compilation (<10ms)
  const compilation = useMemo(() => {
    return compileVeriTeXWorkspace(project.files, project.rootFile);
  }, [project.files, project.rootFile]);

  // AST Context Slicer for low-token AI queries
  const astSlice = useMemo(() => {
    return sliceLatexASTContext(project.files, activeFile?.path || project.rootFile, cursorLine);
  }, [project.files, activeFile?.path, project.rootFile, cursorLine]);

  // Syntax Highlighted HTML code for background layer
  const highlightedSource = useMemo(() => {
    if (!activeFile) return '';
    return highlightLatexSyntax(activeFile.content, activeFile.type);
  }, [activeFile]);

  // Initialize P2P Collaboration Session on Mount
  useEffect(() => {
    if (typeof window === 'undefined') return;

    // Check for ?room= in URL
    const params = new URLSearchParams(window.location.search);
    const roomParam = params.get('room') || 'veritex-mesh';
    setCollabRoomId(roomParam);

    try {
      const session = new VeriTeXCollabSession(roomParam);
      collabSessionRef.current = session;
      session.onPeersChange((peers) => {
        setCollabPeers(peers);
      });
    } catch {
      // Offline fallback
    }

    return () => {
      collabSessionRef.current?.destroy();
    };
  }, []);

  // Sync cursor presence to collaboration peers
  useEffect(() => {
    collabSessionRef.current?.updatePresence(activeFile?.path || 'main.tex', cursorLine);
  }, [activeFile?.path, cursorLine]);

  // Synchronize Scroll between Textarea, Highlight Backdrop, and Gutter
  const handleEditorScroll = (e: React.UIEvent<HTMLTextAreaElement>) => {
    const { scrollTop, scrollLeft } = e.currentTarget;
    if (backdropRef.current) {
      backdropRef.current.scrollTop = scrollTop;
      backdropRef.current.scrollLeft = scrollLeft;
    }
    if (gutterRef.current) {
      gutterRef.current.scrollTop = scrollTop;
    }
  };

  // Keyboard shortcut listener (Ctrl+K, Ctrl+P, Ctrl+B, Ctrl+S)
  useEffect(() => {
    const handleGlobalKeyDown = (e: KeyboardEvent) => {
      const isMac = typeof navigator !== 'undefined' && /Mac|iPod|iPhone|iPad/.test(navigator.platform);
      const mod = isMac ? e.metaKey : e.ctrlKey;

      if (mod && e.key.toLowerCase() === 'k') {
        e.preventDefault();
        setInlineAiOpen((prev) => !prev);
      } else if (mod && e.key.toLowerCase() === 'p') {
        e.preventDefault();
        setCommandPaletteOpen((prev) => !prev);
      } else if (mod && e.key.toLowerCase() === 'b') {
        e.preventDefault();
        setSidebarCollapsed((prev) => !prev);
      } else if (mod && e.key === 'Enter') {
        // Accept first pending patch or trigger compile
        if (pendingPatches.length > 0) {
          e.preventDefault();
          handleAcceptPatch(pendingPatches[0]);
        }
      } else if (e.key === 'Escape') {
        if (completions) setCompletions(null);
        if (inlineAiOpen) setInlineAiOpen(false);
        if (commandPaletteOpen) setCommandPaletteOpen(false);
      }
    };

    window.addEventListener('keydown', handleGlobalKeyDown);
    return () => window.removeEventListener('keydown', handleGlobalKeyDown);
  }, [completions, inlineAiOpen, commandPaletteOpen, pendingPatches]);

  // Detect Math at Cursor for Floating HUD Inspector
  useEffect(() => {
    if (!activeFile) return;
    const math = detectActiveMathAtCursor(activeFile.content, cursorOffset);
    setActiveMathFormula(math);
  }, [activeFile, cursorOffset]);

  // Update active file content
  const updateFileContent = useCallback((path: string, newContent: string) => {
    setProject((prev) => ({
      ...prev,
      updatedAt: Date.now(),
      files: {
        ...prev.files,
        [path]: {
          ...prev.files[path],
          content: newContent,
          updatedAt: Date.now(),
        },
      },
    }));
  }, []);

  // Bracket Auto-Closing, Smart Indentation, and LSP Autocomplete in Textarea KeyDown
  const handleEditorKeyDown = (e: React.KeyboardEvent<HTMLTextAreaElement>) => {
    const textarea = e.currentTarget;
    const { selectionStart, selectionEnd, value } = textarea;

    // 1. If Autocomplete menu is open, handle keyboard navigation
    if (completions && completions.items.length > 0) {
      if (e.key === 'ArrowDown') {
        e.preventDefault();
        setSelectedCompletionIdx((prev) => (prev + 1) % completions.items.length);
        return;
      }
      if (e.key === 'ArrowUp') {
        e.preventDefault();
        setSelectedCompletionIdx((prev) => (prev - 1 + completions.items.length) % completions.items.length);
        return;
      }
      if (e.key === 'Enter' || e.key === 'Tab') {
        e.preventDefault();
        insertAutocompleteItem(completions.items[selectedCompletionIdx]);
        return;
      }
      if (e.key === 'Escape') {
        e.preventDefault();
        setCompletions(null);
        return;
      }
    }

    // 2. Bracket Auto-Closing
    const pairs: Record<string, string> = {
      '{': '}',
      '[': ']',
      '(': ')',
      $: '$',
    };

    if (pairs[e.key] && selectionStart === selectionEnd) {
      e.preventDefault();
      const closeChar = pairs[e.key];
      const nextContent = value.slice(0, selectionStart) + e.key + closeChar + value.slice(selectionEnd);
      updateFileContent(activeFile.path, nextContent);
      setTimeout(() => {
        if (editorRef.current) {
          editorRef.current.selectionStart = selectionStart + 1;
          editorRef.current.selectionEnd = selectionStart + 1;
        }
      }, 0);
      return;
    }

    // 3. Backspace inside empty pair deletes both
    if (e.key === 'Backspace' && selectionStart === selectionEnd && selectionStart > 0) {
      const charBefore = value[selectionStart - 1];
      const charAfter = value[selectionStart];
      if (
        (charBefore === '{' && charAfter === '}') ||
        (charBefore === '[' && charAfter === ']') ||
        (charBefore === '(' && charAfter === ')') ||
        (charBefore === '$' && charAfter === '$')
      ) {
        e.preventDefault();
        const nextContent = value.slice(0, selectionStart - 1) + value.slice(selectionStart + 1);
        updateFileContent(activeFile.path, nextContent);
        setTimeout(() => {
          if (editorRef.current) {
            editorRef.current.selectionStart = selectionStart - 1;
            editorRef.current.selectionEnd = selectionStart - 1;
          }
        }, 0);
        return;
      }
    }

    // 4. Enter inside braces expands with smart indentation
    if (e.key === 'Enter' && selectionStart === selectionEnd && selectionStart > 0) {
      const charBefore = value[selectionStart - 1];
      const charAfter = value[selectionStart];
      if (charBefore === '{' && charAfter === '}') {
        e.preventDefault();
        const nextContent = value.slice(0, selectionStart) + '\n  \n' + value.slice(selectionEnd);
        updateFileContent(activeFile.path, nextContent);
        setTimeout(() => {
          if (editorRef.current) {
            editorRef.current.selectionStart = selectionStart + 3;
            editorRef.current.selectionEnd = selectionStart + 3;
          }
        }, 0);
        return;
      }
    }

    // 5. Tab key indentation
    if (e.key === 'Tab') {
      e.preventDefault();
      const nextContent = value.slice(0, selectionStart) + '  ' + value.slice(selectionEnd);
      updateFileContent(activeFile.path, nextContent);
      setTimeout(() => {
        if (editorRef.current) {
          editorRef.current.selectionStart = selectionStart + 2;
          editorRef.current.selectionEnd = selectionStart + 2;
        }
      }, 0);
      return;
    }
  };

  // Textarea input change + LSP trigger detection
  const handleEditorChange = (e: React.ChangeEvent<HTMLTextAreaElement>) => {
    const content = e.target.value;
    const selStart = e.target.selectionStart;
    updateFileContent(activeFile.path, content);
    setCursorOffset(selStart);

    // Compute active line number
    const lineNum = content.slice(0, selStart).split('\n').length;
    setCursorLine(lineNum);

    // Compute line prefix up to cursor for deterministic LSP autocomplete
    const lineStartIdx = content.lastIndexOf('\n', selStart - 1) + 1;
    const linePrefix = content.slice(lineStartIdx, selStart);

    const match = getDeterministicCompletions(
      linePrefix,
      compilation.labels,
      compilation.bibEntries,
      compilation.macros
    );

    if (match && match.items.length > 0) {
      setCompletions(match);
      setSelectedCompletionIdx(0);
    } else {
      setCompletions(null);
    }
  };

  // Insert selected autocomplete item
  const insertAutocompleteItem = (item: AutocompleteItem) => {
    if (!editorRef.current || !completions) return;
    const textarea = editorRef.current;
    const content = textarea.value;
    const selStart = textarea.selectionStart;

    const lineStartIdx = content.lastIndexOf('\n', selStart - 1) + 1;
    const linePrefix = content.slice(lineStartIdx, selStart);

    // Determine how much prefix to replace
    let replaceFrom = selStart;
    if (item.kind === 'citation') {
      const match = linePrefix.match(/\\cite[a-zA-Z]*\{([^}]*)$/);
      if (match) replaceFrom = selStart - match[1].length;
    } else if (item.kind === 'reference') {
      const match = linePrefix.match(/\\(?:eq)?ref\{([^}]*)$/);
      if (match) replaceFrom = selStart - match[1].length;
    } else if (item.kind === 'environment') {
      const match = linePrefix.match(/\\begin\{([a-zA-Z*]*)$/);
      if (match) replaceFrom = selStart - match[1].length;
    } else {
      const match = linePrefix.match(/\\([a-zA-Z]*)$/);
      if (match) replaceFrom = selStart - match[1].length;
    }

    const nextContent = content.slice(0, replaceFrom) + item.insertText + content.slice(selStart);
    updateFileContent(activeFile.path, nextContent);
    setCompletions(null);

    setTimeout(() => {
      if (editorRef.current) {
        editorRef.current.focus();
        const newCursorPos = replaceFrom + item.insertText.length;
        editorRef.current.selectionStart = newCursorPos;
        editorRef.current.selectionEnd = newCursorPos;
      }
    }, 0);
  };

  // Accept a surgical patch
  const handleAcceptPatch = (patch: SurgicalDiffPatch) => {
    const { updatedSource, applied } = applySurgicalPatch(activeFile.content, patch);
    if (applied) {
      updateFileContent(activeFile.path, updatedSource);
      setPendingPatches((prev) => prev.filter((p) => p.id !== patch.id));
    }
  };

  // Reject patch
  const handleRejectPatch = (patchId: string) => {
    setPendingPatches((prev) => prev.filter((p) => p.id !== patchId));
  };

  // AI Prompt Dispatch (Inline Ctrl+K or Sidebar Chat)
  const handleSendAiPrompt = async (promptText: string) => {
    if (!promptText.trim()) return;
    setAiGenerating(true);
    setInlineAiOpen(false);

    try {
      const diagnosticsSummary = compilation.diagnostics
        .filter((d) => d.severity === 'error')
        .map((d) => `Line ${d.line}: ${d.message}`)
        .join('\n');

      const response = await invokeVeriTeXAgent(aiConfig, promptText, astSlice, diagnosticsSummary);
      const patches = parseSurgicalDiffs(response);

      if (patches.length > 0) {
        setPendingPatches((prev) => [...prev, ...patches]);
      } else {
        // Fallback: If model answered with direct code without search/replace tags
        const searchCandidate = astSlice.activeBlockContent;
        if (searchCandidate && response.length > 20) {
          const rawCode = response.replace(/^```latex\n?|^```\n?|```$/gm, '').trim();
          setPendingPatches((prev) => [
            ...prev,
            {
              id: `patch-${Date.now()}`,
              searchBlock: searchCandidate,
              replaceBlock: rawCode,
              explanation: `AI Generation for: "${promptText.slice(0, 40)}..."`,
              status: 'pending',
            },
          ]);
        }
      }
    } catch (err: unknown) {
      alert(err instanceof Error ? err.message : 'AI generation error.');
    } finally {
      setAiGenerating(false);
      setAiPromptInput('');
      setInlineAiPrompt('');
    }
  };

  // Split-Pane Drag Resizer
  const handleSplitterMouseDown = (e: React.MouseEvent) => {
    e.preventDefault();
    isDraggingSplitter.current = true;

    const onMouseMove = (moveEvent: MouseEvent) => {
      if (!isDraggingSplitter.current) return;
      const totalWidth = window.innerWidth - (sidebarCollapsed ? 48 : 288);
      const newPercent = Math.max(25, Math.min(75, (moveEvent.clientX / totalWidth) * 100));
      setSplitWidth(newPercent);
    };

    const onMouseUp = () => {
      isDraggingSplitter.current = false;
      window.removeEventListener('mousemove', onMouseMove);
      window.removeEventListener('mouseup', onMouseUp);
    };

    window.addEventListener('mousemove', onMouseMove);
    window.addEventListener('mouseup', onMouseUp);
  };

  // SyncTeX: Double-click in Paper Preview jumps editor to exact line
  const handlePreviewDoubleClick = (e: React.MouseEvent) => {
    let target = e.target as HTMLElement | null;
    while (target && target !== previewScrollRef.current) {
      const lineAttr = target.getAttribute('data-source-line');
      if (lineAttr) {
        const line = parseInt(lineAttr, 10);
        jumpToLine(line);
        break;
      }
      target = target.parentElement;
    }
  };

  // Jump to specific line in editor
  const jumpToLine = (targetLine: number) => {
    if (!editorRef.current || !activeFile) return;
    const lines = activeFile.content.split('\n');
    let offset = 0;
    for (let i = 0; i < Math.min(lines.length, targetLine - 1); i++) {
      offset += lines[i].length + 1;
    }
    editorRef.current.focus();
    editorRef.current.selectionStart = offset;
    editorRef.current.selectionEnd = offset;
    setCursorLine(targetLine);
    setCursorOffset(offset);
    editorRef.current.scrollTop = Math.max(0, (targetLine - 5) * 22);
  };

  // Insert Symbol into active file
  const handleInsertSymbol = (sym: string) => {
    if (!editorRef.current) return;
    const textarea = editorRef.current;
    const start = textarea.selectionStart;
    const end = textarea.selectionEnd;
    const nextContent = activeFile.content.slice(0, start) + sym + activeFile.content.slice(end);
    updateFileContent(activeFile.path, nextContent);
    setTimeout(() => {
      if (editorRef.current) {
        editorRef.current.focus();
        editorRef.current.selectionStart = start + sym.length;
        editorRef.current.selectionEnd = start + sym.length;
      }
    }, 0);
  };

  // Switch template
  const handleLoadTemplate = (tplKey: string) => {
    const tpl = TEMPLATES[tplKey];
    if (!tpl) return;
    if (confirm(`Load template "${tpl.name}"? Current unsaved work will be replaced.`)) {
      const newFiles: Record<string, VirtualFile> = {};
      Object.entries(tpl.files).forEach(([path, content]) => {
        const ext = path.split('.').pop() || 'tex';
        newFiles[path] = {
          path,
          content,
          type: ext as VirtualFile['type'],
          updatedAt: Date.now(),
        };
      });
      setProject({
        id: `tpl-${tplKey}-${Date.now()}`,
        name: tpl.name.replace(/\s+/g, '_'),
        rootFile: 'main.tex',
        createdAt: Date.now(),
        updatedAt: Date.now(),
        files: newFiles,
      });
      setActiveFilePath('main.tex');
      setOpenTabs(Object.keys(newFiles));
    }
  };

  // Export full multi-file .ZIP project
  const handleExportZip = async () => {
    const zip = new JSZip();
    Object.values(project.files).forEach((f) => {
      zip.file(f.path, f.content);
    });
    const blob = await zip.generateAsync({ type: 'blob' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `${project.name || 'veritex-project'}.zip`;
    a.click();
    URL.revokeObjectURL(url);
  };

  // Export to Print / PDF
  const handlePrintPdf = () => {
    window.print();
  };

  // Add new file to project
  const handleCreateNewFile = () => {
    const name = prompt('Enter new file name (e.g. methods.tex, macros.sty, references.bib):');
    if (!name?.trim()) return;
    const clean = name.trim();
    const ext = (clean.split('.').pop()?.toLowerCase() || 'tex') as VirtualFile['type'];
    setProject((prev) => ({
      ...prev,
      files: {
        ...prev.files,
        [clean]: {
          path: clean,
          content: ext === 'bib' ? '% BibTeX entries\n' : '% LaTeX source\n',
          type: ext,
          updatedAt: Date.now(),
        },
      },
    }));
    setActiveFilePath(clean);
    if (!openTabs.includes(clean)) {
      setOpenTabs((prev) => [...prev, clean]);
    }
  };

  // Close tab
  const handleCloseTab = (path: string, e: React.MouseEvent) => {
    e.stopPropagation();
    const remaining = openTabs.filter((t) => t !== path);
    if (remaining.length === 0) return;
    setOpenTabs(remaining);
    if (activeFilePath === path) {
      setActiveFilePath(remaining[remaining.length - 1]);
    }
  };

  // Copy Collaboration Share Link
  const handleCopyCollabLink = () => {
    const url = `${window.location.origin}${window.location.pathname}?room=${collabRoomId}`;
    navigator.clipboard.writeText(url);
    setCollabCopied(true);
    setTimeout(() => setCollabCopied(false), 2500);
  };

  // Live Crossref search
  const handleCrossrefSearch = async () => {
    if (!crossrefQuery.trim()) return;
    setCrossrefLoading(true);
    try {
      const items = await searchAndVerifyCrossrefCitations(crossrefQuery);
      setCrossrefResults(items);
    } catch {
      alert('Crossref lookup failed. Check network connection.');
    } finally {
      setCrossrefLoading(false);
    }
  };

  // Insert Crossref citation into references.bib
  const insertVerifiedCitation = (entry: BibEntry) => {
    const bibFile = project.files['references.bib'] || Object.values(project.files).find((f) => f.type === 'bib');
    const bibPath = bibFile ? bibFile.path : 'references.bib';
    const bibTeX = formatBibEntryToBibTeX(entry);

    const oldContent = project.files[bibPath]?.content || '';
    const newContent = oldContent ? `${oldContent}\n\n${bibTeX}\n` : `${bibTeX}\n`;

    updateFileContent(bibPath, newContent);
    alert(`Added \\cite{${entry.key}} to ${bibPath}!`);
  };

  // Command Palette Items
  const commandPaletteItems = useMemo(() => {
    return [
      {
        id: 'compile',
        title: 'Compile Workspace',
        shortcut: 'Ctrl+Enter',
        action: () => alert(`Compiled in ${compilation.compileTimeMs}ms with ${compilation.diagnostics.length} diagnostics.`),
      },
      {
        id: 'inline_ai',
        title: 'Open Inline AI Command Bar',
        shortcut: 'Ctrl+K',
        action: () => setInlineAiOpen(true),
      },
      {
        id: 'export_zip',
        title: 'Export Multi-File .ZIP Project',
        shortcut: '',
        action: handleExportZip,
      },
      {
        id: 'print_pdf',
        title: 'Print / Export Publication PDF',
        shortcut: 'Ctrl+P',
        action: handlePrintPdf,
      },
      {
        id: 'theme_white',
        title: 'Paper Theme: Standard White Journal',
        shortcut: '',
        action: () => setPaperTheme('white'),
      },
      {
        id: 'theme_ivory',
        title: 'Paper Theme: Warm Ivory Archival',
        shortcut: '',
        action: () => setPaperTheme('ivory'),
      },
      {
        id: 'theme_night',
        title: 'Paper Theme: High-Contrast Night Mode',
        shortcut: '',
        action: () => setPaperTheme('night'),
      },
      {
        id: 'new_file',
        title: 'Create New File...',
        shortcut: '',
        action: handleCreateNewFile,
      },
      {
        id: 'byom_settings',
        title: 'Setup BYOM AI Providers (Gemini, Claude, GPT-4o, Ollama)',
        shortcut: '',
        action: () => setByomModalOpen(true),
      },
      {
        id: 'share_room',
        title: 'Start P2P Real-Time Collaboration Room',
        shortcut: '',
        action: () => setCollabModalOpen(true),
      },
      ...Object.keys(TEMPLATES).map((key) => ({
        id: `template_${key}`,
        title: `Load Template: ${TEMPLATES[key].name}`,
        shortcut: '',
        action: () => handleLoadTemplate(key),
      })),
      ...Object.keys(project.files).map((f) => ({
        id: `open_${f}`,
        title: `Go to File: ${f}`,
        shortcut: '',
        action: () => {
          setActiveFilePath(f);
          if (!openTabs.includes(f)) setOpenTabs((prev) => [...prev, f]);
        },
      })),
    ].filter((item) => item.title.toLowerCase().includes(paletteSearch.toLowerCase()));
  }, [paletteSearch, compilation, project.files, openTabs]);

  return (
    <div className="flex flex-col h-screen w-screen overflow-hidden bg-zinc-950 text-zinc-100 font-sans select-none">
      {/* ========================================================
          1. Peak Studio Top Header & Workbench Nav
          ======================================================== */}
      <header className="h-12 border-b border-zinc-800 bg-zinc-950/95 backdrop-blur px-3 flex items-center justify-between z-20 shrink-0">
        <div className="flex items-center gap-3">
          {/* Brand Logo */}
          <div className="flex items-center gap-2">
            <div className="w-7 h-7 rounded-lg bg-gradient-to-tr from-emerald-600 via-teal-500 to-cyan-400 flex items-center justify-center shadow-lg shadow-emerald-500/20">
              <Zap className="w-4 h-4 text-zinc-950 font-bold fill-current" />
            </div>
            <div className="flex items-baseline gap-1.5">
              <span className="font-extrabold tracking-tight text-base bg-gradient-to-r from-emerald-400 via-teal-300 to-cyan-200 bg-clip-text text-transparent">
                VeriTeX
              </span>
              <span className="text-[10px] uppercase font-bold tracking-widest px-1.5 py-0.5 rounded bg-emerald-500/10 text-emerald-400 border border-emerald-500/20">
                Studio 2.0
              </span>
            </div>
          </div>

          <div className="h-4 w-px bg-zinc-800" />

          {/* Project Name editable */}
          <input
            value={project.name}
            onChange={(e) => setProject((prev) => ({ ...prev, name: e.target.value }))}
            className="bg-transparent hover:bg-zinc-900 focus:bg-zinc-900 border border-transparent focus:border-zinc-700 rounded px-2 py-0.5 text-xs font-medium text-zinc-300 focus:outline-none transition max-w-[180px] truncate"
            title="Rename Project"
          />

          {/* Collaboration Presence Badge */}
          <button
            onClick={() => setCollabModalOpen(true)}
            className="flex items-center gap-1.5 px-2.5 py-1 rounded-full bg-zinc-900 hover:bg-zinc-800 border border-zinc-800 text-[11px] text-zinc-300 transition"
            title="Real-Time Collaborative Room"
          >
            <span className="relative flex h-2 w-2">
              <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-emerald-400 opacity-75"></span>
              <span className="relative inline-flex rounded-full h-2 w-2 bg-emerald-500"></span>
            </span>
            <span className="font-mono">{collabRoomId}</span>
            {collabPeers.length > 0 && (
              <span className="px-1.5 py-0.2 rounded-full bg-emerald-500/20 text-emerald-300 text-[10px] font-bold">
                {collabPeers.length + 1}
              </span>
            )}
          </button>
        </div>

        {/* Center: Mode Switcher & Fast Actions */}
        <div className="flex items-center gap-1 bg-zinc-900/80 p-0.5 rounded-lg border border-zinc-800">
          <button
            onClick={() => setWorkbenchMode('studio')}
            className={`px-3 py-1 rounded-md text-xs font-medium transition flex items-center gap-1.5 ${
              workbenchMode === 'studio'
                ? 'bg-emerald-500/15 text-emerald-300 shadow-sm'
                : 'text-zinc-400 hover:text-zinc-200'
            }`}
          >
            <SplitSquareVertical className="w-3.5 h-3.5" /> Studio
          </button>
          <button
            onClick={() => setWorkbenchMode('focus')}
            className={`px-3 py-1 rounded-md text-xs font-medium transition flex items-center gap-1.5 ${
              workbenchMode === 'focus'
                ? 'bg-emerald-500/15 text-emerald-300 shadow-sm'
                : 'text-zinc-400 hover:text-zinc-200'
            }`}
          >
            <Eye className="w-3.5 h-3.5" /> Focus
          </button>
          <button
            onClick={() => setWorkbenchMode('diff')}
            className={`px-3 py-1 rounded-md text-xs font-medium transition flex items-center gap-1.5 ${
              workbenchMode === 'diff'
                ? 'bg-emerald-500/15 text-emerald-300 shadow-sm'
                : 'text-zinc-400 hover:text-zinc-200'
            }`}
          >
            <GitCompare className="w-3.5 h-3.5" /> Diffs
            {pendingPatches.length > 0 && (
              <span className="w-4 h-4 rounded-full bg-emerald-500 text-zinc-950 font-bold text-[9px] flex items-center justify-center">
                {pendingPatches.length}
              </span>
            )}
          </button>
        </div>

        {/* Right Header Toolbar: Ctrl+K, Command Palette, Compile Status, Export, BYOM Settings */}
        <div className="flex items-center gap-2">
          {/* Ctrl+K Inline AI Trigger */}
          <button
            onClick={() => setInlineAiOpen(true)}
            className="flex items-center gap-1.5 px-2.5 py-1 rounded-md bg-emerald-500/10 hover:bg-emerald-500/20 text-emerald-300 border border-emerald-500/30 text-xs font-semibold shadow-sm transition"
          >
            <Sparkles className="w-3.5 h-3.5 text-emerald-400 animate-pulse" />
            <span>Inline AI</span>
            <kbd className="hidden sm:inline-block px-1 py-0.5 rounded bg-zinc-900 border border-zinc-700 text-[10px] font-mono text-zinc-400">
              Ctrl+K
            </kbd>
          </button>

          {/* Ctrl+P Command Palette */}
          <button
            onClick={() => setCommandPaletteOpen(true)}
            className="flex items-center gap-1 px-2 py-1 rounded-md bg-zinc-900 hover:bg-zinc-800 text-zinc-300 border border-zinc-800 text-xs transition"
            title="Global Command Palette (Ctrl+P)"
          >
            <Command className="w-3.5 h-3.5" />
            <kbd className="hidden md:inline-block text-[10px] font-mono text-zinc-500">Ctrl+P</kbd>
          </button>

          {/* Compilation Diagnostic & Speed Badge */}
          <div
            className={`flex items-center gap-1.5 px-2 py-1 rounded-md text-xs font-mono border ${
              compilation.diagnostics.some((d) => d.severity === 'error')
                ? 'bg-red-500/10 border-red-500/30 text-red-300'
                : 'bg-zinc-900 border-zinc-800 text-emerald-400'
            }`}
            title="Sub-10ms AST Incremental Hot-Patch Engine"
          >
            <Clock className="w-3 h-3 text-emerald-400" />
            <span>{compilation.compileTimeMs}ms</span>
            {compilation.diagnostics.length > 0 && (
              <span className="text-[10px] px-1 rounded bg-zinc-800 text-zinc-300">
                {compilation.diagnostics.length} issues
              </span>
            )}
          </div>

          {/* Export Menu */}
          <div className="flex items-center gap-1">
            <button
              onClick={handleExportZip}
              className="px-2.5 py-1 rounded-md bg-zinc-900 hover:bg-zinc-800 text-zinc-200 border border-zinc-800 text-xs flex items-center gap-1.5 transition"
              title="Download full project .zip"
            >
              <Download className="w-3.5 h-3.5" />
              <span className="hidden sm:inline">ZIP</span>
            </button>
            <button
              onClick={handlePrintPdf}
              className="px-2.5 py-1 rounded-md bg-zinc-900 hover:bg-zinc-800 text-zinc-200 border border-zinc-800 text-xs flex items-center gap-1.5 transition"
              title="Print publication PDF"
            >
              <Printer className="w-3.5 h-3.5" />
              <span className="hidden sm:inline">PDF</span>
            </button>
          </div>

          {/* BYOM Settings Gear */}
          <button
            onClick={() => setByomModalOpen(true)}
            className="p-1.5 rounded-md bg-zinc-900 hover:bg-zinc-800 text-zinc-300 border border-zinc-800 transition"
            title="Configure BYOM AI Models (Gemini, Claude, GPT, DeepSeek, Ollama)"
          >
            <Settings className="w-4 h-4" />
          </button>
        </div>
      </header>

      {/* ========================================================
          2. Main Studio Workspace Layout
          ======================================================== */}
      <div className="flex-1 flex overflow-hidden relative">
        {/* VS Code-Style Left Activity Bar Rail */}
        <aside className="w-12 bg-zinc-950 border-r border-zinc-900 flex flex-col items-center py-2 shrink-0 z-10 justify-between select-none">
          <div className="flex flex-col items-center gap-2">
            <button
              onClick={() => {
                if (sidebarTab === 'explorer' && !sidebarCollapsed) setSidebarCollapsed(true);
                else {
                  setSidebarTab('explorer');
                  setSidebarCollapsed(false);
                }
              }}
              className={`p-2.5 rounded-lg transition relative ${
                sidebarTab === 'explorer' && !sidebarCollapsed
                  ? 'bg-zinc-900 text-emerald-400'
                  : 'text-zinc-500 hover:text-zinc-300 hover:bg-zinc-900/60'
              }`}
              title="Explorer (Files & Bundles)"
            >
              <FolderOpen className="w-4 h-4" />
              {sidebarTab === 'explorer' && !sidebarCollapsed && (
                <span className="absolute left-0 top-1.5 bottom-1.5 w-0.5 bg-emerald-500 rounded-r" />
              )}
            </button>

            <button
              onClick={() => {
                if (sidebarTab === 'outline' && !sidebarCollapsed) setSidebarCollapsed(true);
                else {
                  setSidebarTab('outline');
                  setSidebarCollapsed(false);
                }
              }}
              className={`p-2.5 rounded-lg transition relative ${
                sidebarTab === 'outline' && !sidebarCollapsed
                  ? 'bg-zinc-900 text-emerald-400'
                  : 'text-zinc-500 hover:text-zinc-300 hover:bg-zinc-900/60'
              }`}
              title="Semantic AST Outline"
            >
              <Layers className="w-4 h-4" />
              {sidebarTab === 'outline' && !sidebarCollapsed && (
                <span className="absolute left-0 top-1.5 bottom-1.5 w-0.5 bg-emerald-500 rounded-r" />
              )}
            </button>

            <button
              onClick={() => {
                if (sidebarTab === 'citations' && !sidebarCollapsed) setSidebarCollapsed(true);
                else {
                  setSidebarTab('citations');
                  setSidebarCollapsed(false);
                }
              }}
              className={`p-2.5 rounded-lg transition relative ${
                sidebarTab === 'citations' && !sidebarCollapsed
                  ? 'bg-zinc-900 text-emerald-400'
                  : 'text-zinc-500 hover:text-zinc-300 hover:bg-zinc-900/60'
              }`}
              title="Crossref Verified Citations"
            >
              <BookOpen className="w-4 h-4" />
              {sidebarTab === 'citations' && !sidebarCollapsed && (
                <span className="absolute left-0 top-1.5 bottom-1.5 w-0.5 bg-emerald-500 rounded-r" />
              )}
            </button>

            <button
              onClick={() => {
                if (sidebarTab === 'symbols' && !sidebarCollapsed) setSidebarCollapsed(true);
                else {
                  setSidebarTab('symbols');
                  setSidebarCollapsed(false);
                }
              }}
              className={`p-2.5 rounded-lg transition relative ${
                sidebarTab === 'symbols' && !sidebarCollapsed
                  ? 'bg-zinc-900 text-emerald-400'
                  : 'text-zinc-500 hover:text-zinc-300 hover:bg-zinc-900/60'
              }`}
              title="Mathematical Symbols Palette"
            >
              <Sigma className="w-4 h-4" />
              {sidebarTab === 'symbols' && !sidebarCollapsed && (
                <span className="absolute left-0 top-1.5 bottom-1.5 w-0.5 bg-emerald-500 rounded-r" />
              )}
            </button>

            <button
              onClick={() => {
                if (sidebarTab === 'ai' && !sidebarCollapsed) setSidebarCollapsed(true);
                else {
                  setSidebarTab('ai');
                  setSidebarCollapsed(false);
                }
              }}
              className={`p-2.5 rounded-lg transition relative ${
                sidebarTab === 'ai' && !sidebarCollapsed
                  ? 'bg-zinc-900 text-emerald-400'
                  : 'text-zinc-500 hover:text-zinc-300 hover:bg-zinc-900/60'
              }`}
              title="VeriTeX AI Co-Author"
            >
              <Bot className="w-4 h-4" />
              {sidebarTab === 'ai' && !sidebarCollapsed && (
                <span className="absolute left-0 top-1.5 bottom-1.5 w-0.5 bg-emerald-500 rounded-r" />
              )}
            </button>

            <button
              onClick={() => {
                if (sidebarTab === 'collab' && !sidebarCollapsed) setSidebarCollapsed(true);
                else {
                  setSidebarTab('collab');
                  setSidebarCollapsed(false);
                }
              }}
              className={`p-2.5 rounded-lg transition relative ${
                sidebarTab === 'collab' && !sidebarCollapsed
                  ? 'bg-zinc-900 text-emerald-400'
                  : 'text-zinc-500 hover:text-zinc-300 hover:bg-zinc-900/60'
              }`}
              title="Real-Time Collaboration"
            >
              <Users className="w-4 h-4" />
              {sidebarTab === 'collab' && !sidebarCollapsed && (
                <span className="absolute left-0 top-1.5 bottom-1.5 w-0.5 bg-emerald-500 rounded-r" />
              )}
            </button>
          </div>

          <button
            onClick={() => setSidebarCollapsed(!sidebarCollapsed)}
            className="p-2 text-zinc-500 hover:text-zinc-300 transition"
            title="Toggle Sidebar (Ctrl+B)"
          >
            {sidebarCollapsed ? <PanelLeftOpen className="w-4 h-4" /> : <PanelLeftClose className="w-4 h-4" />}
          </button>
        </aside>

        {/* Collapsible Left Drawer Panels */}
        {!sidebarCollapsed && (
          <aside className="w-64 bg-zinc-950 border-r border-zinc-900 flex flex-col shrink-0 z-10 overflow-hidden">
            {/* Header of Active Panel */}
            <div className="h-9 px-3 border-b border-zinc-900 flex items-center justify-between bg-zinc-950/60">
              <span className="text-xs font-bold uppercase tracking-wider text-zinc-400">
                {sidebarTab === 'explorer' && 'Explorer'}
                {sidebarTab === 'outline' && 'Document Outline'}
                {sidebarTab === 'citations' && 'Citation Firewall'}
                {sidebarTab === 'symbols' && 'LaTeX Symbols'}
                {sidebarTab === 'ai' && 'AI Co-Author'}
                {sidebarTab === 'collab' && 'Collaborators'}
              </span>

              {sidebarTab === 'explorer' && (
                <div className="flex items-center gap-1">
                  <button
                    onClick={handleCreateNewFile}
                    className="p-1 hover:bg-zinc-800 text-zinc-400 hover:text-zinc-200 rounded transition"
                    title="New File"
                  >
                    <Plus className="w-3.5 h-3.5" />
                  </button>
                </div>
              )}
            </div>

            {/* Panel Content Body */}
            <div className="flex-1 overflow-y-auto p-2 space-y-3">
              {/* TAB 1: FILE EXPLORER */}
              {sidebarTab === 'explorer' && (
                <div className="space-y-3">
                  <div className="space-y-0.5">
                    {Object.values(project.files).map((file) => (
                      <div
                        key={file.path}
                        onClick={() => {
                          setActiveFilePath(file.path);
                          if (!openTabs.includes(file.path)) setOpenTabs((prev) => [...prev, file.path]);
                        }}
                        className={`group flex items-center justify-between px-2.5 py-1.5 rounded-md text-xs cursor-pointer transition ${
                          activeFilePath === file.path
                            ? 'bg-emerald-500/15 text-emerald-300 font-medium border border-emerald-500/30'
                            : 'text-zinc-400 hover:bg-zinc-900 hover:text-zinc-200'
                        }`}
                      >
                        <div className="flex items-center gap-2 truncate">
                          {file.type === 'tex' && <FileCode className="w-3.5 h-3.5 text-emerald-400 shrink-0" />}
                          {file.type === 'bib' && <BookOpen className="w-3.5 h-3.5 text-amber-400 shrink-0" />}
                          {file.type === 'sty' && <FileText className="w-3.5 h-3.5 text-purple-400 shrink-0" />}
                          <span className="truncate">{file.path}</span>
                        </div>
                        {file.path === project.rootFile && (
                          <span className="text-[9px] uppercase px-1 rounded bg-zinc-800 text-zinc-400 font-mono">
                            Root
                          </span>
                        )}
                      </div>
                    ))}
                  </div>

                  {/* Template Picker */}
                  <div className="pt-3 border-t border-zinc-900">
                    <div className="text-[11px] font-semibold text-zinc-500 uppercase tracking-wider mb-1.5">
                      Templates
                    </div>
                    <div className="space-y-1">
                      {Object.entries(TEMPLATES).map(([key, tpl]) => (
                        <button
                          key={key}
                          onClick={() => handleLoadTemplate(key)}
                          className="w-full text-left px-2 py-1.5 rounded text-xs text-zinc-400 hover:bg-zinc-900 hover:text-zinc-200 transition"
                        >
                          <div className="font-medium text-zinc-300">{tpl.name}</div>
                          <div className="text-[10px] text-zinc-500 line-clamp-1">{tpl.desc}</div>
                        </button>
                      ))}
                    </div>
                  </div>
                </div>
              )}

              {/* TAB 2: AST OUTLINE */}
              {sidebarTab === 'outline' && (
                <div className="space-y-1">
                  {compilation.outline.length === 0 ? (
                    <div className="text-xs text-zinc-500 py-4 text-center">No sections found in document.</div>
                  ) : (
                    compilation.outline.map((item, idx) => (
                      <button
                        key={idx}
                        onClick={() => jumpToLine(item.line)}
                        style={{ paddingLeft: `${(item.level - 1) * 12 + 8}px` }}
                        className="w-full text-left py-1.5 pr-2 rounded text-xs text-zinc-400 hover:bg-zinc-900 hover:text-zinc-200 truncate flex items-center gap-1.5 transition"
                      >
                        <span className="text-emerald-400 font-mono text-[10px] font-semibold">{item.number}</span>
                        <span className="truncate">{item.title}</span>
                      </button>
                    ))
                  )}
                </div>
              )}

              {/* TAB 3: CITATIONS & CROSSREF */}
              {sidebarTab === 'citations' && (
                <div className="space-y-3">
                  <div>
                    <div className="text-[11px] font-semibold text-zinc-500 uppercase tracking-wider mb-1">
                      Query Crossref API
                    </div>
                    <div className="flex gap-1">
                      <input
                        value={crossrefQuery}
                        onChange={(e) => setCrossrefQuery(e.target.value)}
                        onKeyDown={(e) => e.key === 'Enter' && handleCrossrefSearch()}
                        placeholder="Search paper or DOI..."
                        className="flex-1 bg-zinc-900 border border-zinc-800 rounded px-2 py-1 text-xs text-zinc-200 focus:outline-none focus:border-emerald-500"
                      />
                      <button
                        onClick={handleCrossrefSearch}
                        disabled={crossrefLoading}
                        className="px-2.5 py-1 bg-emerald-600 hover:bg-emerald-500 rounded text-xs font-bold text-white transition"
                      >
                        {crossrefLoading ? <RefreshCw className="w-3 h-3 animate-spin" /> : <Search className="w-3 h-3" />}
                      </button>
                    </div>
                  </div>

                  <div className="space-y-2">
                    {crossrefResults.map((entry) => (
                      <div key={entry.key} className="p-2 rounded bg-zinc-900/90 border border-zinc-800 text-[11px] space-y-1">
                        <div className="font-semibold text-zinc-200 line-clamp-1">{entry.title}</div>
                        <div className="text-zinc-400 line-clamp-1">
                          {entry.author} ({entry.year})
                        </div>
                        <div className="flex items-center justify-between pt-1">
                          <code className="text-[10px] text-emerald-400 font-mono">\cite&#123;{entry.key}&#125;</code>
                          <button
                            onClick={() => insertVerifiedCitation(entry)}
                            className="px-1.5 py-0.5 rounded bg-emerald-500/20 hover:bg-emerald-500/30 text-emerald-300 text-[10px] font-semibold"
                          >
                            + Add to .bib
                          </button>
                        </div>
                      </div>
                    ))}
                  </div>

                  <div className="pt-2 border-t border-zinc-900">
                    <div className="text-[11px] font-semibold text-zinc-500 uppercase tracking-wider mb-1.5">
                      Indexed Bibliography ({Object.keys(compilation.bibEntries).length})
                    </div>
                    <div className="space-y-1 max-h-48 overflow-y-auto">
                      {Object.values(compilation.bibEntries).map((b) => (
                        <div key={b.key} className="p-1.5 rounded bg-zinc-900/50 border border-zinc-800/60 text-xs">
                          <div className="font-mono text-emerald-400 text-[11px] font-bold">{b.key}</div>
                          <div className="text-zinc-400 text-[10px] truncate">{b.title}</div>
                        </div>
                      ))}
                    </div>
                  </div>
                </div>
              )}

              {/* TAB 4: MATH SYMBOLS */}
              {sidebarTab === 'symbols' && (
                <div className="space-y-3">
                  {MATH_SYMBOLS.map((group) => (
                    <div key={group.category} className="space-y-1">
                      <div className="text-[10px] font-bold uppercase tracking-wider text-zinc-500">
                        {group.category}
                      </div>
                      <div className="grid grid-cols-4 gap-1">
                        {group.symbols.map((s) => (
                          <button
                            key={s.label}
                            onClick={() => handleInsertSymbol(s.insert)}
                            className="h-8 rounded bg-zinc-900 hover:bg-zinc-800 border border-zinc-800/80 text-xs font-serif text-zinc-200 flex items-center justify-center hover:text-emerald-400 transition"
                            title={s.desc}
                          >
                            {s.label}
                          </button>
                        ))}
                      </div>
                    </div>
                  ))}
                </div>
              )}

              {/* TAB 5: AI CO-AUTHOR */}
              {sidebarTab === 'ai' && (
                <div className="flex flex-col h-full space-y-3">
                  {/* AST Slicer Token Savings Indicator */}
                  <div className="p-2.5 rounded-lg bg-emerald-500/10 border border-emerald-500/30 text-xs space-y-1">
                    <div className="flex items-center justify-between text-emerald-400 font-semibold">
                      <span className="flex items-center gap-1">
                        <Sparkles className="w-3.5 h-3.5" /> AST Token Slicer
                      </span>
                      <span className="font-mono">{astSlice.savingsPercent}% Saved</span>
                    </div>
                    <div className="text-[11px] text-zinc-400">
                      Scope: <span className="text-zinc-200 font-medium">{astSlice.activeSectionTitle}</span>
                    </div>
                  </div>

                  {/* AI Quick Prompts */}
                  <div className="space-y-1.5">
                    <div className="text-[10px] font-bold uppercase tracking-wider text-zinc-500">
                      Quick Generators
                    </div>
                    <button
                      onClick={() => handleSendAiPrompt('Generate a publication-ready booktabs table comparing models.')}
                      className="w-full text-left p-1.5 rounded bg-zinc-900 hover:bg-zinc-800 text-xs text-zinc-300 transition"
                    >
                      + Booktabs Results Table
                    </button>
                    <button
                      onClick={() => handleSendAiPrompt('Formulate formal convergence proof steps for Theorem.')}
                      className="w-full text-left p-1.5 rounded bg-zinc-900 hover:bg-zinc-800 text-xs text-zinc-300 transition"
                    >
                      + Formal Proof Steps
                    </button>
                    <button
                      onClick={() => handleSendAiPrompt('Convert raw math equations into aligned multi-line environment.')}
                      className="w-full text-left p-1.5 rounded bg-zinc-900 hover:bg-zinc-800 text-xs text-zinc-300 transition"
                    >
                      + \\begin&#123;align&#125; Equations
                    </button>
                  </div>

                  {/* Chat Input */}
                  <div className="space-y-1.5 pt-2">
                    <textarea
                      value={aiPromptInput}
                      onChange={(e) => setAiPromptInput(e.target.value)}
                      onKeyDown={(e) => {
                        if (e.key === 'Enter' && (e.metaKey || e.ctrlKey)) {
                          handleSendAiPrompt(aiPromptInput);
                        }
                      }}
                      placeholder="Instruct VeriTeX AI to refactor, write proofs, or fix errors..."
                      className="w-full h-24 bg-zinc-900 border border-zinc-800 rounded p-2 text-xs text-zinc-200 focus:outline-none focus:border-emerald-500 resize-none font-sans"
                    />
                    <button
                      onClick={() => handleSendAiPrompt(aiPromptInput)}
                      disabled={aiGenerating || !aiPromptInput.trim()}
                      className="w-full py-1.5 rounded bg-emerald-600 hover:bg-emerald-500 disabled:opacity-50 text-white text-xs font-semibold flex items-center justify-center gap-1.5 transition"
                    >
                      {aiGenerating ? (
                        <>
                          <RefreshCw className="w-3.5 h-3.5 animate-spin" /> Synthesizing Patch...
                        </>
                      ) : (
                        <>
                          <Wand2 className="w-3.5 h-3.5" /> Propose Surgical Diff
                        </>
                      )}
                    </button>
                  </div>
                </div>
              )}

              {/* TAB 6: COLLABORATORS */}
              {sidebarTab === 'collab' && (
                <div className="space-y-3">
                  <div className="p-3 rounded-lg bg-zinc-900 border border-zinc-800 space-y-2">
                    <div className="text-xs font-semibold text-zinc-200">Decentralized P2P CRDT</div>
                    <div className="text-[11px] text-zinc-400">
                      Co-authors can edit simultaneously over WebRTC with automatic offline merge.
                    </div>
                    <button
                      onClick={handleCopyCollabLink}
                      className="w-full py-1.5 rounded bg-emerald-600 hover:bg-emerald-500 text-white text-xs font-semibold flex items-center justify-center gap-1.5 transition"
                    >
                      <Share2 className="w-3.5 h-3.5" />
                      {collabCopied ? 'Link Copied!' : 'Copy Invite Link'}
                    </button>
                  </div>

                  <div className="space-y-1.5">
                    <div className="text-[10px] font-bold uppercase tracking-wider text-zinc-500">
                      Active Peers Online ({collabPeers.length + 1})
                    </div>
                    <div className="p-2 rounded bg-zinc-900/60 border border-zinc-800/80 flex items-center justify-between text-xs">
                      <div className="flex items-center gap-2">
                        <span className="w-2.5 h-2.5 rounded-full bg-emerald-500" />
                        <span className="font-medium text-zinc-200">You (Host)</span>
                      </div>
                      <span className="font-mono text-[10px] text-zinc-500">Line {cursorLine}</span>
                    </div>

                    {collabPeers.map((p) => (
                      <div
                        key={p.clientId}
                        className="p-2 rounded bg-zinc-900/60 border border-zinc-800/80 flex items-center justify-between text-xs"
                      >
                        <div className="flex items-center gap-2">
                          <span className="w-2.5 h-2.5 rounded-full" style={{ backgroundColor: p.color }} />
                          <span className="font-medium text-zinc-300">{p.name}</span>
                        </div>
                        <span className="font-mono text-[10px] text-zinc-500">{p.activeFile}:{p.cursorLine}</span>
                      </div>
                    ))}
                  </div>
                </div>
              )}
            </div>
          </aside>
        )}

        {/* ========================================================
            3. Center Editor Pane (Synchronized Dual-Layer Canvas)
            ======================================================== */}
        <main
          style={{ width: workbenchMode === 'focus' ? '100%' : `${splitWidth}%` }}
          className="flex-1 flex flex-col bg-[#0b0d13] overflow-hidden relative"
        >
          {/* Multi-Tab Header Strip */}
          <div className="h-9 bg-zinc-950 border-b border-zinc-900 flex items-center justify-between px-2 shrink-0 overflow-x-auto select-none">
            <div className="flex items-center gap-1 overflow-x-auto">
              {openTabs.map((path) => {
                const isActive = activeFilePath === path;
                const file = project.files[path];
                return (
                  <div
                    key={path}
                    onClick={() => setActiveFilePath(path)}
                    className={`group flex items-center gap-2 px-3 py-1 rounded-t-md text-xs cursor-pointer border-t-2 transition ${
                      isActive
                        ? 'bg-[#0b0d13] text-emerald-300 border-emerald-500 font-medium'
                        : 'bg-zinc-950 text-zinc-500 border-transparent hover:text-zinc-300 hover:bg-zinc-900/50'
                    }`}
                  >
                    {file?.type === 'tex' && <FileCode className="w-3.5 h-3.5 text-emerald-400" />}
                    {file?.type === 'bib' && <BookOpen className="w-3.5 h-3.5 text-amber-400" />}
                    {file?.type === 'sty' && <FileText className="w-3.5 h-3.5 text-purple-400" />}
                    <span>{path}</span>
                    <button
                      onClick={(e) => handleCloseTab(path, e)}
                      className="opacity-0 group-hover:opacity-100 hover:text-red-400 p-0.5 rounded transition"
                    >
                      <X className="w-3 h-3" />
                    </button>
                  </div>
                );
              })}
              <button
                onClick={handleCreateNewFile}
                className="p-1 text-zinc-500 hover:text-zinc-300 hover:bg-zinc-900 rounded transition"
                title="Add New File"
              >
                <Plus className="w-3.5 h-3.5" />
              </button>
            </div>

            {/* Diagnostics Quick Pill */}
            <div className="flex items-center gap-2">
              <span className="text-[11px] font-mono text-zinc-500">
                Words: {compilation.wordCount}
              </span>
            </div>
          </div>

          {/* Breadcrumb Path Bar */}
          <div className="h-6 bg-[#090b10] border-b border-zinc-900/80 px-3 flex items-center text-[11px] text-zinc-500 gap-1.5 shrink-0 select-none">
            <span>{project.name}</span>
            <ChevronRight className="w-3 h-3" />
            <span className="text-zinc-300 font-medium">{activeFile.path}</span>
            <ChevronRight className="w-3 h-3" />
            <span className="text-emerald-400 font-medium">§ {astSlice.activeSectionTitle}</span>
          </div>

          {/* Pending Surgical Diff Review Bar */}
          {pendingPatches.length > 0 && (
            <div className="bg-emerald-950/40 border-b border-emerald-500/30 p-2.5 space-y-2 max-h-48 overflow-y-auto shrink-0 z-10">
              <div className="text-xs font-semibold text-emerald-300 flex items-center justify-between">
                <span>Pending Surgical AI Diffs ({pendingPatches.length})</span>
                <span className="text-[11px] font-mono">
                  {astSlice.savingsPercent}% Context Tokens Saved
                </span>
              </div>
              {pendingPatches.map((patch) => (
                <div key={patch.id} className="p-2 rounded bg-zinc-950 border border-zinc-800 text-xs font-mono space-y-1.5">
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
                      <Check className="w-3 h-3" /> Accept (Ctrl+Enter)
                    </button>
                    <button
                      onClick={() => handleRejectPatch(patch.id)}
                      className="px-2.5 py-1 rounded bg-zinc-800 hover:bg-zinc-700 text-zinc-300 text-xs flex items-center gap-1"
                    >
                      <X className="w-3 h-3" /> Reject (Esc)
                    </button>
                  </div>
                </div>
              ))}
            </div>
          )}

          {/* Synchronized Dual-Layer Editor Canvas */}
          <div className="flex-1 flex overflow-hidden relative">
            {/* Gutter with Line Numbers and Error Squiggly Markers */}
            <div ref={gutterRef} className="editor-gutter">
              {activeFile.content.split('\n').map((_, idx) => {
                const lineNum = idx + 1;
                const hasError = compilation.diagnostics.some(
                  (d) => d.line === lineNum && d.severity === 'error'
                );
                const hasWarning = compilation.diagnostics.some(
                  (d) => d.line === lineNum && d.severity === 'warning'
                );
                const isActive = cursorLine === lineNum;
                return (
                  <div
                    key={lineNum}
                    onClick={() => jumpToLine(lineNum)}
                    className={`editor-gutter-line cursor-pointer ${isActive ? 'active' : ''} ${
                      hasError ? 'has-error' : hasWarning ? 'has-warning' : ''
                    }`}
                    title={
                      hasError
                        ? compilation.diagnostics.find((d) => d.line === lineNum)?.message
                        : `Line ${lineNum}`
                    }
                  >
                    {hasError ? '● ' : ''}
                    {lineNum}
                  </div>
                );
              })}
            </div>

            {/* Canvas Container with Dual Synchronized Layers */}
            <div className="editor-canvas-container flex-1">
              {/* Active Line Highlight Bar */}
              <div
                className="editor-active-line-bg"
                style={{
                  top: `${Math.max(0, (cursorLine - 1) * 22 + 16)}px`,
                  height: '22px',
                }}
              />

              {/* Background Layer: Tokenized Syntax Highlighter */}
              <div
                ref={backdropRef}
                className="editor-layer editor-backdrop"
                dangerouslySetInnerHTML={{ __html: highlightedSource }}
              />

              {/* Foreground Layer: Interactive Textarea */}
              <textarea
                ref={editorRef}
                value={activeFile.content}
                onChange={handleEditorChange}
                onKeyDown={handleEditorKeyDown}
                onScroll={handleEditorScroll}
                onClick={(e) => {
                  const sel = e.currentTarget.selectionStart || 0;
                  setCursorOffset(sel);
                  setCursorLine(e.currentTarget.value.slice(0, sel).split('\n').length);
                }}
                spellCheck={false}
                className="editor-layer editor-textarea"
                placeholder="Type LaTeX here... Press Ctrl+K for inline AI generation."
              />

              {/* Floating Autocomplete IntelliSense Menu */}
              {completions && completions.items.length > 0 && (
                <div
                  className="absolute z-40 w-96 rounded-lg bg-zinc-900 border border-zinc-700 shadow-2xl overflow-hidden font-sans"
                  style={{
                    left: '4rem',
                    top: `${Math.min(300, (cursorLine - 1) * 22 + 40)}px`,
                  }}
                >
                  <div className="px-3 py-1.5 bg-zinc-950 border-b border-zinc-800 flex items-center justify-between text-xs text-zinc-400">
                    <span className="font-semibold text-emerald-400">LSP IntelliSense</span>
                    <span className="text-[10px] font-mono">↑↓ Navigate • ↵ Insert</span>
                  </div>
                  <div className="max-h-56 overflow-y-auto divide-y divide-zinc-800/60">
                    {completions.items.map((item, idx) => {
                      const isSelected = idx === selectedCompletionIdx;
                      return (
                        <div
                          key={idx}
                          onClick={() => insertAutocompleteItem(item)}
                          className={`px-3 py-2 text-xs cursor-pointer flex items-start justify-between gap-2 transition ${
                            isSelected ? 'bg-emerald-500/20 text-emerald-200' : 'text-zinc-300 hover:bg-zinc-800/60'
                          }`}
                        >
                          <div className="flex-1 truncate">
                            <div className="font-mono font-medium text-emerald-300">{item.label}</div>
                            <div className="text-[11px] text-zinc-400 truncate">{item.detail}</div>
                            {item.documentation && isSelected && (
                              <div className="text-[10px] text-zinc-500 italic mt-0.5 line-clamp-2">
                                {item.documentation}
                              </div>
                            )}
                          </div>
                          <span
                            className={`text-[9px] uppercase px-1.5 py-0.5 rounded font-mono font-bold ${
                              item.kind === 'citation'
                                ? 'bg-blue-500/20 text-blue-300'
                                : item.kind === 'reference'
                                ? 'bg-sky-500/20 text-sky-300'
                                : item.kind === 'environment'
                                ? 'bg-emerald-500/20 text-emerald-300'
                                : item.kind === 'math'
                                ? 'bg-amber-500/20 text-amber-300'
                                : 'bg-purple-500/20 text-purple-300'
                            }`}
                          >
                            {item.kind}
                          </span>
                        </div>
                      );
                    })}
                  </div>
                </div>
              )}

              {/* Live Floating Math Inspector HUD (Rendered when cursor inside math) */}
              {activeMathFormula && (
                <div
                  ref={mathInspectorRef}
                  className="absolute right-4 bottom-4 z-30 max-w-md p-3 rounded-xl bg-zinc-950/95 border border-emerald-500/40 shadow-2xl backdrop-blur font-sans space-y-2 animate-in fade-in slide-in-from-bottom-2"
                >
                  <div className="flex items-center justify-between text-xs text-zinc-400 border-b border-zinc-800/80 pb-1.5">
                    <span className="flex items-center gap-1.5 font-semibold text-emerald-400">
                      <Sigma className="w-3.5 h-3.5" /> Live Math Inspector
                    </span>
                    <button
                      onClick={() => {
                        navigator.clipboard.writeText(activeMathFormula);
                      }}
                      className="hover:text-emerald-400 transition flex items-center gap-1 text-[11px]"
                      title="Copy formula LaTeX"
                    >
                      <Copy className="w-3 h-3" /> Copy
                    </button>
                  </div>
                  <div
                    className="overflow-x-auto py-2 text-center text-zinc-100"
                    dangerouslySetInnerHTML={{
                      __html: (() => {
                        try {
                          return katex.renderToString(activeMathFormula, {
                            displayMode: true,
                            throwOnError: false,
                          });
                        } catch {
                          return activeMathFormula;
                        }
                      })(),
                    }}
                  />
                  <div className="text-[10px] text-zinc-500 font-mono truncate">
                    {activeMathFormula}
                  </div>
                </div>
              )}

              {/* Inline Ctrl+K Command Bar Popup */}
              {inlineAiOpen && (
                <div className="absolute left-1/2 -translate-x-1/2 top-16 z-50 w-[540px] p-3 rounded-xl bg-zinc-950 border border-emerald-500/50 shadow-2xl font-sans space-y-2.5 animate-in fade-in zoom-in-95">
                  <div className="flex items-center justify-between">
                    <span className="flex items-center gap-1.5 text-xs font-bold text-emerald-400">
                      <Sparkles className="w-4 h-4" /> Inline AI Co-Author (Ctrl+K)
                    </span>
                    <button
                      onClick={() => setInlineAiOpen(false)}
                      className="p-1 hover:text-red-400 text-zinc-500 transition"
                    >
                      <X className="w-3.5 h-3.5" />
                    </button>
                  </div>

                  <input
                    autoFocus
                    value={inlineAiPrompt}
                    onChange={(e) => setInlineAiPrompt(e.target.value)}
                    onKeyDown={(e) => {
                      if (e.key === 'Enter') handleSendAiPrompt(inlineAiPrompt);
                    }}
                    placeholder="Ask VeriTeX AI to generate proof, table, equation or refactor..."
                    className="w-full bg-zinc-900 border border-zinc-800 rounded-lg px-3 py-2 text-xs text-zinc-100 focus:outline-none focus:border-emerald-500"
                  />

                  {/* Quick Action Chips */}
                  <div className="flex flex-wrap gap-1.5">
                    <button
                      onClick={() => handleSendAiPrompt('Generate publication-ready booktabs table')}
                      className="px-2 py-1 rounded bg-zinc-900 hover:bg-zinc-800 text-[11px] text-zinc-300 border border-zinc-800"
                    >
                      + Booktabs Table
                    </button>
                    <button
                      onClick={() => handleSendAiPrompt('Format aligned equation derivation')}
                      className="px-2 py-1 rounded bg-zinc-900 hover:bg-zinc-800 text-[11px] text-zinc-300 border border-zinc-800"
                    >
                      + Aligned Equation
                    </button>
                    <button
                      onClick={() => handleSendAiPrompt('Add formal proof steps with QED')}
                      className="px-2 py-1 rounded bg-zinc-900 hover:bg-zinc-800 text-[11px] text-zinc-300 border border-zinc-800"
                    >
                      + Convergence Proof
                    </button>
                    <button
                      onClick={() => handleSendAiPrompt('Fix compiler errors in this section')}
                      className="px-2 py-1 rounded bg-zinc-900 hover:bg-zinc-800 text-[11px] text-zinc-300 border border-zinc-800"
                    >
                      + Fix Diagnostics
                    </button>
                  </div>
                </div>
              )}
            </div>
          </div>
        </main>

        {/* Resizable Draggable Split-Pane Divider */}
        {workbenchMode !== 'focus' && (
          <div
            onMouseDown={handleSplitterMouseDown}
            onDoubleClick={() => setSplitWidth(50)}
            className="w-1.5 bg-zinc-950 hover:bg-emerald-500/60 active:bg-emerald-500 cursor-col-resize flex items-center justify-center transition-colors z-20 group select-none"
            title="Drag to resize split pane • Double-click to reset 50/50"
          >
            <div className="w-0.5 h-8 bg-zinc-700 group-hover:bg-emerald-400 rounded-full" />
          </div>
        )}

        {/* ========================================================
            4. Live Paper Preview Pane (Right Side)
            ======================================================== */}
        {workbenchMode !== 'focus' && (
          <section className="flex-1 flex flex-col bg-zinc-900 overflow-hidden relative">
            {/* Paper Preview Toolbar: Theme, Zoom, SyncTeX Indicator */}
            <div className="h-9 bg-zinc-950 border-b border-zinc-800 px-3 flex items-center justify-between text-xs text-zinc-400 shrink-0 select-none">
              <div className="flex items-center gap-2">
                <span className="font-semibold text-zinc-300">Live Paper Preview</span>
                <span className="text-[10px] text-emerald-400 font-mono px-1.5 py-0.5 rounded bg-emerald-500/10 border border-emerald-500/20">
                  SyncTeX Active
                </span>
              </div>

              <div className="flex items-center gap-3">
                {/* Theme Selector */}
                <div className="flex items-center gap-1 bg-zinc-900 p-0.5 rounded border border-zinc-800">
                  <button
                    onClick={() => setPaperTheme('white')}
                    className={`px-2 py-0.5 rounded text-[11px] font-medium transition ${
                      paperTheme === 'white' ? 'bg-zinc-800 text-zinc-100' : 'text-zinc-500 hover:text-zinc-300'
                    }`}
                  >
                    White
                  </button>
                  <button
                    onClick={() => setPaperTheme('ivory')}
                    className={`px-2 py-0.5 rounded text-[11px] font-medium transition ${
                      paperTheme === 'ivory' ? 'bg-zinc-800 text-amber-200' : 'text-zinc-500 hover:text-zinc-300'
                    }`}
                  >
                    Ivory
                  </button>
                  <button
                    onClick={() => setPaperTheme('night')}
                    className={`px-2 py-0.5 rounded text-[11px] font-medium transition ${
                      paperTheme === 'night' ? 'bg-zinc-800 text-emerald-300' : 'text-zinc-500 hover:text-zinc-300'
                    }`}
                  >
                    Night
                  </button>
                </div>

                {/* Zoom Controls */}
                <div className="flex items-center gap-1">
                  <button
                    onClick={() => setPaperZoom((prev) => Math.max(50, prev - 15))}
                    className="p-1 hover:bg-zinc-800 rounded text-zinc-400 hover:text-zinc-200"
                    title="Zoom Out"
                  >
                    -
                  </button>
                  <span className="text-[11px] font-mono text-zinc-400 w-9 text-center">{paperZoom}%</span>
                  <button
                    onClick={() => setPaperZoom((prev) => Math.min(200, prev + 15))}
                    className="p-1 hover:bg-zinc-800 rounded text-zinc-400 hover:text-zinc-200"
                    title="Zoom In"
                  >
                    +
                  </button>
                </div>
              </div>
            </div>

            {/* Scrollable Paper Canvas */}
            <div
              ref={previewScrollRef}
              onDoubleClick={handlePreviewDoubleClick}
              className="flex-1 overflow-y-auto p-8 flex justify-center bg-zinc-900/90"
            >
              <div
                id="veritex-printable-paper"
                style={{
                  transform: `scale(${paperZoom / 100})`,
                  transformOrigin: 'top center',
                }}
                className={`veritex-paper-sheet paper-theme-${paperTheme} w-[820px] min-h-[1080px] p-16 shadow-2xl rounded-sm transition-all duration-150 select-text`}
                dangerouslySetInnerHTML={{ __html: compilation.html }}
              />
            </div>
          </section>
        )}
      </div>

      {/* ========================================================
          5. Global Command Palette Modal (Ctrl+P)
          ======================================================== */}
      {commandPaletteOpen && (
        <div
          onClick={() => setCommandPaletteOpen(false)}
          className="fixed inset-0 z-50 bg-black/60 backdrop-blur-sm flex items-start justify-center pt-24 animate-in fade-in"
        >
          <div
            onClick={(e) => e.stopPropagation()}
            className="w-[560px] rounded-xl bg-zinc-950 border border-zinc-800 shadow-2xl overflow-hidden font-sans space-y-2 animate-in zoom-in-95"
          >
            <div className="p-3 border-b border-zinc-800 flex items-center gap-2">
              <Command className="w-4 h-4 text-emerald-400" />
              <input
                autoFocus
                value={paletteSearch}
                onChange={(e) => {
                  setPaletteSearch(e.target.value);
                  setPaletteSelectedIdx(0);
                }}
                onKeyDown={(e) => {
                  if (e.key === 'ArrowDown') {
                    e.preventDefault();
                    setPaletteSelectedIdx((prev) => (prev + 1) % commandPaletteItems.length);
                  } else if (e.key === 'ArrowUp') {
                    e.preventDefault();
                    setPaletteSelectedIdx((prev) => (prev - 1 + commandPaletteItems.length) % commandPaletteItems.length);
                  } else if (e.key === 'Enter') {
                    e.preventDefault();
                    commandPaletteItems[paletteSelectedIdx]?.action();
                    setCommandPaletteOpen(false);
                  }
                }}
                placeholder="Type a command or search files..."
                className="flex-1 bg-transparent text-sm text-zinc-100 focus:outline-none"
              />
              <kbd className="text-[10px] font-mono text-zinc-500 bg-zinc-900 px-1.5 py-0.5 rounded">ESC</kbd>
            </div>

            <div className="max-h-72 overflow-y-auto p-1 divide-y divide-zinc-900">
              {commandPaletteItems.length === 0 ? (
                <div className="p-4 text-center text-xs text-zinc-500">No matching commands.</div>
              ) : (
                commandPaletteItems.map((item, idx) => {
                  const isSelected = idx === paletteSelectedIdx;
                  return (
                    <div
                      key={item.id}
                      onClick={() => {
                        item.action();
                        setCommandPaletteOpen(false);
                      }}
                      className={`px-3 py-2 rounded-md text-xs cursor-pointer flex items-center justify-between transition ${
                        isSelected ? 'bg-emerald-500/20 text-emerald-200' : 'text-zinc-300 hover:bg-zinc-900'
                      }`}
                    >
                      <span>{item.title}</span>
                      {item.shortcut && (
                        <kbd className="font-mono text-[10px] text-zinc-500">{item.shortcut}</kbd>
                      )}
                    </div>
                  );
                })
              )}
            </div>
          </div>
        </div>
      )}

      {/* ========================================================
          6. BYOM AI Gateway Settings Modal
          ======================================================== */}
      {byomModalOpen && (
        <div
          onClick={() => setByomModalOpen(false)}
          className="fixed inset-0 z-50 bg-black/70 backdrop-blur-sm flex items-center justify-center p-4 animate-in fade-in"
        >
          <div
            onClick={(e) => e.stopPropagation()}
            className="w-[640px] max-h-[85vh] overflow-y-auto rounded-2xl bg-zinc-950 border border-zinc-800 shadow-2xl p-6 font-sans space-y-5 animate-in zoom-in-95"
          >
            <div className="flex items-center justify-between border-b border-zinc-800 pb-3">
              <div>
                <h3 className="text-base font-bold text-zinc-100 flex items-center gap-2">
                  <KeyRound className="w-4 h-4 text-emerald-400" />
                  Universal BYOM AI Gateway
                </h3>
                <p className="text-xs text-zinc-400 mt-0.5">
                  Bring Your Own Model: Keys stay strictly in your browser. Zero telemetry.
                </p>
              </div>
              <button
                onClick={() => setByomModalOpen(false)}
                className="p-1 hover:text-red-400 text-zinc-500 transition"
              >
                <X className="w-4 h-4" />
              </button>
            </div>

            {/* Provider Grid */}
            <div className="grid grid-cols-3 gap-2">
              {Object.values(BYOM_PROVIDER_PRESETS).map((p) => (
                <button
                  key={p.id}
                  onClick={() =>
                    setAiConfig((prev) => ({
                      ...prev,
                      provider: p.id,
                      modelId: p.defaultModel,
                      baseUrl: p.defaultBaseUrl,
                    }))
                  }
                  className={`p-3 rounded-xl border text-left transition ${
                    aiConfig.provider === p.id
                      ? 'bg-emerald-500/15 border-emerald-500 text-emerald-300 font-semibold'
                      : 'bg-zinc-900 border-zinc-800 text-zinc-400 hover:bg-zinc-850 hover:text-zinc-200'
                  }`}
                >
                  <div className="text-xs font-bold">{p.name}</div>
                  <div className="text-[10px] text-zinc-500 mt-0.5">{p.badge}</div>
                </button>
              ))}
            </div>

            {/* Configuration Inputs */}
            <div className="space-y-3 bg-zinc-900/60 p-4 rounded-xl border border-zinc-800/80">
              <div className="space-y-1">
                <label className="text-xs font-medium text-zinc-300">Active Model ID</label>
                <input
                  value={aiConfig.modelId}
                  onChange={(e) => setAiConfig((prev) => ({ ...prev, modelId: e.target.value }))}
                  className="w-full bg-zinc-950 border border-zinc-800 rounded px-3 py-1.5 text-xs text-zinc-200 font-mono focus:outline-none focus:border-emerald-500"
                />
              </div>

              <div className="space-y-1">
                <label className="text-xs font-medium text-zinc-300">API Secret Key</label>
                <input
                  type="password"
                  value={aiConfig.apiKey}
                  onChange={(e) => setAiConfig((prev) => ({ ...prev, apiKey: e.target.value }))}
                  placeholder={
                    aiConfig.provider === 'ollama' ? 'No key required for Local Ollama' : 'Paste API Key (sk-...)'
                  }
                  className="w-full bg-zinc-950 border border-zinc-800 rounded px-3 py-1.5 text-xs text-zinc-200 font-mono focus:outline-none focus:border-emerald-500"
                />
              </div>

              <div className="space-y-1">
                <label className="text-xs font-medium text-zinc-300">Base URL</label>
                <input
                  value={aiConfig.baseUrl}
                  onChange={(e) => setAiConfig((prev) => ({ ...prev, baseUrl: e.target.value }))}
                  className="w-full bg-zinc-950 border border-zinc-800 rounded px-3 py-1.5 text-xs text-zinc-200 font-mono focus:outline-none focus:border-emerald-500"
                />
              </div>

              {/* Handshake Benchmark Test */}
              <div className="pt-2 flex items-center justify-between">
                <button
                  onClick={async () => {
                    setHandshakeStatus({ testing: true, message: 'Running capability handshake...' });
                    const res = await verifyProviderConnection(aiConfig);
                    setHandshakeStatus({ testing: false, message: res.message, ok: res.ok });
                  }}
                  className="px-3 py-1.5 rounded-lg bg-zinc-800 hover:bg-zinc-700 text-xs text-zinc-200 font-semibold flex items-center gap-1.5 transition"
                >
                  <RefreshCw className={`w-3.5 h-3.5 ${handshakeStatus?.testing ? 'animate-spin' : ''}`} />
                  Verify Handshake
                </button>

                {handshakeStatus && (
                  <span
                    className={`text-xs font-medium ${
                      handshakeStatus.ok ? 'text-emerald-400' : 'text-amber-400'
                    }`}
                  >
                    {handshakeStatus.message}
                  </span>
                )}
              </div>
            </div>

            <div className="flex justify-end">
              <button
                onClick={() => setByomModalOpen(false)}
                className="px-4 py-2 rounded-xl bg-emerald-600 hover:bg-emerald-500 text-white text-xs font-bold transition"
              >
                Save & Close
              </button>
            </div>
          </div>
        </div>
      )}

      {/* ========================================================
          7. P2P Collaboration Room Modal
          ======================================================== */}
      {collabModalOpen && (
        <div
          onClick={() => setCollabModalOpen(false)}
          className="fixed inset-0 z-50 bg-black/70 backdrop-blur-sm flex items-center justify-center p-4 animate-in fade-in"
        >
          <div
            onClick={(e) => e.stopPropagation()}
            className="w-[500px] rounded-2xl bg-zinc-950 border border-zinc-800 shadow-2xl p-6 font-sans space-y-4 animate-in zoom-in-95"
          >
            <div className="flex items-center justify-between border-b border-zinc-800 pb-3">
              <h3 className="text-base font-bold text-zinc-100 flex items-center gap-2">
                <Users className="w-4 h-4 text-emerald-400" />
                Collaborative Writing Room
              </h3>
              <button
                onClick={() => setCollabModalOpen(false)}
                className="p-1 hover:text-red-400 text-zinc-500 transition"
              >
                <X className="w-4 h-4" />
              </button>
            </div>

            <div className="space-y-3">
              <p className="text-xs text-zinc-400">
                Share this room link with your research team. Edits sync in real time using P2P WebRTC mesh connections with CRDT state resolution.
              </p>

              <div className="p-3 bg-zinc-900 rounded-xl border border-zinc-800 flex items-center justify-between gap-2">
                <code className="text-xs text-emerald-300 font-mono truncate">
                  {typeof window !== 'undefined'
                    ? `${window.location.origin}${window.location.pathname}?room=${collabRoomId}`
                    : `?room=${collabRoomId}`}
                </code>
                <button
                  onClick={handleCopyCollabLink}
                  className="px-3 py-1.5 rounded-lg bg-emerald-600 hover:bg-emerald-500 text-white text-xs font-semibold shrink-0 transition"
                >
                  {collabCopied ? 'Copied!' : 'Copy Link'}
                </button>
              </div>

              <div className="space-y-1.5 pt-2">
                <div className="text-xs font-bold text-zinc-300">Room Members ({collabPeers.length + 1})</div>
                <div className="p-2 rounded bg-zinc-900 border border-zinc-800 text-xs flex items-center justify-between">
                  <span className="text-emerald-400 font-medium">You (Active Author)</span>
                  <span className="font-mono text-zinc-500 text-[11px]">{activeFile.path}:{cursorLine}</span>
                </div>
                {collabPeers.map((p) => (
                  <div key={p.clientId} className="p-2 rounded bg-zinc-900 border border-zinc-800 text-xs flex items-center justify-between">
                    <span style={{ color: p.color }} className="font-medium">{p.name}</span>
                    <span className="font-mono text-zinc-500 text-[11px]">{p.activeFile}:{p.cursorLine}</span>
                  </div>
                ))}
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}

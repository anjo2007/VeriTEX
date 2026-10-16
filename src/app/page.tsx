'use client';

import React, { useEffect, useMemo, useRef, useState, useCallback } from 'react';
import JSZip from 'jszip';
import katex from 'katex';
import {
  AlertCircle,
  AlertTriangle,
  AlignLeft,
  ArrowLeft,
  ArrowRight,
  BookOpen,
  Bot,
  Calendar,
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
  FileSearch,
  FileText,
  FileUp,
  Folder,
  FolderOpen,
  FolderPlus,
  GitCompare,
  Globe,
  Grid,
  HardDrive,
  Home,
  Info,
  KeyRound,
  Layers,
  Layout,
  List,
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
  Upload,
  UploadCloud,
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
import { decompilePdfToLatexProject, DecompilationOutput } from '@/lib/pdf-decompiler';
import {
  compileAndDownloadPdf,
  downloadProjectZipBundle,
  downloadTexSourceFile,
} from '@/lib/real-pdf-compiler';

const FIXED_INITIAL_TIME = 1760500000000;
const STORAGE_PROJECTS_KEY = 'veritex_projects_v2';
const STORAGE_ACTIVE_ID_KEY = 'veritex_active_project_id_v2';

const INITIAL_PROJECT: VeriTeXProject = {
  id: 'proj-riemannian-attention',
  name: 'Spectral_Riemannian_Attention',
  rootFile: 'main.tex',
  createdAt: FIXED_INITIAL_TIME,
  updatedAt: FIXED_INITIAL_TIME,
  files: {
    'main.tex': {
      path: 'main.tex',
      type: 'tex',
      updatedAt: FIXED_INITIAL_TIME,
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
      updatedAt: FIXED_INITIAL_TIME,
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

const TEMPLATES: Record<
  string,
  { name: string; tag: string; desc: string; icon: string; files: Record<string, string> }
> = {
  nature: {
    name: 'Nature / Science Journal',
    tag: 'Journal',
    desc: 'High-impact 2-column layout with abstract, methods, and references.',
    icon: '🌿',
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
    name: 'IEEE Transactions & Conf',
    tag: 'Conference',
    desc: 'Two-column engineering & computer science standard conference layout.',
    icon: '⚡',
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
    name: 'NeurIPS / ICML AI Paper',
    tag: 'Machine Learning',
    desc: 'Premier machine learning format with math environments & booktabs.',
    icon: '🧠',
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
    tag: 'Dissertation',
    desc: 'Modular dissertation structure with \\input{} and relative paths.',
    icon: '🎓',
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
  beamer: {
    name: 'Beamer Presentation Slides',
    tag: 'Slides',
    desc: 'Academic conference slide deck with frames, blocks, and math equations.',
    icon: '📊',
    files: {
      'main.tex': `\\documentclass{beamer}
\\usetheme{Madrid}

\\title{Spectral Attention on Manifolds}
\\subtitle{Sub-Linear Scientific Transformers}
\\author{A. Researcher}
\\institute{VeriTeX Institute}
\\date{\\today}

\\begin{document}

\\frame{\\titlepage}

\\begin{frame}{Motivation}
  \\begin{itemize}
    \\item Traditional attention incurs $\\mathcal{O}(N^2)$ complexity.
    \\item Long sequence scientific data requires sub-quadratic scaling.
    \\item Our approach: Laplace-Beltrami spectral projections.
  \\end{itemize}
\\end{frame}

\\begin{frame}{Formulation}
  \\begin{block}{Geodesic Spectral Attention}
    \\begin{equation}
      \\mathcal{K}_t(x, y) = \\sum_{k=1}^K e^{-\\lambda_k t} \\phi_k(x) \\phi_k(y)
    \\end{equation}
  \\end{block}
\\end{frame}

\\end{document}`,
    },
  },
  resume: {
    name: 'Scientific CV / Resume',
    tag: 'Curriculum Vitae',
    desc: 'Clean, elegant academic curriculum vitae with publications and appointments.',
    icon: '📄',
    files: {
      'main.tex': `\\documentclass[11pt,a4paper]{article}
\\usepackage[margin=1in]{geometry}
\\usepackage{amsmath,amssymb}

\\title{Curriculum Vitae: Dr. Alex Mercer}
\\author{alex.mercer@oxford.edu}
\\date{}

\\begin{document}
\\maketitle

\\section*{Academic Appointments}
\\textbf{Senior Research Fellow in AI for Science} \\hfill 2024--Present \\\\
University of Oxford, Department of Computer Science

\\section*{Education}
\\textbf{Ph.D. in Applied Mathematics} \\hfill 2020--2024 \\\\
University of Cambridge

\\section*{Selected Publications}
\\begin{itemize}
  \\item \\textbf{A. Mercer}, B. Coauthor. \\textit{Sub-Linear Geodesic Attention on Riemannian Manifolds}. NeurIPS 2026.
\\end{itemize}

\\end{document}`,
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
  // Top-Level View: 'dashboard' (Projects Hub) or 'studio' (IDE Workbench)
  const [currentView, setCurrentView] = useState<'dashboard' | 'studio'>('dashboard');

  // SSR Hydration Guard
  const [mounted, setMounted] = useState<boolean>(false);

  // Projects Collection State
  const [projectsList, setProjectsList] = useState<VeriTeXProject[]>([INITIAL_PROJECT]);
  const [activeProjectId, setActiveProjectId] = useState<string>(INITIAL_PROJECT.id);
  const [searchFilter, setSearchFilter] = useState<string>('');
  const [projectSortBy, setProjectSortBy] = useState<'recent' | 'name' | 'files'>('recent');

  // Studio Active Project & File State
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
  const [splitWidth, setSplitWidth] = useState<number>(50);
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
  const collabSessionRef = useRef<any>(null);
  const [collabCopied, setCollabCopied] = useState<boolean>(false);

  // PDF to LaTeX Converter Modal & Pipeline
  const [decompileModalOpen, setDecompileModalOpen] = useState<boolean>(false);
  const [decompileStage, setDecompileStage] = useState<string>('Ready to decompile PDF');
  const [decompilePercent, setDecompilePercent] = useState<number>(0);
  const [decompileLoading, setDecompileLoading] = useState<boolean>(false);
  const [decompileOutput, setDecompileOutput] = useState<DecompilationOutput | null>(null);
  const [decompileActiveTab, setDecompileActiveTab] = useState<string>('main.tex');

  // DOM Refs
  const editorRef = useRef<HTMLTextAreaElement>(null);
  const backdropRef = useRef<HTMLDivElement>(null);
  const gutterRef = useRef<HTMLDivElement>(null);
  const mathInspectorRef = useRef<HTMLDivElement>(null);
  const previewScrollRef = useRef<HTMLDivElement>(null);
  const isInitializedRef = useRef<boolean>(false);

  // Real PDF Compilation & Export Feedback States
  const [pdfCompiling, setPdfCompiling] = useState<{ active: boolean; stage: string; percent: number } | null>(null);
  const [exportNotice, setExportNotice] = useState<string | null>(null);

  // 1. Client-Side Only Mount Initialization
  useEffect(() => {
    setMounted(true);
    try {
      const savedProjects = localStorage.getItem(STORAGE_PROJECTS_KEY);
      if (savedProjects) {
        const parsed = JSON.parse(savedProjects) as VeriTeXProject[];
        if (Array.isArray(parsed) && parsed.length > 0) {
          setProjectsList(parsed);
          const savedActiveId = localStorage.getItem(STORAGE_ACTIVE_ID_KEY) || parsed[0].id;
          const found = parsed.find((p) => p.id === savedActiveId) || parsed[0];
          setActiveProjectId(found.id);
          setProject(found);
          setActiveFilePath(found.rootFile || Object.keys(found.files)[0]);
          setOpenTabs(Object.keys(found.files).slice(0, 3));
        }
      }
    } catch {
      // LocalStorage fallback
    } finally {
      isInitializedRef.current = true;
    }

    // Connect to P2P collaboration room via dynamic import to prevent SSR warnings
    const params = new URLSearchParams(window.location.search);
    const roomParam = params.get('room');
    if (roomParam) {
      setCollabRoomId(roomParam);
      setCurrentView('studio');
    }

    import('@/lib/collab-crdt')
      .then(({ VeriTeXCollabSession }) => {
        try {
          const session = new VeriTeXCollabSession(roomParam || 'veritex-mesh');
          collabSessionRef.current = session;
          session.onPeersChange((peers) => {
            setCollabPeers(peers);
          });
        } catch {
          // Offline fallback
        }
      })
      .catch(() => {});

    return () => {
      collabSessionRef.current?.destroy();
    };
  }, []);

  // Save projects to LocalStorage
  const persistProjects = useCallback((updatedList: VeriTeXProject[]) => {
    setProjectsList(updatedList);
    try {
      localStorage.setItem(STORAGE_PROJECTS_KEY, JSON.stringify(updatedList));
    } catch {
      // Ignore quota errors
    }
  }, []);

  // Sync current active project into projectsList (guarded against template overwrite)
  useEffect(() => {
    if (!mounted || !isInitializedRef.current) return;
    setProjectsList((prevList) => {
      const idx = prevList.findIndex((p) => p.id === project.id);
      let nextList: VeriTeXProject[];
      if (idx !== -1) {
        nextList = [...prevList];
        nextList[idx] = project;
      } else {
        nextList = [project, ...prevList];
      }
      try {
        localStorage.setItem(STORAGE_PROJECTS_KEY, JSON.stringify(nextList));
        localStorage.setItem(STORAGE_ACTIVE_ID_KEY, project.id);
      } catch {
        // Ignore quota errors
      }
      return nextList;
    });
  }, [project, mounted]);

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

  // Global Keyboard Shortcuts (Ctrl+K, Ctrl+P, Ctrl+B)
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
      } else if (e.key === 'Escape') {
        if (completions) setCompletions(null);
        if (inlineAiOpen) setInlineAiOpen(false);
        if (commandPaletteOpen) setCommandPaletteOpen(false);
        if (decompileModalOpen) setDecompileModalOpen(false);
      }
    };

    window.addEventListener('keydown', handleGlobalKeyDown);
    return () => window.removeEventListener('keydown', handleGlobalKeyDown);
  }, [completions, inlineAiOpen, commandPaletteOpen, decompileModalOpen]);

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

  // Bracket Auto-Closing, Smart Indentation, and LSP in Textarea
  const handleEditorKeyDown = (e: React.KeyboardEvent<HTMLTextAreaElement>) => {
    const textarea = e.currentTarget;
    const { selectionStart, selectionEnd, value } = textarea;

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

    const pairs: Record<string, string> = { '{': '}', '[': ']', '(': ')', $: '$' };
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

  // Textarea change + LSP detector
  const handleEditorChange = (e: React.ChangeEvent<HTMLTextAreaElement>) => {
    const content = e.target.value;
    const selStart = e.target.selectionStart;
    updateFileContent(activeFile.path, content);
    setCursorOffset(selStart);

    const lineNum = content.slice(0, selStart).split('\n').length;
    setCursorLine(lineNum);

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

  // Insert autocomplete item
  const insertAutocompleteItem = (item: AutocompleteItem) => {
    if (!editorRef.current || !completions) return;
    const textarea = editorRef.current;
    const content = textarea.value;
    const selStart = textarea.selectionStart;

    const lineStartIdx = content.lastIndexOf('\n', selStart - 1) + 1;
    const linePrefix = content.slice(lineStartIdx, selStart);

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

  // Split-Pane Resizer
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

  // SyncTeX double-click in Paper Preview
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

  // Insert Symbol
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

  // Accept Surgical Patch
  const handleAcceptPatch = (patch: SurgicalDiffPatch) => {
    const { updatedSource, applied } = applySurgicalPatch(activeFile.content, patch);
    if (applied) {
      updateFileContent(activeFile.path, updatedSource);
      setPendingPatches((prev) => prev.filter((p) => p.id !== patch.id));
    }
  };

  // Send AI Prompt
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
        const searchCandidate = astSlice.activeBlockContent;
        if (searchCandidate && response.length > 20) {
          const rawCode = response.replace(/^```latex\n?|^```\n?|```$/gm, '').trim();
          setPendingPatches((prev) => [
            ...prev,
            {
              id: `patch-${Date.now()}`,
              searchBlock: searchCandidate,
              replaceBlock: rawCode,
              explanation: `AI Generation: "${promptText.slice(0, 40)}..."`,
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

  // Real LaTeX PDF Compiler & Direct Binary Download
  const handleCompileAndExportPdf = async (targetProj?: VeriTeXProject) => {
    const proj = targetProj || project;
    try {
      setPdfCompiling({ active: true, stage: 'Compiling LaTeX AST & vector typography...', percent: 15 });
      const res = await compileAndDownloadPdf(proj, 'veritex-printable-paper', (progress) => {
        setPdfCompiling({ active: true, stage: progress.stage, percent: progress.percent });
      });

      if (res.success) {
        setPdfCompiling(null);
        setExportNotice(`Generated & Downloaded ${proj.name}.pdf!`);
        setTimeout(() => setExportNotice(null), 4000);
      } else {
        setPdfCompiling(null);
        alert(`PDF Compilation Error: ${res.error || 'Failed to assemble PDF document'}`);
      }
    } catch (err: unknown) {
      setPdfCompiling(null);
      alert('PDF Compilation Error: ' + (err instanceof Error ? err.message : 'Unknown error'));
    }
  };

  // Export full multi-file .ZIP project
  const handleExportZip = async (targetProj?: VeriTeXProject) => {
    const proj = targetProj || project;
    try {
      await downloadProjectZipBundle(proj);
      setExportNotice(`Downloaded ${proj.name}_source.zip!`);
      setTimeout(() => setExportNotice(null), 4000);
    } catch {
      alert('Failed to package .ZIP bundle.');
    }
  };

  // Export active .tex source file
  const handleExportActiveTex = () => {
    if (!activeFile) return;
    downloadTexSourceFile(activeFile.path, activeFile.content);
    setExportNotice(`Downloaded ${activeFile.path}!`);
    setTimeout(() => setExportNotice(null), 4000);
  };

  // Create Project from Template
  const handleCreateProjectFromTemplate = (tplKey: string) => {
    const tpl = TEMPLATES[tplKey];
    if (!tpl) return;
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

    const newProj: VeriTeXProject = {
      id: `proj-${Date.now()}`,
      name: `${tpl.name.replace(/[^a-zA-Z0-9]/g, '_')}_Paper`,
      rootFile: 'main.tex',
      createdAt: Date.now(),
      updatedAt: Date.now(),
      files: newFiles,
    };

    const updatedList = [newProj, ...projectsList];
    persistProjects(updatedList);
    setActiveProjectId(newProj.id);
    setProject(newProj);
    setActiveFilePath('main.tex');
    setOpenTabs(Object.keys(newFiles));
    setCurrentView('studio');
  };

  // Create Blank Project
  const handleCreateBlankProject = () => {
    const name = prompt('Enter project name:', 'New_Scientific_Paper');
    if (!name?.trim()) return;

    const newProj: VeriTeXProject = {
      id: `proj-${Date.now()}`,
      name: name.trim().replace(/\s+/g, '_'),
      rootFile: 'main.tex',
      createdAt: Date.now(),
      updatedAt: Date.now(),
      files: {
        'main.tex': {
          path: 'main.tex',
          type: 'tex',
          updatedAt: Date.now(),
          content: `\\documentclass[11pt]{article}
\\usepackage{amsmath,amssymb}

\\title{${name.trim()}}
\\author{Author Name}
\\date{\\today}

\\begin{document}
\\maketitle

\\begin{abstract}
Initial project abstract.
\\end{abstract}

\\section{Introduction}
Start writing your scientific paper here...

\\end{document}`,
        },
        'references.bib': {
          path: 'references.bib',
          type: 'bib',
          updatedAt: Date.now(),
          content: `% Bibliography entries\n`,
        },
      },
    };

    const updatedList = [newProj, ...projectsList];
    persistProjects(updatedList);
    setActiveProjectId(newProj.id);
    setProject(newProj);
    setActiveFilePath('main.tex');
    setOpenTabs(['main.tex', 'references.bib']);
    setCurrentView('studio');
  };

  // Import .ZIP Project
  const handleImportZipProject = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;

    try {
      const zip = await JSZip.loadAsync(file);
      const newFiles: Record<string, VirtualFile> = {};

      for (const [relPath, entry] of Object.entries(zip.files)) {
        if (entry.dir) continue;
        if (/\.(tex|bib|sty|cls|csv|txt)$/i.test(relPath)) {
          const content = await entry.async('string');
          const ext = (relPath.split('.').pop()?.toLowerCase() || 'tex') as VirtualFile['type'];
          newFiles[relPath] = { path: relPath, content, type: ext, updatedAt: Date.now() };
        }
      }

      const detectedRoot =
        Object.values(newFiles).find((f) => f.content.includes('\\documentclass'))?.path ||
        Object.keys(newFiles)[0] ||
        'main.tex';

      const newProj: VeriTeXProject = {
        id: `proj-${Date.now()}`,
        name: file.name.replace(/\.zip$/i, ''),
        rootFile: detectedRoot,
        createdAt: Date.now(),
        updatedAt: Date.now(),
        files: newFiles,
      };

      const updatedList = [newProj, ...projectsList];
      persistProjects(updatedList);
      setActiveProjectId(newProj.id);
      setProject(newProj);
      setActiveFilePath(detectedRoot);
      setOpenTabs(Object.keys(newFiles).slice(0, 4));
      setCurrentView('studio');
    } catch {
      alert('Failed to parse .ZIP bundle. Please ensure it contains valid LaTeX files.');
    }
  };

  // Import Project as Folder (HTML5 webkitdirectory)
  const handleImportFolder = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const fileList = e.target.files;
    if (!fileList || fileList.length === 0) return;

    try {
      const newFiles: Record<string, VirtualFile> = {};
      let folderName = 'Imported_Folder_Project';

      for (let i = 0; i < fileList.length; i++) {
        const file = fileList[i];
        const relativePath = (file as any).webkitRelativePath || file.name;
        const parts = relativePath.split('/');
        if (parts.length > 1) {
          folderName = parts[0];
        }

        const cleanRelPath = parts.length > 1 ? parts.slice(1).join('/') : file.name;
        if (/\.(tex|bib|sty|cls|csv|txt|md|json)$/i.test(file.name)) {
          const content = await file.text();
          const ext = (file.name.split('.').pop()?.toLowerCase() || 'tex') as VirtualFile['type'];
          newFiles[cleanRelPath] = {
            path: cleanRelPath,
            content,
            type: ext,
            updatedAt: Date.now(),
          };
        }
      }

      if (Object.keys(newFiles).length === 0) {
        alert('No compatible LaTeX or text files found in the selected folder.');
        return;
      }

      const detectedRoot =
        Object.values(newFiles).find((f) => f.content.includes('\\documentclass'))?.path ||
        newFiles['main.tex']?.path ||
        Object.keys(newFiles)[0];

      // If in Studio, ask to merge or create new
      if (currentView === 'studio' && confirm('Merge these folder files into the current project? (Cancel to create new project)')) {
        setProject((prev) => ({
          ...prev,
          updatedAt: Date.now(),
          files: { ...prev.files, ...newFiles },
        }));
        setOpenTabs((prev) => Array.from(new Set([...prev, ...Object.keys(newFiles).slice(0, 3)])));
        alert(`Imported ${Object.keys(newFiles).length} files into ${project.name}!`);
        return;
      }

      const newProj: VeriTeXProject = {
        id: `proj-${Date.now()}`,
        name: folderName.replace(/[^a-zA-Z0-9]/g, '_'),
        rootFile: detectedRoot,
        createdAt: Date.now(),
        updatedAt: Date.now(),
        files: newFiles,
      };

      const updatedList = [newProj, ...projectsList];
      persistProjects(updatedList);
      setActiveProjectId(newProj.id);
      setProject(newProj);
      setActiveFilePath(detectedRoot);
      setOpenTabs(Object.keys(newFiles).slice(0, 4));
      setCurrentView('studio');
    } catch (err: unknown) {
      alert('Error importing folder: ' + (err instanceof Error ? err.message : 'Unknown error'));
    }
  };

  // Import Individual File(s)
  const handleImportFiles = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const fileList = e.target.files;
    if (!fileList || fileList.length === 0) return;

    try {
      const addedFiles: Record<string, VirtualFile> = {};
      for (let i = 0; i < fileList.length; i++) {
        const file = fileList[i];
        if (/\.(tex|bib|sty|cls|csv|txt|md)$/i.test(file.name)) {
          const content = await file.text();
          const ext = (file.name.split('.').pop()?.toLowerCase() || 'tex') as VirtualFile['type'];
          addedFiles[file.name] = {
            path: file.name,
            content,
            type: ext,
            updatedAt: Date.now(),
          };
        }
      }

      if (Object.keys(addedFiles).length === 0) {
        alert('No compatible text files selected.');
        return;
      }

      // If in Studio mode, add directly to project
      if (currentView === 'studio') {
        setProject((prev) => ({
          ...prev,
          updatedAt: Date.now(),
          files: { ...prev.files, ...addedFiles },
        }));
        setOpenTabs((prev) => Array.from(new Set([...prev, ...Object.keys(addedFiles)])));
        setActiveFilePath(Object.keys(addedFiles)[0]);
        alert(`Imported ${Object.keys(addedFiles).length} file(s) into project!`);
      } else {
        // In Dashboard, create new project from these files
        const rootCandidate =
          Object.values(addedFiles).find((f) => f.content.includes('\\documentclass'))?.path ||
          Object.keys(addedFiles)[0];

        const newProj: VeriTeXProject = {
          id: `proj-${Date.now()}`,
          name: fileList[0].name.replace(/\.[^/.]+$/, '').replace(/[^a-zA-Z0-9]/g, '_'),
          rootFile: rootCandidate,
          createdAt: Date.now(),
          updatedAt: Date.now(),
          files: addedFiles,
        };

        const updatedList = [newProj, ...projectsList];
        persistProjects(updatedList);
        setActiveProjectId(newProj.id);
        setProject(newProj);
        setActiveFilePath(rootCandidate);
        setOpenTabs(Object.keys(addedFiles));
        setCurrentView('studio');
      }
    } catch {
      alert('Error reading uploaded files.');
    }
  };

  // Decompile PDF File
  const handleStartPdfDecompilation = async (file: File) => {
    setDecompileLoading(true);
    setDecompilePercent(10);
    setDecompileStage('Initializing PDF.js engine & extracting layout...');
    setDecompileOutput(null);

    try {
      const output = await decompilePdfToLatexProject(file, aiConfig, (stage, percent) => {
        setDecompileStage(stage);
        setDecompilePercent(percent);
      });

      setDecompileOutput(output);
      setDecompileActiveTab(output.rootFile);
      setDecompileStage('Decompilation complete! Review generated LaTeX files below.');
      setDecompilePercent(100);
    } catch (err: unknown) {
      alert('PDF decompile failed: ' + (err instanceof Error ? err.message : 'Unknown error'));
      setDecompileStage('Error occurred during PDF decompilation.');
    } finally {
      setDecompileLoading(false);
    }
  };

  // Create Project from Decompile Output
  const handleCreateProjectFromDecompile = () => {
    if (!decompileOutput) return;

    const newProj: VeriTeXProject = {
      id: `proj-decomp-${Date.now()}`,
      name: 'Decompiled_PDF_Manuscript',
      rootFile: decompileOutput.rootFile,
      createdAt: Date.now(),
      updatedAt: Date.now(),
      files: decompileOutput.files,
    };

    const updatedList = [newProj, ...projectsList];
    persistProjects(updatedList);
    setActiveProjectId(newProj.id);
    setProject(newProj);
    setActiveFilePath(decompileOutput.rootFile);
    setOpenTabs(Object.keys(decompileOutput.files).slice(0, 4));
    setDecompileModalOpen(false);
    setCurrentView('studio');
  };

  // Duplicate Project
  const handleDuplicateProject = (proj: VeriTeXProject) => {
    const cloned: VeriTeXProject = {
      ...proj,
      id: `proj-${Date.now()}`,
      name: `${proj.name}_Copy`,
      createdAt: Date.now(),
      updatedAt: Date.now(),
    };
    persistProjects([cloned, ...projectsList]);
  };

  // Delete Project
  const handleDeleteProject = (projId: string) => {
    if (projectsList.length <= 1) {
      alert('Cannot delete the only project.');
      return;
    }
    if (confirm('Are you sure you want to delete this project?')) {
      const filtered = projectsList.filter((p) => p.id !== projId);
      persistProjects(filtered);
      if (activeProjectId === projId) {
        const nextActive = filtered[0];
        setActiveProjectId(nextActive.id);
        setProject(nextActive);
        setActiveFilePath(nextActive.rootFile);
        setOpenTabs(Object.keys(nextActive.files).slice(0, 3));
      }
    }
  };

  // Open Project in Studio
  const handleOpenProject = (proj: VeriTeXProject) => {
    setActiveProjectId(proj.id);
    setProject(proj);
    setActiveFilePath(proj.rootFile || Object.keys(proj.files)[0]);
    setOpenTabs(Object.keys(proj.files).slice(0, 4));
    setCurrentView('studio');
  };

  // Add new file in active project
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
    if (!openTabs.includes(clean)) setOpenTabs((prev) => [...prev, clean]);
  };

  // Close Tab
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
    if (typeof window === 'undefined') return;
    const url = `${window.location.origin}${window.location.pathname}?room=${collabRoomId}`;
    navigator.clipboard.writeText(url);
    setCollabCopied(true);
    setTimeout(() => setCollabCopied(false), 2500);
  };

  // Crossref Search
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

  // Insert Crossref citation
  const insertVerifiedCitation = (entry: BibEntry) => {
    const bibFile = project.files['references.bib'] || Object.values(project.files).find((f) => f.type === 'bib');
    const bibPath = bibFile ? bibFile.path : 'references.bib';
    const bibTeX = formatBibEntryToBibTeX(entry);

    const oldContent = project.files[bibPath]?.content || '';
    const newContent = oldContent ? `${oldContent}\n\n${bibTeX}\n` : `${bibTeX}\n`;

    updateFileContent(bibPath, newContent);
    alert(`Added \\cite{${entry.key}} to ${bibPath}!`);
  };

  // Filtered & Sorted Projects for Home Dashboard
  const filteredProjects = useMemo(() => {
    return projectsList
      .filter(
        (p) =>
          p.name.toLowerCase().includes(searchFilter.toLowerCase()) ||
          Object.values(p.files).some((f) => f.content.toLowerCase().includes(searchFilter.toLowerCase()))
      )
      .sort((a, b) => {
        if (projectSortBy === 'recent') return b.updatedAt - a.updatedAt;
        if (projectSortBy === 'name') return a.name.localeCompare(b.name);
        return Object.keys(b.files).length - Object.keys(a.files).length;
      });
  }, [projectsList, searchFilter, projectSortBy]);

  // Command Palette Items
  const commandPaletteItems = useMemo(() => {
    return [
      {
        id: 'home',
        title: 'Go to Projects Home Dashboard',
        shortcut: 'Esc',
        action: () => setCurrentView('dashboard'),
      },
      {
        id: 'studio',
        title: 'Open Studio IDE Workbench',
        shortcut: '',
        action: () => setCurrentView('studio'),
      },
      {
        id: 'pdf_decompile',
        title: 'Open PDF-to-LaTeX Converter Studio',
        shortcut: '',
        action: () => setDecompileModalOpen(true),
      },
      {
        id: 'compile',
        title: 'Compile Workspace',
        shortcut: 'Ctrl+Enter',
        action: () =>
          alert(`Compiled in ${compilation.compileTimeMs}ms with ${compilation.diagnostics.length} diagnostics.`),
      },
      {
        id: 'inline_ai',
        title: 'Open Inline AI Command Bar',
        shortcut: 'Ctrl+K',
        action: () => {
          setCurrentView('studio');
          setInlineAiOpen(true);
        },
      },
      {
        id: 'export_zip',
        title: 'Export Multi-File .ZIP Project',
        shortcut: '',
        action: () => handleExportZip(),
      },
      {
        id: 'compile_pdf',
        title: 'Compile & Download Real PDF (.pdf)',
        shortcut: 'Ctrl+Shift+E',
        action: () => handleCompileAndExportPdf(),
      },
      {
        id: 'new_blank',
        title: 'Create New Blank Project',
        shortcut: '',
        action: handleCreateBlankProject,
      },
      {
        id: 'byom_settings',
        title: 'Setup BYOM AI Providers (Gemini, Claude, GPT, Ollama)',
        shortcut: '',
        action: () => setByomModalOpen(true),
      },
      ...Object.keys(TEMPLATES).map((key) => ({
        id: `template_${key}`,
        title: `Create Project from Template: ${TEMPLATES[key].name}`,
        shortcut: '',
        action: () => handleCreateProjectFromTemplate(key),
      })),
      ...Object.keys(project.files).map((f) => ({
        id: `open_${f}`,
        title: `Go to File: ${f}`,
        shortcut: '',
        action: () => {
          setCurrentView('studio');
          setActiveFilePath(f);
          if (!openTabs.includes(f)) setOpenTabs((prev) => [...prev, f]);
        },
      })),
    ].filter((item) => item.title.toLowerCase().includes(paletteSearch.toLowerCase()));
  }, [paletteSearch, compilation, project.files, openTabs]);

  return (
    <div className="flex flex-col h-screen w-screen overflow-hidden bg-zinc-950 text-zinc-100 font-sans select-none">
      {/* ========================================================
          TOP NAVIGATION BAR (SHARED ACROSS HOME & STUDIO)
          ======================================================== */}
      <header className="h-12 border-b border-zinc-800 bg-zinc-950/95 backdrop-blur px-3 flex items-center justify-between z-20 shrink-0">
        <div className="flex items-center gap-3">
          {/* Brand Logo & View Switcher */}
          <div
            onClick={() => setCurrentView('dashboard')}
            className="flex items-center gap-2 cursor-pointer group"
            title="Go to Home Dashboard"
          >
            <div className="w-7 h-7 rounded-lg bg-gradient-to-tr from-emerald-600 via-teal-500 to-cyan-400 flex items-center justify-center shadow-lg shadow-emerald-500/20 group-hover:scale-105 transition">
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

          {/* Navigation Mode Pill Tabs */}
          <div className="flex items-center gap-1 bg-zinc-900/80 p-0.5 rounded-lg border border-zinc-800">
            <button
              onClick={() => setCurrentView('dashboard')}
              className={`px-3 py-1 rounded-md text-xs font-semibold transition flex items-center gap-1.5 ${
                currentView === 'dashboard'
                  ? 'bg-emerald-500/15 text-emerald-300 shadow-sm border border-emerald-500/30'
                  : 'text-zinc-400 hover:text-zinc-200'
              }`}
            >
              <Home className="w-3.5 h-3.5" /> Dashboard
            </button>
            <button
              onClick={() => setCurrentView('studio')}
              className={`px-3 py-1 rounded-md text-xs font-semibold transition flex items-center gap-1.5 ${
                currentView === 'studio'
                  ? 'bg-emerald-500/15 text-emerald-300 shadow-sm border border-emerald-500/30'
                  : 'text-zinc-400 hover:text-zinc-200'
              }`}
            >
              <Layout className="w-3.5 h-3.5" /> Studio IDE
            </button>
          </div>

          {/* If in Studio mode: Project Name & Collaboration Badge */}
          {currentView === 'studio' && (
            <>
              <input
                value={project.name}
                onChange={(e) => setProject((prev) => ({ ...prev, name: e.target.value }))}
                className="bg-transparent hover:bg-zinc-900 focus:bg-zinc-900 border border-transparent focus:border-zinc-700 rounded px-2 py-0.5 text-xs font-medium text-zinc-300 focus:outline-none transition max-w-[180px] truncate"
                title="Rename Project"
              />

              <button
                onClick={() => setCollabModalOpen(true)}
                className="flex items-center gap-1.5 px-2.5 py-1 rounded-full bg-zinc-900 hover:bg-zinc-800 border border-zinc-800 text-[11px] text-zinc-300 transition"
                title="Real-Time Collaborative Room"
              >
                <span className="relative flex h-2 w-2">
                  <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-emerald-400 opacity-75"></span>
                  <span className="relative inline-flex rounded-full h-2 w-2 bg-emerald-500"></span>
                </span>
                <span suppressHydrationWarning className="font-mono">
                  {mounted ? collabRoomId : 'veritex-mesh'}
                </span>
                {mounted && collabPeers.length > 0 && (
                  <span className="px-1.5 py-0.2 rounded-full bg-emerald-500/20 text-emerald-300 text-[10px] font-bold">
                    {collabPeers.length + 1}
                  </span>
                )}
              </button>
            </>
          )}
        </div>

        {/* Right Header Toolbar */}
        <div className="flex items-center gap-2">
          {/* Universal PDF-to-LaTeX Converter Trigger */}
          <button
            onClick={() => setDecompileModalOpen(true)}
            className="flex items-center gap-1.5 px-2.5 py-1 rounded-md bg-cyan-500/10 hover:bg-cyan-500/20 text-cyan-300 border border-cyan-500/30 text-xs font-semibold transition"
            title="Decompile any PDF document into compilable LaTeX"
          >
            <FileSearch className="w-3.5 h-3.5 text-cyan-400" />
            <span>PDF to LaTeX</span>
          </button>

          {currentView === 'studio' ? (
            <>
              {/* Studio View Mode Switcher */}
              <div className="hidden sm:flex items-center gap-1 bg-zinc-900/80 p-0.5 rounded-lg border border-zinc-800">
                <button
                  onClick={() => setWorkbenchMode('studio')}
                  className={`px-2.5 py-1 rounded-md text-xs font-medium transition ${
                    workbenchMode === 'studio' ? 'bg-emerald-500/15 text-emerald-300' : 'text-zinc-400'
                  }`}
                >
                  Split
                </button>
                <button
                  onClick={() => setWorkbenchMode('focus')}
                  className={`px-2.5 py-1 rounded-md text-xs font-medium transition ${
                    workbenchMode === 'focus' ? 'bg-emerald-500/15 text-emerald-300' : 'text-zinc-400'
                  }`}
                >
                  Focus
                </button>
                <button
                  onClick={() => setWorkbenchMode('diff')}
                  className={`px-2.5 py-1 rounded-md text-xs font-medium transition ${
                    workbenchMode === 'diff' ? 'bg-emerald-500/15 text-emerald-300' : 'text-zinc-400'
                  }`}
                >
                  Diffs ({pendingPatches.length})
                </button>
              </div>

              {/* Ctrl+K Inline AI */}
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

              {/* Command Palette */}
              <button
                onClick={() => setCommandPaletteOpen(true)}
                className="flex items-center gap-1 px-2 py-1 rounded-md bg-zinc-900 hover:bg-zinc-800 text-zinc-300 border border-zinc-800 text-xs transition"
                title="Command Palette (Ctrl+P)"
              >
                <Command className="w-3.5 h-3.5" />
              </button>

              {/* Compilation Badge with Hydration Suppression */}
              <div
                suppressHydrationWarning
                className={`flex items-center gap-1.5 px-2 py-1 rounded-md text-xs font-mono border ${
                  compilation.diagnostics.some((d) => d.severity === 'error')
                    ? 'bg-red-500/10 border-red-500/30 text-red-300'
                    : 'bg-zinc-900 border-zinc-800 text-emerald-400'
                }`}
                title="Sub-10ms Client Compiler"
              >
                <Clock className="w-3 h-3 text-emerald-400" />
                <span suppressHydrationWarning>{mounted ? `${compilation.compileTimeMs}ms` : '<10ms'}</span>
              </div>

              {/* Real PDF Compiler & Direct Download Button */}
              <button
                onClick={() => handleCompileAndExportPdf()}
                disabled={pdfCompiling?.active}
                className="px-3 py-1 rounded-md bg-emerald-600 hover:bg-emerald-500 disabled:opacity-50 text-white font-semibold text-xs flex items-center gap-1.5 shadow-sm shadow-emerald-500/25 transition"
                title="Compile LaTeX into real multi-page binary PDF and download directly"
              >
                {pdfCompiling?.active ? (
                  <RefreshCw className="w-3.5 h-3.5 animate-spin" />
                ) : (
                  <Download className="w-3.5 h-3.5" />
                )}
                <span>{pdfCompiling?.active ? 'Compiling PDF...' : 'Export PDF'}</span>
              </button>

              {/* Source Downloads & More Dropdown */}
              <div className="relative group">
                <button
                  className="px-2 py-1 rounded-md bg-zinc-900 hover:bg-zinc-800 text-zinc-300 border border-zinc-800 text-xs flex items-center gap-1 transition"
                  title="Export Sources (.zip, .tex, etc.)"
                >
                  <FileCode className="w-3.5 h-3.5 text-zinc-400" />
                  <ChevronDown className="w-3 h-3 text-zinc-500" />
                </button>
                <div className="absolute right-0 top-full mt-1 w-52 rounded-xl bg-zinc-900 border border-zinc-800 shadow-2xl py-1.5 hidden group-hover:block z-50">
                  <button
                    onClick={() => handleExportZip()}
                    className="w-full px-3 py-1.5 text-left text-xs text-zinc-200 hover:bg-zinc-800 flex items-center gap-2"
                  >
                    <Download className="w-3.5 h-3.5 text-amber-400" />
                    <span>Download Project (.zip)</span>
                  </button>
                  <button
                    onClick={handleExportActiveTex}
                    className="w-full px-3 py-1.5 text-left text-xs text-zinc-200 hover:bg-zinc-800 flex items-center gap-2"
                  >
                    <FileText className="w-3.5 h-3.5 text-cyan-400" />
                    <span>Download {activeFile?.path || 'source.tex'}</span>
                  </button>
                  <button
                    onClick={() => window.print()}
                    className="w-full px-3 py-1.5 text-left text-xs text-zinc-400 hover:bg-zinc-800 flex items-center gap-2 border-t border-zinc-800/80 mt-1 pt-1.5"
                  >
                    <Printer className="w-3.5 h-3.5 text-zinc-500" />
                    <span>Browser Print Dialog</span>
                  </button>
                </div>
              </div>
            </>
          ) : (
            <>
              {/* Home Dashboard Right Actions */}
              <button
                onClick={handleCreateBlankProject}
                className="px-3 py-1 rounded-lg bg-emerald-600 hover:bg-emerald-500 text-white text-xs font-semibold flex items-center gap-1.5 shadow-md shadow-emerald-500/20 transition"
              >
                <Plus className="w-3.5 h-3.5" /> New Project
              </button>

              <button
                onClick={() => setCurrentView('studio')}
                className="px-3 py-1 rounded-lg bg-zinc-900 hover:bg-zinc-800 border border-zinc-800 text-zinc-200 text-xs font-semibold flex items-center gap-1.5 transition"
              >
                Open Studio <ArrowRight className="w-3.5 h-3.5" />
              </button>
            </>
          )}

          {/* BYOM Settings */}
          <button
            onClick={() => setByomModalOpen(true)}
            className="p-1.5 rounded-md bg-zinc-900 hover:bg-zinc-800 text-zinc-300 border border-zinc-800 transition"
            title="Configure BYOM AI Models"
          >
            <Settings className="w-4 h-4" />
          </button>
        </div>
      </header>

      {/* ========================================================
          VIEW 1: HOME PAGE DASHBOARD (PROJECTS & WORKSPACE HUB)
          ======================================================== */}
      {currentView === 'dashboard' ? (
        <div className="flex-1 overflow-y-auto bg-gradient-to-b from-zinc-950 via-zinc-900/90 to-zinc-950 p-6 md:p-10 space-y-8">
          <div className="max-w-7xl mx-auto space-y-6">
            {/* Hero Banner & Project Actions */}
            <div className="flex flex-col md:flex-row md:items-center justify-between gap-4 border-b border-zinc-800/80 pb-6">
              <div>
                <h1 className="text-2xl md:text-3xl font-extrabold tracking-tight bg-gradient-to-r from-zinc-100 via-zinc-200 to-zinc-400 bg-clip-text text-transparent">
                  Scientific Workbench & Projects Hub
                </h1>
                <p className="text-sm text-zinc-400 mt-1">
                  High-speed deterministic LaTeX authoring, AST-aware AI co-authors, and decentralized P2P collaboration.
                </p>
              </div>

              {/* Complete Action Buttons Suite */}
              <div className="flex flex-wrap items-center gap-2">
                {/* 1. Blank Project */}
                <button
                  onClick={handleCreateBlankProject}
                  className="px-3.5 py-2 rounded-xl bg-emerald-600 hover:bg-emerald-500 text-white text-xs font-bold flex items-center gap-1.5 shadow-lg shadow-emerald-500/25 transition"
                >
                  <Plus className="w-4 h-4" /> Blank Paper
                </button>

                {/* 2. Import Project Folder (webkitdirectory) */}
                <label className="px-3.5 py-2 rounded-xl bg-zinc-900 hover:bg-zinc-800 border border-zinc-700/80 text-zinc-200 text-xs font-semibold flex items-center gap-1.5 cursor-pointer transition">
                  <FolderPlus className="w-4 h-4 text-amber-400" />
                  <span>Import Folder</span>
                  <input
                    type="file"
                    multiple
                    {...({ webkitdirectory: '', directory: '' } as any)}
                    onChange={handleImportFolder}
                    className="hidden"
                  />
                </label>

                {/* 3. Import File(s) */}
                <label className="px-3.5 py-2 rounded-xl bg-zinc-900 hover:bg-zinc-800 border border-zinc-700/80 text-zinc-200 text-xs font-semibold flex items-center gap-1.5 cursor-pointer transition">
                  <FilePlus2 className="w-4 h-4 text-emerald-400" />
                  <span>Import Files</span>
                  <input
                    type="file"
                    multiple
                    accept=".tex,.bib,.sty,.cls,.csv,.txt,.md"
                    onChange={handleImportFiles}
                    className="hidden"
                  />
                </label>

                {/* 4. Import .ZIP Bundle */}
                <label className="px-3.5 py-2 rounded-xl bg-zinc-900 hover:bg-zinc-800 border border-zinc-700/80 text-zinc-200 text-xs font-semibold flex items-center gap-1.5 cursor-pointer transition">
                  <Upload className="w-4 h-4 text-purple-400" />
                  <span>Import .ZIP</span>
                  <input type="file" accept=".zip" onChange={handleImportZipProject} className="hidden" />
                </label>

                {/* 5. PDF to LaTeX Converter Modal */}
                <button
                  onClick={() => setDecompileModalOpen(true)}
                  className="px-3.5 py-2 rounded-xl bg-cyan-600/20 hover:bg-cyan-600/30 border border-cyan-500/40 text-cyan-300 text-xs font-bold flex items-center gap-1.5 transition"
                >
                  <FileSearch className="w-4 h-4 text-cyan-400" />
                  <span>PDF to LaTeX</span>
                </button>
              </div>
            </div>

            {/* Quick Metrics Cards */}
            <div className="grid grid-cols-2 sm:grid-cols-4 gap-3">
              <div className="p-4 rounded-xl bg-zinc-900/60 border border-zinc-800/80 space-y-1">
                <div className="text-xs font-medium text-zinc-400 flex items-center gap-1.5">
                  <Folder className="w-3.5 h-3.5 text-emerald-400" /> Active Projects
                </div>
                <div className="text-2xl font-bold text-zinc-100">{projectsList.length}</div>
              </div>

              <div className="p-4 rounded-xl bg-zinc-900/60 border border-zinc-800/80 space-y-1">
                <div className="text-xs font-medium text-zinc-400 flex items-center gap-1.5">
                  <FileText className="w-3.5 h-3.5 text-cyan-400" /> Total Virtual Files
                </div>
                <div className="text-2xl font-bold text-zinc-100">
                  {projectsList.reduce((acc, p) => acc + Object.keys(p.files).length, 0)}
                </div>
              </div>

              <div className="p-4 rounded-xl bg-zinc-900/60 border border-zinc-800/80 space-y-1">
                <div className="text-xs font-medium text-zinc-400 flex items-center gap-1.5">
                  <Clock className="w-3.5 h-3.5 text-amber-400" /> Client Compile Engine
                </div>
                <div className="text-2xl font-bold text-emerald-400">&lt; 10ms AST</div>
              </div>

              <div className="p-4 rounded-xl bg-zinc-900/60 border border-zinc-800/80 space-y-1">
                <div className="text-xs font-medium text-zinc-400 flex items-center gap-1.5">
                  <Users className="w-3.5 h-3.5 text-purple-400" /> Collaboration
                </div>
                <div className="text-2xl font-bold text-zinc-100">P2P Mesh</div>
              </div>
            </div>

            {/* Templates Quick-Start Gallery */}
            <div className="space-y-3">
              <div className="flex items-center justify-between">
                <h2 className="text-sm font-bold uppercase tracking-wider text-zinc-400 flex items-center gap-2">
                  <Layers className="w-4 h-4 text-emerald-400" />
                  Academic Publication Templates
                </h2>
                <span className="text-xs text-zinc-500">Instant One-Click Scaffolding</span>
              </div>

              <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-3">
                {Object.entries(TEMPLATES).map(([key, tpl]) => (
                  <div
                    key={key}
                    onClick={() => handleCreateProjectFromTemplate(key)}
                    className="group p-4 rounded-xl bg-zinc-900/70 hover:bg-zinc-900 border border-zinc-800 hover:border-emerald-500/50 cursor-pointer transition shadow-sm space-y-2 flex flex-col justify-between"
                  >
                    <div className="space-y-1.5">
                      <div className="flex items-center justify-between">
                        <span className="text-xl">{tpl.icon}</span>
                        <span className="text-[10px] uppercase font-bold tracking-wider px-2 py-0.5 rounded-full bg-zinc-800 text-zinc-400 group-hover:text-emerald-400 group-hover:bg-emerald-500/10 transition">
                          {tpl.tag}
                        </span>
                      </div>
                      <h3 className="font-bold text-sm text-zinc-200 group-hover:text-emerald-300 transition">
                        {tpl.name}
                      </h3>
                      <p className="text-xs text-zinc-400 line-clamp-2 leading-relaxed">{tpl.desc}</p>
                    </div>

                    <div className="pt-2 flex items-center justify-between text-xs text-emerald-400 font-semibold">
                      <span>Create Project</span>
                      <ArrowRight className="w-3.5 h-3.5 group-hover:translate-x-1 transition" />
                    </div>
                  </div>
                ))}
              </div>
            </div>

            {/* Projects Management Section */}
            <div className="space-y-4 pt-4 border-t border-zinc-800/80">
              <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-3">
                <h2 className="text-sm font-bold uppercase tracking-wider text-zinc-400 flex items-center gap-2">
                  <FolderOpen className="w-4 h-4 text-emerald-400" />
                  Saved Projects ({filteredProjects.length})
                </h2>

                {/* Search & Sort Controls */}
                <div className="flex items-center gap-2">
                  <div className="relative">
                    <Search className="w-3.5 h-3.5 absolute left-3 top-2.5 text-zinc-500" />
                    <input
                      value={searchFilter}
                      onChange={(e) => setSearchFilter(e.target.value)}
                      placeholder="Search projects..."
                      className="bg-zinc-900 border border-zinc-800 rounded-lg pl-8 pr-3 py-1.5 text-xs text-zinc-200 focus:outline-none focus:border-emerald-500 w-48 sm:w-64"
                    />
                  </div>

                  <select
                    value={projectSortBy}
                    onChange={(e) => setProjectSortBy(e.target.value as 'recent' | 'name' | 'files')}
                    className="bg-zinc-900 border border-zinc-800 rounded-lg px-2.5 py-1.5 text-xs text-zinc-300 focus:outline-none"
                  >
                    <option value="recent">Recently Edited</option>
                    <option value="name">Name (A-Z)</option>
                    <option value="files">File Count</option>
                  </select>
                </div>
              </div>

              {/* Projects Grid */}
              <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
                {filteredProjects.map((p) => {
                  const root = p.files[p.rootFile] || Object.values(p.files)[0];
                  const previewText = root?.content
                    ? root.content.replace(/\\[a-zA-Z]+(\{[^}]*\})?/g, ' ').slice(0, 140)
                    : 'Empty document.';

                  return (
                    <div
                      key={p.id}
                      className="group p-5 rounded-2xl bg-zinc-900/60 hover:bg-zinc-900/90 border border-zinc-800/80 hover:border-emerald-500/40 shadow-lg transition flex flex-col justify-between space-y-4"
                    >
                      <div className="space-y-2">
                        <div className="flex items-start justify-between gap-2">
                          <div className="flex items-center gap-2 truncate">
                            <div className="w-8 h-8 rounded-lg bg-emerald-500/10 text-emerald-400 flex items-center justify-center font-bold text-xs shrink-0">
                              TEX
                            </div>
                            <div className="truncate">
                              <h3 className="font-bold text-sm text-zinc-100 truncate group-hover:text-emerald-300 transition">
                                {p.name}
                              </h3>
                              <div className="text-[11px] text-zinc-500 font-mono">{p.rootFile}</div>
                            </div>
                          </div>

                          <div className="flex items-center gap-1 opacity-80 group-hover:opacity-100">
                            <button
                              onClick={() => handleCompileAndExportPdf(p)}
                              className="p-1.5 rounded-md hover:bg-zinc-800 text-zinc-400 hover:text-emerald-400 transition"
                              title="Compile & Download Real PDF"
                            >
                              <FileText className="w-3.5 h-3.5 text-emerald-400" />
                            </button>
                            <button
                              onClick={() => handleExportZip(p)}
                              className="p-1.5 rounded-md hover:bg-zinc-800 text-zinc-400 hover:text-zinc-200 transition"
                              title="Export .ZIP"
                            >
                              <Download className="w-3.5 h-3.5" />
                            </button>
                            <button
                              onClick={() => handleDuplicateProject(p)}
                              className="p-1.5 rounded-md hover:bg-zinc-800 text-zinc-400 hover:text-zinc-200 transition"
                              title="Duplicate Project"
                            >
                              <Copy className="w-3.5 h-3.5" />
                            </button>
                            <button
                              onClick={() => handleDeleteProject(p.id)}
                              className="p-1.5 rounded-md hover:bg-zinc-800 text-zinc-400 hover:text-red-400 transition"
                              title="Delete Project"
                            >
                              <Trash2 className="w-3.5 h-3.5" />
                            </button>
                          </div>
                        </div>

                        <p className="text-xs text-zinc-400 line-clamp-3 leading-relaxed font-serif italic opacity-80">
                          &ldquo;{previewText.trim()}...&rdquo;
                        </p>
                      </div>

                      <div className="pt-3 border-t border-zinc-800/80 flex items-center justify-between text-xs">
                        <span className="text-[11px] text-zinc-500 font-mono">
                          {Object.keys(p.files).length} files
                        </span>

                        <button
                          onClick={() => handleOpenProject(p)}
                          className="px-3 py-1.5 rounded-xl bg-emerald-600/90 hover:bg-emerald-500 text-white font-semibold flex items-center gap-1.5 shadow-md shadow-emerald-500/20 transition"
                        >
                          <span>Open in Studio</span>
                          <ArrowRight className="w-3.5 h-3.5" />
                        </button>
                      </div>
                    </div>
                  );
                })}
              </div>
            </div>
          </div>
        </div>
      ) : (
        /* ========================================================
            VIEW 2: PEAK STUDIO IDE WORKBENCH (FULL EDITOR)
            ======================================================== */
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
                title="Explorer (Files)"
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
                title="Document Outline"
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
                title="Citation Firewall & Crossref"
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
                title="Math Symbol Picker"
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
                title="AI Co-Author"
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
                    <label className="p-1 hover:bg-zinc-800 text-zinc-400 hover:text-zinc-200 rounded transition cursor-pointer" title="Import Folder into Project">
                      <FolderPlus className="w-3.5 h-3.5 text-amber-400" />
                      <input
                        type="file"
                        multiple
                        {...({ webkitdirectory: '', directory: '' } as any)}
                        onChange={handleImportFolder}
                        className="hidden"
                      />
                    </label>
                    <label className="p-1 hover:bg-zinc-800 text-zinc-400 hover:text-zinc-200 rounded transition cursor-pointer" title="Import File(s)">
                      <FilePlus2 className="w-3.5 h-3.5 text-emerald-400" />
                      <input
                        type="file"
                        multiple
                        accept=".tex,.bib,.sty,.cls,.csv,.txt"
                        onChange={handleImportFiles}
                        className="hidden"
                      />
                    </label>
                  </div>
                )}
              </div>

              <div className="flex-1 overflow-y-auto p-2 space-y-3">
                {sidebarTab === 'explorer' && (
                  <div className="space-y-3">
                    <div className="space-y-1">
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

                    <div className="pt-2 border-t border-zinc-900 space-y-1">
                      <button
                        onClick={() => setDecompileModalOpen(true)}
                        className="w-full text-left p-2 rounded-lg bg-cyan-950/30 hover:bg-cyan-950/60 border border-cyan-500/30 text-xs text-cyan-300 font-semibold flex items-center gap-2 transition"
                      >
                        <FileSearch className="w-3.5 h-3.5 text-cyan-400" />
                        <span>Decompile PDF to LaTeX</span>
                      </button>
                    </div>
                  </div>
                )}

                {sidebarTab === 'outline' && (
                  <div className="space-y-1">
                    {compilation.outline.length === 0 ? (
                      <div className="text-xs text-zinc-500 py-4 text-center">No sections found.</div>
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

                {sidebarTab === 'citations' && (
                  <div className="space-y-3">
                    <div className="flex gap-1">
                      <input
                        value={crossrefQuery}
                        onChange={(e) => setCrossrefQuery(e.target.value)}
                        onKeyDown={(e) => e.key === 'Enter' && handleCrossrefSearch()}
                        placeholder="Search Crossref papers / DOI..."
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

                    <div className="space-y-2">
                      {crossrefResults.map((entry) => (
                        <div key={entry.key} className="p-2 rounded bg-zinc-900 border border-zinc-800 text-[11px] space-y-1">
                          <div className="font-semibold text-zinc-200 line-clamp-1">{entry.title}</div>
                          <div className="text-zinc-400 line-clamp-1">
                            {entry.author} ({entry.year})
                          </div>
                          <div className="flex items-center justify-between pt-1">
                            <code className="text-[10px] text-emerald-400 font-mono">\\cite&#123;{entry.key}&#125;</code>
                            <button
                              onClick={() => insertVerifiedCitation(entry)}
                              className="px-1.5 py-0.5 rounded bg-emerald-500/20 hover:bg-emerald-500/30 text-emerald-300 text-[10px] font-semibold"
                            >
                              + Add .bib
                            </button>
                          </div>
                        </div>
                      ))}
                    </div>
                  </div>
                )}

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

                {sidebarTab === 'ai' && (
                  <div className="flex flex-col h-full space-y-3">
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

                    <div className="space-y-1.5">
                      <button
                        onClick={() => handleSendAiPrompt('Generate publication-ready booktabs table')}
                        className="w-full text-left p-1.5 rounded bg-zinc-900 hover:bg-zinc-800 text-xs text-zinc-300 transition"
                      >
                        + Booktabs Results Table
                      </button>
                      <button
                        onClick={() => handleSendAiPrompt('Formulate formal convergence proof steps')}
                        className="w-full text-left p-1.5 rounded bg-zinc-900 hover:bg-zinc-800 text-xs text-zinc-300 transition"
                      >
                        + Formal Proof Steps
                      </button>
                      <button
                        onClick={() => handleSendAiPrompt('Convert equations to align environment')}
                        className="w-full text-left p-1.5 rounded bg-zinc-900 hover:bg-zinc-800 text-xs text-zinc-300 transition"
                      >
                        + Aligned Equations
                      </button>
                    </div>

                    <div className="space-y-1.5 pt-2">
                      <textarea
                        value={aiPromptInput}
                        onChange={(e) => setAiPromptInput(e.target.value)}
                        placeholder="Instruct VeriTeX AI to refactor, write proofs, or fix errors..."
                        className="w-full h-24 bg-zinc-900 border border-zinc-800 rounded p-2 text-xs text-zinc-200 focus:outline-none focus:border-emerald-500 resize-none font-sans"
                      />
                      <button
                        onClick={() => handleSendAiPrompt(aiPromptInput)}
                        disabled={aiGenerating || !aiPromptInput.trim()}
                        className="w-full py-1.5 rounded bg-emerald-600 hover:bg-emerald-500 disabled:opacity-50 text-white text-xs font-semibold flex items-center justify-center gap-1.5 transition"
                      >
                        {aiGenerating ? <RefreshCw className="w-3.5 h-3.5 animate-spin" /> : <Wand2 className="w-3.5 h-3.5" />}
                        Propose Surgical Diff
                      </button>
                    </div>
                  </div>
                )}

                {sidebarTab === 'collab' && (
                  <div className="space-y-3">
                    <div className="p-3 rounded-lg bg-zinc-900 border border-zinc-800 space-y-2">
                      <div className="text-xs font-semibold text-zinc-200">Decentralized P2P CRDT</div>
                      <button
                        onClick={handleCopyCollabLink}
                        className="w-full py-1.5 rounded bg-emerald-600 hover:bg-emerald-500 text-white text-xs font-semibold flex items-center justify-center gap-1.5 transition"
                      >
                        <Share2 className="w-3.5 h-3.5" />
                        {collabCopied ? 'Link Copied!' : 'Copy Invite Link'}
                      </button>
                    </div>
                  </div>
                )}
              </div>
            </aside>
          )}

          {/* Center Editor Pane */}
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

              <div className="flex items-center gap-2">
                <span suppressHydrationWarning className="text-[11px] font-mono text-zinc-500">
                  Words: {mounted ? compilation.wordCount : '...'}
                </span>
              </div>
            </div>

            {/* Breadcrumb Path Bar */}
            <div className="h-6 bg-[#090b10] border-b border-zinc-900/80 px-3 flex items-center text-[11px] text-zinc-500 gap-1.5 shrink-0 select-none">
              <button
                onClick={() => setCurrentView('dashboard')}
                className="hover:text-emerald-400 flex items-center gap-1 transition"
              >
                <ArrowLeft className="w-3 h-3" /> All Projects
              </button>
              <span>/</span>
              <span className="text-zinc-300 font-medium">{project.name}</span>
              <ChevronRight className="w-3 h-3" />
              <span className="text-zinc-400">{activeFile.path}</span>
              <ChevronRight className="w-3 h-3" />
              <span className="text-emerald-400 font-medium">§ {astSlice.activeSectionTitle}</span>
            </div>

            {/* Pending Surgical AI Diffs */}
            {pendingPatches.length > 0 && (
              <div className="bg-emerald-950/40 border-b border-emerald-500/30 p-2.5 space-y-2 max-h-48 overflow-y-auto shrink-0 z-10">
                <div className="text-xs font-semibold text-emerald-300 flex items-center justify-between">
                  <span>Pending Surgical AI Diffs ({pendingPatches.length})</span>
                  <span className="text-[11px] font-mono">{astSlice.savingsPercent}% Tokens Saved</span>
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
                        onClick={() => setPendingPatches((prev) => prev.filter((p) => p.id !== patch.id))}
                        className="px-2.5 py-1 rounded bg-zinc-800 hover:bg-zinc-700 text-zinc-300 text-xs flex items-center gap-1"
                      >
                        <X className="w-3 h-3" /> Reject
                      </button>
                    </div>
                  </div>
                ))}
              </div>
            )}

            {/* Synchronized Dual-Layer Editor Canvas */}
            <div className="flex-1 flex overflow-hidden relative">
              {/* Gutter with Line Numbers and Error Markers */}
              <div ref={gutterRef} className="editor-gutter">
                {activeFile.content.split('\n').map((_, idx) => {
                  const lineNum = idx + 1;
                  const hasError = compilation.diagnostics.some((d) => d.line === lineNum && d.severity === 'error');
                  const hasWarning = compilation.diagnostics.some((d) => d.line === lineNum && d.severity === 'warning');
                  const isActive = cursorLine === lineNum;
                  return (
                    <div
                      key={lineNum}
                      onClick={() => jumpToLine(lineNum)}
                      className={`editor-gutter-line cursor-pointer ${isActive ? 'active' : ''} ${
                        hasError ? 'has-error' : hasWarning ? 'has-warning' : ''
                      }`}
                    >
                      {hasError ? '● ' : ''}
                      {lineNum}
                    </div>
                  );
                })}
              </div>

              {/* Canvas Container */}
              <div className="editor-canvas-container flex-1">
                {/* Active Line Glow */}
                <div
                  className="editor-active-line-bg"
                  style={{
                    top: `${Math.max(0, (cursorLine - 1) * 22 + 16)}px`,
                    height: '22px',
                  }}
                />

                {/* Tokenized Syntax Highlighting Background Layer */}
                <div
                  ref={backdropRef}
                  className="editor-layer editor-backdrop"
                  dangerouslySetInnerHTML={{ __html: highlightedSource }}
                />

                {/* Foreground Interactive Textarea */}
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

                {/* Floating Math Inspector HUD */}
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
                        onClick={() => navigator.clipboard.writeText(activeMathFormula)}
                        className="hover:text-emerald-400 transition flex items-center gap-1 text-[11px]"
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
                  </div>
                )}

                {/* Inline Ctrl+K Prompt Bar */}
                {inlineAiOpen && (
                  <div className="absolute left-1/2 -translate-x-1/2 top-16 z-50 w-[540px] p-3 rounded-xl bg-zinc-950 border border-emerald-500/50 shadow-2xl font-sans space-y-2.5 animate-in fade-in zoom-in-95">
                    <div className="flex items-center justify-between">
                      <span className="flex items-center gap-1.5 text-xs font-bold text-emerald-400">
                        <Sparkles className="w-4 h-4" /> Inline AI Co-Author (Ctrl+K)
                      </span>
                      <button onClick={() => setInlineAiOpen(false)} className="p-1 hover:text-red-400 text-zinc-500">
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

          {/* Resizable Divider */}
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

          {/* Live Paper Preview Pane */}
          {workbenchMode !== 'focus' && (
            <section className="flex-1 flex flex-col bg-zinc-900 overflow-hidden relative">
              <div className="h-9 bg-zinc-950 border-b border-zinc-800 px-3 flex items-center justify-between text-xs text-zinc-400 shrink-0 select-none">
                <div className="flex items-center gap-2">
                  <span className="font-semibold text-zinc-300">Live Paper Preview</span>
                  <span className="text-[10px] text-emerald-400 font-mono px-1.5 py-0.5 rounded bg-emerald-500/10 border border-emerald-500/20">
                    SyncTeX Active
                  </span>
                </div>

                <div className="flex items-center gap-3">
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

                  <div className="flex items-center gap-1">
                    <button
                      onClick={() => setPaperZoom((prev) => Math.max(50, prev - 15))}
                      className="p-1 hover:bg-zinc-800 rounded text-zinc-400"
                    >
                      -
                    </button>
                    <span className="text-[11px] font-mono text-zinc-400 w-9 text-center">{paperZoom}%</span>
                    <button
                      onClick={() => setPaperZoom((prev) => Math.min(200, prev + 15))}
                      className="p-1 hover:bg-zinc-800 rounded text-zinc-400"
                    >
                      +
                    </button>
                  </div>
                </div>
              </div>

              <div
                ref={previewScrollRef}
                onDoubleClick={handlePreviewDoubleClick}
                className="flex-1 overflow-y-auto p-8 flex flex-col items-center gap-8 bg-zinc-900/90"
              >
                {/* Authentic Scientific Paper Sheet */}
                <div
                  id="veritex-printable-paper"
                  style={{
                    transform: `scale(${paperZoom / 100})`,
                    transformOrigin: 'top center',
                  }}
                  className={`veritex-paper-sheet paper-theme-${paperTheme} w-[820px] min-h-[1120px] p-16 shadow-2xl rounded-sm transition-all duration-150 select-text flex flex-col justify-between`}
                >
                  {/* Running Academic Header */}
                  <div className="flex justify-between items-center text-[10px] font-serif border-b border-zinc-200/60 dark:border-zinc-800 pb-2 mb-8 text-zinc-400 select-none">
                    <span className="font-semibold tracking-wide uppercase">{project.name.replace(/_/g, ' ')}</span>
                    <span className="italic font-mono">VeriTeX Typesetting System</span>
                  </div>

                  {/* Main Paper Content */}
                  <div
                    className="flex-1"
                    dangerouslySetInnerHTML={{ __html: compilation.html }}
                  />

                  {/* Running Academic Footer */}
                  <div className="flex justify-between items-center text-[10px] font-serif border-t border-zinc-200/60 dark:border-zinc-800 pt-3 mt-12 text-zinc-400 select-none">
                    <span className="italic font-mono">Typeset via KaTeX & VeriTeX AST Engine</span>
                    <span className="font-bold">Page 1</span>
                  </div>
                </div>
              </div>
            </section>
          )}
        </div>
      )}

      {/* Floating Real PDF Compilation Progress Toast */}
      {pdfCompiling && (
        <div className="fixed bottom-6 right-6 z-50 bg-zinc-900/95 border border-emerald-500/50 shadow-2xl rounded-2xl p-4 flex items-center gap-3.5 max-w-sm backdrop-blur-md animate-in slide-in-from-bottom-5">
          <div className="w-9 h-9 rounded-xl bg-emerald-500/10 text-emerald-400 flex items-center justify-center shrink-0">
            <RefreshCw className="w-4 h-4 animate-spin text-emerald-400" />
          </div>
          <div className="space-y-1 flex-1">
            <div className="flex items-center justify-between text-xs font-bold text-zinc-100">
              <span>Compiling Authentic PDF</span>
              <span className="font-mono text-emerald-400">{pdfCompiling.percent}%</span>
            </div>
            <p className="text-[11px] text-zinc-400 truncate">{pdfCompiling.stage}</p>
            <div className="w-full bg-zinc-800 rounded-full h-1 overflow-hidden mt-1">
              <div
                className="bg-emerald-500 h-1 transition-all duration-300"
                style={{ width: `${pdfCompiling.percent}%` }}
              />
            </div>
          </div>
        </div>
      )}

      {/* Floating Export Success Notification */}
      {exportNotice && (
        <div className="fixed bottom-6 right-6 z-50 bg-emerald-950/90 border border-emerald-500/50 text-emerald-200 shadow-2xl rounded-xl px-4 py-3 flex items-center gap-2.5 text-xs font-semibold backdrop-blur-md animate-in slide-in-from-bottom-5">
          <CheckCircle2 className="w-4 h-4 text-emerald-400" />
          <span>{exportNotice}</span>
        </div>
      )}

      {/* ========================================================
          MODAL 1: INTERACTIVE PDF-TO-LATEX CONVERTER STUDIO
          ======================================================== */}
      {decompileModalOpen && (
        <div
          onClick={() => setDecompileModalOpen(false)}
          className="fixed inset-0 z-50 bg-black/75 backdrop-blur-sm flex items-center justify-center p-4 animate-in fade-in"
        >
          <div
            onClick={(e) => e.stopPropagation()}
            className="w-[840px] max-h-[88vh] overflow-y-auto rounded-2xl bg-zinc-950 border border-zinc-800 shadow-2xl p-6 font-sans space-y-6 animate-in zoom-in-95"
          >
            {/* Header */}
            <div className="flex items-center justify-between border-b border-zinc-800 pb-3">
              <div className="flex items-center gap-2.5">
                <div className="w-8 h-8 rounded-lg bg-cyan-500/10 text-cyan-400 flex items-center justify-center">
                  <FileSearch className="w-4 h-4" />
                </div>
                <div>
                  <h3 className="text-base font-bold text-zinc-100">
                    PDF to LaTeX Decompiler Studio
                  </h3>
                  <p className="text-xs text-zinc-400">
                    4-Stage reverse engineering: extracts text, math equations, tables, and verifies authentic DOIs via Crossref.
                  </p>
                </div>
              </div>
              <button onClick={() => setDecompileModalOpen(false)} className="p-1 hover:text-red-400 text-zinc-500">
                <X className="w-4 h-4" />
              </button>
            </div>

            {/* Upload Box / Drag & Drop */}
            <div className="p-6 rounded-2xl border-2 border-dashed border-zinc-800 hover:border-cyan-500/60 bg-zinc-900/40 text-center space-y-3 transition">
              <UploadCloud className="w-10 h-10 text-cyan-400 mx-auto opacity-80" />
              <div className="space-y-1">
                <div className="text-sm font-semibold text-zinc-200">
                  Upload PDF Research Paper or Book Chapter
                </div>
                <div className="text-xs text-zinc-500">
                  Supports multi-page articles with equations, figures, and bibliographies.
                </div>
              </div>

              <label className="inline-flex items-center gap-2 px-4 py-2 rounded-xl bg-cyan-600 hover:bg-cyan-500 text-white text-xs font-bold cursor-pointer shadow-lg shadow-cyan-500/20 transition">
                <span>Select PDF File</span>
                <input
                  type="file"
                  accept=".pdf"
                  onChange={(e) => {
                    const f = e.target.files?.[0];
                    if (f) handleStartPdfDecompilation(f);
                  }}
                  className="hidden"
                />
              </label>
            </div>

            {/* 4-Stage Stepper Status & Progress Bar */}
            {(decompileLoading || decompileOutput) && (
              <div className="p-4 rounded-xl bg-zinc-900/70 border border-zinc-800 space-y-3">
                <div className="flex items-center justify-between text-xs">
                  <span className="font-semibold text-cyan-300 flex items-center gap-1.5">
                    {decompileLoading && <RefreshCw className="w-3.5 h-3.5 animate-spin text-cyan-400" />}
                    {decompileStage}
                  </span>
                  <span className="font-mono text-cyan-400">{decompilePercent}%</span>
                </div>

                {/* Progress track */}
                <div className="w-full h-2 rounded-full bg-zinc-800 overflow-hidden">
                  <div
                    className="h-full bg-gradient-to-r from-cyan-500 to-emerald-400 transition-all duration-300"
                    style={{ width: `${decompilePercent}%` }}
                  />
                </div>

                {/* 4 Pipeline Stages Badges */}
                <div className="grid grid-cols-4 gap-2 text-[10px] pt-1">
                  <div className={`p-2 rounded-lg border text-center ${decompilePercent >= 20 ? 'bg-cyan-950/40 border-cyan-500/40 text-cyan-300' : 'bg-zinc-950/60 border-zinc-800 text-zinc-600'}`}>
                    1. Layout & Geometry
                  </div>
                  <div className={`p-2 rounded-lg border text-center ${decompilePercent >= 50 ? 'bg-cyan-950/40 border-cyan-500/40 text-cyan-300' : 'bg-zinc-950/60 border-zinc-800 text-zinc-600'}`}>
                    2. Math & Sections
                  </div>
                  <div className={`p-2 rounded-lg border text-center ${decompilePercent >= 75 ? 'bg-cyan-950/40 border-cyan-500/40 text-cyan-300' : 'bg-zinc-950/60 border-zinc-800 text-zinc-600'}`}>
                    3. Crossref Citations
                  </div>
                  <div className={`p-2 rounded-lg border text-center ${decompilePercent >= 95 ? 'bg-emerald-950/40 border-emerald-500/40 text-emerald-300' : 'bg-zinc-950/60 border-zinc-800 text-zinc-600'}`}>
                    4. Syntax Normalization
                  </div>
                </div>
              </div>
            )}

            {/* Generated Code Preview Tabs */}
            {decompileOutput && (
              <div className="space-y-3 pt-2 border-t border-zinc-800">
                <div className="flex items-center justify-between">
                  <div className="flex items-center gap-1 overflow-x-auto">
                    {Object.keys(decompileOutput.files).map((f) => (
                      <button
                        key={f}
                        onClick={() => setDecompileActiveTab(f)}
                        className={`px-3 py-1 rounded-lg text-xs font-mono transition ${
                          decompileActiveTab === f
                            ? 'bg-zinc-800 text-cyan-300 border border-cyan-500/30'
                            : 'text-zinc-500 hover:text-zinc-300'
                        }`}
                      >
                        {f}
                      </button>
                    ))}
                  </div>

                  <div className="text-[11px] text-zinc-400 font-mono">
                    Pages: {decompileOutput.extractedPageCount} • Citations: {decompileOutput.verifiedCitationsCount}
                  </div>
                </div>

                <div className="h-60 overflow-y-auto rounded-xl bg-zinc-950 border border-zinc-800 p-3 font-mono text-xs text-zinc-300 leading-relaxed whitespace-pre-wrap selection:bg-cyan-500/30">
                  {decompileOutput.files[decompileActiveTab]?.content}
                </div>

                <div className="flex items-center justify-between pt-2">
                  <button
                    onClick={() => {
                      const code = decompileOutput.files[decompileActiveTab]?.content || '';
                      navigator.clipboard.writeText(code);
                      alert('Copied to clipboard!');
                    }}
                    className="px-3 py-1.5 rounded-lg bg-zinc-800 hover:bg-zinc-700 text-zinc-200 text-xs font-semibold flex items-center gap-1.5 transition"
                  >
                    <Copy className="w-3.5 h-3.5" /> Copy Code
                  </button>

                  <div className="flex items-center gap-2">
                    <button
                      onClick={async () => {
                        const zip = new JSZip();
                        Object.values(decompileOutput.files).forEach((f) => zip.file(f.path, f.content));
                        const blob = await zip.generateAsync({ type: 'blob' });
                        const url = URL.createObjectURL(blob);
                        const a = document.createElement('a');
                        a.href = url;
                        a.download = 'decompiled-latex-project.zip';
                        a.click();
                        URL.revokeObjectURL(url);
                      }}
                      className="px-3.5 py-1.5 rounded-xl bg-zinc-800 hover:bg-zinc-700 text-zinc-200 text-xs font-bold flex items-center gap-1.5 transition"
                    >
                      <Download className="w-3.5 h-3.5" /> Download .ZIP
                    </button>

                    <button
                      onClick={handleCreateProjectFromDecompile}
                      className="px-4 py-1.5 rounded-xl bg-emerald-600 hover:bg-emerald-500 text-white text-xs font-bold flex items-center gap-1.5 shadow-lg shadow-emerald-500/20 transition"
                    >
                      <span>Open as Project in Studio</span>
                      <ArrowRight className="w-3.5 h-3.5" />
                    </button>
                  </div>
                </div>
              </div>
            )}
          </div>
        </div>
      )}

      {/* ========================================================
          MODAL 2: GLOBAL COMMAND PALETTE (CTRL+P)
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
                placeholder="Type a command, search projects or files..."
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
                      {item.shortcut && <kbd className="font-mono text-[10px] text-zinc-500">{item.shortcut}</kbd>}
                    </div>
                  );
                })
              )}
            </div>
          </div>
        </div>
      )}

      {/* ========================================================
          MODAL 3: BYOM AI GATEWAY MODAL
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
                <p className="text-xs text-zinc-400 mt-0.5">Keys remain strictly in your browser. Zero telemetry.</p>
              </div>
              <button onClick={() => setByomModalOpen(false)} className="p-1 hover:text-red-400 text-zinc-500">
                <X className="w-4 h-4" />
              </button>
            </div>

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

              <div className="pt-2 flex items-center justify-between">
                <button
                  onClick={async () => {
                    setHandshakeStatus({ testing: true, message: 'Testing handshake...' });
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
          MODAL 4: P2P COLLABORATION MODAL
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
              <button onClick={() => setCollabModalOpen(false)} className="p-1 hover:text-red-400 text-zinc-500">
                <X className="w-4 h-4" />
              </button>
            </div>

            <div className="space-y-3">
              <p className="text-xs text-zinc-400">
                Share this room link with your research team. Edits sync in real time using P2P WebRTC mesh connections.
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
                  <span className="text-emerald-400 font-medium">You (Host)</span>
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

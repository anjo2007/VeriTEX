export type FileType = 'tex' | 'bib' | 'sty' | 'cls' | 'csv' | 'image';

export interface VirtualFile {
  path: string;
  content: string;
  type: FileType;
  updatedAt: number;
}

export interface VeriTeXProject {
  id: string;
  name: string;
  rootFile: string;
  createdAt: number;
  updatedAt: number;
  files: Record<string, VirtualFile>;
}

export type WorkbenchMode = 'focus' | 'studio' | 'diff';
export type PaperTheme = 'white' | 'ivory' | 'night';
export type SidebarTab = 'explorer' | 'outline' | 'citations' | 'symbols' | 'ai' | 'collab';

export interface BibEntry {
  key: string;
  type: string;
  title: string;
  author: string;
  year: string;
  journal?: string;
  doi?: string;
  verifiedByCrossref?: boolean;
}

export interface CompilerDiagnostic {
  line: number;
  file: string;
  severity: 'error' | 'warning' | 'info';
  message: string;
  snippet: string;
}

export interface CompilationResult {
  html: string;
  diagnostics: CompilerDiagnostic[];
  labels: Record<string, { number: string; type: string; line: number; file: string }>;
  citations: Record<string, number>;
  bibEntries: Record<string, BibEntry>;
  macros: Record<string, { args: number; body: string }>;
  compileTimeMs: number;
  outline: Array<{ level: number; title: string; line: number; number: string }>;
  wordCount: number;
}

export interface AutocompleteItem {
  label: string;
  insertText: string;
  detail: string;
  documentation?: string;
  kind: 'citation' | 'reference' | 'macro' | 'environment' | 'math' | 'command' | 'snippet';
}

export type AIProviderId = 'gemini' | 'openai' | 'anthropic' | 'deepseek' | 'openrouter' | 'ollama';

export interface AIProviderConfig {
  provider: AIProviderId;
  modelId: string;
  apiKey: string;
  baseUrl: string;
  temperature: number;
  useAstSlicer?: boolean;
  verifyCitations?: boolean;
}

export interface ASTContextSlice {
  activeSectionTitle: string;
  activeBlockContent: string;
  startLine: number;
  endLine: number;
  extractedMacros: string[];
  referencedLabels: string[];
  relevantBibEntries: string[];
  fullDocumentTokens: number;
  slicedTokens: number;
  savingsPercent: number;
}

export interface SurgicalDiffPatch {
  id: string;
  searchBlock: string;
  replaceBlock: string;
  explanation: string;
  status: 'pending' | 'accepted' | 'rejected';
}

export interface CollaboratorPeer {
  clientId: number;
  name: string;
  color: string;
  activeFile: string;
  cursorLine: number;
}

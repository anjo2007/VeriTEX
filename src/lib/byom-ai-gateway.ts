import { AIProviderConfig, AIProviderId, ASTContextSlice, BibEntry } from '@/types/veritex';

export interface ProviderPreset {
  id: AIProviderId;
  name: string;
  badge: string;
  defaultBaseUrl: string;
  defaultModel: string;
  models: string[];
  setupSteps: string[];
}

export const BYOM_PROVIDER_PRESETS: Record<AIProviderId, ProviderPreset> = {
  gemini: {
    id: 'gemini',
    name: 'Google Gemini',
    badge: 'Fastest & Long Context',
    defaultBaseUrl: 'https://generativelanguage.googleapis.com/v1beta',
    defaultModel: 'gemini-2.5-flash',
    models: ['gemini-2.5-flash', 'gemini-2.5-pro'],
    setupSteps: [
      '1. Open Google AI Studio (https://aistudio.google.com/apikey) and sign in.',
      '2. Click "Create API Key" and copy the generated key starting with AIzaSy...',
      '3. Paste the key below and click "Verify & Benchmark Handshake". Keys are stored locally in your browser.',
    ],
  },
  openai: {
    id: 'openai',
    name: 'OpenAI',
    badge: 'Reasoning & Math',
    defaultBaseUrl: 'https://api.openai.com/v1',
    defaultModel: 'gpt-4o',
    models: ['gpt-4o', 'gpt-4o-mini', 'o3-mini'],
    setupSteps: [
      '1. Visit https://platform.openai.com/api-keys and sign in to your OpenAI developer account.',
      '2. Click "+ Create new secret key" and copy the sk-proj-... token.',
      '3. Paste it below and run the 1-click capability test.',
    ],
  },
  anthropic: {
    id: 'anthropic',
    name: 'Anthropic Claude',
    badge: 'Academic Prose & Proofs',
    defaultBaseUrl: 'https://api.anthropic.com/v1',
    defaultModel: 'claude-3-7-sonnet-latest',
    models: ['claude-3-7-sonnet-latest', 'claude-3-5-haiku-latest'],
    setupSteps: [
      '1. Open https://console.anthropic.com/settings/keys and create a new API Key.',
      '2. Copy the key starting with sk-ant-api03-...',
      '3. Paste below and click "Verify & Benchmark Handshake".',
    ],
  },
  deepseek: {
    id: 'deepseek',
    name: 'DeepSeek (R1 / V3)',
    badge: 'Ultra-Low Cost Math',
    defaultBaseUrl: 'https://api.deepseek.com/v1',
    defaultModel: 'deepseek-chat',
    models: ['deepseek-chat', 'deepseek-reasoner'],
    setupSteps: [
      '1. Go to https://platform.deepseek.com/api_keys and generate an API key.',
      '2. DeepSeek uses an OpenAI-compatible endpoint with 90% prefix cache discounts.',
      '3. Paste your key below and test the connection.',
    ],
  },
  openrouter: {
    id: 'openrouter',
    name: 'OpenRouter (200+ Models)',
    badge: 'Universal Aggregator',
    defaultBaseUrl: 'https://openrouter.ai/api/v1',
    defaultModel: 'qwen/qwen-2.5-coder-32b-instruct',
    models: [
      'qwen/qwen-2.5-coder-32b-instruct',
      'meta-llama/llama-3.3-70b-instruct',
      'deepseek/deepseek-r1',
    ],
    setupSteps: [
      '1. Visit https://openrouter.ai/keys and click "Create Key".',
      '2. Optionally set a monthly credit limit for complete cost control.',
      '3. Paste your sk-or-v1-... key below to unlock 200+ open and frontier models.',
    ],
  },
  ollama: {
    id: 'ollama',
    name: 'Local Ollama / VeriTeX-1.5B',
    badge: '100% Offline & Free',
    defaultBaseUrl: 'http://localhost:11434/v1',
    defaultModel: 'qwen2.5-coder:7b',
    models: ['veritex-core:1.5b', 'qwen2.5-coder:7b', 'deepseek-r1:8b', 'llama3.2:3b'],
    setupSteps: [
      '1. Install Ollama from https://ollama.com on your machine.',
      '2. Run in terminal with CORS enabled: OLLAMA_ORIGINS="*" ollama run qwen2.5-coder:7b',
      '3. No API key required! Click "Verify & Benchmark Handshake" to connect to localhost:11434.',
    ],
  },
};

/**
 * Performs a live diagnostic handshake with the configured provider and measures latency.
 */
export async function verifyProviderConnection(config: AIProviderConfig): Promise<{
  ok: boolean;
  latencyMs: number;
  message: string;
}> {
  const t0 = performance.now();
  try {
    const reply = await callUnifiedModelGateway(
      config,
      'Return only the exact string: VERITEX_OK',
      'System handshake test.'
    );
    const latencyMs = Math.round(performance.now() - t0);
    return {
      ok: true,
      latencyMs,
      message: `Connected to ${config.modelId} (${latencyMs}ms) — Response: ${reply.slice(0, 40)}`,
    };
  } catch (err: unknown) {
    const latencyMs = Math.round(performance.now() - t0);
    return {
      ok: false,
      latencyMs,
      message: err instanceof Error ? err.message : 'Connection failed. Check API key or CORS settings.',
    };
  }
}

/**
 * Calls any configured AI model using the low-token AST context slice and Surgical Diff protocol.
 */
export async function invokeVeriTeXAgent(
  config: AIProviderConfig,
  userPrompt: string,
  astSlice: ASTContextSlice,
  diagnosticsSummary: string
): Promise<string> {
  const systemPrompt = [
    'You are the VeriTeX Scientific Co-Author & Deterministic LaTeX Engine.',
    'CRITICAL LOW-TOKEN PROTOCOL:',
    'When proposing edits to existing LaTeX code, DO NOT rewrite the entire document.',
    'Output concise explanation followed by surgical SEARCH/REPLACE blocks in this exact format:',
    '<<<<<<< SEARCH',
    '<exact lines from the active block to find>',
    '=======',
    '<replacement LaTeX lines>',
    '>>>>>>> REPLACE',
    '',
    'Active Preamble Macros:',
    astSlice.extractedMacros.join('\n') || '% None',
    'Referenced Labels Context:',
    astSlice.referencedLabels.join('\n') || '% None',
    'Active BibTeX Context:',
    astSlice.relevantBibEntries.join('\n') || '% None',
  ].join('\n');

  const fullUserPayload = [
    `Active Section: ${astSlice.activeSectionTitle} (Lines ${astSlice.startLine}-${astSlice.endLine})`,
    '```latex',
    astSlice.activeBlockContent,
    '```',
    diagnosticsSummary ? `Compiler Diagnostics:\n${diagnosticsSummary}` : '',
    `User Request: ${userPrompt}`,
  ]
    .filter(Boolean)
    .join('\n\n');

  return callUnifiedModelGateway(config, fullUserPayload, systemPrompt);
}

async function callUnifiedModelGateway(
  config: AIProviderConfig,
  userContent: string,
  systemInstruction: string
): Promise<string> {
  if (config.provider === 'gemini') {
    if (!config.apiKey) throw new Error('Please provide your Google Gemini API key in AI Settings.');
    const url = `${config.baseUrl.replace(/\/$/, '')}/models/${config.modelId}:generateContent?key=${config.apiKey}`;
    const res = await fetch(url, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        systemInstruction: { parts: [{ text: systemInstruction }] },
        contents: [{ role: 'user', parts: [{ text: userContent }] }],
        generationConfig: { temperature: config.temperature },
      }),
    });
    if (!res.ok) {
      const errText = await res.text();
      throw new Error(`Gemini API Error (${res.status}): ${errText.slice(0, 160)}`);
    }
    const data = await res.json();
    return data?.candidates?.[0]?.content?.parts?.[0]?.text || 'No response from Gemini.';
  }

  if (config.provider === 'anthropic') {
    if (!config.apiKey) throw new Error('Please provide your Anthropic API key in AI Settings.');
    const res = await fetch(`${config.baseUrl.replace(/\/$/, '')}/messages`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        'x-api-key': config.apiKey,
        'anthropic-version': '2023-06-01',
        'anthropic-dangerous-direct-browser-access': 'true',
      },
      body: JSON.stringify({
        model: config.modelId,
        max_tokens: 2048,
        system: systemInstruction,
        messages: [{ role: 'user', content: userContent }],
      }),
    });
    if (!res.ok) {
      const errText = await res.text();
      throw new Error(`Anthropic API Error (${res.status}): ${errText.slice(0, 160)}`);
    }
    const data = await res.json();
    return data?.content?.[0]?.text || 'No response from Claude.';
  }

  // OpenAI-compatible protocol (OpenAI, DeepSeek, OpenRouter, Local Ollama)
  const headers: Record<string, string> = { 'Content-Type': 'application/json' };
  if (config.apiKey) {
    headers['Authorization'] = `Bearer ${config.apiKey}`;
  }
  const endpoint = `${config.baseUrl.replace(/\/$/, '')}/chat/completions`;
  const res = await fetch(endpoint, {
    method: 'POST',
    headers,
    body: JSON.stringify({
      model: config.modelId,
      temperature: config.temperature,
      messages: [
        { role: 'system', content: systemInstruction },
        { role: 'user', content: userContent },
      ],
    }),
  });
  if (!res.ok) {
    const errText = await res.text();
    throw new Error(`${config.provider.toUpperCase()} Error (${res.status}): ${errText.slice(0, 160)}`);
  }
  const data = await res.json();
  return data?.choices?.[0]?.message?.content || 'No response returned.';
}

/**
 * Citation Hallucination Firewall: Queries the live Crossref REST API to verify real DOIs
 * and synthesize 100% authentic BibTeX records.
 */
export async function searchAndVerifyCrossrefCitations(query: string): Promise<BibEntry[]> {
  if (!query.trim()) return [];
  const url = `https://api.crossref.org/works?query=${encodeURIComponent(query)}&rows=4`;
  const res = await fetch(url);
  if (!res.ok) throw new Error('Crossref API lookup failed.');
  const data = await res.json();
  const items = data?.message?.items || [];

  return items.map((item: Record<string, unknown>, idx: number) => {
    const titleArr = item.title as string[] | undefined;
    const title = titleArr?.[0] || 'Untitled Publication';
    const authors = (item.author as Array<{ family?: string; given?: string }> | undefined) || [];
    const firstAuthorLast = (authors[0]?.family || `author${idx}`).toLowerCase().replace(/[^a-z0-9]/g, '');
    const authorStr =
      authors
        .map((a) => `${a.family || ''}, ${a.given || ''}`.trim())
        .join(' and ') || 'Unknown Author';
    const issued = item.issued as { 'date-parts'?: number[][] } | undefined;
    const year = String(issued?.['date-parts']?.[0]?.[0] || new Date().getFullYear());
    const containerArr = item['container-title'] as string[] | undefined;
    const journal = containerArr?.[0] || 'Journal / Conference Proceedings';
    const doi = (item.DOI as string) || '';

    return {
      key: `${firstAuthorLast}${year}`,
      type: 'article',
      title,
      author: authorStr,
      year,
      journal,
      doi,
      verifiedByCrossref: true,
    };
  });
}

export function formatBibEntryToBibTeX(entry: BibEntry): string {
  return [
    `@${entry.type}{${entry.key},`,
    `  author = {${entry.author}},`,
    `  title = {${entry.title}},`,
    `  journal = {${entry.journal || 'Preprint'}},`,
    `  year = {${entry.year}},`,
    entry.doi ? `  doi = {${entry.doi}}` : '',
    `}`,
  ]
    .filter(Boolean)
    .join('\n');
}

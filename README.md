# VeriTeX ⚡

> **Open-Source High-Speed Scientific Workbench & Low-Token BYOM LaTeX IDE**

[![Next.js](https://img.shields.io/badge/Next.js-15-black?style=flat&logo=next.js)](https://nextjs.org/)
[![React](https://img.shields.io/badge/React-19-blue?style=flat&logo=react)](https://react.dev/)
[![TypeScript](https://img.shields.io/badge/TypeScript-5.6-blue?style=flat&logo=typescript)](https://www.typescriptlang.org/)
[![Tailwind CSS](https://img.shields.io/badge/Tailwind-3.4-38B2AC?style=flat&logo=tailwind-css)](https://tailwindcss.com/)
[![License](https://img.shields.io/badge/License-Apache_2.0-green.svg)](LICENSE)

VeriTeX is a modern, high-performance LaTeX editing workbench and scientific publishing suite designed for researchers, academics, and engineers. It features real-time client-side compilation, AST-aware context slicing to minimize LLM token costs, surgical diff patch application, Bring-Your-Own-Model (BYOM) AI integration, and decentralized P2P collaborative editing.

---

## ✨ Key Features

- ⚡ **Real-Time Client-Side LaTeX Compiler**:
  - Incremental compilation powered by KaTeX and regex-based AST parsing.
  - Multi-file workspace support (`\input{}`, `\include{}`).
  - Deep verification: theorem numbering, nested tables, bibtex citation resolution, and multi-line environment validation.
  - Interactive diagnostics with exact error line navigation.

- 🧠 **BYOM (Bring-Your-Own-Model) AI Gateway**:
  - Native support for Google Gemini, OpenAI, Anthropic Claude, DeepSeek, and Local Ollama / vLLM instances.
  - Zero lock-in: API keys remain client-side in encrypted local storage.
  - Live Crossref citation lookup and verification.

- ✂️ **AST-Aware Context Slicer**:
  - Automatically isolates only the relevant section, imported macros, and cited bibliography entries.
  - Reduces LLM context token consumption by up to **80–90%** compared to full-document injection.

- 🩹 **Surgical Patch Engine**:
  - High-precision search/replace patch parser (`<<<< SEARCH ==== REPLACE >>>>`).
  - Safely handles LaTeX macro expansion and regex escape tokens (`$&`, `$1`, `\frac`, etc.) without text corruption.

- 👥 **Decentralized P2P Collaboration**:
  - Real-time collaborative editing using **Yjs CRDTs** over WebRTC signaling.
  - Offline-first persistence via IndexedDB with automatic peer awareness indicators.

- 📄 **Reverse PDF-to-LaTeX Decompiler**:
  - Ingest PDF files, extract structure and text using PDF.js, and convert into modular editable LaTeX documents.

- 🚀 **Specialized 1.5B Parameter Model Training Pipeline**:
  - Includes `training/train_veritex_core_1_5b.py` with QLoRA instruction tuning and synthetic dataset generation.
  - One-click export to GGUF for local Ollama and WebGPU deployment.

---

## 🛠️ Project Structure

```
VeriTEX/
├── src/
│   ├── app/                     # Next.js App Router (UI & Workbench Layout)
│   │   ├── globals.css          # Styling & KaTeX custom themes
│   │   ├── layout.tsx           # Root metadata & font configuration
│   │   └── page.tsx             # Main VeriTeX Studio IDE interface
│   ├── lib/                     # Core Engines & Libraries
│   │   ├── byom-ai-gateway.ts   # Multi-provider AI Gateway (Gemini, OpenAI, Claude, DeepSeek, Ollama)
│   │   ├── collab-crdt.ts       # Yjs CRDT & WebRTC collaboration provider
│   │   ├── latex-ast-slicer.ts  # AST context slicer & surgical patch engine
│   │   ├── latex-compiler.ts    # Incremental LaTeX compiler & KaTeX renderer
│   │   └── pdf-decompiler.ts    # PDF to LaTeX extraction pipeline
│   └── types/
│       └── veritex.ts           # TypeScript interfaces and domain schemas
├── training/                    # Custom Model Training Pipeline
│   ├── requirements.txt         # PyTorch, Unsloth, HuggingFace dependencies
│   └── train_veritex_core_1_5b.py # Fine-tuning script for VeriTeX-Core-1.5B
├── public/                      # Static assets & icons
├── run_tests.mjs                # Verification test suite
├── Dockerfile                   # Production container definition
├── docker-compose.yml           # Multi-container orchestration
├── next.config.mjs              # Next.js configuration
├── package.json                 # Node dependencies and scripts
├── tailwind.config.ts           # Tailwind CSS theme configuration
└── tsconfig.json                # TypeScript compiler settings
```

---

## 🚀 Getting Started

### Prerequisites

- **Node.js**: v18.17+ or v20+
- **npm** or **pnpm** / **yarn**
- *(Optional)* **Docker** & **Docker Compose**

### Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/anjo2007/VeriTEX.git
   cd VeriTEX
   ```

2. **Install dependencies**:
   ```bash
   npm install
   ```

3. **Start the development server**:
   ```bash
   npm run dev
   ```
   Open [http://localhost:3000](http://localhost:3000) in your browser.

---

## 🧪 Running Tests

VeriTeX includes a comprehensive verification suite validating multi-file compilation, citation resolution, nested environment parsing, surgical patching, and macro extraction:

```bash
npm test
```

---

## 🐳 Docker Deployment

To build and run VeriTeX using Docker:

```bash
docker compose up --build -d
```

The application will be accessible at `http://localhost:3000`.

---

## 🤖 Training VeriTeX-Core-1.5B

A training pipeline is provided to fine-tune `Qwen/Qwen2.5-Coder-1.5B-Instruct` on surgical LaTeX patches:

```bash
cd training
pip install -r requirements.txt
python train_veritex_core_1_5b.py
```

---

## 📄 License

This project is licensed under the Apache-2.0 License. See the [LICENSE](LICENSE) file for details.

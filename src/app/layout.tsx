import type { Metadata } from 'next';
import 'katex/dist/katex.min.css';
import './globals.css';

export const metadata: Metadata = {
  title: 'VeriTeX Studio — High-Speed Open-Source Scientific Workbench',
  description:
    'Deterministic LaTeX IDE with <10ms live AST compilation, low-token BYOM AI Agent, unlimited CRDT collaboration, and 4-stage PDF decompiler.',
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en" className="dark">
      <body className="bg-zinc-950 text-zinc-100 antialiased selection:bg-emerald-500/30">
        {children}
      </body>
    </html>
  );
}

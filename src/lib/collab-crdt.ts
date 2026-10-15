import * as Y from 'yjs';
import { WebrtcProvider } from 'y-webrtc';
import { IndexeddbPersistence } from 'y-indexeddb';
import { CollaboratorPeer } from '@/types/veritex';

const PEER_COLORS = ['#10b981', '#3b82f6', '#f59e0b', '#ec4899', '#8b5cf6', '#06b6d4'];

export class VeriTeXCollabSession {
  public ydoc: Y.Doc;
  public filesMap: Y.Map<string>;
  private webrtcProvider: WebrtcProvider | null = null;
  private idbPersistence: IndexeddbPersistence | null = null;
  private localPeerName: string;
  private localPeerColor: string;

  constructor(roomId: string, userName?: string) {
    this.ydoc = new Y.Doc();
    this.filesMap = this.ydoc.getMap<string>('veritex-files');
    this.localPeerName = userName || `Researcher-${Math.floor(100 + Math.random() * 899)}`;
    this.localPeerColor = PEER_COLORS[Math.floor(Math.random() * PEER_COLORS.length)];

    if (typeof window !== 'undefined') {
      this.idbPersistence = new IndexeddbPersistence(`veritex-idb-${roomId}`, this.ydoc);
      try {
        this.webrtcProvider = new WebrtcProvider(`veritex-room-${roomId}`, this.ydoc, {
          signaling: ['wss://signaling.yjs.dev'],
        });
        this.updatePresence('main.tex', 1);
      } catch {
        // Gracefully operate in local-first CRDT mode if WebRTC signaling is unreachable
      }
    }
  }

  public updatePresence(activeFile: string, cursorLine: number): void {
    if (!this.webrtcProvider) return;
    this.webrtcProvider.awareness.setLocalStateField('user', {
      name: this.localPeerName,
      color: this.localPeerColor,
      activeFile,
      cursorLine,
    });
  }

  public onPeersChange(callback: (peers: CollaboratorPeer[]) => void): void {
    if (!this.webrtcProvider) return;
    this.webrtcProvider.awareness.on('change', () => {
      const states = this.webrtcProvider?.awareness.getStates();
      if (!states) return;
      const peers: CollaboratorPeer[] = [];
      states.forEach((state: Record<string, unknown>, clientId: number) => {
        if (state.user) {
          const u = state.user as {
            name: string;
            color: string;
            activeFile?: string;
            cursorLine?: number;
          };
          peers.push({
            clientId,
            name: u.name,
            color: u.color,
            activeFile: u.activeFile || 'main.tex',
            cursorLine: u.cursorLine || 1,
          });
        }
      });
      callback(peers);
    });
  }

  public destroy(): void {
    this.webrtcProvider?.destroy();
    this.idbPersistence?.destroy();
    this.ydoc.destroy();
  }
}

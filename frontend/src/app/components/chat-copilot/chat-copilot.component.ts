import { Component, OnInit } from '@angular/core';
import { CommonModule } from '@angular/common';
import { FormsModule } from '@angular/forms';
import { ApiService } from '../../services/api.service';

@Component({
  selector: 'app-chat-copilot',
  standalone: true,
  imports: [CommonModule, FormsModule],
  template: `
    <!-- Floating chat button -->
    <button class="chat-fab" (click)="toggleChat()" [class.active]="isOpen">
      <svg *ngIf="!isOpen" xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="white">
        <path d="M20 2H4c-1.1 0-2 .9-2 2v18l4-4h14c1.1 0 2-.9 2-2V4c0-1.1-.9-2-2-2zm0 14H5.17L4 17.17V4h16v12zM7 9h2v2H7zm4 0h2v2h-2zm4 0h2v2h-2z"/>
      </svg>
      <span *ngIf="isOpen">&times;</span>
    </button>

    <!-- Chat panel -->
    <div class="chat-panel" *ngIf="isOpen">
      <div class="chat-header">
        <div class="chat-header-left">
          <span class="ai-icon">
            <svg xmlns="http://www.w3.org/2000/svg" width="18" height="18" viewBox="0 0 24 24" fill="white">
              <path d="M20 2H4c-1.1 0-2 .9-2 2v18l4-4h14c1.1 0 2-.9 2-2V4c0-1.1-.9-2-2-2zm0 14H5.17L4 17.17V4h16v12z"/>
            </svg>
          </span>
          <div>
            <strong>CorroSight Copilot</strong>
            <small>{{ aiConfigured ? 'Grok 4.1 Fast' : 'Fallback mode' }}</small>
          </div>
        </div>
        <button class="summarize-btn" (click)="summarizeResults()" [disabled]="isLoading" title="Summarize all results">Summarize</button>
      </div>

      <!-- API key banner -->
      <div class="api-key-banner" *ngIf="!aiConfigured && !keyDismissed">
        <div class="api-key-row">
          <input type="password" [(ngModel)]="apiKeyInput" placeholder="Enter xAI API key (starts with xai-)..." class="api-key-input" (keydown.enter)="connectApiKey()" [disabled]="isConnecting" />
          <button class="connect-btn" (click)="connectApiKey()" [disabled]="!apiKeyInput.trim() || isConnecting">{{ isConnecting ? '...' : 'Connect' }}</button>
          <button class="dismiss-btn" (click)="keyDismissed = true">&times;</button>
        </div>
        <small class="api-key-error" *ngIf="apiKeyError">{{ apiKeyError }}</small>
        <small class="api-key-hint" *ngIf="!apiKeyError">Enter your xAI API key for AI-powered responses. Get one at console.x.ai</small>
      </div>

      <div class="connected-banner" *ngIf="showConnectedBanner">
        Connected to Grok 4.1 Fast
      </div>

      <div class="chat-messages" #messagesContainer>
        <!-- Welcome message -->
        <div class="msg assistant" *ngIf="messages.length === 0">
          <div class="msg-bubble">
            I'm your CorroSight AI assistant. I have full access to the analysis results from your ILI inspection runs. Ask me anything!
          </div>
        </div>

        <!-- Suggested questions -->
        <div class="suggestions" *ngIf="messages.length === 0">
          <button *ngFor="let q of suggestedQuestions" (click)="sendMessage(q)" class="suggestion-chip">{{q}}</button>
        </div>

        <div *ngFor="let msg of messages" class="msg" [class.user]="msg.role === 'user'" [class.assistant]="msg.role === 'assistant'">
          <div class="msg-bubble" [innerHTML]="formatMessage(msg.content)"></div>
        </div>

        <div class="msg assistant" *ngIf="isLoading">
          <div class="msg-bubble typing">
            <span class="dot"></span><span class="dot"></span><span class="dot"></span>
          </div>
        </div>
      </div>

      <div class="chat-input">
        <input type="text" [(ngModel)]="userInput" (keydown.enter)="sendMessage()" placeholder="Ask about your pipeline data..." [disabled]="isLoading" />
        <button (click)="sendMessage()" [disabled]="isLoading || !userInput.trim()">Send</button>
      </div>
    </div>
  `,
  styles: [`
    .chat-fab {
      position: fixed; bottom: 24px; right: 24px; z-index: 1000;
      width: 56px; height: 56px; border-radius: 50%;
      background: linear-gradient(135deg, #1a237e, #7c4dff);
      color: white; border: none; font-size: 18px; font-weight: 700;
      cursor: pointer; box-shadow: 0 4px 16px rgba(26,35,126,0.4);
      transition: all 0.3s; display: flex; align-items: center; justify-content: center;
    }
    .chat-fab:hover { transform: scale(1.1); }
    .chat-fab.active { background: #455a64; font-size: 24px; }

    .chat-panel {
      position: fixed; bottom: 92px; right: 24px; z-index: 999;
      width: 420px; height: 560px; background: white;
      border-radius: 16px; box-shadow: 0 8px 32px rgba(0,0,0,0.2);
      display: flex; flex-direction: column; overflow: hidden;
    }

    .chat-header {
      background: linear-gradient(135deg, #1a237e, #283593);
      color: white; padding: 16px; flex-shrink: 0;
      display: flex; justify-content: space-between; align-items: center;
    }
    .chat-header-left { display: flex; align-items: center; gap: 12px; }
    .chat-header-left strong { font-size: 15px; display: block; }
    .chat-header-left small { font-size: 11px; opacity: 0.8; }
    .ai-icon {
      width: 36px; height: 36px; border-radius: 50%;
      background: rgba(255,255,255,0.2); display: flex;
      align-items: center; justify-content: center;
    }
    .summarize-btn {
      background: rgba(255,255,255,0.2); color: white; border: 1px solid rgba(255,255,255,0.3);
      padding: 5px 12px; border-radius: 14px; font-size: 11px; cursor: pointer;
      font-weight: 500; transition: all 0.2s;
    }
    .summarize-btn:hover { background: rgba(255,255,255,0.3); }
    .summarize-btn:disabled { opacity: 0.5; cursor: default; }

    .api-key-banner {
      background: #fff3e0; padding: 10px 14px; border-bottom: 1px solid #ffe0b2; flex-shrink: 0;
    }
    .api-key-row { display: flex; gap: 6px; align-items: center; }
    .api-key-input {
      flex: 1; padding: 6px 10px; border: 1px solid #ddd; border-radius: 6px;
      font-size: 12px; outline: none;
    }
    .api-key-input:focus { border-color: #7c4dff; }
    .connect-btn {
      padding: 6px 12px; background: #1a237e; color: white; border: none;
      border-radius: 6px; font-size: 11px; cursor: pointer; font-weight: 500;
    }
    .connect-btn:disabled { background: #ccc; }
    .dismiss-btn {
      background: none; border: none; font-size: 16px; color: #999; cursor: pointer; padding: 0 4px;
    }
    .api-key-hint { display: block; font-size: 10px; color: #999; margin-top: 4px; }
    .api-key-error { display: block; font-size: 11px; color: #c62828; margin-top: 4px; font-weight: 500; }
    .connected-banner {
      background: #e8f5e9; color: #2e7d32; padding: 8px 14px; text-align: center;
      font-size: 12px; font-weight: 500; flex-shrink: 0;
      animation: fadeOut 3s forwards; animation-delay: 2s;
    }
    @keyframes fadeOut { to { opacity: 0; height: 0; padding: 0; overflow: hidden; } }

    .chat-messages {
      flex: 1; overflow-y: auto; padding: 16px;
      display: flex; flex-direction: column; gap: 12px;
    }

    .msg { display: flex; }
    .msg.user { justify-content: flex-end; }
    .msg.assistant { justify-content: flex-start; }

    .msg-bubble {
      max-width: 85%; padding: 10px 14px; border-radius: 12px;
      font-size: 13px; line-height: 1.5; word-wrap: break-word;
    }
    .msg.user .msg-bubble {
      background: #1a237e; color: white; border-bottom-right-radius: 4px;
    }
    .msg.assistant .msg-bubble {
      background: #f5f5f5; color: #333; border-bottom-left-radius: 4px;
    }
    .msg.assistant .msg-bubble h2 { font-size: 15px; margin: 12px 0 6px; color: #1a237e; }
    .msg.assistant .msg-bubble h3 { font-size: 14px; margin: 10px 0 4px; color: #283593; }
    .msg.assistant .msg-bubble h4 { font-size: 13px; margin: 8px 0 4px; color: #3949ab; }
    .msg.assistant .msg-bubble ul, .msg.assistant .msg-bubble ol {
      margin: 6px 0; padding-left: 24px !important;
    }
    .msg.assistant .msg-bubble ul { list-style-type: disc !important; }
    .msg.assistant .msg-bubble ol { list-style-type: decimal !important; }
    .msg.assistant .msg-bubble li { margin: 4px 0; display: list-item !important; }
    .msg.assistant .msg-bubble p { margin: 2px 0; }
    .msg.assistant .msg-bubble pre {
      background: #263238; color: #e0e0e0; padding: 8px 10px;
      border-radius: 6px; overflow-x: auto; font-size: 11px; margin: 6px 0;
    }
    .msg.assistant .msg-bubble code {
      background: #e8eaf6; padding: 1px 4px; border-radius: 3px; font-size: 12px;
    }
    .msg.assistant .msg-bubble pre code {
      background: none; padding: 0; font-size: 11px;
    }
    .msg.assistant .msg-bubble hr {
      border: none; border-top: 1px solid #ddd; margin: 8px 0;
    }
    .msg.assistant .msg-bubble strong { font-weight: 600; }

    .suggestions {
      display: flex; flex-wrap: wrap; gap: 6px; margin-top: 4px;
    }
    .suggestion-chip {
      background: #e8eaf6; color: #1a237e; border: 1px solid #c5cae9;
      border-radius: 16px; padding: 6px 12px; font-size: 12px;
      cursor: pointer; transition: all 0.2s;
    }
    .suggestion-chip:hover { background: #c5cae9; }

    .typing { display: flex; gap: 4px; align-items: center; padding: 12px 16px; }
    .dot {
      width: 8px; height: 8px; background: #999; border-radius: 50%;
      animation: bounce 1.4s infinite ease-in-out;
    }
    .dot:nth-child(2) { animation-delay: 0.2s; }
    .dot:nth-child(3) { animation-delay: 0.4s; }
    @keyframes bounce {
      0%, 80%, 100% { transform: scale(0.6); opacity: 0.4; }
      40% { transform: scale(1); opacity: 1; }
    }

    .chat-input {
      display: flex; padding: 12px; border-top: 1px solid #eee; gap: 8px; flex-shrink: 0;
    }
    .chat-input input {
      flex: 1; padding: 10px 14px; border: 1px solid #ddd;
      border-radius: 20px; font-size: 13px; outline: none;
    }
    .chat-input input:focus { border-color: #7c4dff; }
    .chat-input button {
      padding: 8px 18px; background: #1a237e; color: white;
      border: none; border-radius: 20px; font-size: 13px;
      cursor: pointer; font-weight: 500;
    }
    .chat-input button:disabled { background: #ccc; cursor: default; }
  `]
})
/**
 * Floating AI chat panel available on all pages, with API key configuration,
 * suggested questions, and markdown rendering for responses.
 */
export class ChatCopilotComponent implements OnInit {
  isOpen = false;
  isLoading = false;
  userInput = '';
  messages: { role: string; content: string }[] = [];

  aiConfigured = false;
  apiKeyInput = '';
  keyDismissed = false;
  showConnectedBanner = false;
  isConnecting = false;
  apiKeyError = '';

  suggestedQuestions = [
    'Summarize the pipeline condition',
    'Which anomalies need repair before 2028?',
    'Compare growth between runs',
    'What are the top 5 risks?',
    'When should the next inspection be?',
  ];

  constructor(private api: ApiService) {}

  /** Checks if the AI backend is already configured on component init. */
  ngOnInit(): void {
    this.api.getAiStatus().subscribe({
      next: (res: any) => { this.aiConfigured = res.configured; },
      error: () => {}
    });
  }

  /** Toggles the chat panel open/closed. */
  toggleChat(): void {
    this.isOpen = !this.isOpen;
  }

  /** Validates and sets the xAI API key at runtime; shows a connected banner on success. */
  connectApiKey(): void {
    const key = this.apiKeyInput.trim();
    if (!key) return;
    this.apiKeyError = '';
    this.isConnecting = true;
    this.api.setApiKey(key).subscribe({
      next: (res: any) => {
        this.isConnecting = false;
        this.aiConfigured = res.configured;
        if (res.configured) {
          this.apiKeyInput = '';
          this.apiKeyError = '';
          this.showConnectedBanner = true;
          setTimeout(() => { this.showConnectedBanner = false; }, 5000);
        } else {
          this.apiKeyError = res.error || 'Invalid API key. Make sure it starts with "xai-".';
        }
      },
      error: () => {
        this.isConnecting = false;
        this.apiKeyError = 'Connection failed. Check that the backend is running.';
      }
    });
  }

  /** Sends a pre-built comprehensive summary prompt to the AI copilot. */
  summarizeResults(): void {
    this.sendMessage('Provide a comprehensive summary of all pipeline analysis results, including key findings, risk hotspots, growth trends, and recommended actions.');
  }

  /** Sends the user message with chat history to the AI copilot endpoint. */
  sendMessage(text?: string): void {
    const msg = text || this.userInput.trim();
    if (!msg) return;

    this.messages.push({ role: 'user', content: msg });
    this.userInput = '';
    this.isLoading = true;

    this.api.chat(msg, this.messages.slice(0, -1)).subscribe({
      next: (res: any) => {
        this.messages.push({ role: 'assistant', content: res.response });
        this.isLoading = false;
      },
      error: () => {
        this.messages.push({ role: 'assistant', content: 'Sorry, I encountered an error. Please try again.' });
        this.isLoading = false;
      }
    });
  }

  /**
   * Converts markdown to HTML with support for code blocks, headers, lists, bold/italic, and inline code.
   * Uses a placeholder-based approach to protect code blocks during processing.
   */
  formatMessage(text: string): string {
    if (!text) return '';

    // Normalize line endings
    text = text.replace(/\r\n/g, '\n');

    // 1. Protect code blocks
    const codeBlocks: string[] = [];
    text = text.replace(/```(\w*)\n?([\s\S]*?)```/g, (_match, _lang, code) => {
      codeBlocks.push(`<pre><code>${this._escapeHtml(code.trim())}</code></pre>`);
      return `%%CODEBLOCK_${codeBlocks.length - 1}%%`;
    });

    // 2. Protect inline code
    const inlineCode: string[] = [];
    text = text.replace(/`([^`]+)`/g, (_match, code) => {
      inlineCode.push(`<code>${this._escapeHtml(code)}</code>`);
      return `%%INLINE_${inlineCode.length - 1}%%`;
    });

    // 3. Pre-process: remove blank lines between consecutive list items
    //    so that "1. foo\n\n2. bar" becomes "1. foo\n2. bar"
    text = text.replace(/^(\d+\.\s+.+)\n\n+(?=\d+\.\s+)/gm, '$1\n');
    text = text.replace(/^([-*]\s+.+)\n\n+(?=[-*]\s+)/gm, '$1\n');

    // 4. Process line by line
    const lines = text.split('\n');
    const processed: string[] = [];
    let inList = false;
    let listType = '';

    const isOlItem = (l: string) => /^\d+\.\s+/.test(l);
    const isUlItem = (l: string) => /^[-*]\s+/.test(l);

    for (let i = 0; i < lines.length; i++) {
      const line = lines[i];

      // Horizontal rule
      if (/^---+$/.test(line.trim())) {
        if (inList) { processed.push(listType === 'ul' ? '</ul>' : '</ol>'); inList = false; }
        processed.push('<hr>');
        continue;
      }

      // Headers (#### before ### before ## before #)
      const headerMatch = line.match(/^(#{1,4})\s+(.+)/);
      if (headerMatch) {
        if (inList) { processed.push(listType === 'ul' ? '</ul>' : '</ol>'); inList = false; }
        const level = Math.min(headerMatch[1].length + 1, 4); // # → h2, ## → h3, ### → h4, #### → h4
        processed.push(`<h${level}>${headerMatch[2]}</h${level}>`);
        continue;
      }

      // Ordered list
      if (isOlItem(line)) {
        if (!inList || listType !== 'ol') {
          if (inList) processed.push(listType === 'ul' ? '</ul>' : '</ol>');
          processed.push('<ol>');
          inList = true; listType = 'ol';
        }
        processed.push(line.replace(/^\d+\.\s+(.+)/, '<li>$1</li>'));
        continue;
      }

      // Unordered list
      if (isUlItem(line)) {
        if (!inList || listType !== 'ul') {
          if (inList) processed.push(listType === 'ul' ? '</ul>' : '</ol>');
          processed.push('<ul>');
          inList = true; listType = 'ul';
        }
        processed.push(line.replace(/^[-*]\s+(.+)/, '<li>$1</li>'));
        continue;
      }

      // Non-list line: close any open list
      if (inList) {
        processed.push(listType === 'ul' ? '</ul>' : '</ol>');
        inList = false;
      }

      // Empty line or regular text
      if (line.trim() === '') {
        processed.push('<br>');
      } else {
        processed.push(`<p>${line}</p>`);
      }
    }
    if (inList) processed.push(listType === 'ul' ? '</ul>' : '</ol>');

    let result = processed.join('');

    // 5. Inline formatting
    result = result.replace(/\*\*(.+?)\*\*/g, '<strong>$1</strong>');
    result = result.replace(/\*(.+?)\*/g, '<em>$1</em>');

    // 6. Restore code blocks and inline code
    codeBlocks.forEach((block, i) => {
      result = result.replace(`%%CODEBLOCK_${i}%%`, block);
    });
    inlineCode.forEach((code, i) => {
      result = result.replace(`%%INLINE_${i}%%`, code);
    });

    // 7. Clean up excessive breaks
    result = result.replace(/(<br>){3,}/g, '<br><br>');

    return result;
  }

  /** Escapes HTML entities in code block content to prevent XSS. */
  private _escapeHtml(text: string): string {
    return text
      .replace(/&/g, '&amp;')
      .replace(/</g, '&lt;')
      .replace(/>/g, '&gt;');
  }
}
